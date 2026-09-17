"""Perception — the seeing side of the orchestration layer.

Owns the lazily-loaded detectors (OCR, icons), the camera-frame
acquisition helpers, the phone-screen watchdog, and the startup camera
settle (exposure tune). It never gestures: it borrows the rig for
devices, transforms, parking, and the busy lock, and hands
frames/element listings back to whoever asked (the orchestrator's tool
ops, the observer's callbacks, the /api/phone/watch route).

Focus deliberately has NO machinery here — the lens position is a
calibration constant (see hardware/focus.py).

Pixel work itself still lives in physiclaw.core.vision — this class
only coordinates it.
"""

import logging
import threading
import time

from physiclaw.common.config import CONFIG
from physiclaw.common.listing import format_elements
from physiclaw.core.hardware import exposure
from physiclaw.core.hardware.frame_reader import FrameReader
from physiclaw.core.orchestration.rig import HardwareRig
from physiclaw.core.vision import quality
from physiclaw.core.vision.blackout import is_blackout
from physiclaw.core.vision.icon_detect import IconDetector
from physiclaw.core.vision.ocr import OCRReader, results_to_elements
from physiclaw.core.vision.preprocess import (
    crop_to_phone_screen,
    phone_screen_crop_box,
)
from physiclaw.core.vision.ui_elements import detect_ui_elements
from physiclaw.core.vision.util import bbox_on_screen, find_numpad_digit
from physiclaw.core.vision.watchdog import Watchdog

log = logging.getLogger(__name__)


class _NullOCR:
    """Stand-in reader after OCR construction failed — `detect` keeps its
    icons channel without re-attempting the heavy model load (and
    re-warning) on every frame. Deliberately NOT installed in the
    `_ocr_reader` cache slot: the keypad poll (`_ocr_elements`) must
    keep raising loudly."""

    def read(self, frame, crop_box=None) -> list:
        return []


class Perception:
    """Detectors + frame acquisition over a borrowed rig."""

    # Delay before a scheduled settle task (exposure re-tune) contends
    # for the hardware: the view that triggered it still holds the rig
    # lock while its gesture finishes parking.
    RETUNE_DELAY_SECONDS = 2.0

    # Spacing between keypad-detection OCR passes. The first pass is
    # immediate — the keypad may already be up and is short-lived — then
    # after each miss the poll waits this long and tries again.
    NUMPAD_OCR_INTERVAL_SECONDS = 1.0

    # Long-edge cap for the keypad-detection OCR input — much smaller than
    # the LLM-view budget (CONFIG.compact.max_image_edge_px, ~1566) the
    # other detect paths use. OCR cost scales with pixels and the keypad
    # lives only seconds, so the input is shrunk hard; the big,
    # high-contrast passcode digits survive the downscale where fine app
    # text would not. Tune against the poll's per-pass log: if the keypad
    # stops reading (few text elems, always a miss), raise this.
    NUMPAD_OCR_MAX_EDGE = 900

    # Eye-failover streak: consecutive blind views (dropped frame or secure
    # blackout) before switching to the next usable eye. Blindness is sticky
    # per frame, so 1 would flap on a decoder glitch; 3 rides those out.
    EYE_FAILOVER_STREAK = 3
    # While degraded (active eye isn't the preferred one), every Nth good
    # view probes the preferred eye with one frame and moves back on a good
    # one — self-healing without ping-pong (probes need evidence both ways).
    EYE_REPROBE_EVERY = 10

    def __init__(self, rig: HardwareRig):
        self._rig = rig
        # Serialize first-use model construction (the watch route and a
        # concurrent tool op could otherwise double-load a heavy model).
        # One lock per detector so loading one model never blocks a
        # caller that wants the other, already-cached one.
        self._ocr_init_lock = threading.Lock()
        self._icon_init_lock = threading.Lock()
        self._ocr_reader: OCRReader | None = None
        self._icon_detector: IconDetector | None = None
        # Set once when OCR construction fails — `detect` degrades to
        # icons-only from then on instead of re-loading models per frame.
        self._ocr_error: str | None = None
        self._watchdog = Watchdog()
        self._last_tune: exposure.TuneResult | None = None
        # Single-flight guard for the background re-tune.
        self._pending_lock = threading.Lock()
        self._retune_pending = False
        # Eye-failover state (see camera_view): consecutive blind views, and
        # good views since the last recovery probe while degraded.
        self._eye_misses = 0
        self._eye_good = 0

    # ─── Lazy detectors ───────────────────────────────────────

    def ocr_reader(self) -> OCRReader:
        """Lazy-load and cache the OCR reader. Double-checked: the warm
        path returns the cached instance without touching the lock."""
        if self._ocr_reader is None:
            with self._ocr_init_lock:
                if self._ocr_reader is None:
                    self._ocr_reader = OCRReader()
        return self._ocr_reader

    def icon_detector(self) -> IconDetector:
        """Lazy-load and cache the icon detector (see ocr_reader)."""
        if self._icon_detector is None:
            with self._icon_init_lock:
                if self._icon_detector is None:
                    self._icon_detector = IconDetector()
        return self._icon_detector

    # ─── Frame acquisition ────────────────────────────────────

    def _frame_is_stale(self, cam) -> bool:
        """Whether the camera's latest frame is too old to trust.

        `fresh_frame` hides staleness by design (stale fallback beats
        nothing for a single view); failover must see through it, or a
        stalled stream looks like a frozen screen forever. Mock-safe:
        doubles without a real age API (spec'd MagicMocks return mocks
        from it) opt out instead of exploding the comparison.
        """
        age_fn = getattr(cam, "frame_age", None)
        if not callable(age_fn):
            return False
        try:
            age = age_fn()
            return age is not None and age > FrameReader.FRESH_MAX_AGE_SECONDS
        except Exception:
            return False

    def _holding_lock(self) -> bool:
        """Whether this thread holds the rig lock — failover mutates the
        active eye (aliases + mapping halves) and is only safe serialized.
        Background paths (watchdog peek, retune meter) read whatever eye is
        active and never switch."""
        try:
            self._rig.assert_locked()
        except RuntimeError:
            return False
        return True

    def _note_eye_miss(self) -> bool:
        """Count one blind view; fail over at the streak threshold.

        Returns True when the eye actually switched (caller re-grabs once).
        No usable alternate: reset and report False, behaving exactly like
        the pre-failover code (raise on None, ship the frame on blackout).
        """
        self._eye_misses += 1
        if self._eye_misses < self.EYE_FAILOVER_STREAK:
            return False
        target = self._rig.next_eye()
        self._eye_misses = 0
        if target is None:
            return False
        old = self._rig.active_eye
        self._rig.use_eye(target)
        self._eye_good = 0
        log.warning("eye failover %s -> %s (blind streak)", old, target)
        return True

    def _maybe_reprobe_eye(self) -> None:
        """While degraded, probe the next eye every Nth good view and move
        back on a good frame. The probe rides use_eye (public, logged), so
        a still-blind preferred eye just flips straight back — one wasted
        frame per interval, no stranding either way."""
        if self._rig.active_eye == self._rig.preferred_eye:
            return  # not degraded — nothing to recover
        degraded_from = self._rig.active_eye
        target = self._rig.next_eye()
        if target is None or target != self._rig.preferred_eye:
            return
        self._eye_good += 1
        if self._eye_good < self.EYE_REPROBE_EVERY:
            return
        self._eye_good = 0
        try:
            self._rig.use_eye(target)
            probe = self._rig.require_cam().snapshot()
        except Exception:
            return  # preferred eye still down — stay degraded silently
        if probe is None or is_blackout(probe):
            try:
                self._rig.use_eye(degraded_from)
            except Exception:
                pass
            return
        log.warning("eye recovered -> %s", target)

    def camera_view(self):
        """Capture a frame from the overhead camera. Returns BGR numpy array.

        Takes the frame as-is — the stylus may be visible.
        Call rig.park() first if an unobstructed view is needed.
        Frame is already rotated to portrait by the camera.
        Blind views (dropped frame, secure blackout) count toward eye
        failover; unlocked callers get frames without failover.
        """
        cam = self._rig.require_cam()
        frame = cam.snapshot()
        if not self._holding_lock():
            if frame is None:
                raise RuntimeError("Camera capture failed")
            return frame
        if frame is None or self._frame_is_stale(cam) or is_blackout(frame):
            if self._note_eye_miss():
                frame = self._rig.require_cam().snapshot()
            if frame is None:
                raise RuntimeError("Camera capture failed")
            return frame
        self._eye_misses = 0
        self._maybe_reprobe_eye()
        return frame

    def cropped_view(self):
        """The camera view cropped to the phone screen (0-1 span) — the
        frame shape `detect` expects and the observer grabs."""
        return crop_to_phone_screen(self.camera_view(), self._rig.transforms)

    # ─── Detection ────────────────────────────────────────────

    def detect(self, frame) -> tuple[str, object]:
        """Icon detection + OCR on a frame. Caller holds the lock.

        ``frame`` must already span the phone screen (0-1) — camera
        views need to be cropped via ``cropped_view`` first.
        Returns (formatted element listing, annotated frame).
        """
        # The lazy cache constructs RapidOCR here, BEFORE _detect_texts'
        # degrade guard can catch a broken install (missing/corrupt models
        # raise in __init__, not read()). Same contract applies: the text
        # channel degrades, icons must survive. The failure is memoized —
        # warned once, then a null reader per frame instead of re-loading
        # models on every peek. The keypad poll deliberately keeps
        # raising — its caller needs text or nothing.
        if self._ocr_error is not None:
            ocr: OCRReader | _NullOCR = _NullOCR()
        else:
            try:
                ocr = self.ocr_reader()
            except Exception as ex:
                self._ocr_error = str(ex)
                log.warning("OCR reader unavailable — detecting icons only: %s", ex)
                ocr = _NullOCR()
        elements, annotated = detect_ui_elements(
            frame,
            icon_detector=self.icon_detector(),
            ocr_reader=ocr,
        )
        return format_elements(e.to_element() for e in elements), annotated

    def _ocr_elements(self, frame, max_edge: int | None = None) -> list[dict]:
        """OCR one already-grabbed frame into on-screen text elements —
        used by the keypad poll, which must OCR the exact frame it
        metered, not a fresh grab. Derives the crop box from the frame
        itself, so callers can't pair a box with a different frame.
        ``max_edge`` (px), when set, downscales the OCR input to that long
        edge for speed (see OCRReader.read)."""
        crop_box = phone_screen_crop_box(frame, self._rig.transforms)
        results = self.ocr_reader().read(frame, crop_box=crop_box, max_edge=max_edge)
        elements = results_to_elements(results, self._rig.transforms)
        return [e for e in elements if bbox_on_screen(e["bbox"])]

    def wait_for_numpad_digit(
        self, digit: str, timeout: float = 10.0
    ) -> list[float] | None:
        """Poll the camera until the passcode keypad shows `digit`;
        return its bbox, or None at `timeout`. Caller must hold the lock.

        Deliberately simple: OCR immediately — the keypad may already be
        up and is the scarce, short-lived resource — then retry every
        NUMPAD_OCR_INTERVAL_SECONDS until found or timeout. No quality
        gating and no exposure tune: the keypad's big high-contrast digits
        read even on a dark or slightly soft crop, so trying always beats
        delaying the first OCR past the keypad's lifetime."""
        self._rig.assert_locked()
        self._rig.park()
        start = time.monotonic()
        deadline = start + timeout
        ocr_passes = 0
        while True:
            grab_start = time.monotonic()
            frame = self.camera_view()
            ocr_start = time.monotonic()
            elements = self._ocr_elements(frame, max_edge=self.NUMPAD_OCR_MAX_EDGE)
            bbox = find_numpad_digit(elements, digit)
            ocr_passes += 1
            now = time.monotonic()
            log.info(
                "numpad poll #%d @%.1fs: grab %.2fs, ocr %.2fs, %d text elems — %s",
                ocr_passes,
                grab_start - start,
                ocr_start - grab_start,
                now - ocr_start,
                len(elements),
                "found" if bbox is not None else "miss",
            )
            if bbox is not None:
                log.info(
                    "numpad digit %r found in %.1fs (%d OCR pass(es))",
                    digit,
                    time.monotonic() - start,
                    ocr_passes,
                )
                return bbox
            if time.monotonic() >= deadline:
                break
            time.sleep(self.NUMPAD_OCR_INTERVAL_SECONDS)
        log.info(
            "numpad digit %r not found — gave up after %.1fs (%d OCR pass(es))",
            digit,
            time.monotonic() - start,
            ocr_passes,
        )
        return None

    # ─── Watchdog ────────────────────────────────────────────

    def watch(self) -> dict:
        """Poll the camera for wake events. Returns ``{"wake": bool, "reason": str}``."""
        with self._rig.engaged():
            frame = self._rig.require_cam().peek()
            if frame is None:
                return {"wake": False, "reason": ""}
            return self._watchdog.poll(frame, self._rig.transforms)

    # ─── Camera settle (exposure tune) ───────────────────────

    def _screen_crop(self):
        """Settled phone-screen crop for the tune meter, or None.

        The one home for the meter's acquisition contract: wait out the
        driver's property-apply latency (a few fresh frames, bounded),
        take the latest frame, crop to the phone screen — never meter
        the whole frame, the dark desk around the phone would dominate
        the statistics."""
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None:
            return None
        if not cam.wait_frames(exposure.SETTLE_FRAMES, timeout=5.0):
            return None
        frame = cam.peek()
        if frame is None:
            return None
        return crop_to_phone_screen(frame, t)

    def _with_rig_parked(self, name: str, body) -> bool:
        """Acquire the rig, park the arm (a stylus in frame would
        pollute any meter), run `body`, always release. Returns False
        on a busy skip OR a failure — the rig-lock discipline for every
        settle entry point, defined once. The catch below is what makes
        the settle paths' "fail-open, never raises" claim actually true:
        `park()` is a serial arm move that can raise (timeout, alarm),
        and a settle must never take readiness or a session down with it."""
        try:
            self._rig.acquire()
        except RuntimeError:
            log.info("%s: hardware busy — skipped", name)
            return False
        try:
            self._rig.park()
            body()
            return True
        except Exception:
            log.exception("%s: failed — leaving camera as-is", name)
            return False
        finally:
            self._rig.release()

    def settle_camera(self) -> None:
        """Startup settle for every ready-flipping path (warm start,
        the /api/ready flip): verify/converge exposure. Fail-open.
        Focus needs no settling — the lens is pinned from the moment
        the camera opens."""
        self.tune_exposure()

    def tune_exposure(self) -> None:
        """Standalone verify-and-converge for camera exposure — the
        background re-tune target. Fail-open: never raises; skipped
        when the hardware is busy (the trigger conditions persist, so
        it gets re-scheduled).

        Meters the PHONE-SCREEN crop, never the whole frame (see
        `_screen_crop`). macOS is tunable only when the UVC helper
        controls the camera (AVFoundation ignores exposure props);
        otherwise it exits at `exposure_tunable` and firmware AE stays
        in charge. Worst case a few seconds, bounded by `wait_frames`
        timeouts. A failed tune reverts to auto — the runtime quality
        monitor keeps warning the agent, so nothing is lost."""
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None or not cam.exposure_tunable:
            return
        self._with_rig_parked("exposure tune", self.tune_now)

    def tune_now(self) -> None:
        """Verify-and-converge with the rig lock ALREADY HELD and the arm
        parked — the synchronous path: the observer calls this mid-grab
        whenever `needs_inline_fix` flags a view (mis-exposed with no
        hold, a held scene that moved, or an in-band view clearing a
        stale hold), so the
        corrected frame ships to the agent instead of a mis-exposed one
        with a warning. Fail-open: never raises, no-op when the rig
        can't tune."""
        self._rig.assert_locked()
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None or not cam.exposure_tunable:
            return
        # Wall-clock the whole search: a tune meters SETTLE_FRAMES per step
        # and can block on `wait_frames` timeouts, so the elapsed is the
        # honest cost the observer paid mid-grab — worth surfacing to spot a
        # tune that's dragging out the view (e.g. a driver ignoring writes).
        started = time.monotonic()
        try:

            def meter():
                crop = self._screen_crop()
                return None if crop is None else quality.assess(crop)

            result = exposure.converge(
                meter,
                cam.set_auto_exposure,
                cam.set_manual_exposure,
                start=CONFIG.camera.exposure,
                prefer_auto=CONFIG.camera.auto_exposure,
                # The -4 ceiling costs ~16fps — affordable only under a
                # pinned lens; live AF hunts outlive the view settle
                # there (measured), so unpinned rigs keep the old -5.
                max_exposure=(
                    exposure.MAX_EXPOSURE
                    if cam.focus_pinned
                    else exposure.UNPINNED_MAX_EXPOSURE
                ),
            )
            self._last_tune = result
            log.info(
                "exposure tune: %s (%.2fs)", result.detail, time.monotonic() - started
            )
        except Exception:
            log.exception(
                "exposure tune failed after %.2fs — leaving camera as-is",
                time.monotonic() - started,
            )
            # Record a deferred sentinel: without a result the hold
            # has no state, and a flaky camera on a dark scene would
            # re-fire the multi-second tune on every judged view. No
            # baseline → dark views hold, blown views stay urgent, and
            # the first in-band view clears it.
            self._last_tune = exposure.TuneResult(
                "auto",
                None,
                False,
                "tune crashed — holding until the scene changes",
                deferred=True,
            )

    def needs_inline_fix(self, report: quality.QualityReport) -> bool:
        """Should this view's exposure be fixed before it ships?

        The single predicate behind both the observer's inline fix and
        the background safety net — and it is entirely
        `exposure.wants_retry`'s call: dark and blown views fire it,
        failed-tune holds bound it (a scene the search already proved
        unfixable doesn't re-fire until it changes), and clean views
        clear stale holds with one cheap verify-tune. One home, next
        to the defer semantics it inverts."""
        cam = self._rig.cam
        if cam is None or not cam.exposure_tunable:
            return False
        return exposure.wants_retry(self._last_tune, report)

    def on_quality_report(self, report: quality.QualityReport, streak: int) -> None:
        """Background exposure re-tune, fed every judged view by the
        GestureObserver.

        The safety net behind the observer's inline fix (same
        `needs_inline_fix` predicate): the inline path normally corrects
        the view before it ships, so it fires only when that fix
        crashed, was skipped, or didn't recover the frame. Never blocks
        the view that reported: runs on a detached thread
        (single-flight) and skips itself if the hardware is busy.

        Blur deliberately triggers nothing — focus is a calibration
        constant (see hardware/focus.py)."""
        if self.needs_inline_fix(report):
            if report.blown:
                self._schedule_retune(
                    f"washed-out view ({report.clip_pct:.0%} clip, streak {streak})"
                )
            elif report.dark:
                self._schedule_retune(
                    f"dark view (median {report.median_luma:.0f}, streak {streak})"
                )
            else:
                self._schedule_retune("in-band view — clearing a stale hold")

    def _schedule_retune(self, reason: str) -> None:
        """Run the re-tune on a detached daemon thread, at most one in
        flight. The delay lets the triggering view's gesture release the
        rig lock first; a still-busy rig makes the task skip harmlessly
        (the trigger conditions persist, so it will be re-scheduled).
        The run revalidates against `_last_tune`: if an inline fix (or
        another path) tuned while this task waited, its conclusion is
        fresher than the trigger — repeating the probe would re-learn
        the same answer seconds later."""
        with self._pending_lock:
            if self._retune_pending:
                return
            self._retune_pending = True
        log.info("exposure re-tune scheduled: %s", reason)
        token = self._last_tune

        def run() -> None:
            try:
                time.sleep(self.RETUNE_DELAY_SECONDS)
                if self._last_tune is not token:
                    log.info("exposure re-tune superseded — skipped")
                    return
                self.tune_exposure()
            finally:
                with self._pending_lock:
                    self._retune_pending = False

        threading.Thread(target=run, name="exposure re-tune", daemon=True).start()
