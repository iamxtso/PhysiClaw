"""Tests for the dual-link fallback policy (no devices needed).

Eye failover (Perception, sticky with recovery probes) and hand failover
(orchestrator `_execute`, retried once on the next usable hand) run
against a real HardwareRig wired with spec'd doubles and real mappings.
"""

from __future__ import annotations

import numpy as np
import pytest

from physiclaw.core.calibration import Calibration
from physiclaw.core.hardware.device import DeviceTimeout
from physiclaw.core.hardware.scrcpy import ScrcpyArm
from physiclaw.core.orchestration import gestures
from physiclaw.core.orchestration.orchestrator import PhysiClaw
from physiclaw.core.orchestration.perception import Perception
from physiclaw.core.orchestration.rig import (
    PHYSICAL_BACKEND,
    SCRCPY_BACKEND,
    HardwareRig,
    _BackendState,
)

DIAG = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
PIXEL = np.array([[1024.0, 0.0, 0.0], [0.0, 576.0, 0.0]])
BLACK = np.zeros((16, 16, 3), dtype=np.uint8)
GOOD = np.random.default_rng(7).integers(0, 256, size=(16, 16, 3)).astype(np.uint8)


def _dual_rig(arm_double, cam_double, *, eye=SCRCPY_BACKEND, hand=PHYSICAL_BACKEND):
    """Both links connected and mapped; eye/hand placeable independently."""
    rig = HardwareRig()
    scrcpy_arm, phys_arm = arm_double(), arm_double()
    scrcpy_cam, phys_cam = cam_double(), cam_double()
    scrcpy_cam.snapshot.return_value = BLACK.copy()
    phys_cam.snapshot.return_value = GOOD.copy()
    rig.calibration = Calibration()
    if eye == SCRCPY_BACKEND:
        rig._cam = scrcpy_cam
        rig.calibration.pct_to_cam = np.eye(2, 3)
        rig.calibration.cam_size = (1024, 576)
        rig.calibration.cam_rotation = -1
        rig._backends[PHYSICAL_BACKEND] = _BackendState(
            cam=phys_cam, pct_to_cam=np.eye(2, 3), cam_size=(1920, 1080), cam_rotation=0
        )
    else:
        rig._cam = phys_cam
        rig.calibration.pct_to_cam = np.eye(2, 3)
        rig.calibration.cam_size = (1920, 1080)
        rig.calibration.cam_rotation = 0
        rig._backends[SCRCPY_BACKEND] = _BackendState(
            cam=scrcpy_cam,
            pct_to_cam=np.eye(2, 3),
            cam_size=(1024, 576),
            cam_rotation=-1,
        )
    rig._eye = eye
    if hand == PHYSICAL_BACKEND:
        rig._arm = phys_arm
        rec = rig._backends.setdefault(SCRCPY_BACKEND, _BackendState())
        if rec.arm is None:
            rec.arm = scrcpy_arm
        rig.calibration.pct_to_grbl = DIAG.copy()
        rig._origin_pinned = True
        rec.pct_to_grbl = PIXEL.copy()
        rec.pinned = True
    else:
        rig._arm = scrcpy_arm
        rig.calibration.pct_to_grbl = PIXEL.copy()
        rig._origin_pinned = True
        rig._backends[PHYSICAL_BACKEND] = _BackendState(
            arm=phys_arm, pct_to_grbl=DIAG.copy(), pinned=True
        )
    rig._hand = hand
    return rig


# ─── eye failover ──────────────────────────────────────────────


def test_eye_fails_over_on_blackout_streak(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=SCRCPY_BACKEND)
    p = Perception(rig)
    rig.acquire()
    try:
        assert p.camera_view() is not None  # miss 1: black shipped
        assert p.camera_view() is not None  # miss 2: black shipped
        frame = p.camera_view()  # miss 3: switch + fresh grab
        assert rig.active_eye == PHYSICAL_BACKEND
        np.testing.assert_array_equal(frame, GOOD)
    finally:
        rig.release()


def test_eye_no_failover_without_lock(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=SCRCPY_BACKEND)
    p = Perception(rig)
    for _ in range(5):
        p.camera_view()
    assert rig.active_eye == SCRCPY_BACKEND


def test_eye_reprobe_recovers_preferred(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND)
    p = Perception(rig)
    p._eye_good = Perception.EYE_REPROBE_EVERY - 1
    rig.acquire()
    try:
        p.camera_view()
        # scrcpy parked cam serves BLACK → flips straight back.
        assert rig.active_eye == PHYSICAL_BACKEND
    finally:
        rig.release()


def test_eye_reprobe_moves_back_on_good_frame(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND)
    rig._backends[SCRCPY_BACKEND].cam.snapshot.return_value = GOOD.copy()
    p = Perception(rig)
    p._eye_good = Perception.EYE_REPROBE_EVERY - 1
    rig.acquire()
    try:
        p.camera_view()
        assert rig.active_eye == SCRCPY_BACKEND
    finally:
        rig.release()


# ─── hand failover ─────────────────────────────────────────────


def _hand_rig(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND, hand=PHYSICAL_BACKEND)
    rig._backends[SCRCPY_BACKEND].cam = None  # hands only for this test
    return rig


def test_hand_fails_over_once_on_timeout(arm_double, cam_double):
    rig = _hand_rig(arm_double, cam_double)
    phys_arm = rig._arm
    scrcpy_arm = rig._backends[SCRCPY_BACKEND].arm
    phys_arm.tap.side_effect = [DeviceTimeout("link dead"), None]
    p = PhysiClaw()
    p.rig = rig
    rig.acquire()
    try:
        assert (
            p._execute(gestures.Tap([0.4, 0.5, 0.6, 0.7]))
            == "Tapped at bbox [0.4, 0.5, 0.6, 0.7]"
        )
        assert rig.active_hand == SCRCPY_BACKEND
        scrcpy_arm.tap.assert_called_once_with()
        # Sticky: the next gesture goes straight to scrcpy.
        p._execute(gestures.Tap([0.4, 0.5, 0.6, 0.7]))
        assert scrcpy_arm.tap.call_count == 2
        assert phys_arm.tap.call_count == 1
    finally:
        rig.release()


def test_hand_reraises_without_alternate(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND, hand=PHYSICAL_BACKEND)
    rig._backends[SCRCPY_BACKEND].arm = None  # no alternate hand
    rig._arm.tap.side_effect = DeviceTimeout("link dead")
    p = PhysiClaw()
    p.rig = rig
    rig.acquire()
    try:
        with pytest.raises(DeviceTimeout):
            p._execute(gestures.Tap([0.4, 0.5, 0.6, 0.7]))
    finally:
        rig.release()


# ─── hand recovery probing ─────────────────────────────────────


def test_hand_recovers_on_live_probe(arm_double, cam_double):
    rig = _hand_rig(arm_double, cam_double)
    phys_arm = rig._arm
    phys_arm.tap.side_effect = DeviceTimeout("link dead")
    p = PhysiClaw()
    p.rig = rig
    p._gestures_since_probe = PhysiClaw.HAND_REPROBE_EVERY - 1
    p._last_hand_probe = 0.0  # wall-clock gate open (monotonic epoch)
    rig.acquire()
    try:
        # Probe runs first: the parked scrcpy hand answers health() →
        # recovery moves there before the gesture, so no tap ever fails.
        p._execute(gestures.Tap([0.4, 0.5, 0.6, 0.7]))
        assert rig.active_hand == SCRCPY_BACKEND
        assert phys_arm.tap.call_count == 0
    finally:
        rig.release()


def test_hand_probe_moves_back_when_preferred_answers(arm_double, cam_double):
    rig = _hand_rig(arm_double, cam_double)
    rig._hand = SCRCPY_BACKEND
    rig._arm = rig._backends[SCRCPY_BACKEND].arm
    p = PhysiClaw()
    p.rig = rig
    p._gestures_since_probe = PhysiClaw.HAND_REPROBE_EVERY - 1
    p._last_hand_probe = 0.0
    rig.acquire()
    try:
        p._maybe_recover_hand()
        # Preferred scrcpy hand is a MagicMock (truthy health) → moves back.
        assert rig.active_hand == SCRCPY_BACKEND
    finally:
        rig.release()


def test_hand_probe_stays_when_preferred_silent(arm_double, cam_double):
    rig = _hand_rig(arm_double, cam_double)
    rig._hand = SCRCPY_BACKEND
    rig._arm = rig._backends[SCRCPY_BACKEND].arm
    rig._backends[PHYSICAL_BACKEND] = _BackendState()  # preferred gone
    p = PhysiClaw()
    p.rig = rig
    # Preferred per default order is scrcpy — which IS active: no-op.
    p._maybe_recover_hand()
    assert rig.active_hand == SCRCPY_BACKEND


def test_probe_hand_absent_is_false(arm_double, cam_double):
    rig = HardwareRig()
    assert rig.probe_hand(SCRCPY_BACKEND) is False
    assert rig.probe_hand(PHYSICAL_BACKEND) is False


def test_status_degraded_flags(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND, hand=PHYSICAL_BACKEND)
    rig._backends.clear()  # pure physical rig: nothing better available
    out = rig.status()
    assert out["eye_degraded"] is False
    assert out["hand_degraded"] is False


def test_status_degraded_when_preferred_usable(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND, hand=PHYSICAL_BACKEND)
    out = rig.status()  # scrcpy parked + mapped on both channels
    assert out["eye_degraded"] is True
    assert out["hand_degraded"] is True
    rig._backends[SCRCPY_BACKEND].cam = None  # eye alternate gone
    assert rig.status()["eye_degraded"] is False


# ─── liveness probes ───────────────────────────────────────────


class _StubSession:
    video_size = (1024, 576)

    def __init__(self, alive=True):
        self._alive = alive

    def check_control_open(self):
        if not self._alive:
            raise DeviceTimeout("control socket closed")


def test_probe_hand_scrcpy_live(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND, hand=PHYSICAL_BACKEND)
    rig._backends[SCRCPY_BACKEND].arm = ScrcpyArm(_StubSession(alive=True))
    rig._backends[SCRCPY_BACKEND].session = rig._backends[SCRCPY_BACKEND].arm._session
    rig._backends[SCRCPY_BACKEND].cam.health.return_value = True
    assert rig.probe_hand(SCRCPY_BACKEND) is True


def test_probe_hand_scrcpy_dead_control(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND, hand=PHYSICAL_BACKEND)
    rig._backends[SCRCPY_BACKEND].arm = ScrcpyArm(_StubSession(alive=False))
    rig._backends[SCRCPY_BACKEND].session = rig._backends[SCRCPY_BACKEND].arm._session
    assert rig.probe_hand(SCRCPY_BACKEND) is False


def test_probe_hand_grbl_uses_health(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=PHYSICAL_BACKEND, hand=PHYSICAL_BACKEND)
    rig._arm.health.return_value = True
    assert rig.probe_hand(PHYSICAL_BACKEND) is True
    rig._arm.health.return_value = False
    assert rig.probe_hand(PHYSICAL_BACKEND) is False


def test_eye_fails_over_on_stale_stream(arm_double, cam_double):
    rig = _dual_rig(arm_double, cam_double, eye=SCRCPY_BACKEND)
    rig._cam.frame_age.return_value = 99.0  # stalled but serving old pixels
    p = Perception(rig)
    rig.acquire()
    try:
        p.camera_view()
        p.camera_view()
        frame = p.camera_view()
        assert rig.active_eye == PHYSICAL_BACKEND
        np.testing.assert_array_equal(frame, GOOD)
    finally:
        rig.release()


def test_hand_recovery_bounces_dead_preferred_hand(arm_double):
    """probe silent + revive heals → moves back with exactly one bounce."""
    from unittest.mock import MagicMock

    class _FlappingSession:
        def __init__(self):
            self.bounces = 0

        def check_control_open(self):
            if self.bounces == 0:
                raise DeviceTimeout("control dead")

        def restart(self):
            self.bounces += 1

    rig = HardwareRig()
    rig._arm = arm_double()
    rig.calibration = Calibration()
    rig.calibration.pct_to_grbl = DIAG.copy()
    rig._origin_pinned = True
    rig._backends[SCRCPY_BACKEND] = _BackendState(
        arm=MagicMock(spec=ScrcpyArm),
        pct_to_grbl=PIXEL.copy(),
        pinned=True,
    )
    rig._backends[SCRCPY_BACKEND].session = _FlappingSession()
    rig._hand = PHYSICAL_BACKEND
    p = PhysiClaw()
    p.rig = rig
    p._gestures_since_probe = PhysiClaw.HAND_REPROBE_EVERY - 1
    p._last_hand_probe = 0.0
    rig.acquire()
    try:
        p._maybe_recover_hand()
        assert rig.active_hand == SCRCPY_BACKEND
        assert rig._backends[SCRCPY_BACKEND].session.bounces == 1
    finally:
        rig.release()
