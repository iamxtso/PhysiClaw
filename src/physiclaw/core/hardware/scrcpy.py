"""scrcpy-backed camera + touch arm (server 4.1 wire protocol).

A first-class hardware backend beside the GRBL stylus arm and the USB
camera: ``ScrcpySession`` owns one scrcpy-server instance (video + control
sockets over adb), ``ScrcpyCamera`` decodes its H264 stream through the
same :class:`FrameReader` the USB camera uses, and ``ScrcpyArm`` injects
touches through its control socket.

Work units are video pixels: the calibration affine maps screen 0-1 to
pixel space (installed analytically by ``HardwareRig.connect_scrcpy`` via
the existing ``install_arm_calibration`` path), so every rig primitive —
``move_to_bbox_center``, ``swipe_from_bbox``, ``park`` — flows unchanged.
Contactless moves (``rapid_to``, ``park``) send nothing: there is no
physical tip to lift, the cursor just records where the next DOWN lands.

Hardware-layer rules apply (see ``hardware/__init__.py``): no imports
from bridge, calibration, orchestration, or server. The 4.1 control and
stream framing below was verified byte-for-byte against the real server.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from io import BufferedReader
from typing import BinaryIO

import cv2

from physiclaw.common.dumps import save_raw_camera, save_snapshot
from physiclaw.core.hardware.device import DeviceNotFound, DeviceTimeout, ProtocolError
from physiclaw.core.hardware.frame_reader import FrameReader

log = logging.getLogger(__name__)

SERVER_VERSION = (
    "4.1"  # fallback when the client binary is absent (see _server_version)
)
SERVER_JAR_NAME = "scrcpy-server.jar"


def _server_version() -> str:
    """The server version the device jar must print back on start.

    The server hard-fails on mismatch, so derive it from the installed
    client (``scrcpy --version`` → ``scrcpy X.Y``) rather than trusting the
    fallback: a client/jar skew then fails here with a clear message,
    not as a cryptic server exit in connect().
    """
    exe = shutil.which("scrcpy")
    if exe is None:
        log.warning("scrcpy client not on PATH — assuming server %s", SERVER_VERSION)
        return SERVER_VERSION
    try:
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=15
        )
        version = out.stdout.strip().split()[1]
    except (IndexError, OSError, subprocess.SubprocessError) as e:
        raise DeviceNotFound(f"cannot read scrcpy version: {e}")
    log.debug("scrcpy client version %s", version)
    return version


# Control message types (server 4.1 ControlMessage — append-only enum).
TYPE_INJECT_TOUCH_EVENT = 2
TYPE_BACK_OR_SCREEN_ON = 4
TYPE_GET_CLIPBOARD = 8
TYPE_SET_CLIPBOARD = 9

# Device message types (server → client on the control socket).
DEVICE_MSG_CLIPBOARD = 0
DEVICE_MSG_ACK_CLIPBOARD = 1

# MotionEvent actions.
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2

COPY_KEY_NONE = 0
PRESSURE_FULL = 0xFFFF  # u16 fixed point 1.0 (Binary.u16FixedPointToFloat)

# Stream framing: every video packet carries a 12-byte header. Session meta
# (resolution) has the top bit of the first word set; frame meta is
# pts (8 bytes, CONFIG/KEY flags in the top bits) + size (4 bytes).
SESSION_META_BIT = 0x80000000

# How long connect() waits for the server to accept the video socket.
SERVER_START_TIMEOUT_SECONDS = 20.0
# How long the camera waits for the first decoded frame.
FIRST_FRAME_TIMEOUT_SECONDS = 10.0
# close(): how long to wait for the cap lock (Camera.close precedent — a
# native release() racing cap.read() can segfault; leak instead of hang).
CLOSE_LOCK_TIMEOUT_SECONDS = 3.0
# Per-exchange bound on the control socket (probes included).
CONTROL_SOCK_TIMEOUT_SECONDS = 5.0


def _read_exact(f, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("scrcpy socket closed")
        buf += chunk
    return buf


def find_server_jar(explicit: str | None = None) -> str:
    """Locate the device-side scrcpy-server jar.

    Explicit path wins, then ``PHYSICLAW_SCRCPY_SERVER``, then the share
    directory beside a ``scrcpy`` binary on PATH (brew layout). Raises
    ``DeviceNotFound`` — same taxonomy as GRBL detection.
    """
    candidates = [p for p in (explicit, os.environ.get("PHYSICLAW_SCRCPY_SERVER")) if p]
    scrcpy_bin = shutil.which("scrcpy")
    if scrcpy_bin is not None:
        candidates.append(
            os.path.join(
                os.path.dirname(scrcpy_bin), "..", "share", "scrcpy", "scrcpy-server"
            )
        )
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            return path
    raise DeviceNotFound(
        "scrcpy-server jar not found — pass server_path or set PHYSICLAW_SCRCPY_SERVER"
    )


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def parse_list_displays(text: str) -> list[tuple[int, int, int]]:
    """Parse ``scrcpy --list-displays`` output into [(display_id, w, h)].

    Matches the client's ``--display-id=N    (WxH)`` lines; anything else
    (server INFO banners, adb device lists) is ignored so a client-version
    wobble degrades to an empty list, not a crash.
    """
    out = []
    for line in text.splitlines():
        m = re.search(r"--display-id=(\d+)\s+\((\d+)x(\d+)\)", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def list_displays(serial: str | None = None) -> list[tuple[int, int, int]]:
    """List the device's displays via ``scrcpy --list-displays``.

    Returns [(display_id, w, h)] for ``ScrcpySession(display_id=...)`` to
    pick from (the camera-index enumeration analog). Raises
    ``DeviceNotFound`` when the client binary is missing or reports none.
    """
    exe = shutil.which("scrcpy")
    if exe is None:
        raise DeviceNotFound("scrcpy client not on PATH — needed for --list-displays")
    cmd = [exe, "--list-displays"]
    if serial is not None:
        cmd += ["--serial", serial]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    displays = parse_list_displays(r.stdout + r.stderr)
    if not displays:
        raise DeviceNotFound("scrcpy --list-displays reported no displays")
    return displays


class ScrcpySession:
    """One scrcpy-server instance: lifecycle, sockets, and the wire lock.

    Owns the adb-pushed server process, the two local forwards, and the
    accepted video + control sockets. The camera parses the video socket,
    the arm sends on the control socket through :meth:`control` (the
    transport-lock equivalent — every send/ack-read pair is atomic).
    ``restart()`` bounces the whole server; devices re-grab sockets from
    here afterwards, so a reconnecting camera can never strand the arm.
    """

    def __init__(
        self,
        serial: str | None = None,
        *,
        display_id: int = 0,
        server_path: str | None = None,
        max_size: int = 1024,
        video_bit_rate: int = 4_000_000,
        max_fps: int = 30,
    ) -> None:
        self.display_id = display_id
        self.serial = serial
        self.server_path = find_server_jar(server_path)
        self.server_version = _server_version()
        self.max_size = max_size
        self.video_bit_rate = video_bit_rate
        self.max_fps = max_fps
        self.device_name = ""
        self._video_size: tuple[int, int] = (0, 0)
        self._size_cond = threading.Condition()
        self._video_sock: socket.socket | None = None
        self._video_file: BufferedReader | None = None
        self._control_sock: socket.socket | None = None
        self._control_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._video_port = _free_port()
        self._control_port = _free_port()
        self._closed = False

    # ─── adb ──────────────────────────────────────────────────────

    def _adb_base(self) -> list[str]:
        cmd = ["adb"]
        if self.serial is not None:
            cmd += ["-s", self.serial]
        return cmd

    def _adb(self, *args: str, **kwargs):
        cmd = self._adb_base() + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)

    # ─── lifecycle ────────────────────────────────────────────────

    def connect(self) -> None:
        """Push the jar, start the server, forward, and open both sockets.

        The server accepts video first, then control — connect in that
        order or the streams swap. Raises ``DeviceNotFound`` (no adb /
        no device / server died on start) or ``DeviceTimeout``.
        """
        if shutil.which("adb") is None:
            raise DeviceNotFound("adb not on PATH")
        if self._closed:
            raise DeviceNotFound("session closed")
        r = self._adb("push", self.server_path, "/data/local/tmp/" + SERVER_JAR_NAME)
        if r.returncode != 0:
            raise DeviceNotFound(f"adb push failed: {r.stderr.strip()}")
        server_args = " ".join(
            [
                self.server_version,
                "scid=%s" % ("-1" if self.display_id == 0 else "%x" % self.display_id),
                "display_id=%d" % self.display_id,
                "log_level=info",
                "video=true",
                "audio=false",
                "control=true",
                "cleanup=false",
                "tunnel_forward=true",
                "send_device_meta=true",
                "send_frame_meta=true",
                "send_dummy_byte=true",
                # The pump needs codec + session meta; the server may one day
                # flip these defaults, so pin what we parse (unknown keys
                # only warn, pinned keys keep working).
                "send_stream_meta=true",
                # Screen-off shows black, which the blackout eye-failover
                # would mistake for a secure surface — keep it lit.
                "power_on=true",
                "max_size=%d" % self.max_size,
                "video_bit_rate=%d" % self.video_bit_rate,
                "video_codec=h264",
                "max_fps=%d" % self.max_fps,
                "stay_awake=true",
                "show_touches=false",
            ]
        )
        self._proc = subprocess.Popen(
            self._adb_base()
            + [
                "shell",
                f"CLASSPATH=/data/local/tmp/{SERVER_JAR_NAME} app_process / "
                f"com.genymobile.scrcpy.Server {server_args}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        time.sleep(1.0)
        if self._proc.poll() is not None:
            raise DeviceNotFound("scrcpy-server exited on start — see adb logcat")
        name = self.socket_name
        self._adb("forward", f"tcp:{self._video_port}", f"localabstract:{name}")
        self._adb("forward", f"tcp:{self._control_port}", f"localabstract:{name}")
        self._open_sockets()
        log.info(
            "scrcpy session: %s video=%s",
            self.device_name or "device",
            self._video_size,
        )

    def _open_sockets(self) -> None:
        """Accept-order open: video first, then control, then video header."""
        deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
        video = None
        while time.monotonic() < deadline:
            try:
                video = socket.create_connection(
                    ("127.0.0.1", self._video_port), timeout=2
                )
                break
            except OSError:
                if self._proc is not None and self._proc.poll() is not None:
                    raise DeviceNotFound("scrcpy-server died while waiting for video")
                time.sleep(0.5)
        if video is None:
            raise DeviceTimeout("scrcpy video socket never accepted")
        try:
            control = socket.create_connection(
                ("127.0.0.1", self._control_port), timeout=10
            )
        except OSError as e:
            video.close()
            raise DeviceTimeout(f"scrcpy control socket refused: {e}")
        self._video_sock = video
        # Bound every control exchange (probes included): a dead peer turns
        # into DeviceTimeout within seconds, never a wedged thread.
        control.settimeout(CONTROL_SOCK_TIMEOUT_SECONDS)
        self._control_sock = control
        self._video_file = video.makefile("rb")
        try:
            assert _read_exact(self._video_file, 1) == b"\x00", "missing dummy byte"
            self.device_name = (
                _read_exact(self._video_file, 64)
                .split(b"\x00")[0]
                .decode("utf-8", "replace")
            )
            codec = _read_exact(self._video_file, 4)
        except (EOFError, AssertionError) as e:
            raise ProtocolError(f"bad scrcpy video header: {e}")
        if codec != b"h264":
            raise ProtocolError(f"unsupported scrcpy codec {codec!r} — h264 only")

    def restart(self) -> None:
        """Bounce the server and reconnect both sockets (camera _reopen path)."""
        self._close_sockets()
        self._kill_server()
        self.connect()

    def close(self) -> None:
        """Release sockets, server, and forwards. Idempotent (Device protocol)."""
        if self._closed:
            return
        self._closed = True
        self._close_sockets()
        self._kill_server()
        try:
            self._adb("forward", "--remove", f"tcp:{self._video_port}")
            self._adb("forward", "--remove", f"tcp:{self._control_port}")
        except Exception:
            pass

    def _close_sockets(self) -> None:
        for attr in ("_video_file", "_video_sock", "_control_sock"):
            obj = getattr(self, attr)
            setattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass

    def _kill_server(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    # ─── video size (session meta, updated by the camera pump) ────

    @property
    def socket_name(self) -> str:
        """Abstract socket: bare "scrcpy" for display 0, scid-suffixed after."""
        if self.display_id == 0:
            return "scrcpy"
        return "scrcpy_%08x" % self.display_id

    def check_control_open(self) -> None:
        """Raise DeviceTimeout unless the control socket is locally open.

        Zero bytes on the wire (getpeername only) — safe to call on any
        cadence. Catches a locally-closed socket, not a dead peer; remote
        death surfaces as video staleness (same process) or a failed send.
        """
        with self._control_lock:
            sock = self._control_sock
            if sock is None:
                raise DeviceTimeout("scrcpy control socket closed")
            try:
                sock.getpeername()
            except OSError as e:
                raise DeviceTimeout(f"scrcpy control socket dead: {e}")

    # ─── video size (session meta, updated by the camera pump) ────

    @property
    def video_size(self) -> tuple[int, int]:
        with self._size_cond:
            return self._video_size

    def _note_size(self, w: int, h: int) -> None:
        with self._size_cond:
            self._video_size = (w, h)
            self._size_cond.notify_all()

    def wait_video_size(self, timeout: float = 10.0) -> tuple[int, int]:
        """Block until the first session-meta packet reports the size."""
        with self._size_cond:
            if not self._size_cond.wait_for(
                lambda: self._video_size != (0, 0), timeout=timeout
            ):
                raise DeviceTimeout("scrcpy never reported its video size")
            return self._video_size

    # ─── control channel ──────────────────────────────────────────

    @contextmanager
    def control(
        self,
    ) -> Generator[tuple[Callable[[bytes], None], Callable[[int], bytes]], None, None]:
        """Yield (send, recv_exact) under the wire lock — one atomic exchange."""
        with self._control_lock:
            if self._control_sock is None:
                raise DeviceTimeout("scrcpy control socket closed")
            yield self._send_locked, self._recv_locked

    def _send_locked(self, data: bytes) -> None:
        try:
            assert self._control_sock is not None
            self._control_sock.sendall(data)
        except OSError as e:
            raise DeviceTimeout(f"scrcpy control send failed: {e}")

    def _recv_locked(self, n: int) -> bytes:
        buf = b""
        try:
            while len(buf) < n:
                assert self._control_sock is not None
                chunk = self._control_sock.recv(n - len(buf))
                if not chunk:
                    raise DeviceTimeout("scrcpy control socket closed")
                buf += chunk
        except DeviceTimeout:
            raise
        except OSError as e:
            raise DeviceTimeout(f"scrcpy control recv failed: {e}")
        return buf


def parse_video_header(hdr: bytes) -> tuple[str, tuple[int, int] | int]:
    """Split one 12-byte video header: session meta or frame meta.

    Returns ``("session", (w, h))`` when the top bit marks resolution, else
    ``("frame", size)``. The camera pump branches on this; unit-testable
    without sockets (FrameReader's scripted-callable precedent).
    """
    word0 = struct.unpack(">I", hdr[:4])[0]
    if word0 & SESSION_META_BIT:
        w, h = struct.unpack(">II", hdr[4:12])
        return ("session", (w, h))
    return ("frame", struct.unpack(">I", hdr[8:12])[0])


class ScrcpyCamera:
    """The scrcpy video stream as a Camera-shaped device.

    H264 packets are pumped into a FIFO that ``cv2.VideoCapture`` (ffmpeg
    backend) decodes — no new dependencies. Frames flow through the same
    :class:`FrameReader` the USB camera uses, so ``health``,
    ``wait_frames``, staleness reconnect, and the fatal-drought guard come
    for free. Digital frames need no exposure or focus: ``exposure_tunable``
    is False (perception skips every settle) and the focus stubs report
    unpinned.
    """

    def __init__(self, session: ScrcpySession) -> None:
        self._session = session
        # scrcpy orients the video itself — no calibration rotation, ever.
        self.rotation: int = -1
        self._tmpdir = tempfile.mkdtemp(prefix="scrcpy-video-")
        self._fifo = os.path.join(self._tmpdir, "video.h264")
        os.mkfifo(self._fifo)
        self._cap_lock = threading.Lock()
        self._writer: BinaryIO | None = None
        self.cap: cv2.VideoCapture | None = None
        self._closed = False
        self._reader = FrameReader(
            read_frame=self._locked_read,
            reopen=self._reopen,
            label="scrcpy-camera",
        )
        self._open()
        try:
            first = self._acquire_first_frame()
        except Exception:
            self._release_cap()
            raise
        self._reader.publish(first)
        self._reader.start()

    # ─── cv2 lifecycle ────────────────────────────────────────────

    def _open(self) -> None:
        """Start the pump, open the decoder, block on the FIFO rendezvous."""
        threading.Thread(target=self._pump, name="scrcpy-pump", daemon=True).start()
        with self._cap_lock:
            self.cap = cv2.VideoCapture(self._fifo)
            if not self.cap.isOpened():
                raise DeviceNotFound("cv2 could not open the scrcpy H264 stream")

    def _pump(self) -> None:
        """Parse the 4.1 stream; write Annex-B payloads to the FIFO."""
        try:
            out = open(self._fifo, "wb", buffering=0)
        except OSError:
            return
        self._writer = out
        try:
            f = self._session._video_file
            while True:
                hdr = _read_exact(f, 12)
                kind, value = parse_video_header(hdr)
                if kind == "session":
                    assert isinstance(value, tuple)
                    self._session._note_size(*value)
                    continue
                assert isinstance(value, int)
                out.write(_read_exact(f, value))
        except (EOFError, OSError, ValueError):
            pass  # generation over, drought calls _reopen (ValueError: buffered read after close)
        finally:
            try:
                out.close()
            except OSError:
                pass
            self._writer = None

    def _acquire_first_frame(self):
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            with self._cap_lock:
                assert self.cap is not None
                ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.1)
        raise DeviceTimeout("scrcpy produced no decodable frame")

    def _locked_read(self):
        with self._cap_lock:
            if self.cap is None:
                return False, None
            return self.cap.read()

    def _reopen(self) -> None:
        """Bounce the server and rebuild the decode chain (FrameReader hook).

        Never raises: a restart mid-outage just fails, and the next drought
        cycle retries — so one outage can't permanently kill the eye. A
        half-rebuilt chain reads (False, None) until the retry lands.
        """
        try:
            self._release_cap()
            self._session.restart()
            self._open()
            self._reader.note_reopened()
        except Exception:
            log.warning(
                "scrcpy camera reopen failed — retrying on next drought", exc_info=True
            )

    def _release_cap(self) -> None:
        if self._cap_lock.acquire(timeout=CLOSE_LOCK_TIMEOUT_SECONDS):
            try:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
            finally:
                self._cap_lock.release()
        # else: wedged reader holds the lock — leak the handle (Camera.close
        # precedent) rather than deadlock the supervisor.

    # ─── Device protocol ──────────────────────────────────────────

    def health(self) -> bool:
        """Live check — same FrameReader semantics as the USB camera."""
        return self._reader.healthy()

    def frame_age(self):
        """Seconds since the last published frame (None if never any)."""
        return self._reader.frame_age()

    def close(self) -> None:
        """Stop the reader, release the decoder, remove the FIFO. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._reader.stop()
        except Exception:
            pass
        self._release_cap()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ─── Camera-shaped surface ────────────────────────────────────

    @property
    def video_size(self) -> tuple[int, int]:
        return self._session.video_size

    @property
    def exposure_tunable(self) -> bool:
        return False

    def set_auto_exposure(self) -> None:
        pass

    def set_manual_exposure(self, value: int) -> None:
        pass

    def focus_lockable(self) -> bool:
        return False

    def focus_pinned(self) -> bool:
        return False

    def lock_focus(self) -> bool:
        return False

    def unlock_focus(self) -> None:
        pass

    def read_focus(self) -> float | None:
        return None

    def apply_focus(self, value: float) -> bool:
        return False

    def wait_frames(self, n: int, timeout: float = 5.0) -> bool:
        """Block until the reader publishes `n` MORE frames (or timeout)."""
        return self._reader.wait_frames(n, timeout=timeout)

    def _fresh_frame(self):
        frame = self._reader.fresh_frame()
        if frame is None:
            return None
        out = frame.copy()
        save_raw_camera(out)  # no-op unless --save-raw-camera
        return out

    def raw_frame(self):
        """Fresh BGR frame without rotation (always unrotated here)."""
        return self._fresh_frame()

    def _rotate(self, frame):
        if self.rotation == -1:
            return frame
        return cv2.rotate(frame, self.rotation)

    def peek(self):
        """Fresh BGR frame, rotation applied. High-frequency polling path."""
        frame = self._fresh_frame()
        if frame is None:
            return None
        return self._rotate(frame)

    def snapshot(self, bbox=None):
        """Fresh BGR frame; draws `bbox` and archives when configured."""
        frame = self.peek()
        if frame is None:
            return None
        if bbox is not None:
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
        save_snapshot(frame)
        return frame


class ScrcpyArm:
    """The scrcpy control socket as a StylusArm-shaped device.

    Work units are video pixels: ``rapid_to``/``swipe_to`` take the same
    calibrated coordinates the rig feeds the GRBL arm, and the analytic
    scrcpy affine makes those pixels. Contactless moves send nothing —
    there is no tip — the cursor just records where the next DOWN lands,
    clamped on-screen so ``park`` (off-screen by definition) is a safe
    no-op. Press gestures inject DOWN/UP at the cursor; ``lift_stylus``
    releases a stuck press, mirroring the coil release.
    """

    TAP_DURATION = 0.08
    LONG_PRESS_DURATION = 1.2
    DOUBLE_TAP_GAP = 0.12  # down-to-down, under the ~300ms OS window

    def __init__(self, session: ScrcpySession) -> None:
        self._session = session
        w, h = session.video_size
        if (w, h) == (0, 0):
            w, h = session.wait_video_size()
        self._vw, self._vh = w, h
        unit = max(1, min(w, h))
        # Pixel-space twins of StylusArm's mm constants.
        self.MOVE_DIRECTIONS = {
            "right": (1.0, 0.0),
            "left": (-1.0, 0.0),
            "bottom": (0.0, 1.0),
            "top": (0.0, -1.0),
            "top-left": (-1.0, -1.0),
            "top-right": (1.0, -1.0),
            "bottom-left": (-1.0, 1.0),
            "bottom-right": (1.0, 1.0),
        }
        self.MOVE_DISTANCES = {
            "large": 0.35 * unit,
            "medium": 0.12 * unit,
            "small": 0.04 * unit,
            "nudge": 0.01 * unit,
        }
        self.SWIPE_DISTANCE = 0.25 * unit
        # Speed name → (interpolated MOVE count, step delay).
        self.SWIPE_SPEEDS = {
            "slow": (16, 0.030),
            "medium": (10, 0.016),
            "fast": (5, 0.008),
        }
        self._cursor = (0.0, 0.0)
        self._pressed = False
        self._pointer_id = 0  # single-finger; one id is enough
        self._clip_seq = 0
        self._closed = False

    # ─── setup / lifecycle ────────────────────────────────────────

    def setup(self) -> None:
        """Prove the session is alive without touching the device.

        Video session-meta received (the server streams) plus a locally
        open control socket. Deliberately NOT a clipboard GET roundtrip:
        the server stays silent on an empty clipboard, so GET can't tell
        "working but empty" from "dead" — and hanging on the answer
        would stall every connect on fresh devices.
        """
        if self._session.video_size == (0, 0):
            raise DeviceTimeout("scrcpy server not streaming")
        self._session.check_control_open()

    def health(self) -> bool:
        """True when the session is alive and this arm isn't closed/broken."""
        if self._closed:
            return False
        proc = self._session._proc
        return proc is not None and proc.poll() is None

    def close(self) -> None:
        """Release any stuck press. Idempotent. The session outlives us."""
        if self._closed:
            return
        self._closed = True
        try:
            self.lift_stylus()
        except Exception:
            pass

    def wait_idle(self, timeout: float = 10) -> None:
        """No-op: control sends are synchronous, there is no planner queue."""
        pass

    def position(self) -> tuple[float, float]:
        """Cursor in work coordinates (video pixels)."""
        return self._cursor

    # ─── coordinate frame (no-ops: pixel space needs no frame) ────

    def unlock(self) -> None:
        pass

    def set_origin(self) -> None:
        """No-op — no work frame to declare (contrast StylusArm.set_origin)."""
        pass

    def set_work_position(self, x: float, y: float) -> None:
        """No-op — pixel space is pinned by construction (warm-start safe)."""
        pass

    def return_to_origin(self) -> None:
        """Reset the cursor to (0, 0); nothing physical to home."""
        self._cursor = (0.0, 0.0)

    def set_direction_mapping(self, right_vec: tuple, down_vec: tuple) -> None:
        """Build MOVE_DIRECTIONS — same signature as StylusArm (the rig's
        ``_apply_bundle_to_arm`` calls it with the affine columns; for the
        analytic scrcpy affine those are the pixel axes)."""
        rx, ry = right_vec
        dx, dy = down_vec
        self.MOVE_DIRECTIONS = {
            "right": (rx, ry),
            "left": (-rx, -ry),
            "bottom": (dx, dy),
            "top": (-dx, -dy),
            "top-left": (-rx - dx, -ry - dy),
            "top-right": (rx - dx, ry - dy),
            "bottom-left": (-rx + dx, -ry + dy),
            "bottom-right": (rx + dx, ry + dy),
        }

    # ─── wire ─────────────────────────────────────────────────────

    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        return (
            max(0, min(int(x), self._vw - 1)),
            max(0, min(int(y), self._vh - 1)),
        )

    def _touch(self, send, action: int, x: float, y: float) -> None:
        px, py = self._clamp(x, y)
        send(
            struct.pack(
                ">BBqiiHHHii",
                TYPE_INJECT_TOUCH_EVENT,
                action,
                self._pointer_id,
                px,
                py,
                self._vw,
                self._vh,
                PRESSURE_FULL,
                0,
                0,
            )
        )

    def _down(self, x: float, y: float) -> None:
        with self._session.control() as (send, _):
            self._touch(send, ACTION_DOWN, x, y)
        self._pressed = True

    def _up(self, x: float, y: float) -> None:
        with self._session.control() as (send, _):
            self._touch(send, ACTION_UP, x, y)
        self._pressed = False

    # ─── press gestures (at the cursor, like StylusArm) ───────────

    def tap(self) -> None:
        """Single tap at the cursor."""
        x, y = self._cursor
        self._down(x, y)
        time.sleep(self.TAP_DURATION)
        self._up(x, y)

    def double_tap(self) -> None:
        """Double tap at the cursor, inside the OS double-tap window."""
        x, y = self._cursor
        self._down(x, y)
        time.sleep(self.TAP_DURATION)
        self._up(x, y)
        time.sleep(self.DOUBLE_TAP_GAP)
        self._down(x, y)
        time.sleep(self.TAP_DURATION)
        self._up(x, y)

    def long_press(self) -> None:
        """Long press at the cursor."""
        x, y = self._cursor
        self._down(x, y)
        time.sleep(self.LONG_PRESS_DURATION)
        self._up(x, y)

    def lift_stylus(self) -> None:
        """Release a stuck press — the coil-release equivalent for teardown."""
        if self._pressed:
            x, y = self._cursor
            try:
                self._up(x, y)
            finally:
                self._pressed = False

    # ─── moves ────────────────────────────────────────────────────

    def rapid_to(self, x: float, y: float, speed: int = 8000) -> None:
        """Record the cursor (clamped on-screen). Sends nothing — contactless
        moves are free in pixel space, which is also what makes ``park`` a
        safe no-op. `speed` is accepted for StylusArm signature parity."""
        self._cursor = self._clamp(x, y)

    def _linear_move(
        self, send, x0: float, y0: float, x1: float, y1: float, speed: str
    ) -> None:
        steps, delay = self.SWIPE_SPEEDS[speed]
        for i in range(1, steps + 1):
            t = i / steps
            self._touch(send, ACTION_MOVE, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            time.sleep(delay)

    def swipe_to(
        self,
        x: float,
        y: float,
        speed: str = "medium",
        start_dwell: float = 0.0,
        end_dwell: float = 0.0,
    ) -> None:
        """Swipe from the cursor to work coordinate (x, y): press, slide,
        release. `start_dwell`/`end_dwell` mirror StylusArm.swipe_to's edge
        anchors (touch-down registration, app-switcher hold)."""
        if speed not in self.SWIPE_SPEEDS:
            raise ValueError(
                f"speed must be one of {list(self.SWIPE_SPEEDS)}, got {speed!r}"
            )
        x0, y0 = self._cursor
        with self._session.control() as (send, _):
            self._touch(send, ACTION_DOWN, x0, y0)
            self._pressed = True
            try:
                if start_dwell:
                    time.sleep(start_dwell)
                self._linear_move(send, x0, y0, x, y, speed)
                if end_dwell:
                    time.sleep(end_dwell)
                self._touch(send, ACTION_UP, x, y)
            finally:
                self._pressed = False
        self._cursor = self._clamp(x, y)

    def jog(self, dx: float, dy: float) -> None:
        """Relative cursor move of (dx, dy) pixels. No screen contact."""
        x, y = self._cursor
        self._cursor = self._clamp(x + dx, y + dy)

    def move(self, direction: str, distance: str = "medium") -> None:
        """Cursor move relative to now (same direction/distance names as
        StylusArm; distances resolve to pixels from the video size)."""
        if self.MOVE_DIRECTIONS is None:
            raise RuntimeError("MOVE_DIRECTIONS not set — run calibration first")
        mx, my = self.MOVE_DIRECTIONS[direction]
        d = self.MOVE_DISTANCES[distance]
        mag = (mx**2 + my**2) ** 0.5 or 1
        self.jog(mx / mag * d, my / mag * d)

    def swipe(self, direction: str, speed: str = "medium") -> None:
        """Swipe from the cursor in a cardinal direction."""
        if self.MOVE_DIRECTIONS is None:
            raise RuntimeError("MOVE_DIRECTIONS not set — run calibration first")
        mx, my = self.MOVE_DIRECTIONS[direction]
        mag = (mx**2 + my**2) ** 0.5 or 1
        x0, y0 = self._cursor
        d = self.SWIPE_DISTANCE
        self.swipe_to(x0 + mx / mag * d, y0 + my / mag * d, speed)

    # ─── keys + clipboard ─────────────────────────────────────────

    def back(self) -> None:
        """BACK key (coordinate-free)."""
        with self._session.control() as (send, _):
            for action in (ACTION_DOWN, ACTION_UP):
                send(struct.pack(">BB", TYPE_BACK_OR_SCREEN_ON, action))
                time.sleep(0.05)

    def set_clipboard(self, text: str) -> None:
        """Set the device clipboard (server acks with our sequence)."""
        raw = text.encode("utf-8")
        self._clip_seq += 1
        with self._session.control() as (send, recv):
            send(
                struct.pack(">BqB", TYPE_SET_CLIPBOARD, self._clip_seq, 0)
                + struct.pack(">I", len(raw))
                + raw
            )
            if recv(1) != bytes((DEVICE_MSG_ACK_CLIPBOARD,)):
                raise ProtocolError("missing clipboard ack")
            if struct.unpack(">q", recv(8))[0] != self._clip_seq:
                raise ProtocolError("clipboard ack sequence mismatch")

    def get_clipboard(self) -> str:
        """Read the device clipboard (the setup() probe — no side effects)."""
        with self._session.control() as (send, recv):
            send(bytes((TYPE_GET_CLIPBOARD, COPY_KEY_NONE)))
            if recv(1) != bytes((DEVICE_MSG_CLIPBOARD,)):
                raise ProtocolError("not a clipboard reply")
            size = struct.unpack(">I", recv(4))[0]
            return recv(size).decode("utf-8", "replace")
