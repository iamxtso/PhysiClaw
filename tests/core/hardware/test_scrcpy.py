"""Tests for `physiclaw.core.hardware.scrcpy` — wire framing and arm logic.

No device needed: control exchanges run against a scripted fake session,
video framing is pure parsing, and the pixel-space mapping is plain math.
Byte layouts are pinned here so a scrcpy-server upgrade that moves them
fails loudly instead of tapping wrong.
"""

from __future__ import annotations

import struct
from collections import deque
from contextlib import contextmanager

import numpy as np
import pytest

from physiclaw.core.calibration import ScreenTransforms
from physiclaw.core.hardware import scrcpy
from physiclaw.core.hardware.scrcpy import ScrcpyArm, parse_video_header

VW, VH = 1024, 576


class _FakeSession:
    """Scripted stand-in for ScrcpySession: records sends, replays recvs."""

    def __init__(self, replies=()) -> None:
        self.sent: list[bytes] = []
        self._replies = deque(replies)
        self._video_size = (VW, VH)
        self._proc = None

    @property
    def video_size(self):
        return self._video_size

    def wait_video_size(self, timeout=10.0):
        return self._video_size

    @contextmanager
    def control(self):
        yield self._send, self._recv

    def _send(self, data: bytes) -> None:
        self.sent.append(data)

    def _recv(self, n: int) -> bytes:
        return self._replies.popleft()


def _touch_fields(raw: bytes):
    assert len(raw) == 32
    t, action, pid, x, y, w, h, pressure, abtn, btn = struct.unpack(">BBqiiHHHii", raw)
    assert t == scrcpy.TYPE_INJECT_TOUCH_EVENT
    return action, pid, x, y, w, h, pressure


# ─── video header branch ─────────────────────────────────────────


def test_parse_session_meta_reports_size():
    hdr = struct.pack(">III", 0x80000000, VW, VH)
    assert parse_video_header(hdr) == ("session", (VW, VH))


def test_parse_frame_meta_reports_size():
    hdr = struct.pack(">qI", 123456, 41)
    assert parse_video_header(hdr) == ("frame", 41)


def test_parse_config_frame_is_still_a_frame():
    hdr = struct.pack(">qI", 1 << 62, 17)  # CONFIG flag set, no pts
    assert parse_video_header(hdr) == ("frame", 17)


# ─── pixel-space mapping (the analytic scrcpy affine) ────────────


def test_scrcpy_affine_maps_pct_to_video_pixels():
    t = ScreenTransforms(
        pct_to_grbl=np.array([[VW, 0.0, 0.0], [0.0, VH, 0.0]]),
        pct_to_cam=np.eye(2, 3),
        cam_size=(VW, VH),
    )
    assert t.pct_to_grbl_mm(0.5, 0.5) == (512.0, 288.0)
    assert t.pct_to_grbl_mm(0.0, 0.0) == (0.0, 0.0)
    assert t.pct_to_cam_pixel(0.5, 0.5) == (512, 288)


# ─── press gestures ──────────────────────────────────────────────


def test_tap_is_down_then_up_at_cursor():
    arm = ScrcpyArm(_FakeSession())
    arm.rapid_to(512, 288)
    arm.tap()
    assert len(arm._session.sent) == 2
    a0, _, x0, y0, w, h, p = _touch_fields(arm._session.sent[0])
    a1, _, x1, y1, _, _, _ = _touch_fields(arm._session.sent[1])
    assert (a0, a1) == (scrcpy.ACTION_DOWN, scrcpy.ACTION_UP)
    assert (x0, y0) == (x1, y1) == (512, 288)
    assert (w, h, p) == (VW, VH, 0xFFFF)


def test_double_tap_is_four_messages():
    arm = ScrcpyArm(_FakeSession())
    arm.rapid_to(100, 100)
    arm.double_tap()
    actions = [_touch_fields(m)[0] for m in arm._session.sent]
    assert actions == [
        scrcpy.ACTION_DOWN,
        scrcpy.ACTION_UP,
        scrcpy.ACTION_DOWN,
        scrcpy.ACTION_UP,
    ]


def test_park_clamps_offscreen_and_sends_nothing():
    arm = ScrcpyArm(_FakeSession())
    arm.rapid_to(-102.4, -28.8)  # PARK_PCT through the pixel affine
    assert arm.position() == (0, 0)
    assert arm._session.sent == []


def test_lift_releases_only_when_pressed():
    session = _FakeSession()
    arm = ScrcpyArm(session)
    arm.lift_stylus()
    assert session.sent == []
    arm._pressed = True
    arm.lift_stylus()
    assert len(session.sent) == 1
    assert _touch_fields(session.sent[0])[0] == scrcpy.ACTION_UP


def test_swipe_to_slides_down_move_up():
    session = _FakeSession()
    arm = ScrcpyArm(session)
    arm.rapid_to(512, 100)
    arm.swipe_to(512, 300, "fast")
    actions = [_touch_fields(m)[0] for m in session.sent]
    assert actions[0] == scrcpy.ACTION_DOWN
    assert actions[-1] == scrcpy.ACTION_UP
    assert set(actions[1:-1]) == {scrcpy.ACTION_MOVE}
    assert arm.position() == (512, 300)


def test_swipe_to_rejects_unknown_speed():
    arm = ScrcpyArm(_FakeSession())
    with pytest.raises(ValueError):
        arm.swipe_to(10, 10, "ludicrous")


def test_move_normalizes_cardinal_distance():
    arm = ScrcpyArm(_FakeSession())
    arm.rapid_to(512, 288)
    arm.move("right", "medium")
    x, y = arm.position()
    assert y == 288
    assert x == int(512 + arm.MOVE_DISTANCES["medium"])


# ─── clipboard framing ───────────────────────────────────────────


def test_set_clipboard_frame_and_ack():
    session = _FakeSession(replies=[bytes((1,)), struct.pack(">q", 1)])
    arm = ScrcpyArm(session)
    arm.set_clipboard("hi")
    raw = session.sent[0]
    assert raw[0] == scrcpy.TYPE_SET_CLIPBOARD
    seq, paste, ln = struct.unpack(">qBI", raw[1:14])
    assert (seq, paste, ln) == (1, 0, 2)
    assert raw[14:] == b"hi"


def test_get_clipboard_frame():
    session = _FakeSession(replies=[bytes((0,)), struct.pack(">I", 3), b"abc"])
    arm = ScrcpyArm(session)
    assert arm.get_clipboard() == "abc"
    assert session.sent[0] == bytes((scrcpy.TYPE_GET_CLIPBOARD, scrcpy.COPY_KEY_NONE))


def test_back_is_down_up_pair():
    session = _FakeSession()
    arm = ScrcpyArm(session)
    arm.back()
    assert session.sent == [
        struct.pack(">BB", scrcpy.TYPE_BACK_OR_SCREEN_ON, scrcpy.ACTION_DOWN),
        struct.pack(">BB", scrcpy.TYPE_BACK_OR_SCREEN_ON, scrcpy.ACTION_UP),
    ]


# ─── session helpers (no device) ─────────────────────────────────


def test_parse_list_displays():
    from physiclaw.core.hardware.scrcpy import parse_list_displays

    text = (
        "[server] INFO: List of displays:\n"
        "    --display-id=0    (1920x1080)\n"
        "    --display-id=1    (800x600)\n"
        "scrcpy 4.1 <https://github.com/Genymobile/scrcpy>\n"
    )
    assert parse_list_displays(text) == [(0, 1920, 1080), (1, 800, 600)]
    assert parse_list_displays("no displays here") == []


def test_socket_name_follows_display(mocker):
    import physiclaw.core.hardware.scrcpy as m

    mocker.patch.object(m, "find_server_jar", return_value="/tmp/x.jar")
    mocker.patch.object(m, "_server_version", return_value="4.1")
    assert m.ScrcpySession(display_id=0).socket_name == "scrcpy"
    assert m.ScrcpySession(display_id=1).socket_name == "scrcpy_00000001"


def test_server_version_parsed_from_client(mocker):
    import physiclaw.core.hardware.scrcpy as m

    mocker.patch.object(m.shutil, "which", return_value="/bin/scrcpy")
    proc = mocker.Mock()
    proc.stdout = "scrcpy 4.1 <https://github.com/Genymobile/scrcpy>\n"
    mocker.patch.object(m.subprocess, "run", return_value=proc)
    assert m._server_version() == "4.1"


def test_server_version_falls_back_without_client(mocker):
    import physiclaw.core.hardware.scrcpy as m

    mocker.patch.object(m.shutil, "which", return_value=None)
    assert m._server_version() == m.SERVER_VERSION


def test_server_args_pin_display_scid_and_cleanup(mocker):
    import physiclaw.core.hardware.scrcpy as m

    mocker.patch.object(m, "find_server_jar", return_value="/tmp/x.jar")
    mocker.patch.object(m, "_server_version", return_value="4.1")
    d0 = m.ScrcpySession(display_id=0)._server_args()
    assert "scid=-1" in d0
    assert "display_id=0" in d0
    assert "cleanup=true" in d0
    assert "cleanup=false" not in d0
    d1 = m.ScrcpySession(display_id=1)._server_args()
    assert "scid=1 " in d1  # trailing space: not scid=10/11/...
    assert "display_id=1" in d1
    assert "scid=-1" not in d1
