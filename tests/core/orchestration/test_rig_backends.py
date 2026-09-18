"""Tests for the rig's dual-link backend registry (physical + scrcpy).

Eye and hand switch independently; mapping halves travel with the device
so the crop always matches the frame's source and the affine always
matches the arm. Fakes are spec'd doubles; the calibration container is
real (mapping assertions need real affines).
"""

from __future__ import annotations

import numpy as np
import pytest

from physiclaw.core.calibration import Calibration
from physiclaw.core.hardware.device import DeviceTimeout
from physiclaw.core.orchestration.rig import (
    PHYSICAL_BACKEND,
    SCRCPY_BACKEND,
    HardwareRig,
    _BackendState,
)

DIAG = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
PIXEL = np.array([[1024.0, 0.0, 0.0], [0.0, 576.0, 0.0]])


def _live_rig(arm_double, cam_double):
    """A rig with a calibrated physical pair (grbl affine + pin)."""
    rig = HardwareRig()
    rig._arm = arm_double()
    rig._cam = cam_double()
    rig.calibration = Calibration()
    rig.calibration.pct_to_grbl = DIAG.copy()
    rig.calibration.pct_to_cam = np.eye(2, 3)
    rig.calibration.cam_size = (1920, 1080)
    rig.calibration.cam_rotation = 0
    rig._origin_pinned = True
    return rig


def _park_scrcpy(rig, arm_double, cam_double):
    """Park a calibrated scrcpy pair into the rig's records."""
    rig._backends[SCRCPY_BACKEND] = _BackendState(
        arm=arm_double(),
        cam=cam_double(),
        pct_to_grbl=PIXEL.copy(),
        pinned=True,
        pct_to_cam=np.eye(2, 3),
        cam_size=(1024, 576),
        cam_rotation=-1,
    )


# ─── hand switching ────────────────────────────────────────────


def test_hand_switch_roundtrip_preserves_mappings(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    phys_arm = rig._arm
    _park_scrcpy(rig, arm_double, cam_double)
    rig.acquire()
    try:
        rig.use_hand(SCRCPY_BACKEND)
        assert rig._hand == SCRCPY_BACKEND
        assert rig._arm is not phys_arm  # alias moved to the parked arm
        np.testing.assert_array_equal(rig.calibration.pct_to_grbl, PIXEL)
        # Outgoing physical pair parked away with its mapping.
        saved = rig._backends[PHYSICAL_BACKEND]
        assert saved.arm is phys_arm
        np.testing.assert_array_equal(saved.pct_to_grbl, DIAG)

        rig.use_hand(PHYSICAL_BACKEND)
        assert rig._arm is phys_arm
        np.testing.assert_array_equal(rig.calibration.pct_to_grbl, DIAG)
        assert rig._origin_pinned is True
    finally:
        rig.release()


def test_hand_switch_parks_outgoing_arm(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    _park_scrcpy(rig, arm_double, cam_double)
    rig.acquire()
    try:
        rig.use_hand(SCRCPY_BACKEND)
    finally:
        rig.release()
    # PARK_PCT (-0.1, -0.05) through the diag affine.
    rig._backends[PHYSICAL_BACKEND].arm.rapid_to.assert_called_once_with(-1.0, -1.0)


def test_switch_unconnected_backend_raises(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    rig.acquire()
    try:
        with pytest.raises(RuntimeError, match="not connected"):
            rig.use_hand(SCRCPY_BACKEND)
        with pytest.raises(RuntimeError, match="not connected"):
            rig.use_eye(SCRCPY_BACKEND)
        with pytest.raises(ValueError, match="unknown backend"):
            rig.use_hand("nope")
    finally:
        rig.release()


# ─── eye switching ─────────────────────────────────────────────


def test_eye_switch_moves_cam_mapping(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    phys_cam = rig._cam
    _park_scrcpy(rig, arm_double, cam_double)
    rig.acquire()
    try:
        rig.use_eye(SCRCPY_BACKEND)
        assert rig._eye == SCRCPY_BACKEND
        assert rig._cam is not phys_cam
        assert rig.calibration.cam_size == (1024, 576)
        assert rig.calibration.cam_rotation == -1
        saved = rig._backends[PHYSICAL_BACKEND]
        assert saved.cam is phys_cam
        assert saved.cam_size == (1920, 1080)
    finally:
        rig.release()


# ─── fallback selection ────────────────────────────────────────


def test_next_hand_skips_uncalibrated_link(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    assert rig.next_hand() is None  # scrcpy absent entirely
    rig._backends[SCRCPY_BACKEND] = _BackendState(arm=arm_double())  # no mapping
    assert rig.next_hand() is None  # present but useless
    rig._backends[SCRCPY_BACKEND].pct_to_grbl = PIXEL.copy()
    assert rig.next_hand() == SCRCPY_BACKEND


def test_next_eye_respects_order(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    _park_scrcpy(rig, arm_double, cam_double)
    assert rig.next_eye() == SCRCPY_BACKEND  # digital first by default
    flipped = HardwareRig(eye_order=(PHYSICAL_BACKEND, SCRCPY_BACKEND))
    assert flipped.next_eye() is None  # nothing parked; physical is active


# ─── status ────────────────────────────────────────────────────


def test_status_reports_backend_table(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    _park_scrcpy(rig, arm_double, cam_double)
    out = rig.status()
    assert out["active_eye"] == PHYSICAL_BACKEND
    assert out["active_hand"] == PHYSICAL_BACKEND
    assert out["backends"][PHYSICAL_BACKEND]["arm"] is True
    assert out["backends"][SCRCPY_BACKEND]["camera"] is True
    assert out["backends"][SCRCPY_BACKEND]["calibrated"] is True


# ─── scrcpy-only clipboard/screenshot (no bridge) ──────────────


def _scrcpy_only_rig(arm_double, cam_double):
    """Active scrcpy pair, mappings installed, no bridge attached."""
    from unittest.mock import MagicMock

    from physiclaw.core.hardware.scrcpy import ScrcpyArm, ScrcpyCamera

    rig = HardwareRig()
    rig._arm = MagicMock(spec=ScrcpyArm)
    rig._cam = MagicMock(spec=ScrcpyCamera)
    rig._eye = SCRCPY_BACKEND
    rig._hand = SCRCPY_BACKEND
    rig.calibration = Calibration()
    rig.calibration.pct_to_grbl = PIXEL.copy()
    rig._origin_pinned = True
    rig.calibration.pct_to_cam = np.eye(2, 3)
    rig.calibration.cam_size = (1024, 576)
    rig.calibration.cam_rotation = -1
    return rig


def test_sync_clipboard_direct_path(arm_double, cam_double):
    rig = _scrcpy_only_rig(arm_double, cam_double)
    rig.acquire()
    try:
        assert rig.sync_clipboard("hi", timeout=5) is True
        rig._arm.set_clipboard.assert_called_once_with("hi")
    finally:
        rig.release()


def test_sync_clipboard_grbl_without_bridge_raises(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    assert rig._bridge is None
    rig.acquire()
    try:
        with pytest.raises(RuntimeError, match="bridge page or the scrcpy hand"):
            rig.sync_clipboard("hi", timeout=5)
    finally:
        rig.release()


def test_sync_clipboard_scrcpy_hand_ignores_bridge(
    arm_double, cam_double, bridge_double
):
    """Active scrcpy hand + attached bridge: direct control path wins — the
    iOS bridge+AT pipeline must never run against an Android screen."""
    from unittest.mock import MagicMock

    from physiclaw.core.hardware.scrcpy import ScrcpyArm

    rig = _live_rig(arm_double, cam_double)
    rig._arm = MagicMock(spec=ScrcpyArm)
    rig._hand = SCRCPY_BACKEND
    rig.attach_bridge(bridge_double())
    rig.acquire()
    try:
        assert rig.sync_clipboard("hi", timeout=5) is True
        rig._arm.set_clipboard.assert_called_once_with("hi")
        rig._bridge.send_text.assert_not_called()
    finally:
        rig.release()


def test_sync_clipboard_physical_hand_uses_bridge(
    arm_double, cam_double, bridge_double, at_double
):
    """Physical hand keeps the iOS bridge+AT path (bridge present)."""
    rig = _live_rig(arm_double, cam_double)
    rig.attach_bridge(bridge_double())
    rig._bridge.wait_clipboard.return_value = True
    rig._assistive_touch = at_double()
    rig.acquire()
    try:
        assert rig.sync_clipboard("hi", timeout=5) is True
        rig._bridge.send_text.assert_called_once_with("hi")
    finally:
        rig.release()


def test_take_screenshot_direct_path(arm_double, cam_double):
    rig = _scrcpy_only_rig(arm_double, cam_double)
    rig._cam.snapshot.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    rig.acquire()
    try:
        data = rig.take_screenshot()
        assert data[:2] == b"\xff\xd8"  # JPEG magic
    finally:
        rig.release()


def test_take_screenshot_none_frame_returns_none(arm_double, cam_double):
    rig = _scrcpy_only_rig(arm_double, cam_double)
    rig._cam.snapshot.return_value = None
    rig.acquire()
    try:
        assert rig.take_screenshot() is None
    finally:
        rig.release()


def test_take_screenshot_usb_without_bridge_raises(arm_double, cam_double):
    rig = _live_rig(arm_double, cam_double)
    rig.acquire()
    try:
        with pytest.raises(RuntimeError, match="bridge page or the scrcpy eye"):
            rig.take_screenshot()
    finally:
        rig.release()


# ─── preference order ──────────────────────────────────────────


def test_default_order_prefers_scrcpy():
    rig = HardwareRig()
    assert rig._eye_order == (SCRCPY_BACKEND, PHYSICAL_BACKEND)
    assert rig._hand_order == (SCRCPY_BACKEND, PHYSICAL_BACKEND)


def test_order_follows_config_preferred(mocker):
    from types import SimpleNamespace

    mocker.patch(
        "physiclaw.core.orchestration.rig.CONFIG",
        SimpleNamespace(backend=SimpleNamespace(preferred="physical")),
    )
    rig = HardwareRig()
    assert rig._eye_order == (PHYSICAL_BACKEND, SCRCPY_BACKEND)
    assert rig.preferred_eye == PHYSICAL_BACKEND


def test_explicit_order_beats_config(mocker):
    from types import SimpleNamespace

    mocker.patch(
        "physiclaw.core.orchestration.rig.CONFIG",
        SimpleNamespace(backend=SimpleNamespace(preferred="physical")),
    )
    rig = HardwareRig(eye_order=(SCRCPY_BACKEND, PHYSICAL_BACKEND))
    assert rig._eye_order == (SCRCPY_BACKEND, PHYSICAL_BACKEND)


# ─── hand revival ──────────────────────────────────────────────


class _DeadSession:
    """control stays shut; restart raises (server never comes back)."""

    def __init__(self):
        self.bounces = 0

    def check_control_open(self):
        raise DeviceTimeout("control dead")

    def restart(self):
        self.bounces += 1
        raise DeviceTimeout("server never came back")


def test_revive_hand_without_session_is_false(arm_double):
    from physiclaw.core.hardware.arm import StylusArm

    rig = HardwareRig()
    rig._backends[PHYSICAL_BACKEND] = _BackendState(arm=arm_double())
    assert isinstance(rig._backends[PHYSICAL_BACKEND].arm, StylusArm)
    assert rig.revive_hand(PHYSICAL_BACKEND) is False


def test_revive_hand_failed_restart_is_false():
    from unittest.mock import MagicMock

    from physiclaw.core.hardware.scrcpy import ScrcpyArm

    rig = HardwareRig()
    rig._backends[SCRCPY_BACKEND] = _BackendState(arm=MagicMock(spec=ScrcpyArm))
    rig._backends[SCRCPY_BACKEND].session = _DeadSession()
    assert rig.revive_hand(SCRCPY_BACKEND) is False
    assert rig._backends[SCRCPY_BACKEND].session.bounces == 1


def test_revive_hand_bounce_then_probe_true():
    from unittest.mock import MagicMock

    from physiclaw.core.hardware.scrcpy import ScrcpyArm

    class _HealingSession:
        def __init__(self):
            self.bounces = 0

        def check_control_open(self):
            if self.bounces == 0:
                from physiclaw.core.hardware.device import DeviceTimeout

                raise DeviceTimeout("control dead")

        def restart(self):
            self.bounces += 1

    rig = HardwareRig()
    rig._backends[SCRCPY_BACKEND] = _BackendState(arm=MagicMock(spec=ScrcpyArm))
    rig._backends[SCRCPY_BACKEND].session = _HealingSession()
    assert rig.probe_hand(SCRCPY_BACKEND) is False
    assert rig.revive_hand(SCRCPY_BACKEND) is True


def test_take_screenshot_scrcpy_eye_ignores_bridge(
    arm_double, cam_double, bridge_double
):
    """Active scrcpy eye + attached bridge: video frame direct, no AT tap,
    no Shortcut upload wait."""
    rig = _scrcpy_only_rig(arm_double, cam_double)
    rig._cam.snapshot.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    rig.attach_bridge(bridge_double())
    rig.acquire()
    try:
        data = rig.take_screenshot()
        assert data[:2] == b"\xff\xd8"  # JPEG magic
        rig._bridge.clear_screenshot.assert_not_called()
    finally:
        rig.release()
