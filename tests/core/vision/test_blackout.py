"""Tests for `physiclaw.core.vision.blackout` — secure-surface detection."""

from __future__ import annotations

import numpy as np

from physiclaw.core.vision import blackout
from physiclaw.core.vision.blackout import frame_stats, is_blackout


def test_pure_black_frame_is_blackout():
    assert is_blackout(np.zeros((64, 64, 3), dtype=np.uint8)) is True


def test_near_black_noise_floor_is_blackout():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 4, size=(64, 64, 3)).astype(np.uint8)
    assert is_blackout(frame) is True


def test_dark_mode_ui_is_not_blackout():
    # Dark but structured: near-black background + bright text rows.
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[::8, 8:56] = 220
    assert is_blackout(frame) is False


def test_normal_frame_is_not_blackout():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(64, 64, 3)).astype(np.uint8)
    assert is_blackout(frame) is False


def test_gray_input_scores_like_bgr():
    frame = np.zeros((32, 32), dtype=np.uint8)
    assert is_blackout(frame) is True
    mean, std = frame_stats(frame)
    assert (mean, std) == (0.0, 0.0)


def test_thresholds_come_from_config():
    assert blackout.BLACKOUT_MEAN_LUMA == 8.0
    assert blackout.BLACKOUT_STD_LUMA == 5.0
