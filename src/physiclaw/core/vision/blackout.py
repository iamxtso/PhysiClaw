"""Secure-surface blackout detection — is the scrcpy eye blind?

Some surfaces (navigation, DRM video, banking) set FLAG_SECURE: the
compositor hands screen capture a black frame while the physical screen
stays live. The USB camera still sees the phone, so a blackout is the
eye-fallback trigger: the frame carries no information, and no exposure
or brightness tuning can recover it (contrast with quality.py's DARK —
a dim-but-legible screen).

The rule is two-factor like BLOWN: mean luma near zero AND near-zero
spread. A dark-mode UI is dark but structured (high std); a secure
surface is uniformly ~0. Thresholds live in VisionConfig beside the
other detection thresholds.
"""

from __future__ import annotations

import numpy as np

from physiclaw.common.config import CONFIG
from physiclaw.core.vision.preprocess import grayscale

BLACKOUT_MEAN_LUMA = CONFIG.vision.blackout_mean_luma
BLACKOUT_STD_LUMA = CONFIG.vision.blackout_std_luma


def frame_stats(frame: np.ndarray) -> tuple[float, float]:
    """Mean + std of luma — the logged observables for threshold tuning."""
    gray = grayscale(frame)
    return float(gray.mean()), float(gray.std())


def is_blackout(frame: np.ndarray) -> bool:
    """True when the frame is a secure-surface black frame (eye is blind).

    Accepts BGR or gray. Small frames are scored as-is — blackness is
    scale-invariant, unlike sharpness, so no working-width normalization.
    """
    mean, std = frame_stats(frame)
    return mean < BLACKOUT_MEAN_LUMA and std < BLACKOUT_STD_LUMA
