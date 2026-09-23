"""Pure-numpy detectors and image geometry. No policy code lives here: targets
(``targets.py``) turn these into Sightings and behaviours (``behaviors.py``)
act on them.

Detection is a red-pixel threshold so everything runs identically in check
(replayed or random frames), sim (a red sphere from ``--sim-target``) and on
the robot (hold up something red).
"""
from __future__ import annotations

import math

import numpy as np

from config import HEAD_CAMERA_FOVY


def red_blob(image: np.ndarray, min_fraction: float = 0.002):
    """Centre of the red pixels as ``(u, v, fraction)`` with u, v in [-1, 1]
    (u positive to the right of the image, v down), or None when fewer than
    ``min_fraction`` of the pixels are red."""
    img = image.astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = (r > 120) & (r - g > 60) & (r - b > 60)
    n = int(mask.sum())
    if n < min_fraction * mask.size:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    u = 2.0 * xs.mean() / (w - 1) - 1.0
    v = 2.0 * ys.mean() / (h - 1) - 1.0
    return float(u), float(v), n / mask.size


def bearing(u: float, image_shape, fovy_deg: float = HEAD_CAMERA_FOVY) -> float:
    """Horizontal angle in rad (positive to the right) of normalised image column ``u``."""
    h, w = image_shape[:2]
    half_w = math.tan(math.radians(fovy_deg) / 2) * w / h
    return math.atan(u * half_w)


def elevation(v: float, image_shape, fovy_deg: float = HEAD_CAMERA_FOVY) -> float:
    """Vertical angle in rad (positive up) of normalised image row ``v`` (v is
    +1 at the bottom), relative to the optical axis. The head camera's axis
    itself points ``config.HEAD_CAMERA_PITCH`` below horizontal."""
    return -math.atan(v * math.tan(math.radians(fovy_deg) / 2))
