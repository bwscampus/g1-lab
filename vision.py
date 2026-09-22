"""Pure-numpy vision helpers and the example camera policies.

Detection is a red-pixel threshold so the examples behave identically in check
(replayed or random frames), sim (a red sphere from ``--sim-target``) and on the
robot (hold up something red). Swap ``red_blob`` for a real detector and the
policies stay the same.
"""
from __future__ import annotations

import math

import numpy as np

from config import HEAD_CAMERA_FOVY, joint_index
from policy import Obs, Pose, ReactivePolicy

WAIST_YAW = joint_index("waist_yaw")


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


class Look(ReactivePolicy):
    """Turn the waist to keep the red blob centred; the arms stay at STAND.
    Each new frame moves the yaw target by ``gain`` times the blob's bearing,
    so it converges over a few frames without overshooting on noisy detections."""

    name = "look"

    def __init__(self, duration: float = 15.0, *, gain: float = 0.6, yaw_max: float = 0.8,
                 **kw) -> None:
        super().__init__(duration, **kw)
        self.gain = gain
        self.yaw_max = yaw_max

    def track(self, t: float, obs: Obs) -> Pose:
        blob = red_blob(obs.frame.image)
        if blob is None:
            return {}                       # nothing red: hold
        u, _, _ = blob
        # u > 0 is to the robot's right; positive waist yaw turns it left.
        yaw = self.cmd[WAIST_YAW] - self.gain * bearing(u, obs.frame.image.shape)
        return {WAIST_YAW: float(np.clip(yaw, -self.yaw_max, self.yaw_max))}
