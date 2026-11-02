"""Aiming the right arm from what the robot measures: the URDF's chains (``config.WAIST_CHAIN``,
``config.R_ARM_CHAIN``, the head camera's mount), the waist and arm joints and the IMU.

Everything is in the **IMU's frame**: at the pelvis, oriented as the IMU's quaternion says
(its tilt and its heading), x forward, y left, z up. The heading is kept on purpose: raising the arm
with the waist turned twists the pelvis over the planted feet (about 10 degrees on the twin), and a
target fixed in this frame stays where it is while the robot twists under it, so the miss after a
point measures the twist and the next aim corrects it. Without an IMU the pelvis is the frame. One camera has no depth, so a pixel is a ray; with a distance
along it (the model's estimate) it is a point, and the arm can be aimed at that point from the
shoulder, which sits 12 cm right of and 19 cm below the camera.

    pixel_point(x, y, w, h, distance_m, waist, imu)   the point a pixel shows at that distance
    project(point, w, h, waist, imu)                  where a point appears in the image
    arm_ray(arm_q, waist, imu)                        the straight right arm's line: shoulder, direction
    solve_point(target, ...)                          the waist yaw and shoulder pitch whose arm line
                                                      passes through the target
    miss(target, arm_q, waist, imu)                   how far left/above the arm's line the target lies

Pure numpy: the robot and the twin run the same code; the twin's own kinematics only check it
(tests/core/test_aim.py).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from g1.core.config import HEAD_CAMERA_FOVY, HEAD_CAMERA_PITCH, HEAD_CAMERA_POS, R_ARM_CHAIN, WAIST_CHAIN


def _quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _axis(axis, a: float) -> np.ndarray:
    k = np.asarray(axis, dtype=float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def tilt(quat) -> np.ndarray:
    """The pelvis orientation from the IMU (w, x, y, z) with its heading removed: how the
    feet's frame is tipped. Level when there is no IMU."""
    if quat is None:
        return np.eye(3)
    r = _quat(quat)
    return _axis((0, 0, 1), -math.atan2(r[1, 0], r[0, 0])) @ r


def _walk(chain, angles, R: np.ndarray, p: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each link's (orientation, origin) down a chain, from a base pose."""
    out = []
    for (pos, quat, axis), a in zip(chain, angles):
        p = p + R @ np.asarray(pos)
        R = R @ _quat(quat) @ _axis(axis, a)
        out.append((R, p))
    return out


def orientation(imu) -> np.ndarray:
    """The pelvis in the IMU's frame: its quaternion (w, x, y, z), or level and straight without one."""
    return np.eye(3) if imu is None else _quat(imu)


def torso(waist: Sequence[float], imu=None) -> tuple[np.ndarray, np.ndarray]:
    """torso_link's orientation and origin; ``waist`` = (yaw, roll, pitch) rad."""
    return _walk(WAIST_CHAIN, waist, orientation(imu), np.zeros(3))[-1]


def camera(waist, imu=None) -> tuple[np.ndarray, np.ndarray]:
    """The head camera's orientation (x along its view) and position."""
    R, p = torso(waist, imu)
    return R @ _axis((0, 1, 0), HEAD_CAMERA_PITCH), p + R @ np.asarray(HEAD_CAMERA_POS)


def _focal(height: int) -> float:
    return (height / 2) / math.tan(math.radians(HEAD_CAMERA_FOVY) / 2)


def pixel_point(x_px: float, y_px: float, width: int, height: int, distance_m: float, waist, imu=None) -> np.ndarray:
    """The point the pixel shows ``distance_m`` from the camera."""
    R, c = camera(waist, imu)
    f = _focal(height)
    ray = R @ np.array([1.0, -(x_px - width / 2) / f, -(y_px - height / 2) / f])
    return c + distance_m * ray / np.linalg.norm(ray)


def project(point, width: int, height: int, waist, imu=None) -> Optional[tuple[float, float]]:
    """Where a point appears in the image (pixels; may lie outside it), or None if behind the camera."""
    R, c = camera(waist, imu)
    v = R.T @ (np.asarray(point) - c)
    if v[0] <= 1e-6:
        return None
    f = _focal(height)
    return width / 2 - f * v[1] / v[0], height / 2 - f * v[2] / v[0]


def arm_ray(arm_q: Sequence[float], waist, imu=None) -> tuple[np.ndarray, np.ndarray]:
    """The right arm's line, shoulder to wrist: the shoulder's position and the unit direction.
    ``arm_q`` = the 7 right-arm joints (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw)."""
    links = _walk(R_ARM_CHAIN, arm_q, *torso(waist, imu))
    shoulder, wrist = links[0][1], links[-1][1]
    d = wrist - shoulder
    return shoulder, d / np.linalg.norm(d)


def _angles(v) -> tuple[float, float]:
    """(left, up) of a direction, degrees: its azimuth and elevation."""
    return math.degrees(math.atan2(v[1], v[0])), math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))


def miss(target, arm_q, waist, imu=None) -> tuple[float, float]:
    """How far the target lies (left, up) of the arm's line, degrees, seen from the shoulder."""
    shoulder, d = arm_ray(arm_q, waist, imu)
    (al, au), (tl, tu) = _angles(d), _angles(np.asarray(target) - shoulder)
    return math.degrees(math.remainder(math.radians(tl - al), 2 * math.pi)), tu - au


def solve_point(target, arm_rest: Sequence[float], waist_roll: float, waist_pitch: float, imu=None, *,
                yaw0: float = 0.0, iterations: int = 20) -> tuple[float, float]:
    """(waist yaw, shoulder pitch), rad, that put the arm's line through ``target``. ``arm_rest`` is
    the 7 arm joints to hold (its shoulder pitch is the first guess); only yaw and pitch change."""
    x = np.array([yaw0, float(arm_rest[0])])

    def err(x):
        q = [x[1], *arm_rest[1:]]
        left, up = miss(target, q, (x[0], waist_roll, waist_pitch), imu)
        return np.radians([left, up])

    for _ in range(iterations):
        e = err(x)
        if np.max(np.abs(e)) < 1e-5:
            break
        h = 1e-5
        J = np.column_stack([(err(x + dx) - e) / h for dx in (np.array([h, 0]), np.array([0, h]))])
        x = x - np.linalg.lstsq(J, e, rcond=None)[0]
    return float(x[0]), float(x[1])
