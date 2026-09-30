"""Environment interface.

An Env is the thing a program's actions are sent to. Both envs expose the
same calls so the runner loop is identical:

    with env:                      # setup / teardown (release hardware, close viewer)
        obs = env.observe(env.reset())   # current joint state (29,) + latest camera frame
        ...
        obs = env.observe(env.step(action))  # apply targets, advance one control tick
    env.report()                   # optional summary after the run

Camera frames are optional: ``frame()`` returns the latest one (or None) and
``clock()`` is the timebase its ``stamp`` is on, so ``observe`` can compute the
frame's age. The runner sets ``use_camera`` before ``setup`` from
``Runnable.uses_camera``; envs only open a camera when it is set.
"""
from __future__ import annotations

import argparse
import math
import signal
import time
from contextlib import contextmanager
from typing import ClassVar

import numpy as np

from g1.camera import Frame
from g1.core.action import Action, Obs


class EnvAbort(Exception):
    """Raised by an env to stop the run early; the runner still calls report()."""


@contextmanager
def shield_sigint(message: str):
    """Ignore Ctrl-C for the duration (it only prints ``message``), so the safe
    return and the robot's hand-over can never be cut short by a key press.
    There is deliberately no escape hatch. No-op off the main thread."""
    def handler(signum, frame):
        print(f"\n{message}", flush=True)
    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:                 # not the main thread: signals are not ours to handle
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


class Env:
    name: ClassVar[str] = "env"
    use_camera: bool = False

    @property
    def can_walk(self) -> bool:
        """Whether this env accepts Action.base right now."""
        return False

    @property
    def has_loco(self) -> bool:
        """Whether this env can run Action.command (onboard LocoClient gestures)."""
        return False

    def base_pose(self):
        """The env's own (x, y, yaw) base estimate, or None."""
        return None

    def joint_vel(self):
        """Measured joint velocities (29,) rad/s, or None."""
        return None

    def joint_torque(self):
        """Measured joint torques (29,) N m, or None."""
        return None

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register env-specific CLI flags."""

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "Env":
        self.setup()
        return self

    def __exit__(self, *exc) -> None:
        self.teardown()

    def setup(self) -> None: ...
    def teardown(self) -> None: ...

    # -- control -----------------------------------------------------------
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: Action) -> np.ndarray:
        raise NotImplementedError

    def report(self) -> bool:
        """Print a summary; return False if the run should count as failed."""
        return True

    # -- observation -------------------------------------------------------
    def frame(self) -> Frame | None:
        """Latest camera frame, or None. Never blocks."""
        return None

    def clock(self) -> float:
        """Seconds on the timebase frame stamps use (wall clock by default)."""
        return time.monotonic()

    def observe(self, q: np.ndarray) -> Obs:
        f = self.frame()
        now = self.clock()
        age = math.inf if f is None else max(0.0, now - f.stamp)
        return Obs(q, f, age, self.base_pose(), self.joint_vel(), self.joint_torque())
