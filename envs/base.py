"""Environment interface.

An Env is the thing a policy's actions are sent to. All three stages expose the
same calls so the runner loop is identical:

    with env:                      # setup / teardown (release hardware, close viewer)
        obs = env.observe(env.reset())   # current joint state (29,) + latest camera frame
        ...
        obs = env.observe(env.step(action))  # apply targets, advance one control tick
    env.report()                   # optional summary after the run

Camera frames are optional: ``frame()`` returns the latest one (or None) and
``clock()`` is the timebase its ``stamp`` is on, so ``observe`` can compute the
frame's age. The runner sets ``use_camera`` before ``setup`` from
``Policy.uses_camera``; envs only open a camera when it is set. A vision model
is plugged in the same way: the runner sets ``perceiver``, ``observe`` offers it
each frame and attaches its latest ``Percept`` with the age of the frame that
percept describes (on the same clock, so the model's latency is included). The
model only runs when a policy asks (``perceiver.request``).
"""
from __future__ import annotations

import argparse
import math
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from camera import Frame
from policy import Action, Obs

if TYPE_CHECKING:
    from perception import Perceiver


class EnvAbort(Exception):
    """Raised by an env to stop the run early; the runner still calls report()."""


class Env:
    name: ClassVar[str] = "env"
    use_camera: bool = False
    perceiver: "Perceiver | None" = None

    @property
    def can_walk(self) -> bool:
        """Whether this env accepts Action.base right now."""
        return False

    def base_pose(self):
        """The env's own (x, y, yaw) base estimate, or None."""
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
        p = None
        if self.perceiver is not None:
            if f is not None:
                self.perceiver.offer(f)
            p = self.perceiver.latest()
        p_age = math.inf if p is None else max(0.0, now - p.frame_stamp)
        return Obs(q, f, age, p, p_age, self.perceiver, self.base_pose())
