"""Environment interface.

An Env is the thing a policy's actions are sent to. All three stages expose the
same four calls so the runner loop is identical:

    with env:                      # setup / teardown (release hardware, close viewer)
        q = env.reset()            # current joint state (29,)
        ...
        q = env.step(action)       # apply targets, advance one control tick, return new state
    env.report()                   # optional summary after the run
"""
from __future__ import annotations

import argparse
from typing import ClassVar

import numpy as np

from policy import Action


class EnvAbort(Exception):
    """Raised by an env to stop the run early; the runner still calls report()."""


class Env:
    name: ClassVar[str] = "env"

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
