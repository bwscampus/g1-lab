"""Environment registry. ``--env <name>`` picks one of these.

Two stages: ``sim`` (MuJoCo, and the checks that gate a run) then ``robot``.
There is no separate check env — ``--env sim --headless`` is the fast,
windowless pre-flight, and it validates more than the old one did because it
watches the *measured* joint angles as well as the commands.
"""
from __future__ import annotations

from envs.base import Env, EnvAbort, shield_sigint
from envs.monitor import JointMonitor, Violation
from envs.robot import RobotEnv
from envs.sim import SimEnv

ENVS: dict[str, type[Env]] = {
    SimEnv.name: SimEnv,
    RobotEnv.name: RobotEnv,
}

__all__ = ["Env", "EnvAbort", "ENVS", "SimEnv", "RobotEnv", "JointMonitor", "Violation"]
