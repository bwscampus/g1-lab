"""Environment registry. ``--env <name>`` picks one of these."""
from __future__ import annotations

from envs.base import Env, EnvAbort
from envs.check import CheckEnv
from envs.robot import RobotEnv
from envs.sim import SimEnv

ENVS: dict[str, type[Env]] = {
    CheckEnv.name: CheckEnv,
    SimEnv.name: SimEnv,
    RobotEnv.name: RobotEnv,
}

__all__ = ["Env", "EnvAbort", "ENVS", "CheckEnv", "SimEnv", "RobotEnv"]
