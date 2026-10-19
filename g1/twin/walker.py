"""The walker: Unitree's trained G1 gait network (unitree_rl_gym), the stand-in
for the robot's own closed-source locomotion controller.

At 50 Hz it reads the pelvis orientation and angular velocity, the 12 leg joints
and a velocity command, and returns 12 leg targets for a PD loop. It is an LSTM
whose state lives in the TorchScript module, so each ``Walker`` keeps its own
copy and swaps it in around every call: two twins never share a gait, and a run
is repeatable.

Its observation has no base velocity, so on its own it drifts (it was trained
with the arms fixed in one pose). ``Onboard`` closes that loop with the twin's
velocity estimate, as the robot's controller does with its state estimator.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from g1.core.config import CONTROL_DT
from g1.twin import assets


@dataclass(frozen=True)
class Trained:
    """Fixed by training, not tunable: unitree_rl_gym deploy/deploy_mujoco/configs/g1.yaml
    and legged_gym/envs/base/legged_robot_config.py at commit 276801e."""
    kp: tuple = (100, 100, 100, 150, 40, 40) * 2          # hip pitch/roll/yaw, knee, ankle pitch/roll; per leg
    kd: tuple = (2, 2, 2, 4, 2, 2) * 2
    default: tuple = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2   # rad: the stance the network acts around
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    action_scale: float = 0.25
    cmd_scale: tuple = (2.0, 2.0, 0.25)
    cmd_max: tuple = (1.0, 1.0, 1.0)                       # m/s, m/s, rad/s: the command range it was trained on
    gait_period_s: float = 0.8
    hidden: int = 64


G1 = Trained()
KP = np.array(G1.kp, dtype=float)
KD = np.array(G1.kd, dtype=float)
DEFAULT = np.array(G1.default, dtype=float)

_NET = None


def network():
    """The TorchScript module, loaded once per process."""
    global _NET
    if _NET is None:
        import torch
        assets.require()
        torch.set_num_threads(1)              # one thread: repeatable, and no contention with the render
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)   # torch.jit.load's deprecation notice
            _NET = torch.jit.load(str(assets.WALKER))
        _NET.eval()
    return _NET


def gravity_in_body(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


class Walker:
    def __init__(self) -> None:
        import torch
        self.torch = torch
        self.net = network()
        self.reset()

    def reset(self) -> None:
        t = self.torch
        self.h = t.zeros(1, 1, G1.hidden)
        self.c = t.zeros(1, 1, G1.hidden)
        self.action = np.zeros(12, dtype=np.float32)
        self.phase_t = 0.0

    def targets(self, cmd, quat, omega, q_legs, dq_legs) -> np.ndarray:
        """One 50 Hz step: the 12 leg targets (rad). ``omega`` is the pelvis angular
        velocity in its own frame, ``cmd`` (vx, vy, vyaw) in the pelvis frame."""
        self.phase_t += CONTROL_DT
        phase = self.phase_t % G1.gait_period_s / G1.gait_period_s
        cmd = np.clip(np.asarray(cmd, dtype=float), -np.array(G1.cmd_max), np.array(G1.cmd_max))
        obs = np.concatenate([
            np.asarray(omega) * G1.ang_vel_scale,
            gravity_in_body(quat),
            cmd * np.array(G1.cmd_scale),
            (np.asarray(q_legs) - DEFAULT) * G1.dof_pos_scale,
            np.asarray(dq_legs) * G1.dof_vel_scale,
            self.action,
            [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
        ]).astype(np.float32)
        t, net = self.torch, self.net
        with t.no_grad():
            net.hidden_state.copy_(self.h)
            net.cell_state.copy_(self.c)
            out = net(t.from_numpy(obs).unsqueeze(0))
            self.h = net.hidden_state.clone()
            self.c = net.cell_state.clone()
        self.action = out.numpy().squeeze().astype(np.float32)
        return self.action * G1.action_scale + DEFAULT
