"""The walker: NVIDIA's GR00T whole-body controller (GR00T-WholeBodyControl,
``decoupled_wbc``, the HOMIE line), the stand-in for the robot's own closed-source
locomotion controller.

Two small networks run at 50 Hz on the CPU (onnxruntime): **Balance** while the
command is (almost) zero, so the robot stands still instead of marching in place,
and **Walk** otherwise. Each reads the last 6 frames of: the command (vx, vy,
vyaw), a body height and torso orientation target, the pelvis angular velocity
and tilt, all 29 joint angles and velocities (so it sees what the arms are doing)
and its own last action; it returns targets for the 12 leg joints and the waist.

It tracks a velocity command only roughly on this model (it was trained on one
without torque limits), so ``Onboard`` closes that loop with the twin's velocity
estimate, as the robot's controller does with its state estimator. Measured
against the previous walker (unitree_rl_gym) on the twin, the pair stands 7x
calmer after a move, stands still within about a second of a stop, takes a 120 N
sideways push the old one fell to, and lands a 3 m walk within 4 cm.

Each ``Walker`` keeps its own history, so two twins never share one and a run
is repeatable. Weights: NVIDIA Open Model License (``g1 twin fetch``).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from g1.twin import assets


@dataclass(frozen=True)
class Trained:
    """Fixed by training, not tunable: GR00T-WholeBodyControl decoupled_wbc/sim2mujoco
    resources/robots/g1/g1_gear_wbc.yaml and scripts/run_mujoco_gear_wbc.py at 45c03fa."""
    kp: tuple = (150, 150, 150, 200, 40, 40) * 2 + (250, 250, 250)   # legs (hip p/r/y, knee, ankle p/r), waist
    kd: tuple = (2, 2, 2, 4, 2, 2) * 2 + (5, 5, 5)
    default: tuple = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2 + (0.0, 0.0, 0.0)   # rad: the stance it acts around
    ang_vel_scale: float = 0.5
    dof_vel_scale: float = 0.05
    action_scale: float = 0.25
    cmd_scale: tuple = (2.0, 2.0, 0.5)
    height: float = 0.74                       # m: the body height it is asked to hold
    history: int = 6                           # frames
    frame: int = 86                            # values per frame
    walk_above: float = 0.05                   # |command| above this runs Walk, at or below Balance


G1 = Trained()
KP = np.array(G1.kp, dtype=float)
KD = np.array(G1.kd, dtype=float)
DEFAULT = np.array(G1.default, dtype=float)
JOINTS = len(G1.default)                       # 15: legs and waist

_NETS = None


def networks():
    """The Balance and Walk sessions, loaded once per process, one thread each (repeatable,
    and no contention with the render)."""
    global _NETS
    if _NETS is None:
        import onnxruntime as ort
        assets.require()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        load = lambda p: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        _NETS = (load(assets.WALKER_BALANCE), load(assets.WALKER_WALK))
    return _NETS


def gravity_in_body(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


class Walker:
    def __init__(self) -> None:
        self.balance, self.walk = networks()
        self.input = self.balance.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        self.history = collections.deque([np.zeros(G1.frame, np.float32)] * G1.history, maxlen=G1.history)
        self.action = np.zeros(JOINTS, dtype=np.float32)
        self.walking = False                   # the last step ran Walk

    def targets(self, cmd, quat, omega, q, dq) -> np.ndarray:
        """One 50 Hz step: the 15 leg and waist targets (rad). ``cmd`` (vx, vy, vyaw) is in the
        pelvis heading frame, ``omega`` the pelvis angular velocity in its own frame, ``q``/``dq``
        all 29 joints in DDS order."""
        cmd = np.asarray(cmd, dtype=float)
        o = np.zeros(G1.frame, np.float32)
        o[0:3] = cmd * np.array(G1.cmd_scale)
        o[3] = G1.height                       # o[4:7]: the torso orientation target, level
        o[7:10] = np.asarray(omega) * G1.ang_vel_scale
        o[10:13] = gravity_in_body(quat)
        qs = np.array(q, dtype=float)
        qs[:JOINTS] -= DEFAULT                 # the arms are measured from zero, as in training
        o[13:42] = qs
        o[42:71] = np.asarray(dq) * G1.dof_vel_scale
        o[71:86] = self.action
        self.history.append(o)
        self.walking = bool(np.linalg.norm(cmd) > G1.walk_above)
        net = self.walk if self.walking else self.balance
        x = np.concatenate(self.history)[None].astype(np.float32)
        self.action = net.run(None, {self.input: x})[0].squeeze().astype(np.float32)
        return self.action * G1.action_scale + DEFAULT
