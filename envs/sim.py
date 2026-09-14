"""Stage 2: MuJoCo replay in the interactive viewer.

Loads the Menagerie ``unitree_g1`` scene (position actuators, kp=500), starts
at the ``stand`` keyframe and drives the actuators with the policy's targets,
blended with the hold pose by the arm_sdk weight exactly as the check env does.
Joints the policy does not command hold the keyframe pose.

By default the pelvis is welded to the world so an arm-only policy can be
viewed without the robot needing to balance. Pass ``--free-base`` to drop the
weld (the stiff PD on the legs keeps it standing for a while, but it will not
balance).

macOS: the viewer must run under ``mjpython``:
    mjpython run.py --env sim --policy tpose
Pass ``--headless`` to run the physics without a window (e.g. in CI).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from config import CONTROL_DT, NUM_JOINTS
from policy import Action
from envs.base import Env

_LOCAL_MENAGERIE = Path.home() / "Robotics" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


def load_model():
    """Return an MjModel of the G1 scene. Priority: $G1_MJCF, pip menagerie, ~/Robotics clone."""
    import mujoco

    env_path = os.environ.get("G1_MJCF")
    if env_path:
        return mujoco.MjModel.from_xml_path(env_path)
    try:
        import mujoco_menagerie
        return mujoco_menagerie.load("unitree_g1")
    except Exception:
        pass
    if _LOCAL_MENAGERIE.exists():
        return mujoco.MjModel.from_xml_path(str(_LOCAL_MENAGERIE))
    raise FileNotFoundError(
        "No G1 MJCF found. Set G1_MJCF=/path/to/unitree_g1/scene.xml or pip install mujoco-menagerie")


class SimEnv(Env):
    name = "sim"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("sim")
        g.add_argument("--headless", action="store_true", help="run physics without the viewer")
        g.add_argument("--free-base", action="store_true",
                       help="do not weld the pelvis to the world")
        g.add_argument("--realtime", type=float, default=1.0,
                       help="playback speed multiplier (default 1.0)")
        g.add_argument("--hold-end", type=float, default=2.0,
                       help="seconds to keep the viewer open after the policy finishes")

    def setup(self) -> None:
        import mujoco
        import mujoco.viewer

        self.mujoco = mujoco
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        if self.model.nu != NUM_JOINTS:
            raise RuntimeError(f"expected {NUM_JOINTS} actuators, model has {self.model.nu}")
        # Menagerie puts the free joint first, so joint i sits at qpos[7 + i].
        self.qpos_idx = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
                                  for i in range(self.model.nu)])
        self.qvel_idx = np.array([self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
                                  for i in range(self.model.nu)])
        self.substeps = max(1, int(round(CONTROL_DT / self.model.opt.timestep)))

        self._pin_base = not self.args.free_base
        self.viewer = None
        if not self.args.headless:
            if sys.platform == "darwin" and not getattr(mujoco.viewer, "_MJPYTHON", None):
                raise SystemExit("On macOS the viewer needs mjpython:\n"
                                 "    mjpython run.py --env sim ...\n"
                                 "or pass --headless.")
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def teardown(self) -> None:
        if self.viewer is not None:
            t_end = time.time() + self.args.hold_end
            while self.viewer.is_running() and time.time() < t_end:
                self._physics_tick()
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)
            self.viewer.close()

    def reset(self) -> np.ndarray:
        m, d = self.model, self.data
        key = self.mujoco.mj_name2id(m, self.mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.mujoco.mj_resetDataKeyframe(m, d, key)
        self._base_qpos = d.qpos[:7].copy()
        self.hold = d.qpos[self.qpos_idx].copy()
        d.ctrl[:] = self.hold
        self.mujoco.mj_forward(m, d)
        if self.viewer is not None:
            self.viewer.sync()
        self._wall = time.time()
        return self._q()

    def _q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def _physics_tick(self) -> None:
        d = self.data
        if self._pin_base:
            # Pin the pelvis by overriding the free-joint state each substep
            # (no model edit needed, so the same MJCF serves both modes).
            d.qpos[:7] = self._base_qpos
            d.qvel[:6] = 0.0
        self.mujoco.mj_step(self.model, d)

    def step(self, action: Action) -> np.ndarray:
        if self.viewer is not None and not self.viewer.is_running():
            raise KeyboardInterrupt("viewer closed")
        j = list(action.joints)
        ctrl = self.hold.copy()
        ctrl[j] = (1.0 - action.weight) * self.hold[j] + action.weight * action.q[j]
        self.data.ctrl[:] = ctrl
        for _ in range(self.substeps):
            self._physics_tick()
        if self.viewer is not None:
            self.viewer.sync()
            # pace to wall-clock
            self._wall += CONTROL_DT / max(self.args.realtime, 1e-6)
            lag = self._wall - time.time()
            if lag > 0:
                time.sleep(lag)
        return self._q()

    def report(self) -> bool:
        q = self._q()
        print(f"sim: finished at sim time {self.data.time:.2f} s; "
              f"pelvis z = {self.data.qpos[2]:.3f} m")
        return bool(np.all(np.isfinite(q)))
