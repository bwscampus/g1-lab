"""The twin in-process: ``RobotEnv`` with its link swapped for the loopback
(``g1/twin/loopback.py``), stepped in lockstep on sim time. It is the test suite's
env, not a user command: headless it runs about 30x real time and two runs are
bit-identical, so ~130 tests run in a few minutes. ``g1 run TASK --sim`` reaches the
same twin over DDS instead (``g1/twin/main.py:TwinProcess``).

Everything the robot code does is the same: the bring-up, the rt/arm_sdk messages,
LocoClient.Move from the BaseCommander, the AudioClient calls, the strict monitor
and the twin's verdict (``RobotEnv`` with ``args.sim``). The settings come from
``g1.cli.run_settings(..., **overrides)``; ``mode`` gantry hangs it from a rope,
``camera_dir`` replays image files instead of rendering, ``viewer`` opens the
MuJoCo window (mjpython on macOS).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from g1.camera import Camera, DirCamera
from g1.core.config import HEAD_CAMERA_SIZE
from g1.envs.base import Env
from g1.envs.robot import RobotEnv


def _size(text: str) -> tuple[int, int]:
    try:
        h, w = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected HxW, e.g. 720x1280") from None
    return h, w


def _xyz(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return parts[0], parts[1], parts[2]


class SimEnv(RobotEnv):
    name = "sim"

    def __init__(self, args: argparse.Namespace) -> None:
        Env.__init__(self, args)
        self.link = None                   # the loopback, once setup() has built the twin
        self.twin = None
        self.replay = None
        self.renderer = None
        self._t0 = 0.0

    # -- what RobotEnv asks of its subclass ------------------------------------------------------
    @property
    def can_walk(self) -> bool:
        return (self.args.mode or "standing") == "standing"      # known before setup(): the runner gates on it

    def _check_args(self) -> None:
        if self.args.mode is None:
            self.args.mode = "standing"
        self.args.sim = True               # RobotEnv's strict monitor and the twin's verdict
        self.args.iface = "twin"

    def _open_camera(self):
        if self.args.camera_dir is not None:
            self.replay = DirCamera(self.args.camera_dir, fps=self.args.camera_fps)
            self.replay.start()            # a recorded feed replaces the render
            return self.replay
        return self.link.camera

    # -- lifecycle --------------------------------------------------------------------------------
    def setup(self) -> None:
        import mujoco
        import mujoco.viewer
        from g1.twin.core import Twin
        from g1.twin.loopback import LoopbackLink
        from g1.twin.onboard import FSM_DAMP, FSM_MAIN

        self._check_args()
        gantry = self.args.mode == "gantry"
        fsm = self.args.twin_fsm if self.args.twin_fsm is not None else (FSM_DAMP if gantry else FSM_MAIN)
        self.twin = Twin(scene=self.args.scene, objects=self.args.sim_objects, obstacle=self.args.sim_obstacle,
                         camera_size=self.args.camera_size, fsm=fsm, band=gantry, voice="sim")
        viewer = None
        if not self.args.headless:
            if sys.platform == "darwin" and not getattr(mujoco.viewer, "_MJPYTHON", None):
                raise SystemExit("On macOS the MuJoCo window needs mjpython (run the tests headless)")
            viewer = mujoco.viewer.launch_passive(self.twin.model, self.twin.data)
        realtime = self.args.realtime
        if realtime is None and viewer is not None:
            realtime = 1.0
        renderer = camera = None
        if self.use_camera and self.args.camera_dir is None:
            renderer = mujoco.Renderer(self.twin.model, *(self.args.camera_size or HEAD_CAMERA_SIZE))
            camera = Camera()
        self.renderer = renderer
        self.link = LoopbackLink(self.twin, realtime=realtime, viewer=viewer, renderer=renderer, camera=camera,
                                 camera_every=self.args.camera_every, stamp=self.clock)
        super().setup()

    def reset(self) -> np.ndarray:
        q = super().reset()                 # the robot's own hand-over: release, a fresh LowState
        self._t0 = self.twin.time           # the program's clock and base pose start here
        self.twin.mark_start()
        self.monitor.reset()
        return q

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            link = self.link
            viewer = getattr(link, "viewer", None)
            if viewer is not None:
                t_end = time.time() + self.args.hold_end
                link.realtime = link.realtime or 1.0
                try:
                    while viewer.is_running() and time.time() < t_end:
                        link.tick()
                except KeyboardInterrupt:
                    pass
                viewer.close()
            if self.renderer is not None:
                self.renderer.close()

    # -- observation ------------------------------------------------------------------------------
    def frame(self):
        if self.replay is not None:
            return self.replay.poll(self.clock())
        return None if self.camera is None else self.camera.latest()

    def source(self):
        if self.replay is not None:
            return self.replay
        return self.camera

    def clock(self) -> float:
        return 0.0 if self.twin is None else self.twin.time - self._t0

    def base_pose(self):
        return None if self.twin is None else self.twin.base_pose()

    def _true_base(self):
        return self.base_pose()

    @property
    def model(self):
        return self.twin.model

    @property
    def data(self):
        return self.twin.data

    # the monitor is the run's verdict; forward what callers and tests read
    @property
    def violations(self):
        return self.monitor.violations

    @property
    def q_min(self):
        return self.monitor.q_min

    @property
    def q_max(self):
        return self.monitor.q_max

    @property
    def cmd_min(self):
        return self.monitor.cmd_min

    @property
    def cmd_max(self):
        return self.monitor.cmd_max

    @property
    def peak_vel(self):
        return self.monitor.peak_vel

    @property
    def base_path(self) -> float:
        return self.monitor.base_path

    @property
    def base_ticks(self) -> int:
        return self.monitor.base_ticks

    @property
    def base_peak(self):
        return self.monitor.base_peak
