"""The twin: a MuJoCo G1 with a free base, torque motors and the onboard
controller, that real robot code talks to through the SDK's interfaces.

    twin = Twin(scene="room", fsm=200)
    twin.call("sport", 7101, '{"data": 200}')     # what LocoClient.SetFsmId sends
    twin.on_arm_sdk(ArmSdkCommand.of(lowcmd))      # what rt/arm_sdk carries
    twin.advance(0.02)                             # physics at 500 Hz, the controller at 50 Hz
    twin.lowstate()                                # what rt/lowstate carries

Two transports drive it: ``loopback.py`` (in-process, lockstep on sim time:
the tests) and ``bridge.py`` (DDS on the loopback interface,
real time: ``g1 twin``). One lock guards the state, so DDS handler threads and
the physics loop never interleave mid-step.

``--mode gantry`` hangs the torso from a rope above its standing height (slack
while it stands, taut when it sags), so Damp leaves it hanging instead of fallen.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from g1.core import limits
from g1.core.config import CONTROL_DT, NUM_JOINTS
from g1.twin import model as twin_model
from g1.twin.onboard import FSM_MAIN, STAND, ArmSdkCommand, Onboard
from g1.twin.services import Services

SUBSTEPS = int(round(CONTROL_DT / twin_model.PHYSICS_DT))
SETTLE_S = 1.0                    # dropped onto its feet and let settle before t = 0
START_Z = 0.80                    # pelvis height it is dropped from (stands at about 0.78)
BAND_KP = limits.get("twin_band_kp")
BAND_KD = limits.get("twin_band_kd")


@dataclass
class LowState:
    """One rt/lowstate sample: per motor q, dq, tau_est; the pelvis IMU."""
    tick: int
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    quat: np.ndarray              # w, x, y, z
    gyro: np.ndarray              # rad/s, pelvis frame
    acc: np.ndarray               # m/s^2, pelvis frame


class Twin:
    def __init__(self, *, scene: str = "none", objects=(), obstacle=None, camera_size=None,
                 fsm: int = FSM_MAIN, band: bool = False, voice: str = "sim") -> None:
        import mujoco
        self.mujoco = mujoco
        self.model = twin_model.load_model(obstacle, scene=scene, objects=objects, camera_size=camera_size)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.band = band
        self.start_fsm = fsm
        self.onboard = Onboard(self.model, fsm)
        self.services = Services(self.onboard, lambda: float(self.data.time), voice, summary=self.report)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._sensor = {n: (self.model.sensor(n).adr[0], self.model.sensor(n).dim[0])
                        for n in ("imu_quat", "imu_gyro", "imu_acc")}
        self.reset()

    def reset(self) -> None:
        """Standing at the origin facing +x, settled, at t = 0, in the starting FSM."""
        m, d = self.model, self.data
        with self.lock:
            self.mujoco.mj_resetData(m, d)
            d.qpos[2] = START_Z
            d.qpos[7:7 + NUM_JOINTS] = STAND
            self.mujoco.mj_forward(m, d)
            self.onboard.reset(FSM_MAIN)            # stand it up, whatever it will start in
            self._anchor = None
            for _ in range(int(SETTLE_S / twin_model.PHYSICS_DT)):
                self._substep()
            self._anchor = (float(d.xpos[self.torso][0]), float(d.xpos[self.torso][1]),
                            float(d.xpos[self.torso][2]) - 0.01)
            d.time = 0.0
            self.onboard.reset(self.start_fsm)
            self.services.log.clear()
            self.steps = 0
            self.mark_start()

    # -- the SDK's side ---------------------------------------------------------------------------
    def call(self, service: str, api: int, parameter: str) -> tuple[int, str]:
        with self.lock:
            return self.services.call(service, api, parameter)

    def on_arm_sdk(self, cmd: ArmSdkCommand) -> None:
        with self.lock:
            self.onboard.on_arm_sdk(cmd)

    def lowstate(self) -> LowState:
        d = self.data
        s = lambda n: d.sensordata[self._sensor[n][0]:self._sensor[n][0] + self._sensor[n][1]].copy()
        with self.lock:
            return LowState(self.steps, d.qpos[7:7 + NUM_JOINTS].copy(), d.qvel[6:6 + NUM_JOINTS].copy(),
                            d.actuator_force.copy(), s("imu_quat"), s("imu_gyro"), s("imu_acc"))

    # -- physics ----------------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self) -> None:
        """One control tick: the controller's 50 Hz update, then SUBSTEPS physics steps."""
        with self.lock:
            self.onboard.tick(self.data, self.time)
            for _ in range(SUBSTEPS):
                self._substep()
            self.steps += 1

    def advance(self, dt: float) -> None:
        end = self.time + dt - 1e-9
        while self.time < end:
            self.step()

    def _substep(self) -> None:
        d = self.data
        d.ctrl[:] = self.onboard.torques(d, float(d.time))
        if self.band and self._anchor is not None:
            self._rope()
        self.mujoco.mj_step(self.model, d)

    def _rope(self) -> None:
        """A rope from above the standing torso: pulls only when the torso sags below it."""
        d, b = self.data, self.torso
        ax, ay, az = self._anchor
        x, y, z = d.xpos[b]
        vel = np.zeros(6)
        self.mujoco.mj_objectVelocity(self.model, d, self.mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
        f = np.zeros(3)
        if z < az:
            f[2] = BAND_KP * (az - z) - BAND_KD * vel[5]
            f[0] = 0.2 * BAND_KP * (ax - x) - 0.2 * BAND_KD * vel[3]
            f[1] = 0.2 * BAND_KP * (ay - y) - 0.2 * BAND_KD * vel[4]
        d.xfrc_applied[b, :3] = f

    # -- what a report needs --------------------------------------------------------------------
    def truth(self) -> tuple[float, float, float]:
        """The pelvis (x, y, yaw) in the world, from the simulator, not an estimate."""
        w, x, y, z = self.data.qpos[3:7]
        return (float(self.data.qpos[0]), float(self.data.qpos[1]),
                math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def mark_start(self) -> None:
        """Measure base_pose() from here (the program's start)."""
        self.start = self.truth()

    def base_pose(self) -> tuple[float, float, float]:
        """Truth relative to where the run started, in the start's heading frame."""
        x0, y0, yaw0 = self.start
        x, y, yaw = self.truth()
        c, s = math.cos(yaw0), math.sin(yaw0)
        dx, dy = x - x0, y - y0
        return (c * dx + s * dy, -s * dx + c * dy, math.remainder(yaw - yaw0, 2 * math.pi))

    @property
    def pelvis_z(self) -> float:
        return float(self.data.qpos[2])

    def report(self) -> dict:
        """What the twin knows that a robot cannot say: the Summary call of its ``twin`` service."""
        return {"fallen_at": self.onboard.fallen_at, "base_pose": list(self.base_pose()),
                "pelvis_z": self.pelvis_z, "fsm": self.onboard.fsm, "summary": self.summary()}

    def summary(self) -> str:
        o = self.onboard
        x, y, yaw = self.base_pose()
        parts = [f"pelvis z = {self.pelvis_z:.3f} m", f"FSM {o.fsm}",
                 f"base at ({x:+.2f}, {y:+.2f}) m yaw {math.degrees(yaw):+.0f} deg (truth)"]
        if o.walk_ticks:
            parts.append(f"walked {o.walk_ticks * CONTROL_DT:.1f} s")
        if o.recoveries:
            parts.append(f"{o.recoveries} recovery step(s)")
        sat = {i: n for i, n in enumerate(o.saturated) if n}
        if sat:
            from g1.core.config import JOINT_NAMES
            worst = sorted(sat.items(), key=lambda kv: -kv[1])[:3]
            parts.append("torque limit hit: " + ", ".join(f"{JOINT_NAMES[i]} {n * twin_model.PHYSICS_DT:.2f} s"
                                                          for i, n in worst))
        if o.waist_raised:
            parts.append(f"waist held at kp >= {limits.get('twin_walk_waist_kp'):g} while walking, "
                         f"{o.waist_raised * twin_model.PHYSICS_DT:.1f} s (limit twin_walk_waist_kp)")
        if o.fallen_at is not None:
            parts.append(f"FELL at t = {o.fallen_at:.2f} s")
        return "; ".join(parts)
