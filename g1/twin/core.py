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
from g1.twin.onboard import FSM_MAIN, FSM_SIT, STAND, ArmSdkCommand, Onboard
from g1.twin.services import Services

SUBSTEPS = int(round(CONTROL_DT / twin_model.PHYSICS_DT))
SETTLE_S = limits.get("twin_settle_s")         # dropped onto its feet and let settle before t = 0
START_Z = limits.get("twin_start_z_m")         # pelvis height it is dropped from
BAND_KP = limits.get("twin_band_kp")
BAND_KD = limits.get("twin_band_kd")
ROPE_SLACK = limits.get("twin_rope_slack_m")   # below the standing torso: slack in any stance, so the feet carry it
ROPE_LATERAL = limits.get("twin_rope_lateral")  # sideways stiffness and damping, as a fraction


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
                 fsm: int = FSM_MAIN, band: bool = False, voice: str = "sim", start=None) -> None:
        import mujoco
        self.mujoco = mujoco
        self.model = twin_model.load_model(obstacle, scene=scene, objects=objects, camera_size=camera_size)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.band = band
        self.start_pose = tuple(float(v) for v in start) if start else (0.0, 0.0, 0.0)   # x, y, yaw_deg
        self.start_fsm = fsm
        self.onboard = Onboard(self.model, fsm)
        self.services = Services(self.onboard, lambda: float(self.data.time), voice, summary=self.report)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._sensor = {n: (self.model.sensor(n).adr[0], self.model.sensor(n).dim[0])
                        for n in ("imu_quat", "imu_gyro", "imu_acc")}
        self.reset()

    def reset(self) -> None:
        """Standing at the start pose (the origin facing +x by default), settled, at t = 0, in the
        starting FSM."""
        m, d = self.model, self.data
        with self.lock:
            self.mujoco.mj_resetData(m, d)
            x0, y0, yaw = self.start_pose
            half = math.radians(yaw) / 2
            d.qpos[0:3] = (x0, y0, START_Z)
            d.qpos[3:7] = (math.cos(half), 0.0, 0.0, math.sin(half))
            d.qpos[7:7 + NUM_JOINTS] = STAND
            self.mujoco.mj_forward(m, d)
            self.onboard.reset(FSM_MAIN)            # stand it up, whatever it will start in: the walker
            self._anchor = None                     # balances it through its full loop, 50 Hz and 500 Hz
            for _ in range(int(round(SETTLE_S / CONTROL_DT))):
                self.onboard.tick(d, float(d.time))
                for _ in range(SUBSTEPS):
                    self._substep()
            self._anchor = (float(d.xpos[self.torso][0]), float(d.xpos[self.torso][1]),
                            float(d.xpos[self.torso][2]) - ROPE_SLACK)
            d.time = 0.0
            self.onboard.reset(self.start_fsm, keep_gait=True)   # the walker keeps its history
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
        self.onboard.hanging = bool(z < az)          # the rope carries it: a tilt is a swing, not a fall
        if z < az:
            f[2] = BAND_KP * (az - z) - BAND_KD * vel[5]
            f[0] = ROPE_LATERAL * (BAND_KP * (ax - x) - BAND_KD * vel[3])
            f[1] = ROPE_LATERAL * (BAND_KP * (ay - y) - BAND_KD * vel[4])
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
        from g1.core.config import JOINT_NAMES
        return {"fallen_at": self.onboard.fallen_at, "base_pose": list(self.base_pose()),
                "pelvis_z": self.pelvis_z, "fsm": self.onboard.fsm,
                "torque_limit_s": {JOINT_NAMES[i]: round(int(n) * twin_model.PHYSICS_DT, 4)
                                   for i, n in enumerate(self.onboard.saturated) if n},
                "summary": self.summary()}

    def summary(self) -> str:
        o = self.onboard
        x, y, yaw = self.base_pose()
        parts = [f"pelvis z = {self.pelvis_z:.3f} m", f"FSM {o.fsm}",
                 f"base at ({x:+.2f}, {y:+.2f}) m yaw {math.degrees(yaw):+.0f} deg (truth)"]
        if o.walk_ticks:
            parts.append(f"walked {o.walk_ticks * CONTROL_DT:.1f} s")
        sat = {i: n for i, n in enumerate(o.saturated) if n}
        if sat:
            from g1.core.config import JOINT_NAMES
            worst = sorted(sat.items(), key=lambda kv: -kv[1])[:3]
            parts.append("torque limit hit: " + ", ".join(f"{JOINT_NAMES[i]} {n * twin_model.PHYSICS_DT:.2f} s"
                                                          for i, n in worst))
        if o.waist_raised:
            parts.append(f"waist held by the walker over arm_sdk for {o.waist_raised * twin_model.PHYSICS_DT:.1f} s "
                         f"(the robot holds it steady while it walks)")
        if o.fsm == FSM_SIT and o.fallen_at is None:
            seat = self.seat()
            parts.append(f"seated on {seat}" if seat else "seated with no seat under it")
        if o.fallen_at is not None:
            parts.append(f"FELL at t = {o.fallen_at:.2f} s")
        return "; ".join(parts)

    def seat(self) -> str | None:
        """The chair seat the robot rests on (a ``*_seat`` geom it touches), if any."""
        m, d = self.model, self.data
        for c in d.contact[:d.ncon]:
            for g in (c.geom1, c.geom2):
                name = m.geom(g).name
                if name.endswith("_seat"):
                    return name[:-len("_seat")]
        return None
