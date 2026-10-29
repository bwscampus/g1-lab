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

A task's ``success`` (task.json) is scored here, every control tick, from the
simulator's truth (``Evaluator``): the robot cannot say whether it pointed at the
laptop, the twin can. The verdict reaches the records through the Summary call,
never the model.
"""
from __future__ import annotations

import math
import threading
import time
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
TRUTH_PERIOD = limits.get("twin_truth_period_s")   # the base truth kept for the host's trace
# the right arm's pointing ray: shoulder to wrist (the arm is straight when it points)
ARM_RAY = ("right_shoulder_pitch_link", "right_wrist_yaw_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
SUCCESS_KEYS = ("point_at", "within_deg", "for_s", "feet_drift_max_m", "feet_yaw_drift_max_deg", "no_contact")


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


class Evaluator:
    """A task's ``success`` criteria, measured on the twin every control tick:

        point_at                the object the right arm must point at (a body name, or its prefix: laptop)
        within_deg, for_s       ... the arm's ray (shoulder to wrist) within within_deg of the object's
                                centre, for for_s seconds without a break
        feet_drift_max_m        how far the feet (their midpoint) may move from where they stood: did it
                                walk or slide. Not the pelvis: it twists over planted feet when an arm
                                swings out with the waist turned
        feet_yaw_drift_max_deg  ... and how far the line between them may turn
        no_contact              geoms (or bodies) by name or prefix the robot must never touch

    A fall always fails. ``result()`` is the verdict and what was measured."""

    def __init__(self, twin: "Twin", criteria: dict) -> None:
        unknown = sorted(set(criteria) - set(SUCCESS_KEYS))
        if unknown:
            raise ValueError(f"success: unknown {unknown} (known: {', '.join(SUCCESS_KEYS)})")
        if "point_at" in criteria and not {"within_deg", "for_s"} <= set(criteria):
            raise ValueError("success: point_at needs within_deg and for_s")
        self.twin = twin
        self.criteria = dict(criteria)
        m, mj = twin.model, twin.mujoco
        body = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
        self.ray = tuple(body(n) for n in ARM_RAY)
        self.target_geoms: list[int] = []
        if "point_at" in criteria:
            name = criteria["point_at"]
            bodies = [b for b in range(m.nbody) if m.body(b).name == name] or \
                     [b for b in range(m.nbody) if m.body(b).name.startswith(name)]
            if not bodies:
                raise ValueError(f"success: point_at {name!r}: no such object in the scene")
            self.target_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bodies[0]]
        self.feet = tuple(body(n) for n in FEET)
        robot_root = body("pelvis")
        self.robot = np.array([m.body_rootid[m.geom_bodyid[g]] == robot_root for g in range(m.ngeom)])
        names = criteria.get("no_contact", [])

        def matches(g: int) -> bool:
            for n in (m.geom(g).name, m.body(m.geom_bodyid[g]).name):
                if any(n == k or n.startswith(k) for k in names):
                    return True
            return False
        self.forbidden = np.array([bool(names) and not self.robot[g] and matches(g) for g in range(m.ngeom)])
        self.reset()

    def reset(self) -> None:
        self.error_min = math.inf
        self.within_s = 0.0           # the current unbroken stretch within within_deg
        self.within_best_s = 0.0
        self.error_at_best = math.nan
        self.drift_max = 0.0
        self.yaw_drift_max = 0.0
        self.feet0 = self._feet()
        self.contacts: dict[str, float] = {}     # "robot geom/object geom" -> first time

    def _feet(self) -> tuple[np.ndarray, float]:
        """The feet's midpoint (x, y) and the heading of the line from the right foot to the left."""
        left, right = (self.twin.data.xpos[b][:2].copy() for b in self.feet)
        return (left + right) / 2, math.atan2(left[1] - right[1], left[0] - right[0])

    def target(self) -> np.ndarray:
        d = self.twin.data
        return d.geom_xpos[self.target_geoms].mean(axis=0)

    def pointing_error_deg(self) -> float:
        d = self.twin.data
        shoulder, wrist = (d.xpos[b] for b in self.ray)
        arm, to = wrist - shoulder, self.target() - shoulder
        c = float(np.dot(arm, to) / (np.linalg.norm(arm) * np.linalg.norm(to) + 1e-12))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    def tick(self, dt: float) -> None:
        c = self.criteria
        if self.target_geoms:
            err = self.pointing_error_deg()
            self.error_min = min(self.error_min, err)
            self.within_s = self.within_s + dt if err <= c["within_deg"] else 0.0
            if self.within_s > self.within_best_s:
                self.within_best_s, self.error_at_best = self.within_s, err
        (mid0, yaw0), (mid, yaw) = self.feet0, self._feet()
        self.drift_max = max(self.drift_max, float(np.linalg.norm(mid - mid0)))
        self.yaw_drift_max = max(self.yaw_drift_max, abs(math.degrees(math.remainder(yaw - yaw0, 2 * math.pi))))
        if self.forbidden.any():
            m, d = self.twin.model, self.twin.data
            for con in d.contact[:d.ncon]:
                a, b = int(con.geom1), int(con.geom2)
                for r, o in ((a, b), (b, a)):
                    if self.robot[r] and self.forbidden[o]:
                        self.contacts.setdefault(f"{m.geom(r).name or m.body(m.geom_bodyid[r]).name}/{m.geom(o).name}",
                                                 round(self.twin.time, 3))

    def result(self) -> dict:
        c = self.criteria
        x, y, yaw = self.twin.base_pose()
        measured: dict = {"feet_drift_max_m": round(self.drift_max, 4),
                          "feet_yaw_drift_max_deg": round(self.yaw_drift_max, 2),
                          "pelvis_end": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
                          "contacts": self.contacts, "fell": self.twin.onboard.fallen_at is not None}
        failed = []
        if self.target_geoms:
            measured.update(pointing_error_min_deg=round(self.error_min, 2) if math.isfinite(self.error_min) else None,
                            pointing_within_s=round(self.within_best_s, 3),
                            pointing_error_at_best_deg=None if math.isnan(self.error_at_best)
                            else round(self.error_at_best, 2))
            if self.within_best_s + 1e-9 < c["for_s"]:
                failed.append("point_at")
        if "feet_drift_max_m" in c and self.drift_max > c["feet_drift_max_m"]:
            failed.append("feet_drift_max_m")
        if "feet_yaw_drift_max_deg" in c and self.yaw_drift_max > c["feet_yaw_drift_max_deg"]:
            failed.append("feet_yaw_drift_max_deg")
        if self.contacts:
            failed.append("no_contact")
        if measured["fell"]:
            failed.append("fell")
        return {"passed": not failed, "failed": failed, "criteria": c, "measured": measured}


class Twin:
    def __init__(self, *, scene: str = "none", objects=(), obstacle=None, camera_size=None,
                 fsm: int = FSM_MAIN, band: bool = False, voice: str = "sim", start=None,
                 success: dict | None = None) -> None:
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
        self.evaluator = Evaluator(self, success) if success else None
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
            self.truth_log: list[list[float]] = []    # [monotonic s, x, y, yaw]: the host's trace lines up on it
            self._truth_at = -math.inf
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
            if self.evaluator is not None:
                self.evaluator.tick(CONTROL_DT)
            if self.time - self._truth_at >= TRUTH_PERIOD - 1e-9:
                self._truth_at = self.time
                self.truth_log.append([round(time.monotonic(), 4), *(round(v, 4) for v in self.base_pose())])

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
        """Measure base_pose() from here (the program's start); the scoring starts again too."""
        self.start = self.truth()
        if getattr(self, "evaluator", None) is not None:
            self.evaluator.reset()

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
                "evaluation": None if self.evaluator is None else self.evaluator.result(),
                "truth": self.truth_log, "summary": self.summary()}

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
        if self.evaluator is not None:
            r = self.evaluator.result()
            parts.append("success criteria " + ("met" if r["passed"] else f"NOT met ({', '.join(r['failed'])})"))
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
