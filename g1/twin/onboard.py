"""The onboard controller, as the twin plays it: everything the real G1 does
behind its SDK services that g1-lab relies on.

  FSM        0 zero torque, 1 damp, 4 locked stand (every joint to the stand pose
             over ``twin_lock_stand_s``), 200 / 500 main operation; a SetFsmId is
             reported by GetFsmId after ``twin_fsm_delay_s``
  legs       in 200 / 500: a stand controller (stiff ankles, pelvis-tilt feedback)
             while the base command is zero, the walker (``walker.py``) while it is
             not; handed over in double support. Move is a 1 s dead-man, as on the
             robot
  upper body held at the stand pose by the onboard gains, blended with the last
             ``rt/arm_sdk`` message exactly as arm_sdk blends: target, kp and kd all
             ``(1 - w) * onboard + w * arm_sdk`` with ``w = motor_cmd[29].q``
  gestures   SetArmTask 0/1 wave, 2/3 shake: a keyframed right arm on the onboard
             channel, so it shows only while arm_sdk has let go (w = 0)
  waist      while it walks, the waist (yaw, roll, pitch) is held at least
             ``twin_walk_waist_kp`` stiff whatever arm_sdk sends, as the robot's controller holds
             it steady for a walk; standing, arm_sdk's gains apply as sent (so a bow at kp 60
             sags under the torso's weight, as it would on the robot). ``waist_raised`` counts the physics
             steps it applied, and the report says so
  falls      the pelvis below ``twin_fall_z_m`` or tilted past ``twin_fall_tilt_deg``
             damps the robot, like the robot's fall protection

Every physics step computes the motor torques from these targets
(``tau = kp (q* - q) + kd (dq* - dq) + tau_ff``) and clips them to the motors'
limits, so a PD loop runs at 500 Hz as it does on the motor drivers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from g1.core import limits
from g1.core.config import ARM_SDK_WEIGHT_IDX, CONTROL_DT, NUM_JOINTS, STAND_Q, UPPER_BODY
from g1.twin import walker as gait

FSM_ZERO, FSM_DAMP, FSM_LOCKED_STAND, FSM_MAIN, FSM_START = 0, 1, 4, 200, 500
FSMS = frozenset({FSM_ZERO, FSM_DAMP, FSM_LOCKED_STAND, FSM_MAIN, FSM_START})
WALKING_FSMS = frozenset({FSM_MAIN, FSM_START})

UP = np.array(UPPER_BODY)                       # 12..28
STAND = STAND_Q.copy()
STAND[:12] = gait.DEFAULT                      # the legs stand where the walker acts around

# The stand controller: tuned in the twin's Phase 0 spike (stands 20 s within 1.4 cm, takes
# 60 N pushes of 0.1 s forward and sideways). Leg order: hip pitch/roll/yaw, knee, ankle pitch/roll.
STAND_KP = np.array([200, 200, 200, 300, 200, 100] * 2, dtype=float)
STAND_KD = np.array([5, 5, 5, 8, 5, 3] * 2, dtype=float)
STAND_TILT_GAIN = 1.0           # rad of ankle per rad of pelvis tilt
STAND_RATE_GAIN = 0.1           # rad of ankle per rad/s of pelvis rate
# Handing the legs between the two. The stand controller takes over only after the
# walker has braked (no command for STOP_AFTER_S, both feet down, slow, upright); it
# hands back to the walker, to step, when the robot leans or drifts past what an
# ankle can hold (the arms swinging forward can put the centre of mass past the toes).
STOP_AFTER_S = 0.4              # two half-steps of the 0.8 s gait
SETTLE_COS = math.cos(math.radians(4.0))
RECOVER_COS = math.cos(math.radians(10.0))   # the hand-over itself can tilt it 6 deg for a moment
RECOVER_SPEED = 0.3             # m/s

FSM_DELAY = limits.get("twin_fsm_delay_s")
LOCK_S = limits.get("twin_lock_stand_s")
DAMP_KD = limits.get("twin_damp_kd")
FALL_Z = limits.get("twin_fall_z_m")
FALL_COS = math.cos(math.radians(limits.get("twin_fall_tilt_deg")))
VEL_FB = limits.get("twin_vel_feedback")
VEL_I = limits.get("twin_vel_integral")
WALK_BLEND = limits.get("twin_walk_blend_s")
STAND_BLEND = limits.get("twin_stand_blend_s")
STOP_SPEED = limits.get("twin_stop_speed")
GESTURE_S = {"wave": limits.get("wave_hand_seconds"), "shake": limits.get("shake_hand_seconds")}

WAIST = [12, 13, 14]            # waist yaw, roll, pitch
WALK_WAIST_KP = limits.get("twin_walk_waist_kp")
WAIST_HOLD_S = limits.get("twin_waist_hold_s")
PHYSICS_STEP = limits.get("twin_physics_dt_s")
UPPER_KP = np.array([limits.get("twin_waist_kp")] * 3 + [limits.get("twin_arm_kp")] * 14)
UPPER_KD = np.array([limits.get("twin_waist_kd")] * 3 + [limits.get("twin_arm_kd")] * 14)


@dataclass
class ArmSdkCommand:
    """The last rt/arm_sdk message: per joint q, dq, tau, kp, kd, and the weight."""
    q: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    kp: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS))
    weight: float = 0.0

    @classmethod
    def of(cls, msg) -> "ArmSdkCommand":
        """From a LowCmd_ (or anything with ``motor_cmd[i].q/dq/tau/kp/kd``)."""
        m = msg.motor_cmd
        get = lambda attr: np.array([getattr(m[i], attr) for i in range(NUM_JOINTS)], dtype=float)
        return cls(get("q"), get("dq"), get("tau"), get("kp"), get("kd"), float(m[ARM_SDK_WEIGHT_IDX].q))


def _smooth(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3 - 2 * a)


def gesture_pose(kind: str, a: float, t: float) -> np.ndarray:
    """The upper-body target of a gesture at fraction ``a`` of its length: raise the
    right arm over the first fifth, move, lower it over the last fifth."""
    q = STAND[UP].copy()
    env = _smooth(a / 0.2) * _smooth((1.0 - a) / 0.2)
    k = lambda j: j - 12                       # joint index -> upper-body slot
    if kind == "wave":
        pose = {22: -0.3, 23: -1.3, 24: 0.0, 25: 1.0}
        wiggle = {24: 0.35 * math.sin(2 * math.pi * 1.25 * t)}
    else:                                      # shake: hand forward, small up-down
        pose = {22: -0.9, 23: -0.15, 24: 0.0, 25: 0.6}
        wiggle = {22: 0.08 * math.sin(2 * math.pi * 2.0 * t)}
    for j, v in pose.items():
        q[k(j)] += env * (v - STAND[j])
    for j, v in wiggle.items():
        q[k(j)] += env * v
    return q


class Onboard:
    def __init__(self, model, fsm: int = FSM_MAIN) -> None:
        import mujoco
        self.model = model
        self.lo, self.hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.walker = gait.Walker()
        self.reset(fsm)

    def reset(self, fsm: int = FSM_MAIN) -> None:
        self.fsm = fsm
        self.pending: tuple[int, float] | None = None
        self.arm: ArmSdkCommand | None = None
        self.cmd = np.zeros(3)
        self.cmd_until = -math.inf
        self.last_cmd = np.zeros(3)
        self.gesture: tuple[str, float] | None = None
        self.legs = "stand"                     # stand | walk | settle
        self.stand_share = 1.0                  # 1: the stand controller owns the legs, 0: the walker
        self.walk_q = gait.DEFAULT.copy()
        self.vel = np.zeros(3)                  # filtered pelvis-frame (vx, vy, vyaw)
        self.integral = np.zeros(3)
        self.lock_from: np.ndarray | None = None
        self.lock_t0 = 0.0
        self.fallen_at: float | None = None
        self.saturated = np.zeros(NUM_JOINTS, dtype=int)   # physics steps each motor spent at its limit
        self.walk_ticks = 0
        self.idle_since: float | None = 0.0     # when the base command last went to zero
        self.recoveries = 0                     # times the stand controller handed back to the walker to step
        self.waist_raised = 0                   # physics steps the walk's waist hold overrode arm_sdk's kp
        self.waist_hold = 0.0                   # 0..1: how far the walk's waist hold is applied
        self.walker.reset()

    # -- the services' side ---------------------------------------------------------------------
    def set_fsm(self, fsm: int, t: float) -> int:
        if fsm not in FSMS:
            return 3203                          # the twin does not play this FSM
        self.pending = (int(fsm), t + FSM_DELAY)
        return 0

    def set_velocity(self, v, duration: float, t: float) -> None:
        self.cmd = np.asarray(v, dtype=float)
        self.cmd_until = t + float(duration)

    def arm_task(self, task: int, t: float) -> None:
        self.gesture = ("wave" if int(task) in (0, 1) else "shake", t)

    def on_arm_sdk(self, cmd: ArmSdkCommand) -> None:
        self.arm = cmd

    # -- 50 Hz ------------------------------------------------------------------------------------
    def tick(self, d, t: float) -> None:
        if self.pending is not None and t >= self.pending[1]:
            self._enter(self.pending[0], d, t)
            self.pending = None
        self._fall_check(d, t)
        if self.gesture is not None and t - self.gesture[1] > GESTURE_S[self.gesture[0]]:
            self.gesture = None
        if self.fsm not in WALKING_FSMS:
            return
        c = self.cmd if t < self.cmd_until else np.zeros(3)
        moving = bool(np.any(np.abs(c) > 1e-3))
        if abs(c[2] - self.last_cmd[2]) > 1e-3:
            self.integral[2] = 0.0              # a new turn rate: the old one's lag would carry past the stop
        self.last_cmd = c.copy()
        if moving:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = t
        upright = 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)         # cos of the pelvis tilt
        speed = math.hypot(*self.vel[:2])
        if self.legs == "stand" and (moving or upright < RECOVER_COS or speed > RECOVER_SPEED):
            self.legs = "walk"                  # a command, or a lean the ankles cannot hold: step
            self.walker.reset()
            self.integral[:] = 0.0
            self.recoveries += not moving
        elif self.legs == "walk" and not moving:
            if (t - self.idle_since >= STOP_AFTER_S and self._both_feet_down(d) and speed < STOP_SPEED
                    and upright > SETTLE_COS):
                self.legs = "settle"
        if self.legs in ("walk", "settle"):
            goal = c if self.legs == "walk" else np.zeros(3)
            err = goal - self.vel
            self.integral = np.clip(self.integral + err * CONTROL_DT * VEL_I, -0.5, 0.5)
            self.walk_q = self.walker.targets(goal + VEL_FB * err + self.integral, d.qpos[3:7], d.qvel[3:6],
                                              d.qpos[7:19], d.qvel[6:18])
            self.walk_ticks += 1
        if self.legs == "walk":
            self.stand_share = max(0.0, self.stand_share - CONTROL_DT / WALK_BLEND)
        elif self.legs == "settle":
            self.stand_share = min(1.0, self.stand_share + CONTROL_DT / STAND_BLEND)
            if self.stand_share >= 1.0:
                self.legs = "stand"

    def _enter(self, fsm: int, d, t: float) -> None:
        self.fsm = fsm
        if fsm == FSM_LOCKED_STAND:
            self.lock_from = d.qpos[7:7 + NUM_JOINTS].copy()
            self.lock_t0 = t
        if fsm in WALKING_FSMS:
            self.legs, self.stand_share = "stand", 1.0
            self.walker.reset()
        if fsm in (FSM_ZERO, FSM_DAMP):
            self.cmd_until = -math.inf
            self.gesture = None

    def _fall_check(self, d, t: float) -> None:
        if self.fallen_at is not None:
            return
        w, x, y, z = d.qpos[3:7]
        upright = 1 - 2 * (x * x + y * y)       # cos of the pelvis tilt
        if d.qpos[2] < FALL_Z or upright < FALL_COS:
            self.fallen_at = t
            if self.fsm != FSM_ZERO:
                self.fsm, self.pending = FSM_DAMP, None
                self.cmd_until = -math.inf

    def _both_feet_down(self, d) -> bool:
        down = set()
        m = self.model
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            for foot, other in ((b1, b2), (b2, b1)):
                if foot in self.feet and m.body_weldid[other] == 0:
                    down.add(foot)
        return len(down) == 2

    # -- 500 Hz -----------------------------------------------------------------------------------
    def torques(self, d, t: float) -> np.ndarray:
        q, dq = d.qpos[7:7 + NUM_JOINTS], d.qvel[6:6 + NUM_JOINTS]
        if self.fsm == FSM_ZERO:
            return np.zeros(NUM_JOINTS)
        if self.fsm == FSM_DAMP:
            return np.clip(-DAMP_KD * dq, self.lo, self.hi)
        self._estimate(d)
        target, kp, kd = STAND.copy(), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)
        dq_target, ff = np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)
        # legs: the stand controller, blended with the walker
        w_, x, y, z = d.qpos[3:7]
        roll = math.atan2(2 * (w_ * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w_ * y - z * x))))
        stand = gait.DEFAULT.copy()
        stand[[4, 10]] += STAND_TILT_GAIN * pitch + STAND_RATE_GAIN * d.qvel[4]
        stand[[5, 11]] += -STAND_TILT_GAIN * roll - STAND_RATE_GAIN * d.qvel[3]
        stand[[1, 7]] += 0.5 * STAND_TILT_GAIN * roll
        upper = STAND[UP].copy()
        if self.fsm == FSM_LOCKED_STAND and self.lock_from is not None:
            a = _smooth((t - self.lock_t0) / LOCK_S)
            stand = (1 - a) * self.lock_from[:12] + a * stand
            upper = (1 - a) * self.lock_from[UP] + a * upper
        s = self.stand_share if self.fsm in WALKING_FSMS else 1.0
        target[:12] = s * stand + (1 - s) * self.walk_q
        kp[:12] = s * STAND_KP + (1 - s) * gait.KP
        kd[:12] = s * STAND_KD + (1 - s) * gait.KD
        # upper body: the onboard hold (or a gesture), blended with arm_sdk
        if self.gesture is not None and self.fsm in WALKING_FSMS:
            kind, t0 = self.gesture
            upper = gesture_pose(kind, (t - t0) / GESTURE_S[kind], t - t0)
        okp, okd = UPPER_KP, UPPER_KD
        if self.arm is not None and self.fsm in WALKING_FSMS:
            w = min(1.0, max(0.0, self.arm.weight))
            a = self.arm
            target[UP] = (1 - w) * upper + w * a.q[UP]
            kp[UP] = (1 - w) * okp + w * a.kp[UP]
            kd[UP] = (1 - w) * okd + w * a.kd[UP]
            dq_target[UP] = w * a.dq[UP]
            ff[UP] = w * a.tau[UP]
        else:
            target[UP], kp[UP], kd[UP] = upper, okp, okd
        walking = self.fsm in WALKING_FSMS and (self.legs != "stand" or self.stand_share < 1.0)
        if walking:                                            # on as fast as the legs hand over, off slowly:
            self.waist_hold = min(1.0, self.waist_hold + PHYSICS_STEP / WALK_BLEND)
        else:                                                  # never a jump in stiffness
            self.waist_hold = max(0.0, self.waist_hold - PHYSICS_STEP / WAIST_HOLD_S)
        h = self.waist_hold
        if h > 0 and WALK_WAIST_KP > 0 and np.any(kp[WAIST] < WALK_WAIST_KP):
            kp[WAIST] += h * np.maximum(0.0, WALK_WAIST_KP - kp[WAIST])
            kd[WAIST] += h * np.maximum(0.0, UPPER_KD[np.array(WAIST) - 12] - kd[WAIST])
            self.waist_raised += 1
        tau = kp * (target - q) + kd * (dq_target - dq) + ff
        out = np.clip(tau, self.lo, self.hi)
        self.saturated += (out != tau)
        return out

    def _estimate(self, d) -> None:
        """The pelvis velocity in its own heading frame, low-passed (time constant 0.1 s at
        500 Hz). The robot's controller gets this from its state estimator; the twin reads it."""
        w, x, y, z = d.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy = d.qvel[0], d.qvel[1]
        v = np.array([c * vx + s * vy, -s * vx + c * vy, d.qvel[5]])
        self.vel += (v - self.vel) * 0.02
