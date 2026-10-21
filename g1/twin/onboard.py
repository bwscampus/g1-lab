"""The onboard controller, as the twin plays it: everything the real G1 does
behind its SDK services that g1-lab relies on.

  FSM        0 zero torque, 1 damp, 4 locked stand (every joint to the stand pose
             over ``twin_lock_stand_s``, the legs then held by the walker's Balance, never
             stepping), 200 / 500 main operation; a SetFsmId is
             reported by GetFsmId after ``twin_fsm_delay_s``
  legs       in 200 / 500: the walker (``walker.py``, GR00T: Balance at zero command, Walk
             otherwise) drives the legs and the waist, its velocity command corrected
             by the twin's own velocity estimate (``twin_vel_feedback``,
             ``twin_vel_integral``). Move is a 1 s dead-man, as on the robot
  upper body held at the stand pose by the onboard gains, blended with the last
             ``rt/arm_sdk`` message exactly as arm_sdk blends: target, kp and kd all
             ``(1 - w) * onboard + w * arm_sdk`` with ``w = motor_cmd[29].q``
  sit        3: legs and waist from where they are to a seated pose over ``twin_sit_s``, the
             twin's guess at the robot's own sit (no walker, nothing balances it: it sits down where
             it stands, onto a chair if the seat's front edge is within about 0.25 m behind the
             heels). 801 (sit to stand) is not played: a stand-up from a chair needs balance the
             twin's PD cannot give, so it answers 3203 and a sim run ends seated
  gestures   SetArmTask 0/1 wave, 2/3 shake: a keyframed right arm on the onboard
             channel, so it shows only while arm_sdk has let go (w = 0)
  waist      the walker's, which does not observe the yaw arm_sdk turns it to (see
             ``_walker_joints``) while the arms are near rest; while it walks it holds the waist whatever arm_sdk sends, as the
             robot's controller holds it steady for a walk (on over ``twin_walk_blend_s``, off
             over ``twin_waist_hold_s``); standing, arm_sdk's blend applies as sent (so a bow
             at kp 60 sags under the torso's weight, as it would on the robot).
             ``waist_raised`` counts the physics steps the hold overrode arm_sdk, and the
             report says so
  falls      the pelvis below ``twin_fall_z_m`` or tilted past ``twin_fall_tilt_deg``
             damps the robot, like the robot's fall protection; a robot the gantry rope is
             carrying (``hanging``) only counts as fallen by height

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
FSM_SIT = 3
FSMS = frozenset({FSM_ZERO, FSM_DAMP, FSM_SIT, FSM_LOCKED_STAND, FSM_MAIN, FSM_START})
WALKING_FSMS = frozenset({FSM_MAIN, FSM_START})
ARM_SDK_FSMS = WALKING_FSMS | {FSM_SIT}        # where arm_sdk blends into the upper body

UP = np.array(UPPER_BODY)                       # 12..28
WAIST = np.array([12, 13, 14])                  # waist yaw, roll, pitch
WAIST_YAW = 12
ARMS = np.arange(15, 29)
STAND = STAND_Q.copy()
STAND[:12] = gait.DEFAULT[:12]                  # the legs stand where the walker acts around

FSM_DELAY = limits.get("twin_fsm_delay_s")
LOCK_S = limits.get("twin_lock_stand_s")
DAMP_KD = limits.get("twin_damp_kd")
FALL_Z = limits.get("twin_fall_z_m")
FALL_COS = math.cos(math.radians(limits.get("twin_fall_tilt_deg")))
VEL_FB = limits.get("twin_vel_feedback")
VEL_I = limits.get("twin_vel_integral")
WALK_BLEND = limits.get("twin_walk_blend_s")
WAIST_HOLD_S = limits.get("twin_waist_hold_s")
PHYSICS_STEP = limits.get("twin_physics_dt_s")
GESTURE_S = {"wave": limits.get("wave_hand_seconds"), "shake": limits.get("shake_hand_seconds")}
WALK_FLOOR = gait.G1.walk_above + limits.get("twin_walk_floor_margin")   # a corrected walk never drops into Balance
VEL_I_MAX = limits.get("twin_vel_integral_max")
VEL_FILTER = limits.get("twin_vel_filter")
GESTURE_EDGE = limits.get("twin_gesture_edge")
WAVE = (limits.get("twin_wave_amp_rad"), limits.get("twin_wave_hz"))
SHAKE = (limits.get("twin_shake_amp_rad"), limits.get("twin_shake_hz"))

ARMS_OUT = limits.get("twin_mask_arms_out_rad")
SIT_S = limits.get("twin_sit_s")
SEATED_FALL_Z = limits.get("twin_seated_fall_z_m")
SIT_KP, SIT_KD = limits.get("twin_sit_kp"), limits.get("twin_sit_kd")
# legs and waist (DDS 0..14): hip pitch/roll/yaw, knee, ankle pitch/roll per leg, then the waist
# (found on the twin: thighs level, shins tipped back, so the pelvis lands 0.25-0.35 m behind the heels)
SEATED = np.array((-1.3, 0.0, 0.0, 1.3, 0.2, 0.0) * 2 + (0.0, 0.0, 0.0))

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
    right arm over the first GESTURE_EDGE of it, move, lower it over the last."""
    q = STAND[UP].copy()
    env = _smooth(a / GESTURE_EDGE) * _smooth((1.0 - a) / GESTURE_EDGE)
    k = lambda j: j - 12                       # joint index -> upper-body slot
    if kind == "wave":
        pose = {22: -0.3, 23: -1.3, 24: 0.0, 25: 1.0}
        wiggle = {24: WAVE[0] * math.sin(2 * math.pi * WAVE[1] * t)}
    else:                                      # shake: hand forward, small up-down
        pose = {22: -0.9, 23: -0.15, 24: 0.0, 25: 0.6}
        wiggle = {22: SHAKE[0] * math.sin(2 * math.pi * SHAKE[1] * t)}
    for j, v in pose.items():
        q[k(j)] += env * (v - STAND[j])
    for j, v in wiggle.items():
        q[k(j)] += env * v
    return q


class Onboard:
    def __init__(self, model, fsm: int = FSM_MAIN) -> None:
        self.model = model
        self.lo, self.hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
        self.walker = gait.Walker()
        self.reset(fsm)

    def reset(self, fsm: int = FSM_MAIN, keep_gait: bool = False) -> None:
        """Clear the controller; ``keep_gait`` keeps the walker's history, its last targets
        and the velocity estimate (the twin's reset settles it standing first)."""
        self.fsm = fsm
        self.pending: tuple[int, float] | None = None
        self.arm: ArmSdkCommand | None = None
        self.cmd = np.zeros(3)
        self.cmd_until = -math.inf
        self.last_cmd = np.zeros(3)
        self.gesture: tuple[str, float] | None = None
        self.walking = False                    # the walker is stepping to a command (Walk, not Balance)
        if not keep_gait:
            self.walk_q = gait.DEFAULT.copy()   # the walker's 15 leg and waist targets
            self.vel = np.zeros(3)              # filtered pelvis-frame (vx, vy, vyaw)
            self.walker.reset()
        self.integral = np.zeros(3)
        self.lock_from: np.ndarray | None = None
        self.lock_t0 = 0.0
        self.fallen_at: float | None = None
        self.hanging = False                    # the gantry rope is carrying it (set by the twin)
        self.saturated = np.zeros(NUM_JOINTS, dtype=int)   # physics steps each motor spent at its limit
        self.walk_ticks = 0
        self.waist_raised = 0                   # physics steps the walk's waist hold overrode arm_sdk
        self.waist_hold = 0.0                   # 0..1: how far the walk's waist hold is applied

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
        if self.fsm == FSM_LOCKED_STAND:        # a standing hold: Balance, never a step
            self.walk_q = self.walker.targets(np.zeros(3), d.qpos[3:7], d.qvel[3:6], *self._walker_joints(d))
            return
        if self.fsm not in WALKING_FSMS:
            return
        c = self.cmd if t < self.cmd_until else np.zeros(3)
        if abs(c[2] - self.last_cmd[2]) > 1e-3:
            self.integral[2] = 0.0              # a new turn rate: the old one's lag would carry past the stop
        self.last_cmd = c.copy()
        if np.linalg.norm(c) > gait.G1.walk_above:
            # Walk, its command corrected by the measured velocity: the network alone falls short
            err = c - self.vel
            self.integral = np.clip(self.integral + err * CONTROL_DT * VEL_I, -VEL_I_MAX, VEL_I_MAX)
            goal = c + VEL_FB * err + self.integral
            n = np.linalg.norm(goal)
            if n < WALK_FLOOR:
                goal = goal / max(n, 1e-9) * WALK_FLOOR
            self.walk_ticks += 1
        else:
            goal = np.zeros(3)                  # Balance: stand still, no loop to invent a shuffle
            self.integral[:] = 0.0
        self.walk_q = self.walker.targets(goal, d.qpos[3:7], d.qvel[3:6], *self._walker_joints(d))
        self.walking = self.walker.walking


    def _walker_joints(self, d) -> tuple[np.ndarray, np.ndarray]:
        """The joints the walker observes: measured, less the waist yaw arm_sdk asks for while
        the arms are near rest. Trained with the waist its own (yaw near 0), it reads a look as
        a fall to catch: it twists the hips and creeps the feet (20 deg in 15 s at 60 deg, a
        spin past 120). Hiding the turn stops that, but with an arm out it would balance for
        the arm where it is not (out front instead of to the side) and tip, so the mask fades
        out as the arms leave rest (from half of ``twin_mask_arms_out_rad`` to all of it) and the walker sees the
        truth, creeping a little while the arm is out. The robot's controller is assumed to
        hold its stance throughout (check on the robot: a 145 deg look held, the feet watched)."""
        q, dq = d.qpos[7:7 + NUM_JOINTS].copy(), d.qvel[6:6 + NUM_JOINTS].copy()
        if self.arm is not None:
            arms_out = float(np.max(np.abs(q[ARMS] - STAND_Q[ARMS])))
            # as much of arm_sdk's yaw as reaches the waist: its weight, less the walk's hold
            w = (min(1.0, max(0.0, self.arm.weight)) * (1.0 - self.waist_hold)
                 * (1.0 - _smooth(2.0 * arms_out / ARMS_OUT - 1.0)))      # whole up to half of it
            q[WAIST_YAW] -= w * (self.arm.q[WAIST_YAW] - gait.DEFAULT[WAIST_YAW])
            dq[WAIST_YAW] -= w * self.arm.dq[WAIST_YAW]
        return q, dq

    def _enter(self, fsm: int, d, t: float) -> None:
        before, self.fsm = self.fsm, fsm
        if fsm in (FSM_LOCKED_STAND, FSM_SIT):
            self.lock_from = d.qpos[7:7 + NUM_JOINTS].copy()
            self.lock_t0 = t
        if fsm == FSM_SIT:
            self.cmd_until = -math.inf
            self.gesture = None
            self.walking = False
        balancing = WALKING_FSMS | {FSM_LOCKED_STAND}
        if fsm in balancing and before not in balancing:    # the walker takes over a limp robot: fresh history
            self.walker.reset()
            self.walk_q = gait.DEFAULT.copy()
            self.walking = False
        if fsm in (FSM_ZERO, FSM_DAMP):
            self.cmd_until = -math.inf
            self.gesture = None

    def _fall_check(self, d, t: float) -> None:
        if self.fallen_at is not None:
            return
        w, x, y, z = d.qpos[3:7]
        upright = 1 - 2 * (x * x + y * y)       # cos of the pelvis tilt
        floor = SEATED_FALL_Z if self.fsm == FSM_SIT else FALL_Z
        if d.qpos[2] < floor or (upright < FALL_COS and not self.hanging):
            self.fallen_at = t
            if self.fsm != FSM_ZERO:
                self.fsm, self.pending = FSM_DAMP, None
                self.cmd_until = -math.inf

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
        kp[:12], kd[:12] = gait.KP[:12], gait.KD[:12]
        upper = STAND[UP].copy()
        okp, okd = UPPER_KP.copy(), UPPER_KD.copy()
        if self.fsm == FSM_SIT and self.lock_from is not None:
            a = _smooth((t - self.lock_t0) / SIT_S)       # down to the seated pose; nothing balances it
            legs = (1 - a) * self.lock_from[:15] + a * SEATED
            target[:12] = legs[:12]
            upper[:3] = legs[12:15]
            kp[:12], kd[:12] = SIT_KP, SIT_KD
            okp[:3], okd[:3] = SIT_KP, SIT_KD
        elif self.fsm == FSM_LOCKED_STAND and self.lock_from is not None:
            a = _smooth((t - self.lock_t0) / LOCK_S)        # into the stand, then held there by Balance
            target[:12] = (1 - a) * self.lock_from[:12] + a * self.walk_q[:12]
            upper = (1 - a) * self.lock_from[UP] + a * upper
            upper[:3] = (1 - a) * self.lock_from[WAIST] + a * self.walk_q[12:15]
            okp[:3], okd[:3] = gait.KP[12:15], gait.KD[12:15]
        else:                                               # 200 / 500: the walker's legs and waist
            target[:12] = self.walk_q[:12]
            upper[:3] = self.walk_q[12:15]
            okp[:3], okd[:3] = gait.KP[12:15], gait.KD[12:15]
        # upper body: the onboard hold (or a gesture), blended with arm_sdk
        if self.gesture is not None and self.fsm in WALKING_FSMS:
            kind, t0 = self.gesture
            upper[3:] = gesture_pose(kind, (t - t0) / GESTURE_S[kind], t - t0)[3:]
        if self.arm is not None and self.fsm in ARM_SDK_FSMS:
            w = min(1.0, max(0.0, self.arm.weight))
            a = self.arm
            target[UP] = (1 - w) * upper + w * a.q[UP]
            kp[UP] = (1 - w) * okp + w * a.kp[UP]
            kd[UP] = (1 - w) * okd + w * a.kd[UP]
            dq_target[UP] = w * a.dq[UP]
            ff[UP] = w * a.tau[UP]
        else:
            target[UP], kp[UP], kd[UP] = upper, okp, okd
        # the walk's waist hold: on as fast as a walk starts, off slowly, never a jump in stiffness
        if self.fsm in WALKING_FSMS and self.walking:
            self.waist_hold = min(1.0, self.waist_hold + PHYSICS_STEP / WALK_BLEND)
        else:
            self.waist_hold = max(0.0, self.waist_hold - PHYSICS_STEP / WAIST_HOLD_S)
        h = self.waist_hold
        if h > 0 and self.fsm in WALKING_FSMS:
            held = self.arm is not None and self.arm.weight > 0
            target[WAIST] = h * self.walk_q[12:15] + (1 - h) * target[WAIST]
            kp[WAIST] = h * gait.KP[12:15] + (1 - h) * kp[WAIST]
            kd[WAIST] = h * gait.KD[12:15] + (1 - h) * kd[WAIST]
            dq_target[WAIST] *= 1 - h
            ff[WAIST] *= 1 - h
            self.waist_raised += held
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
        self.vel += (v - self.vel) * VEL_FILTER
