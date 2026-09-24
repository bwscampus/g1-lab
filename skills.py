"""Skills: the one building block, and the only format a behaviour is made of.

  Policy   the executor contract: reset/step at 50 Hz. The only thing an env runs.
  Skill    a name, a JSON-schema parameter menu, and ``segments()`` -> pose
           segments. Never executed directly: ``skill_policy`` turns it into a
           SegmentPolicy, and a Routine concatenates several with bookends.

"Walk forward" and "lift the arm" are the same format because a Segment may
hold a base velocity for its duration as easily as a pose. A skill may be any
length: the agent records a step every ``STEP_MAX`` seconds while one runs, so
the camera and the joint angles are captured at that cadence regardless.

Parameters are bound at construction (``Turn(angle_deg=45)``) and validated
against ``params``, which is also the menu a vision model chooses from.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from config import CONTROL_DT, UPPER_BODY, joint_index
from poses import ARMS_UP, SIXSEVEN, STAND
from policy import Policy, Segment, SegmentPolicy

STEP_MAX = 3.0        # s: how often a running skill is observed and recorded
WALK_SPEED = 0.2      # m/s (< BASE_VEL_MAX[0]; the speed proven on the robot)
TURN_RATE = 0.4       # rad/s (< BASE_VEL_MAX[2])
WALK_MAX_M = 3.0      # one decision may cross a room; it is still watched every STEP_MAX
TURN_MAX_DEG = 180.0
MIN_SEGMENT = 0.5     # s: a shorter entry segment could recentre a held waist too fast
WAIST_YAW = joint_index("waist_yaw")
L_ELBOW, R_ELBOW = 18, 25
ELBOW_REST = 0.0      # 90-degree bend, forearm horizontal
ELBOW_UP = -0.45      # fingertips at shoulder height


def _ticks(seconds: float) -> float:
    """Round a duration up to whole control ticks (and at least MIN_SEGMENT), so a
    velocity held for it integrates to exactly the intended distance or angle."""
    return max(math.ceil(seconds / CONTROL_DT - 1e-9), round(MIN_SEGMENT / CONTROL_DT)) * CONTROL_DT


def _num(desc: str, lo: float, hi: float, default: Optional[float] = None) -> dict:
    spec = {"type": "number", "description": desc, "minimum": lo, "maximum": hi}
    if default is not None:
        spec["default"] = default
    return spec


class Skill:
    """Subclass and implement ``segments()``; declare ``params`` for anything
    tunable. Instances carry bound, validated arguments."""

    name: str = "skill"
    description: str = ""
    params: dict = {"type": "object", "properties": {}, "required": []}
    needs_base: bool = False        # hidden where the env cannot walk
    terminal: bool = False          # ends an agent run (`done`)
    internal: bool = False          # bookends: never in a menu, never CLI-chainable
    allows_start: bool = False      # may use the reserved "start" goal
    joints: list[int] = UPPER_BODY

    def __init__(self, **args) -> None:
        self.args, self.notes = validate_args(type(self), args)
        for k, v in self.args.items():
            setattr(self, k, v)

    def segments(self) -> tuple[Segment, ...]:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments())

    @classmethod
    def schema(cls) -> dict:
        return {"name": cls.name, "description": cls.description, "params": cls.params}

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


# --------------------------------------------------------------------------
# Locomotion: a segment holding a base velocity
# --------------------------------------------------------------------------

class WalkForward(Skill):
    name = "walk_forward"
    description = "Walk straight ahead by distance_m metres (at 0.2 m/s), then stand still."
    params = {"type": "object", "required": ["distance_m"],
              "properties": {"distance_m": _num("metres to walk", 0.1, WALK_MAX_M)}}
    needs_base = True

    def segments(self) -> tuple[Segment, ...]:
        duration = _ticks(self.distance_m / WALK_SPEED)
        v = self.distance_m / duration
        return (Segment(STAND, duration, base=(v, 0.0, 0.0), label=f"walk {self.distance_m:.2f} m"),)


class Turn(Skill):
    name = "turn"
    description = "Turn in place by angle_deg degrees: positive = LEFT (counter-clockwise), negative = RIGHT."
    params = {"type": "object", "required": ["angle_deg"],
              "properties": {"angle_deg": _num("degrees, + left / - right", -TURN_MAX_DEG, TURN_MAX_DEG)}}
    needs_base = True

    def segments(self) -> tuple[Segment, ...]:
        rad = math.radians(self.angle_deg)
        if abs(rad) < 1e-6:
            return (Segment(STAND, MIN_SEGMENT, label="turn 0"),)
        duration = _ticks(abs(rad) / TURN_RATE)
        return (Segment(STAND, duration, base=(0.0, 0.0, rad / duration),
                        label=f"turn {self.angle_deg:+.0f} deg"),)


# --------------------------------------------------------------------------
# Upper body
# --------------------------------------------------------------------------

class Look(Skill):
    name = "look"
    description = ("Turn only the waist (not the feet) by yaw_deg and hold it, so the next image looks "
                   "sideways: positive = LEFT, negative = RIGHT. The next skill recentres the waist.")
    params = {"type": "object", "required": ["yaw_deg"],
              "properties": {"yaw_deg": _num("degrees, + left / - right", -45.0, 45.0)}}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({**STAND, WAIST_YAW: math.radians(self.yaw_deg)}, 1.5,
                        label=f"look {self.yaw_deg:+.0f} deg"),)


class Hold(Skill):
    name = "hold"
    description = "Stand still for seconds, keeping the current pose and waist direction."
    params = {"type": "object", "required": [],
              "properties": {"seconds": _num("how long to wait", 0.1, 30.0, default=1.0)}}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({}, self.seconds, label=f"hold {self.seconds:.1f} s"),)


class TPose(Skill):
    name = "tpose"
    description = "Raise both arms straight out to the sides (a T-pose), hold, then lower them."
    params = {"type": "object", "required": [],
              "properties": {"hold": _num("seconds to hold the pose", 0.5, 30.0, default=5.0),
                             "rise": _num("seconds to raise and lower", 1.0, 10.0, default=3.0)}}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment(ARMS_UP, self.rise, label="arms up"),
                Segment(ARMS_UP, self.hold, label="holding T-pose"),
                Segment(STAND, self.rise, label="arms down"))


class SixSeven(Skill):
    """Palms up, elbows bent 90 degrees, then the forearms see-saw: the left
    swings up until the fingertips reach shoulder height, and as it starts back
    down the right swings up, overlapping, for ``reps`` rounds each.

    Joint conventions (verified by rendering the Menagerie model):
      * elbow 0.0 is the 90-degree bend with the forearm forward; ~1.57 is a
        straight arm; more negative bends the forearm up
      * wrist roll -1.57 (left) / +1.57 (right) turns the palms up, thumbs outward
    """

    name = "sixseven"
    description = "A two-handed see-saw wave: palms up, forearms swing up and down in turn, then arms down."
    params = {"type": "object", "required": [],
              "properties": {"reps": {"type": "integer", "description": "swings per hand",
                                      "minimum": 1, "maximum": 10, "default": 3},
                             "swing_time": _num("seconds per swing", 0.3, 2.0, default=0.8),
                             "settle": _num("seconds to reach the palms-up pose", 0.5, 5.0, default=3.0),
                             "hold": _num("seconds to hold before and after", 0.1, 5.0, default=1.0)}}

    def _swings(self) -> tuple[Segment, ...]:
        """Each segment raises one hand while lowering the other, so the next hand
        starts up exactly when the previous one starts down."""
        order = [("left", L_ELBOW), ("right", R_ELBOW)] * self.reps
        out: list[Segment] = []
        prev = None
        for i, (side, joint) in enumerate(order):
            goal = {joint: ELBOW_UP}
            if prev is not None:
                goal[prev] = ELBOW_REST
            out.append(Segment(goal, self.swing_time, label=f"{side} hand up ({i // 2 + 1}/{self.reps})"))
            prev = joint
        if prev is not None:
            out.append(Segment({prev: ELBOW_REST}, self.swing_time))   # last hand back down
        return tuple(out)

    def segments(self) -> tuple[Segment, ...]:
        return (Segment(SIXSEVEN, self.settle, label="palms up, elbows 90"),
                Segment(SIXSEVEN, self.hold, label="holding"),
                *self._swings(),
                Segment(SIXSEVEN, self.hold, label="holding"),
                Segment(STAND, 1.2, label="arms down"))


class Done(Skill):
    name = "done"
    description = "Stop: the goal is achieved (found=true) or cannot be achieved (found=false)."
    params = {"type": "object", "required": ["found"],
              "properties": {"found": {"type": "boolean", "description": "goal achieved?"},
                             "note": {"type": "string", "description": "one short sentence", "default": ""}}}
    terminal = True

    def segments(self) -> tuple[Segment, ...]:
        return ()


# --------------------------------------------------------------------------
# Bookends: internal skills a Routine or an agent wraps around the rest
# --------------------------------------------------------------------------

class Takeover(Skill):
    """Ramp the arm_sdk weight 0->1 while holding the pose observed at reset
    (nothing should move), then go to STAND."""

    name = "takeover"
    internal = True
    allows_start = True
    params = {"type": "object", "required": [],
              "properties": {"ramp": _num("weight ramp seconds", 0.0, 10.0, default=2.0),
                             "to_stand": _num("seconds to reach STAND", 0.0, 10.0, default=3.0)}}

    def segments(self) -> tuple[Segment, ...]:
        out = []
        if self.ramp > 0:
            out.append(Segment("start", self.ramp, weight=lambda a: a, label="taking over (hold)"))
        if self.to_stand > 0:
            out.append(Segment(STAND, self.to_stand, label="moving to stand"))
        return tuple(out)


class Handback(Skill):
    """Go to STAND, then hold it while ramping the weight 1->0 so the onboard
    controller takes the arms back smoothly."""

    name = "handback"
    internal = True
    params = {"type": "object", "required": [],
              "properties": {"to_stand": _num("seconds to reach STAND", 0.0, 10.0, default=3.0),
                             "ramp": _num("weight ramp seconds", 0.0, 10.0, default=2.0)}}

    def segments(self) -> tuple[Segment, ...]:
        out = []
        if self.to_stand > 0:
            out.append(Segment(STAND, self.to_stand, label="returning to stand"))
        if self.ramp > 0:
            out.append(Segment(STAND, self.ramp, weight=lambda a: 1.0 - a, label="handing back"))
        return tuple(out)


SKILLS: dict[str, type[Skill]] = {s.name: s for s in
                                  (WalkForward, Turn, Look, Hold, TPose, SixSeven, Done, Takeover, Handback)}


def menu(allow_base: bool = True) -> list[type[Skill]]:
    """The skills a decider may choose from; no bookends, and no walking where
    the env cannot walk."""
    return [s for s in SKILLS.values() if not s.internal and (allow_base or not s.needs_base)]


def validate_args(skill: "type[Skill]", args: dict) -> tuple[dict, list[str]]:
    """Coerce, default and range-clamp ``args`` against the skill's schema.
    Unknown or missing required keys and wrong types raise ValueError; clamps
    come back as notes."""
    props = skill.params.get("properties", {})
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key in args:
        if key not in props:
            raise ValueError(f"{skill.name}: unknown argument {key!r} (allowed: {sorted(props)})")
    for key in skill.params.get("required", []):
        if key not in args:
            raise ValueError(f"{skill.name}: missing argument {key!r}")
    for key, spec in props.items():
        if key not in args:
            if "default" in spec:
                out[key] = spec["default"]
            continue
        v = args[key]
        kind = spec.get("type")
        try:
            if kind == "boolean":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")
            elif kind == "number":
                v = float(v)
            elif kind == "integer":
                v = int(float(v))
            elif kind == "string":
                v = str(v)
        except (TypeError, ValueError):
            raise ValueError(f"{skill.name}: argument {key!r} must be {kind}, got {args[key]!r}") from None
        if kind in ("number", "integer"):
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if lo is not None and v < lo:
                notes.append(f"{key} {v} raised to {lo}"); v = lo
            if hi is not None and v > hi:
                notes.append(f"{key} {v} lowered to {hi}"); v = hi
        out[key] = v
    return out, notes


def parse_skill(item: str) -> Skill:
    """``name[:arg[:arg…]]`` with args positional in schema order or ``k=v``."""
    parts = item.split(":")
    name = parts[0].strip()
    if name not in SKILLS:
        raise KeyError(name)
    cls = SKILLS[name]
    keys = list(cls.params.get("properties", {}))
    args: dict = {}
    for i, raw in enumerate(p.strip() for p in parts[1:] if p.strip()):
        if "=" in raw:
            k, v = raw.split("=", 1)
            args[k.strip()] = v.strip()
        elif i < len(keys):
            args[keys[i]] = raw
        else:
            raise ValueError(f"{name}: too many arguments in {item!r}")
    return cls(**args)


def skill_segments(skill: Skill) -> list[Segment]:
    """A skill's segments with labels prefixed by its name. Only the Takeover
    bookend may use the reserved "start" goal."""
    out = []
    for seg in skill.segments():
        if seg.goal == "start" and not skill.allows_start:
            raise ValueError(f"skill {skill.name!r} uses the reserved 'start' goal")
        out.append(Segment(seg.goal, seg.duration, seg.weight,
                           f"{skill.name}: {seg.label}" if seg.label else "", seg.base))
    return out


def skill_policy(skill: Skill, joints: Optional[list[int]] = None,
                 name: Optional[str] = None) -> Policy:
    """A runnable policy for one skill, with no bookends."""
    return SegmentPolicy(skill_segments(skill), joints=list(joints or skill.joints),
                         name=name or skill.name)


def describe_menu(skills: Sequence[type[Skill]]) -> str:
    """One line per skill for --list."""
    lines = []
    for s in skills:
        ps = ", ".join(f"{k}: {v.get('type')}" + (f" [{v['minimum']}, {v['maximum']}]" if "minimum" in v else "")
                       for k, v in s.params.get("properties", {}).items())
        lines.append(f"  {s.name}({ps})" + ("  [needs --walk]" if s.needs_base else ""))
    return "\n".join(lines)
