"""Skills: the decision-level unit, and the one format for "walk forward" and
"lift the arm".

  Policy   the executor contract: reset/step at 50 Hz; the only thing an env runs
  Motion   a scripted building block: pose segments, no sensing, no bookends
  Skill    a name, a parameter schema (the menu a model chooses from), a hard
           duration bound, and build(**args) -> Motion (walk, turn, arms up) or a
           bounded Policy. Skill -> builds -> Motion -> runs as -> SegmentPolicy.

Locomotion is a Segment with a base velocity held for its duration, exactly as
an arm move is a Segment with a pose. Every skill is bounded by STEP_MAX so the
camera is re-read every step, and every skill except ``hold`` and ``look``
ends at STAND, so skill boundaries are continuous by construction.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

from config import CONTROL_DT, UPPER_BODY, joint_index
from motions import SixSeven
from motions.poses import ARMS_UP, STAND
from policy import Motion, Policy, Segment, SegmentPolicy

STEP_MAX = 3.0        # s: every skill's max_duration is at most this (enforced by a test)
WALK_SPEED = 0.2      # m/s (< BASE_VEL_MAX[0]; the user's proven speed)
TURN_RATE = 0.4       # rad/s (< BASE_VEL_MAX[2])
WALK_MAX_M = 0.6      # = STEP_MAX * WALK_SPEED
TURN_MAX_DEG = 68.0   # 1.19 rad = 2.97 s
MIN_SEGMENT = 0.5     # s: a shorter first segment could recentre a held waist too fast
WAIST_YAW = joint_index("waist_yaw")


def _ticks(seconds: float) -> float:
    """Round a duration up to whole control ticks (and at least MIN_SEGMENT), so a
    velocity held for it integrates to exactly the intended distance or angle."""
    return max(math.ceil(seconds / CONTROL_DT - 1e-9), round(MIN_SEGMENT / CONTROL_DT)) * CONTROL_DT


class Segments(Motion):
    """A Motion from a literal tuple of segments."""

    def __init__(self, name: str, segments: Sequence[Segment]) -> None:
        self.name = name
        self._segments = tuple(segments)

    def segments(self) -> tuple[Segment, ...]:
        return self._segments


class Skill:
    name: str = "skill"
    description: str = ""
    params: dict = {"type": "object", "properties": {}, "required": []}
    needs_base: bool = False
    terminal: bool = False
    joints: list[int] = UPPER_BODY

    def build(self, **args) -> Union[Motion, Policy]:
        raise NotImplementedError

    def max_duration(self, **args) -> float:
        built = self.build(**args)
        return built.duration if isinstance(built, Motion) else STEP_MAX

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "params": self.params}


def _num(name: str, desc: str, lo: float, hi: float) -> dict:
    return {"type": "number", "description": desc, "minimum": lo, "maximum": hi}


class WalkForward(Skill):
    name = "walk_forward"
    description = "Walk straight ahead by distance_m metres (at 0.2 m/s), then stand still."
    params = {"type": "object", "properties": {"distance_m": _num("distance_m", "metres to walk", 0.1, WALK_MAX_M)},
              "required": ["distance_m"]}
    needs_base = True

    def build(self, distance_m: float) -> Motion:
        duration = _ticks(distance_m / WALK_SPEED)
        v = distance_m / duration
        return Segments("walk", (Segment(STAND, duration, base=(v, 0.0, 0.0), label=f"walk {distance_m:.2f} m"),))


class Turn(Skill):
    name = "turn"
    description = "Turn in place by angle_deg degrees: positive = LEFT (counter-clockwise), negative = RIGHT."
    params = {"type": "object", "properties": {"angle_deg": _num("angle_deg", "degrees, + left / - right",
                                                                  -TURN_MAX_DEG, TURN_MAX_DEG)},
              "required": ["angle_deg"]}
    needs_base = True

    def build(self, angle_deg: float) -> Motion:
        rad = math.radians(angle_deg)
        if abs(rad) < 1e-6:
            return Segments("turn", (Segment(STAND, MIN_SEGMENT, label="turn 0"),))
        duration = _ticks(abs(rad) / TURN_RATE)
        return Segments("turn", (Segment(STAND, duration, base=(0.0, 0.0, rad / duration),
                                         label=f"turn {angle_deg:+.0f} deg"),))


class Look(Skill):
    name = "look"
    description = ("Turn only the waist (not the feet) by yaw_deg and hold it, so the next image looks "
                   "sideways: positive = LEFT, negative = RIGHT. The next skill recentres the waist.")
    params = {"type": "object", "properties": {"yaw_deg": _num("yaw_deg", "degrees, + left / - right", -45.0, 45.0)},
              "required": ["yaw_deg"]}

    def build(self, yaw_deg: float) -> Motion:
        return Segments("look", (Segment({**STAND, WAIST_YAW: math.radians(yaw_deg)}, 1.5,
                                         label=f"look {yaw_deg:+.0f} deg"),))


class HoldStill(Skill):
    name = "hold"
    description = "Stand still for seconds (keeps the current waist direction)."
    params = {"type": "object", "properties": {"seconds": _num("seconds", "how long to wait", 0.5, STEP_MAX)},
              "required": ["seconds"]}

    def build(self, seconds: float) -> Motion:
        return Segments("hold", (Segment({}, seconds, label=f"hold {seconds:.1f} s"),))


class ArmsUp(Skill):
    name = "arms_up"
    description = "Raise both arms straight out to the sides (T-pose), then lower them."

    def build(self) -> Motion:
        return Segments("arms_up", (Segment(ARMS_UP, 1.2, label="arms up"), Segment(ARMS_UP, 0.6, label="holding"),
                                    Segment(STAND, 1.2, label="arms down")))


class Wave(Skill):
    name = "wave"
    description = "A short two-handed wave (palms up, forearms swing), then arms back down."

    def build(self) -> Motion:
        # SixSeven sized to the 3 s bound; every joint stays under check's 4 rad/s gate.
        return Segments("wave", (*SixSeven(reps=1, swing_time=0.4, settle=0.7, hold=0.1).segments(),
                                 Segment(STAND, 0.8, label="arms down")))


class Done(Skill):
    name = "done"
    description = "Stop: the goal is achieved (found=true) or cannot be achieved (found=false)."
    params = {"type": "object", "properties": {"found": {"type": "boolean", "description": "goal achieved?"},
                                               "note": {"type": "string", "description": "one short sentence"}},
              "required": ["found"]}
    terminal = True

    def build(self, found: bool, note: str = "") -> Motion:
        return Segments("done", ())


SKILLS: dict[str, Skill] = {s.name: s for s in (WalkForward(), Turn(), Look(), HoldStill(), ArmsUp(), Wave(), Done())}


def menu(allow_base: bool = True) -> list[Skill]:
    """The skills a decider may choose from; base skills only where the env can walk."""
    return [s for s in SKILLS.values() if allow_base or not s.needs_base]


def validate_args(skill: Skill, args: dict) -> tuple[dict, list[str]]:
    """Coerce and range-clamp ``args`` against the skill's schema. Unknown or
    missing keys and wrong types raise ValueError; clamps are returned as notes."""
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
            continue
        v = args[key]
        kind = spec.get("type")
        try:
            if kind == "boolean":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")
            elif kind == "number":
                v = float(v)
            elif kind == "integer":
                v = int(v)
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


def parse_skill(item: str) -> tuple[Skill, dict]:
    """``name[:arg[:arg…]]`` with args positional in schema order or ``k=v``."""
    parts = item.split(":")
    name = parts[0].strip()
    if name not in SKILLS:
        raise KeyError(name)
    skill = SKILLS[name]
    keys = list(skill.params.get("properties", {}))
    args: dict = {}
    for i, raw in enumerate(p.strip() for p in parts[1:] if p.strip()):
        if "=" in raw:
            k, v = raw.split("=", 1)
            args[k.strip()] = v.strip()
        elif i < len(keys):
            args[keys[i]] = raw
        else:
            raise ValueError(f"{name}: too many arguments in {item!r}")
    validated, _ = validate_args(skill, args)
    return skill, validated


def skill_policy(skill: Skill, args: dict, joints: Optional[list[int]] = None,
                 name: Optional[str] = None) -> Policy:
    """A runnable policy for one skill (no bookends), labels prefixed by the skill name."""
    built = skill.build(**args)
    if isinstance(built, Policy):
        return built
    segs = [Segment(s.goal, s.duration, s.weight, f"{skill.name}: {s.label}" if s.label else "", s.base)
            for s in built.segments()]
    return SegmentPolicy(segs, joints=list(joints or skill.joints), name=name or skill.name)


def describe_menu(skills: Sequence[Skill]) -> str:
    """One line per skill for --list."""
    lines = []
    for s in skills:
        ps = ", ".join(f"{k}: {v.get('type')}" + (f" [{v['minimum']}, {v['maximum']}]" if "minimum" in v else "")
                       for k, v in s.params.get("properties", {}).items())
        lines.append(f"  {s.name}({ps})" + ("  [needs --walk]" if s.needs_base else ""))
    return "\n".join(lines)
