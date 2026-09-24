"""Skills: the one building block, and the only format a behaviour is made of.

  Policy   the executor contract: reset/step at 50 Hz. The only thing an env runs.
  Skill    ``segments()`` -> pose segments, plus the joints it commands. Never
           executed directly: ``skill_policy`` turns it into a SegmentPolicy,
           and a Routine concatenates several with bookends.

"Walk forward" and "lift the arm" are the same format because a Segment may
hold a base velocity for its duration as easily as a pose. A skill may be any
length: the agent records a step every ``STEP_MAX`` seconds while one runs, so
the camera and the joint angles are captured at that cadence regardless.

**The catalog is a JSON file**, ``configs/skills.json`` (the layout of
GPT-Policy's ``tools.json``): every skill's name, the class that implements it,
whether it is enabled / terminal / internal / needs the base, the one-line
``description`` and the longer ``prompt`` a model reads, and its ``parameters``
as a JSON schema. The classes here hold only code. ``load_catalog`` validates
the file and binds that metadata onto the classes, so ``SKILLS``, ``menu()``
and ``parse_skill()`` see one consistent set. Three renderings feed a model:
``prompt_catalog`` (bullets for the system prompt), ``function_schemas`` (the
OpenAI-style function list) and ``output_schema`` (what one reply must match).

Parameters are bound at construction (``Turn(angle_deg=45, note=...)``) and
validated against the schema.
"""
from __future__ import annotations

import importlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

from config import CONTROL_DT, JOINT_HI, JOINT_LO, JOINT_NAMES, NUM_JOINTS, STAND_Q, UPPER_BODY, joint_index
from poses import ARMS_UP, SIXSEVEN, STAND
from policy import Policy, Segment, SegmentPolicy

STEP_MAX = 3.0        # s: how often a running skill is observed and recorded
WALK_SPEED = 0.2      # m/s (< BASE_VEL_MAX[0]; the speed proven on the robot)
SIDE_SPEED = 0.15     # m/s (< BASE_VEL_MAX[1])
TURN_RATE = 0.4       # rad/s (< BASE_VEL_MAX[2])
MIN_SEGMENT = 0.5     # s: a shorter entry segment could recentre a held waist too fast
WAIST_YAW = joint_index("waist_yaw")
ARM_JOINTS = [JOINT_NAMES[j] for j in UPPER_BODY]      # the 17 joints arm_sdk may command, by name
# The only onboard LocoClient calls a skill may make. Everything else (FSM, damp, torque,
# sit, squat, stand height) is unreachable from a model reply by construction.
LOCO_METHODS = frozenset({"WaveHand", "ShakeHand"})
L_ELBOW, R_ELBOW = 18, 25
ELBOW_REST = 0.0      # 90-degree bend, forearm horizontal
ELBOW_UP = -0.45      # fingertips at shoulder height

CATALOG_PATH = Path(__file__).parent / "configs" / "skills.json"
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
STATIC_TEMPLATES = {"arm_joints"}          # resolvable without a menu, so bound onto the class


def _ticks(seconds: float) -> float:
    """Round a duration up to whole control ticks (and at least MIN_SEGMENT), so a
    velocity held for it integrates to exactly the intended distance or angle."""
    return max(math.ceil(seconds / CONTROL_DT - 1e-9), round(MIN_SEGMENT / CONTROL_DT)) * CONTROL_DT


class Skill:
    """Subclass and implement ``segments()``. Everything a model reads
    (``name``, ``description``, ``prompt``, ``params``, the flags) is bound
    from the catalog; instances carry bound, validated arguments."""

    name: str = "skill"
    description: str = ""
    prompt: str = ""
    params: dict = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    enabled: bool = True
    offer: bool = True              # in the model's menu (False: CLI presets and replay only)
    needs_base: bool = False        # hidden where the env cannot walk
    needs_loco: bool = False        # hidden where the env has no onboard LocoClient (sim)
    terminal: bool = False          # ends an agent run (`done`, `give_up`)
    internal: bool = False          # bookends: never in a menu, never CLI-chainable
    allows_start: bool = False      # may use the reserved "start" goal
    command: Optional[str] = None   # the LocoClient method a gesture skill calls (LOCO_METHODS)
    joints: list[int] = UPPER_BODY
    note: str = ""                  # the model's evidence + intent, bound like any argument

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
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in self.args.items() if k != 'note')})"


# --------------------------------------------------------------------------
# Locomotion: a segment holding a base velocity
# --------------------------------------------------------------------------

class Move(Skill):
    """One relative base displacement in the start frame, their ``move_to``
    for a base: translate at (vx, vy), then rotate, so the dead-reckoned end
    pose is exactly (dx, dy, dyaw). Zero parts are skipped."""

    def segments(self) -> tuple[Segment, ...]:
        out = []
        if abs(self.dx_m) > 1e-6 or abs(self.dy_m) > 1e-6:
            duration = _ticks(max(abs(self.dx_m) / WALK_SPEED, abs(self.dy_m) / SIDE_SPEED))
            out.append(Segment(STAND, duration, base=(self.dx_m / duration, self.dy_m / duration, 0.0),
                               label=f"move {self.dx_m:+.2f} m forward, {self.dy_m:+.2f} m left"))
        rad = math.radians(self.dyaw_deg)
        if abs(rad) > 1e-6:
            duration = _ticks(abs(rad) / TURN_RATE)
            out.append(Segment(STAND, duration, base=(0.0, 0.0, rad / duration),
                               label=f"turn {self.dyaw_deg:+.0f} deg"))
        if not out:
            out.append(Segment(STAND, MIN_SEGMENT, label="move 0"))
        return tuple(out)


class WalkForward(Skill):
    def segments(self) -> tuple[Segment, ...]:
        duration = _ticks(self.distance_m / WALK_SPEED)
        v = self.distance_m / duration
        return (Segment(STAND, duration, base=(v, 0.0, 0.0), label=f"walk {self.distance_m:.2f} m"),)


class Turn(Skill):
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

class ArmPath(Skill):
    """Joint-space waypoints over the arm_sdk joints, their ``move_eef_chunk``
    without IK. Each waypoint's ``joints`` merges onto the previous pose."""

    def __init__(self, **args) -> None:
        super().__init__(**args)
        if not self.waypoints:
            raise ValueError("arm_path: at least one waypoint")
        for i, wp in enumerate(self.waypoints):
            if not isinstance(wp, dict) or not isinstance(wp.get("joints"), dict) or not wp["joints"]:
                raise ValueError(f"arm_path: waypoint {i} needs a non-empty joints object")
            for name, v in wp["joints"].items():
                if name not in ARM_JOINTS:
                    raise ValueError(f"arm_path: unknown joint {name!r} (allowed: {ARM_JOINTS})")
                j = joint_index(name)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"arm_path: {name} must be a number") from None
                if not (JOINT_LO[j] <= v <= JOINT_HI[j]):
                    raise ValueError(f"arm_path: {name}={v} outside [{JOINT_LO[j]:.3f}, {JOINT_HI[j]:.3f}]")
            seconds = wp.get("seconds", 2.0)
            if not isinstance(seconds, (int, float)) or not (0.5 <= seconds <= 10.0):
                raise ValueError(f"arm_path: waypoint {i} seconds must be within [0.5, 10]")

    def segments(self) -> tuple[Segment, ...]:
        n = len(self.waypoints)
        return tuple(Segment({joint_index(k): float(v) for k, v in wp["joints"].items()},
                             float(wp.get("seconds", 2.0)), label=f"waypoint {i + 1}/{n}")
                     for i, wp in enumerate(self.waypoints))


class Look(Skill):
    def segments(self) -> tuple[Segment, ...]:
        return (Segment({**STAND, WAIST_YAW: math.radians(self.yaw_deg)}, 1.5,
                        label=f"look {self.yaw_deg:+.0f} deg"),)


class Hold(Skill):
    def segments(self) -> tuple[Segment, ...]:
        return (Segment({}, self.seconds, label=f"hold {self.seconds:.1f} s"),)


class TPose(Skill):
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


# --------------------------------------------------------------------------
# Onboard gestures: the LocoClient does the motion, arm_sdk lets go meanwhile
# --------------------------------------------------------------------------

class Gesture(Skill):
    """Hand the arms to the onboard controller (weight 1->0), call the
    LocoClient method once, hold at weight 0 for the gesture's length, then
    take the arms back (0->1). Robot-only: sim has no onboard gestures."""

    needs_loco = True
    ramp = 1.0

    def call_args(self) -> dict:
        return {}

    def segments(self) -> tuple[Segment, ...]:
        assert self.command is not None
        return (Segment(STAND, self.ramp, weight=lambda a: 1.0 - a, label="handing the arms to the onboard controller"),
                Segment(STAND, self.seconds, weight=lambda a: 0.0, command=(self.command, self.call_args()),
                        label=f"{self.name} (onboard)"),
                Segment(STAND, self.ramp, weight=lambda a: a, label="taking the arms back"))


class WaveHand(Gesture):
    command = "WaveHand"

    def call_args(self) -> dict:
        return {"turn_flag": bool(self.turn_flag)}


class ShakeHand(Gesture):
    command = "ShakeHand"

    def call_args(self) -> dict:
        return {"stage": int(self.stage)}


# --------------------------------------------------------------------------
# Tools that move nothing: the dry run and the two terminals
# --------------------------------------------------------------------------

class Check(Skill):
    """``check(skill, arguments)``: the agent plans the named skill from its
    current commanded pose through a JointMonitor and reports the verdict.
    No segments of its own, so it can never be chained or run."""

    def segments(self) -> tuple[Segment, ...]:
        return ()

    def target(self) -> Skill:
        """The skill this check is about, built with its arguments (a note is
        not required of the inner skill)."""
        if self.skill not in SKILLS:
            raise ValueError(f"check: unknown skill {self.skill!r}")
        cls = SKILLS[self.skill]
        if cls.terminal or cls.internal or cls is Check:
            raise ValueError(f"check: {self.skill} is not a movement")
        args = dict(self.arguments)
        if "note" in cls.params.get("properties", {}):
            args.setdefault("note", self.note or "checked")
        return cls(**args)


class Done(Skill):
    def segments(self) -> tuple[Segment, ...]:
        return ()


class GiveUp(Skill):
    def segments(self) -> tuple[Segment, ...]:
        return ()


# --------------------------------------------------------------------------
# Bookends: internal skills a Routine or an agent wraps around the rest
# --------------------------------------------------------------------------

class Takeover(Skill):
    """Ramp the arm_sdk weight 0->1 while holding the pose observed at reset
    (nothing should move), then go to STAND."""

    allows_start = True

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

    def segments(self) -> tuple[Segment, ...]:
        out = []
        if self.to_stand > 0:
            out.append(Segment(STAND, self.to_stand, label="returning to stand"))
        if self.ramp > 0:
            out.append(Segment(STAND, self.ramp, weight=lambda a: 1.0 - a, label="handing back"))
        return tuple(out)


# --------------------------------------------------------------------------
# The catalog
# --------------------------------------------------------------------------

class Catalog:
    """The validated skill catalog. Binds each entry's metadata onto its class
    and renders the three model-facing views."""

    def __init__(self, data: dict, source: str = "inline") -> None:
        if not isinstance(data, dict):
            raise ValueError(f"skill catalog {source}: root must be an object")
        if data.get("version") != 1:
            raise ValueError(f"skill catalog {source}: must have version 1")
        schemas = data.get("schemas", {})
        selection = data.get("selection_description")
        raw = data.get("skills")
        if not isinstance(schemas, dict):
            raise ValueError(f"skill catalog {source}: schemas must be an object")
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError(f"skill catalog {source}: selection_description must be text")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"skill catalog {source}: skills must be a non-empty array")
        self.source = source
        self.schemas = deepcopy(schemas)
        self.selection_description = self._format(selection)
        self.skills: dict[str, type[Skill]] = {}
        self.entries: dict[str, dict] = {}
        seen: set[str] = set()
        for i, entry in enumerate(raw):
            cls = self._bind(entry, i)
            if cls.name in seen:
                raise ValueError(f"skill catalog {source}: duplicate skill {cls.name}")
            seen.add(cls.name)
            self.entries[cls.name] = entry
            if cls.enabled:
                self.skills[cls.name] = cls
        if not [c for c in self.skills.values() if not c.internal]:
            raise ValueError(f"skill catalog {source}: enables no skills")

    def _bind(self, entry: Any, i: int) -> type[Skill]:
        src = self.source
        if not isinstance(entry, dict):
            raise ValueError(f"skill catalog {src}: skills[{i}] must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ValueError(f"skill catalog {src}: skills[{i}].name is invalid")
        path = entry.get("skill")
        if not isinstance(path, str) or "." not in path:
            raise ValueError(f"skill catalog {src}: {name} must name its class as module.Class")
        module, _, attr = path.rpartition(".")
        try:
            cls = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"skill catalog {src}: {name}: cannot import {path} ({e})") from None
        if not isinstance(cls, type) or not issubclass(cls, Skill):
            raise ValueError(f"skill catalog {src}: {name}: {path} is not a Skill")
        flags = {}
        for key in ("enabled", "offer", "terminal", "internal", "needs_base", "needs_loco"):
            v = entry.get(key, key in ("enabled", "offer"))
            if not isinstance(v, bool):
                raise ValueError(f"skill catalog {src}: {name}.{key} must be boolean")
            flags[key] = v
        if cls.command is not None and cls.command not in LOCO_METHODS:
            raise ValueError(f"skill catalog {src}: {name} calls {cls.command!r}, not an allowed onboard method "
                             f"{sorted(LOCO_METHODS)}")
        if cls.command is not None and not flags["needs_loco"]:
            raise ValueError(f"skill catalog {src}: {name} calls the onboard controller and must set needs_loco")
        for key in ("description", "prompt"):
            v = entry.get(key)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"skill catalog {src}: {name} must have {key} text")
        params = entry.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object" \
                or not isinstance(params.get("properties"), dict):
            raise ValueError(f"skill catalog {src}: {name}.parameters must be an object schema")
        resolved = self._resolve(params, templates=False)
        unknown = set(resolved.get("required", [])) - set(resolved["properties"])
        if unknown:
            raise ValueError(f"skill catalog {src}: {name} requires undeclared {sorted(unknown)}")
        resolved.setdefault("required", [])
        resolved.setdefault("additionalProperties", False)
        cls.name = name
        cls.description = self._format(entry["description"])
        cls.prompt = self._format(entry["prompt"])
        cls.params = resolved
        for key, v in flags.items():
            setattr(cls, key, v)
        return cls

    # -- schema helpers ----------------------------------------------------------
    @staticmethod
    def _format(text: str) -> str:
        return text.replace("{n_joints}", str(NUM_JOINTS))

    def _resolve(self, value: Any, *, templates: bool, menu: Sequence[type[Skill]] = ()) -> Any:
        if isinstance(value, list):
            return [self._resolve(v, templates=templates, menu=menu) for v in value]
        if isinstance(value, str):
            return self._format(value)
        if not isinstance(value, dict):
            return deepcopy(value)
        if set(value) == {"$schema"}:
            ref = value["$schema"]
            if ref not in self.schemas:
                raise ValueError(f"skill catalog {self.source}: unknown schema {ref!r}")
            return self._resolve(self.schemas[ref], templates=templates, menu=menu)
        if set(value) == {"$template"}:
            if not templates and value["$template"] not in STATIC_TEMPLATES:
                return dict(value)
            return self._template(value["$template"], menu)
        return {k: self._resolve(v, templates=templates, menu=menu) for k, v in value.items()}

    def _template(self, name: str, menu: Sequence[type[Skill]]) -> dict:
        if name == "skill_name":
            names = [s.name for s in menu if not s.terminal and not s.internal and s is not Check]
            return {"type": "string", "enum": names, "description": "a skill from this menu"}
        if name == "arm_joints":
            props = {n: {"type": "number", "minimum": round(float(JOINT_LO[joint_index(n)]), 3),
                         "maximum": round(float(JOINT_HI[joint_index(n)]), 3)} for n in ARM_JOINTS}
            return {"type": "object", "properties": props, "additionalProperties": False, "minProperties": 1,
                    "description": "joint name -> target angle in radians; omitted joints keep their pose"}
        raise ValueError(f"skill catalog {self.source}: unknown template {name!r}")

    def parameters(self, cls: type[Skill], menu: Sequence[type[Skill]]) -> dict:
        """The skill's parameter schema with menu-dependent templates filled in."""
        return self._resolve(cls.params, templates=True, menu=menu)

    # -- the three renderings ----------------------------------------------------
    def prompt_catalog(self, menu: Sequence[type[Skill]]) -> str:
        return "\n".join(f"- {s.name}: {s.prompt}" for s in menu)

    def function_schemas(self, menu: Sequence[type[Skill]]) -> list[dict]:
        return [{"type": "function", "function": {"name": s.name, "description": s.description,
                                                  "parameters": self.parameters(s, menu)}}
                for s in menu]

    def output_schema(self, menu: Sequence[type[Skill]], strict: bool = True) -> dict:
        arguments = []
        for s in menu:
            p = self.parameters(s, menu)
            p = _strict(p) if strict else p
            if p not in arguments:
                arguments.append(p)
        return {"type": "object",
                "properties": {"name": {"type": "string", "enum": [s.name for s in menu]},
                               "arguments": {"anyOf": arguments}},
                "required": ["name", "arguments"], "additionalProperties": False,
                "description": self.selection_description}


def _strict(schema: Any) -> Any:
    """Every property required, optional ones nullable: what structured-output
    providers accept. Used for the request only; host validation keeps defaults."""
    if isinstance(schema, list):
        return [_strict(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _strict(v) for k, v in schema.items()}
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        required = set(out.get("required", []))
        for key, spec in out["properties"].items():
            if key not in required:
                spec = {k: v for k, v in spec.items() if k != "default"}
                out["properties"][key] = {"anyOf": [spec, {"type": "null"}]}
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


def load_catalog(path: Path | str | None = None) -> Catalog:
    p = Path(path) if path is not None else CATALOG_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"skill catalog not found: {p}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"skill catalog {p}: {e}") from None
    return Catalog(data, str(p))


CATALOG = load_catalog()
SKILLS: dict[str, type[Skill]] = dict(CATALOG.skills)


def use_catalog(path: Path | str | None) -> Catalog:
    """Swap the process-wide catalog (``--skills FILE``); ``SKILLS`` is updated
    in place so every module that imported it sees the new set."""
    global CATALOG
    CATALOG = load_catalog(path)
    SKILLS.clear()
    SKILLS.update(CATALOG.skills)
    return CATALOG


def menu(allow_base: bool = True, has_loco: bool = False) -> list[type[Skill]]:
    """The skills a decider may choose from: enabled and offered, no bookends,
    no walking where the env cannot walk, no onboard gestures without a LocoClient."""
    return [s for s in SKILLS.values() if not s.internal and s.offer
            and (allow_base or not s.needs_base) and (has_loco or not s.needs_loco)]


def joint_table() -> list[dict]:
    """The arm_sdk joints for the prompt: name, limits, the STAND value."""
    return [{"name": n, "min": round(float(JOINT_LO[joint_index(n)]), 3),
             "max": round(float(JOINT_HI[joint_index(n)]), 3),
             "stand": round(float(STAND_Q[joint_index(n)]), 3)} for n in ARM_JOINTS]


def validate_args(skill: "type[Skill]", args: dict) -> tuple[dict, list[str]]:
    """Coerce, default and range-clamp ``args`` against the skill's schema.
    Unknown or missing required keys and wrong types raise ValueError; clamps
    come back as notes. (jsonschema then checks what is left, in the decider.)"""
    props = skill.params.get("properties", {})
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key in args:
        if key not in props:
            raise ValueError(f"{skill.name}: unknown argument {key!r} (allowed: {sorted(props)})")
    for key in skill.params.get("required", []):
        if key not in args:
            if key == "note":
                continue            # required of a model (the schema check), not of code or the CLI
            raise ValueError(f"{skill.name}: missing argument {key!r}")
    for key, spec in props.items():
        if key not in args:
            if "default" in spec:
                out[key] = spec["default"]
            elif key == "note":
                out[key] = ""
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
                if v is None:
                    v = ""
                v = str(v)
            elif kind == "object":
                if not isinstance(v, dict):
                    raise TypeError
            elif kind == "array":
                if not isinstance(v, list):
                    raise TypeError
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
    """``name[:arg[:arg…]]`` with args positional in schema order or ``k=v``
    (``note`` is skipped: nobody reads it on the CLI)."""
    parts = split_outside(item, ":")
    name = parts[0].strip()
    if name not in SKILLS:
        raise KeyError(name)
    cls = SKILLS[name]
    props = cls.params.get("properties", {})
    keys = [k for k in props if k != "note"]
    args: dict = {}
    for i, raw in enumerate(p.strip() for p in parts[1:] if p.strip()):
        if "=" in raw and not raw.startswith(("{", "[")):
            k, v = raw.split("=", 1)
            args[k.strip()] = v.strip()
        elif i < len(keys):
            args[keys[i]] = raw
        else:
            raise ValueError(f"{name}: too many arguments in {item!r}")
    for k, v in list(args.items()):
        if props.get(k, {}).get("type") in ("object", "array") and isinstance(v, str):
            try:
                args[k] = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"{name}: {k} must be JSON ({e.msg})") from None
    return cls(**args)


def split_outside(item: str, sep: str = ":") -> list[str]:
    """Split on ``sep`` outside JSON brackets and quotes, so a JSON argument survives."""
    out, depth, quote, cur = [], 0, None, []
    for ch in item:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append("".join(cur)); cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return out


def skill_segments(skill: Skill) -> list[Segment]:
    """A skill's segments with labels prefixed by its name. Only the Takeover
    bookend may use the reserved "start" goal."""
    out = []
    for seg in skill.segments():
        if seg.goal == "start" and not skill.allows_start:
            raise ValueError(f"skill {skill.name!r} uses the reserved 'start' goal")
        out.append(Segment(seg.goal, seg.duration, seg.weight,
                           f"{skill.name}: {seg.label}" if seg.label else "", seg.base, seg.command))
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
        ps = ", ".join(f"{k}: {v.get('type', 'enum')}" + (f" [{v['minimum']}, {v['maximum']}]" if "minimum" in v else "")
                       for k, v in s.params.get("properties", {}).items() if k != "note")
        tags = ("  [needs --walk]" if s.needs_base else "") + ("  [robot only]" if s.needs_loco else "") \
            + ("" if s.offer else "  [preset: not offered to the model]")
        lines.append(f"  {s.name}({ps})" + tags)
    return "\n".join(lines)
