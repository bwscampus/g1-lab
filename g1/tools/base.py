"""The Tool: the one building block, and what the model picks.

A tool is a bounded program with parameters, in GPT-Policy's sense: the model
names one and gives its arguments, the host plans the motion from those
arguments, checks it, executes it, waits for the joints to settle and reports
what happened. **No tool watches the camera while it runs** — the model is the
closed loop (observe, decide, act, feed back), so every tool ends by
construction: a motion has a duration, a query has none.

One class holds everything the model reads and everything the robot runs::

    class Bow(Tool):
        name = "bow"
        prompt = "Bow from the waist by angle_deg degrees over down_s seconds ({limit:hold_seconds_max} ...)"
        params = {"angle_deg": num(5, 30, 15, "how far to bend, degrees"),
                  "down_s": num(0.5, 5, 1.5, "seconds to bend down"), ...}

        def segments(self):
            return (Segment({WAIST_PITCH: math.radians(self.angle_deg)}, self.down_s, label="bowing"),
                    Segment(STAND, self.up_s, label="straightening up"))

``params`` maps argument names to JSON-schema fragments (``num``, ``integer``,
``text``, ``flag``, or a dict); a property without a ``default`` is required;
a ``limit("name")`` marker anywhere in a spec, and ``{limit:name}`` in the
prompt, are filled from ``configs/limits.json`` when read, so text and schema
can never quote a number the host does not enforce. Every motion tool also
takes a ``note`` — required of the model, defaulted to "" for code and the CLI.

Arguments are validated and clamped at construction (``Turn(angle_deg=45)``)
and become attributes. The 50 Hz player (``reset``/``step``) is part of the
class and is not overridden by tools: it interpolates ``segments()`` from the
pose the tool was seeded with, so tools chain continuously.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

import numpy as np

from g1.core import limits
from g1.core.action import Action, Obs, Pose, Runnable, Segment, ease
from g1.core.config import LOCO_METHODS, NUM_JOINTS, UPPER_BODY

KINDS = ("motion", "query", "terminal")
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_LIMIT_IN_TEXT = re.compile(r"\{limit:([a-z0-9_]+)\}")

NOTE = {"type": "string", "minLength": 1,
        "description": "1-2 short sentences: the current visual evidence and the purpose of this action."}


# --------------------------------------------------------------------------
# Parameter specs
# --------------------------------------------------------------------------

class Limit:
    """A reference to a limit, resolved when the schema is read (so a swapped
    limits file changes what the model is offered)."""

    def __init__(self, name: str, negate: bool = False) -> None:
        if name not in limits.NAMES:
            raise ValueError(f"unknown limit {name!r} (see configs/limits.json)")
        self.name = name
        self.negate = negate

    def __neg__(self) -> "Limit":
        return Limit(self.name, not self.negate)

    def value(self) -> Any:
        v = limits.get(self.name)
        return -v if self.negate else v

    def __repr__(self) -> str:
        return f"{'-' if self.negate else ''}limit({self.name!r})"


def limit(name: str) -> Limit:
    return Limit(name)


Number = float | int | Limit


def num(lo: Number, hi: Number, default: Optional[Number] = None, description: str = "") -> dict:
    """A continuous quantity: a number with a range (and a default, unless required)."""
    spec: dict[str, Any] = {"type": "number", "minimum": lo, "maximum": hi, "description": description}
    if default is not None:
        spec["default"] = default
    return spec


def integer(lo: Number, hi: Number, default: Optional[Number] = None, description: str = "") -> dict:
    spec: dict[str, Any] = {"type": "integer", "minimum": lo, "maximum": hi, "description": description}
    if default is not None:
        spec["default"] = default
    return spec


def text(description: str = "", required: bool = True) -> dict:
    """Free text; ``required`` means it must be non-empty."""
    spec: dict[str, Any] = {"type": "string", "description": description}
    if required:
        spec["minLength"] = 1
    else:
        spec["default"] = ""
    return spec


def flag(default: bool = False, description: str = "") -> dict:
    return {"type": "boolean", "default": default, "description": description}


def format_text(text_: str) -> str:
    """``{limit:name}`` and ``{n_joints}`` in prompt text."""
    def fill(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in limits.LIMITS:
            raise ValueError(f"unknown limit {name!r} in text")
        return f"{limits.get(name):g}"
    return _LIMIT_IN_TEXT.sub(fill, text_.replace("{n_joints}", str(NUM_JOINTS)))


def resolve(value: Any) -> Any:
    """A spec with every ``Limit`` replaced by its value and text formatted."""
    if isinstance(value, Limit):
        return value.value()
    if isinstance(value, str):
        return format_text(value)
    if isinstance(value, list):
        return [resolve(v) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------
# The Tool
# --------------------------------------------------------------------------

class Tool(Runnable):
    """Subclass, declare the metadata, implement ``segments()``."""

    # -- what the model reads ------------------------------------------------------------
    name: str = ""
    prompt: str = ""                    # what the model reads about this tool (visible tools)
    params: dict = {}                   # argument name -> spec (see num/integer/text/flag)
    kind: str = "motion"                # motion (plays segments) | query (answers) | terminal (ends the run)
    visible: bool = True                # in the model's menu; False: CLI chains and replay only
    needs_base: bool = False            # hidden where the env cannot walk
    needs_loco: bool = False            # hidden where the env has no onboard LocoClient (sim)
    order: int = 50                     # menu position (lower first; terminal tools go last)
    # -- what the robot runs -----------------------------------------------------------
    joints: list[int] = UPPER_BODY
    command: Optional[str] = None       # the LocoClient method a gesture calls, once (config.LOCO_METHODS)
    allows_start: bool = False          # may use the reserved "start" segment goal (the takeover bookend)
    note: str = ""                      # the model's evidence + intent, bound like any argument

    def __init__(self, **args) -> None:
        self.args, self.notes = validate_args(type(self), args)
        for k, v in self.args.items():
            setattr(self, k, v)

    def segments(self) -> tuple[Segment, ...]:
        """The motion, computed from the arguments. ``()`` for a query or terminal tool."""
        return ()

    def query(self, cmd: np.ndarray, joints: Sequence[int]) -> dict:
        """A query tool's answer, given the commanded pose. Never moves anything."""
        raise NotImplementedError(f"{self.name} is not a query tool")

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments())

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in self.args.items() if k != 'note')})"

    # -- the schema --------------------------------------------------------------------------
    @classmethod
    def schema(cls) -> dict:
        """The argument object schema, limits filled in, ``note`` added for
        motion and query tools, a property without a default required."""
        props = {k: resolve(v) for k, v in cls.params.items()}
        if cls.kind != "terminal":
            props.setdefault("note", dict(NOTE))
        required = [k for k, spec in props.items() if "default" not in spec]
        return {"type": "object", "properties": props, "required": required, "additionalProperties": False}

    @classmethod
    def describe(cls) -> str:
        """The prompt with limits filled in."""
        return format_text(cls.prompt)

    @classmethod
    def check_definition(cls, where: str = "") -> None:
        """Fail loudly at import when a tool is declared wrongly."""
        src = f"{where}: " if where else ""
        if not isinstance(cls.name, str) or _NAME.fullmatch(cls.name) is None:
            raise ValueError(f"{src}tool {cls.__name__} needs a snake_case name")
        if cls.kind not in KINDS:
            raise ValueError(f"{src}{cls.name}.kind must be one of {KINDS}")
        if cls.visible and not (isinstance(cls.prompt, str) and cls.prompt.strip()):
            raise ValueError(f"{src}{cls.name} is visible to the model and needs a prompt")
        if not isinstance(cls.params, dict):
            raise ValueError(f"{src}{cls.name}.params must be a dict of argument specs")
        for key, spec in cls.params.items():
            if not isinstance(key, str) or _NAME.fullmatch(key) is None or key == "note":
                raise ValueError(f"{src}{cls.name}: bad argument name {key!r}")
            if isinstance(spec, dict) and set(spec) == {"$template"}:
                continue                                    # filled from the menu (see tools.parameters)
            if not isinstance(spec, dict) or "type" not in spec:
                raise ValueError(f"{src}{cls.name}.{key}: a spec needs a type (use num/integer/text/flag)")
            if spec["type"] in ("number", "integer") and ("minimum" not in spec or "maximum" not in spec):
                raise ValueError(f"{src}{cls.name}.{key}: a number needs minimum and maximum (use num/integer)")
        if cls.command is not None and cls.command not in LOCO_METHODS:
            raise ValueError(f"{src}{cls.name} calls {cls.command!r}, not an allowed onboard method {sorted(LOCO_METHODS)}")
        if cls.command is not None and not cls.needs_loco:
            raise ValueError(f"{src}{cls.name} calls the onboard controller and must set needs_loco = True")
        if any(j < 0 or j >= NUM_JOINTS for j in cls.joints):
            raise ValueError(f"{src}{cls.name}.joints has an invalid joint index")
        cls.describe()                                          # unknown {limit:...} fails here
        cls.schema()

    # -- the player: reset once, step at 50 Hz until None ---------------------------------------
    def reset(self, obs: Obs) -> None:
        """Seed from ``obs.q``. Every chained tool is seeded from the previous
        tool's last *commanded* pose, never the measured one."""
        allowed = set(self.joints)
        self._segments = tuple(self.segments())
        for seg in self._segments:
            if isinstance(seg.goal, dict):
                bad = sorted(set(seg.goal) - allowed)
                if bad:
                    raise ValueError(f"[{self.name}] segment {seg.label!r} sets joints {bad} outside this tool's joints")
            elif seg.goal == "start" and not self.allows_start:
                raise ValueError(f"tool {self.name!r} uses the reserved 'start' goal")
        self._q0 = np.array(obs.q, dtype=float)
        start = {j: float(self._q0[j]) for j in self.joints}
        self._plan: list[tuple[float, float, Pose, Pose, Segment]] = []
        t = 0.0
        prev = dict(start)
        for seg in self._segments:
            goal = dict(start) if seg.goal == "start" else {**prev, **seg.goal}
            self._plan.append((t, t + seg.duration, prev, goal, seg))
            prev = goal
            t += seg.duration
        self.total_time = t
        self._last_label: str | None = None
        self._commanded: set[int] = set()

    def step(self, t: float, obs: Obs) -> Optional[Action]:
        for i, (t0, t1, start, goal, seg) in enumerate(self._plan):
            if t < t1 - 1e-9:
                a = ease((t - t0) / (t1 - t0))
                if seg.label and seg.label != self._last_label:
                    print(f"[{self.name}] {seg.label}")
                    self._last_label = seg.label
                out = self._q0.copy()
                for j in self.joints:
                    out[j] = start[j] + a * (goal[j] - start[j])
                command = None
                if seg.command is not None and i not in self._commanded:
                    self._commanded.add(i)
                    command = seg.command
                return self.action(out, weight=seg.weight(a), base=seg.base, command=command)
        return None

    @classmethod
    def of(cls, segments: Sequence[Segment], *, joints: Optional[Sequence[int]] = None,
           name: str = "segments", allows_start: bool = False) -> "Tool":
        """An anonymous tool that plays the given segments (the safe return, tests)."""
        segs = tuple(segments)
        js = list(joints) if joints is not None else list(cls.joints)

        class Segments(Tool):
            visible = False

            def segments(self) -> tuple[Segment, ...]:
                return segs

        Segments.name = name
        Segments.joints = js
        Segments.allows_start = allows_start
        return Segments()


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

def validate_args(tool: "type[Tool]", args: dict) -> tuple[dict, list[str]]:
    """Coerce, default and range-clamp ``args`` against the tool's schema.
    Unknown or missing required keys and wrong types raise ValueError; clamps
    come back as notes. (jsonschema then checks what is left, in the decider.)"""
    schema = tool.schema()
    props = schema["properties"]
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key in args:
        if key not in props:
            raise ValueError(f"{tool.name}: unknown argument {key!r} (allowed: {sorted(props)})")
    for key in schema["required"]:
        if key not in args:
            if key == "note":
                continue            # required of a model (the schema check), not of code or the CLI
            raise ValueError(f"{tool.name}: missing argument {key!r}")
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
            raise ValueError(f"{tool.name}: argument {key!r} must be {kind}, got {args[key]!r}") from None
        if kind in ("number", "integer"):
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if lo is not None and v < lo:
                notes.append(f"{key} {v} raised to {lo}"); v = lo
            if hi is not None and v > hi:
                notes.append(f"{key} {v} lowered to {hi}"); v = hi
        out[key] = v
    return out, notes
