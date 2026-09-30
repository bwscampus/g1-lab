"""The tools: every ``Tool`` subclass in this package, found by import.

    TOOLS                 name -> class, every tool (visible or not)
    menu(allow_base, has_loco)   the classes a model may choose from
    prompt_catalog / function_schemas / output_schema   the three renderings a model reads
    parse_tool("turn:45") / parse_chain("walk_forward:0.5,hold:1,tpose")   the CLI forms
    Chain                 several tools played in order with the bookends once

Drop a file in this directory with a ``Tool`` subclass and it is in the menu
on the next run (``g1 new tool NAME`` writes one from the template); a bad
definition fails at import, naming the file. Modules whose name starts with
``_`` are not imported (``_template.py``).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import pkgutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

from g1.core import limits
from g1.core.action import Segment
from g1.core.config import JOINT_HI, JOINT_LO, JOINT_NAMES, STAND_Q, UPPER_BODY, joint_index
from g1.tools.base import (NOTE, Limit, Tool, flag, format_text, integer, limit, num, resolve, text,
                           validate_args)

STEP_MAX = limits.get("record_step_s")      # how often a running tool is observed and recorded
ARM_JOINTS = [JOINT_NAMES[j] for j in UPPER_BODY]      # the 17 joints arm_sdk may command, by name
SELECTION_DESCRIPTION = (
    "One tool selection for a Unitree G1 humanoid with {n_joints} joints. The host plans the tool, checks it "
    "against joint and speed limits, runs it to its end (arm_sdk joint targets; LocoClient velocity commands "
    "for the base; the onboard controller for gestures), waits for the joints to settle, then supplies a fresh "
    "observation with the result.")

TOOLS: dict[str, type[Tool]] = {}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def _sort_key(cls: type[Tool]) -> tuple:
    return (cls.kind == "terminal", cls.order, cls.name)


def tools_in(module) -> list[type[Tool]]:
    """The tools a module defines (its own ``Tool`` subclasses with a name)."""
    out = []
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, Tool) and value is not Tool \
                and value.__module__ == module.__name__ and value.name:
            out.append(value)
    return out


def register(cls: type[Tool], where: str = "") -> type[Tool]:
    """Validate a tool class and add it to ``TOOLS``; a duplicate name fails."""
    cls.check_definition(where)
    other = TOOLS.get(cls.name)
    if other is not None and other is not cls:
        raise ValueError(f"{where or cls.__module__}: tool {cls.name!r} is already defined in {other.__module__}")
    TOOLS[cls.name] = cls
    for name in sorted(TOOLS, key=lambda n: _sort_key(TOOLS[n])):
        TOOLS[name] = TOOLS.pop(name)
    return cls


def unregister(name: str) -> None:
    TOOLS.pop(name, None)


def load_module(path: Path | str) -> list[type[Tool]]:
    """Import a tool file by path (a student's file outside the package, a
    test's scaffolded tool) and register what it defines."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"g1.tools._{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = tools_in(module)
    if not found:
        raise ValueError(f"{path}: defines no Tool subclass with a name")
    return [register(cls, str(path)) for cls in found]


def discover() -> dict[str, type[Tool]]:
    package = Path(__file__).parent
    for info in sorted(pkgutil.iter_modules([str(package)]), key=lambda i: i.name):
        if info.name.startswith("_") or info.name == "base":
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for cls in tools_in(module):
            register(cls, str(package / f"{info.name}.py"))
    return TOOLS


discover()


# --------------------------------------------------------------------------
# The menu and its three renderings
# --------------------------------------------------------------------------

def menu(allow_base: bool = True, has_loco: bool = False) -> list[type[Tool]]:
    """The tools a decider may choose from: visible, no walking where the env
    cannot walk, no onboard gestures without a LocoClient."""
    return [t for t in TOOLS.values() if t.visible
            and (allow_base or not t.needs_base) and (has_loco or not t.needs_loco)]


def _template(name: str, context: Sequence[type[Tool]]) -> dict:
    if name == "tool_name":
        names = [t.name for t in context if t.kind == "motion"]
        return {"type": "string", "enum": names, "description": "a tool from this menu"}
    if name == "arm_joints":
        props = {n: {"type": "number", "minimum": round(float(JOINT_LO[joint_index(n)]), 3),
                     "maximum": round(float(JOINT_HI[joint_index(n)]), 3)} for n in ARM_JOINTS}
        return {"type": "object", "properties": props, "additionalProperties": False, "minProperties": 1,
                "description": "joint name -> target angle in radians; omitted joints keep their pose"}
    raise ValueError(f"unknown template {name!r}")


def _fill(value: Any, context: Sequence[type[Tool]]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$template"}:
            return _template(value["$template"], context)
        return {k: _fill(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, context) for v in value]
    return value


def parameters(cls: type[Tool], context: Sequence[type[Tool]] = ()) -> dict:
    """The tool's argument schema with menu-dependent templates filled in."""
    return _fill(cls.schema(), context or list(TOOLS.values()))


def prompt_catalog(context: Sequence[type[Tool]]) -> str:
    return "\n".join(f"- {t.name}: {t.describe()}" for t in context)


def function_schemas(context: Sequence[type[Tool]]) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.describe(),
                                              "parameters": parameters(t, context)}}
            for t in context]


def output_schema(context: Sequence[type[Tool]], strict: bool = True) -> dict:
    arguments = []
    for t in context:
        p = parameters(t, context)
        p = _strict(p) if strict else p
        if p not in arguments:
            arguments.append(p)
    return {"type": "object",
            "properties": {"name": {"type": "string", "enum": [t.name for t in context]},
                           "arguments": {"anyOf": arguments}},
            "required": ["name", "arguments"], "additionalProperties": False,
            "description": format_text(SELECTION_DESCRIPTION)}


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


def joint_table() -> list[dict]:
    """The arm_sdk joints for the prompt: name, limits, the STAND value."""
    return [{"name": n, "min": round(float(JOINT_LO[joint_index(n)]), 3),
             "max": round(float(JOINT_HI[joint_index(n)]), 3),
             "stand": round(float(STAND_Q[joint_index(n)]), 3)} for n in ARM_JOINTS]


def describe_menu(tools: Sequence[type[Tool]]) -> str:
    """One line per tool for ``g1 tools``."""
    lines = []
    for t in tools:
        ps = ", ".join(f"{k}: {v.get('type', 'enum')}" + (f" [{v['minimum']:g}, {v['maximum']:g}]" if "minimum" in v else "")
                       for k, v in t.schema()["properties"].items() if k != "note")
        tags = ("  [needs --walk]" if t.needs_base else "") + ("  [robot only]" if t.needs_loco else "") \
            + ("" if t.visible else "  [preset: not offered to the model]")
        lines.append(f"  {t.name}({ps})" + tags)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The CLI forms
# --------------------------------------------------------------------------

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


def parse_tool(item: str) -> Tool:
    """``name[:arg[:arg…]]`` with args positional in schema order or ``k=v``
    (``note`` is skipped: nobody reads it on the CLI)."""
    parts = split_outside(item, ":")
    name = parts[0].strip()
    if name not in TOOLS:
        raise KeyError(name)
    cls = TOOLS[name]
    props = cls.schema()["properties"]
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


def chain_spec(steps: Sequence[tuple[str, dict]]) -> str:
    """The ``--tools`` string for (name, args) pairs (what a replay needs)."""
    items = []
    for name, args in steps:
        parts = ":".join(f"{k}={json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v}"
                         for k, v in args.items() if k != "note")
        items.append(name + (":" + parts if parts else ""))
    return ",".join(items)


def prefixed(tool: Tool) -> list[Segment]:
    """A tool's segments with labels prefixed by its name."""
    return [Segment(seg.goal, seg.duration, seg.weight,
                    f"{tool.name}: {seg.label}" if seg.label else "", seg.base, seg.command)
            for seg in tool.segments()]


class Chain(Tool):
    """Bookends once, then the given tools in order with a pause between, as
    one continuous command stream (every boundary is inside the monitor's
    velocity gate). Composition is at the segment level on purpose: chaining
    programs would restart each from the env's *measured* q, which on the
    robot lags the command by gravity sag and would jump at every boundary."""

    visible = False

    def __init__(self, *tools: Tool, pause: float = 1.0, name: Optional[str] = None) -> None:
        from g1.tools.control import Handback, Hold, Takeover
        if not tools:
            raise ValueError("a chain needs at least one tool")
        for t in tools:
            if t.kind != "motion" or not t.segments():
                raise ValueError(f"{t.name} cannot be chained; it is not a movement")
        parts: list[Tool] = [Takeover()]
        for i, t in enumerate(tools):
            if i > 0 and pause > 0:
                parts.append(Hold(seconds=pause))
            parts.append(t)
        parts.append(Handback())
        segs: list[Segment] = []
        joints: set[int] = set()
        for part in parts:
            joints.update(part.joints)
            segs.extend(prefixed(part))
        self.tools = tuple(tools)
        self._segs = tuple(segs)
        self.joints = sorted(joints)
        self.name = name or "+".join(t.name for t in tools)
        self.allows_start = True
        self.args, self.notes = {}, []

    def segments(self) -> tuple[Segment, ...]:
        return self._segs


def parse_chain(spec: str, pause: float = 1.0) -> Chain:
    """``walk_forward:0.5,turn:45,tpose`` as one Chain from the start pose."""
    items = [n.strip() for n in split_outside(spec, ",") if n.strip()]
    if not items:
        raise KeyError(spec)
    return Chain(*(parse_tool(item) for item in items), pause=pause)


__all__ = ["TOOLS", "Tool", "Chain", "Limit", "NOTE", "STEP_MAX", "ARM_JOINTS", "SELECTION_DESCRIPTION",
           "limit", "num", "integer", "text", "flag", "resolve", "format_text", "validate_args",
           "menu", "parameters", "prompt_catalog", "function_schemas", "output_schema", "joint_table",
           "describe_menu", "parse_tool", "parse_chain", "chain_spec", "prefixed", "split_outside",
           "register", "unregister", "load_module", "discover", "tools_in"]
