"""Routines: our reference motions, played start to finish with no model in the loop.

A routine is a timeline of **tracks**, one per level of control the SDK offers:

    routines/<name>/routine.json
    {"tracks": {
       "upper_body": {"keyframes": [{"t": 3.0, "joints": {"left_shoulder_roll": 1.57, ...}, "label": "arms up"}, ...]}
                     or {"samples": "motion.npz"},
       "base":       [{"t0": 3.0, "t1": 5.0, "vx": 0.2, "vy": 0.0, "vyaw": 0.0}],
       "onboard":    [{"t": 5.0, "call": "WaveHand", "seconds": 4.0}],
       "full_body":  {...}},
     "notes": "what it does, for people and for talk's menu"}

* ``upper_body``: waist and arm targets (``config.UPPER_BODY``) through arm_sdk while Unitree's
  controller balances. A keyframe's ``t`` is when its pose is reached; the motion between keyframes is
  eased (``core.action.ease``) and a joint a keyframe leaves out keeps its previous target. ``samples``
  is a dense recorded or trained motion (``motion.npz``: ``t`` [N], ``q`` [N, J], ``joints`` [J] names),
  reached smoothly at its first sample and played linearly between samples.
* ``base``: velocity windows through LocoClient (``Move``), within the base ceilings; standing mode only.
* ``onboard``: allow-listed LocoClient gestures (``WaveHand``, ``ShakeHand``) at a keyframe time. The
  timeline pauses there: the arms are handed to the onboard controller, the gesture runs, they are taken
  back (``tools/gestures.py``'s pattern), then the timeline goes on.
* ``full_body``: all 29 joints through low-level commands. Kept in the format; playing one is refused
  until a tracking controller exists.

With an upper-body track, every base edge and onboard time must fall on one of its keyframe (or
sample) times: a smooth segment is never split. A routine compiles to the same ``Segment`` s every
tool plays (``compile_segments``), so it is dry-run like any movement (``RoutineProgram``) and plays
through the same player, takeover and handback (``tools.Chain``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

from g1.core import limits
from g1.core.action import Segment
from g1.core.config import BASE_VEL_MAX, JOINT_HI, JOINT_LO, JOINT_NAMES, UPPER_BODY, joint_index
from g1.core.poses import STAND

DIR = limits.ROOT / "routines"
GESTURES = ("WaveHand", "ShakeHand")       # the onboard calls a routine may make (config.LOCO_METHODS minus Sit)
TIME_EPS = 1e-6

_NUM = {"type": "number"}
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["tracks"],
    "properties": {
        "notes": {"type": "string"},
        "tracks": {
            "type": "object", "additionalProperties": False, "minProperties": 1,
            "properties": {
                "upper_body": {"type": "object", "additionalProperties": False,
                               "oneOf": [{"required": ["keyframes"]}, {"required": ["samples"]}],
                               "properties": {
                                   "keyframes": {"type": "array", "minItems": 1, "items": {
                                       "type": "object", "additionalProperties": False, "required": ["t", "joints"],
                                       "properties": {"t": {"type": "number", "exclusiveMinimum": 0},
                                                      "joints": {"type": "object", "additionalProperties": _NUM},
                                                      "label": {"type": "string"}}}},
                                   "samples": {"type": "string", "pattern": r"^[\w.-]+\.npz$"}}},
                "base": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["t0", "t1"],
                    "properties": {"t0": {"type": "number", "minimum": 0}, "t1": _NUM, "vx": _NUM, "vy": _NUM,
                                   "vyaw": _NUM, "label": {"type": "string"}}}},
                "onboard": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["t", "call", "seconds"],
                    "properties": {"t": {"type": "number", "minimum": 0}, "call": {"enum": list(GESTURES)},
                                   "seconds": {"type": "number", "exclusiveMinimum": 0},
                                   "args": {"type": "object"}, "label": {"type": "string"}}}},
                "full_body": {"type": "object"},
            }},
    },
}


class RoutineError(ValueError):
    """A routine that cannot be loaded or played."""


@dataclass
class Routine:
    name: str
    dir: Path
    notes: str
    tracks: dict
    points: list[tuple[float, dict[int, float], str, str]]   # upper body: (t, goal, label, ease)

    @property
    def full_body(self) -> bool:
        return "full_body" in self.tracks

    @property
    def uses_base(self) -> bool:
        return bool(self.tracks.get("base"))

    @property
    def uses_onboard(self) -> bool:
        return bool(self.tracks.get("onboard"))

    @property
    def upper_body_only(self) -> bool:
        return not (self.full_body or self.uses_base or self.uses_onboard)


def routine_names(root: Optional[Path] = None) -> list[str]:
    root = Path(root) if root is not None else DIR
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "routine.json").is_file())


def find(name_or_dir: str | Path, root: Optional[Path] = None) -> Path:
    p = Path(name_or_dir)
    if (p / "routine.json").is_file():
        return p
    root = Path(root) if root is not None else DIR
    if (root / p / "routine.json").is_file():
        return root / p
    raise RoutineError(f"no routine {str(name_or_dir)!r}; routines: {', '.join(routine_names(root)) or 'none'}")


def _upper_index(name: str, where: str) -> int:
    if name not in JOINT_NAMES:
        raise RoutineError(f"{where}: unknown joint {name!r}")
    j = joint_index(name)
    if j not in UPPER_BODY:
        raise RoutineError(f"{where}: {name} is not an upper-body joint (the legs are Unitree's; "
                           "a full-body routine goes in full_body)")
    return j


def _check_angle(j: int, v: float, where: str) -> float:
    v = float(v)
    if not (JOINT_LO[j] <= v <= JOINT_HI[j]):
        raise RoutineError(f"{where}: {JOINT_NAMES[j]} {v:.3f} rad is outside its limits "
                           f"[{JOINT_LO[j]:.3f}, {JOINT_HI[j]:.3f}]")
    return v


def _keyframe_points(keyframes: list[dict], where: str) -> list[tuple[float, dict[int, float], str, str]]:
    if len(keyframes) > limits.get("routine_keyframes_max"):
        raise RoutineError(f"{where}: {len(keyframes)} keyframes; at most routine_keyframes_max "
                           f"({limits.get('routine_keyframes_max'):g})")
    points, last = [], 0.0
    for i, k in enumerate(keyframes):
        at = f"{where}: keyframe {i + 1}"
        if k["t"] <= last + TIME_EPS:
            raise RoutineError(f"{at}: t {k['t']} must come after {last} (times are when each pose is reached)")
        goal = {_upper_index(n, at): 0.0 for n in k["joints"]}
        for n, v in k["joints"].items():
            goal[joint_index(n)] = _check_angle(joint_index(n), v, at)
        points.append((float(k["t"]), goal, k.get("label") or f"keyframe {i + 1}/{len(keyframes)}", "smooth"))
        last = float(k["t"])
    return points


def _sample_points(path: Path, where: str) -> list[tuple[float, dict[int, float], str, str]]:
    try:
        data = np.load(path, allow_pickle=False)
        t, q, names = np.asarray(data["t"], float), np.asarray(data["q"], float), [str(n) for n in data["joints"]]
    except (OSError, KeyError, ValueError) as e:
        raise RoutineError(f"{where}: cannot read {path.name} (needs t [N], q [N, J], joints [J]): {e}") from None
    if t.ndim != 1 or q.shape != (len(t), len(names)) or len(t) < 1:
        raise RoutineError(f"{where}: {path.name}: t is {t.shape}, q {q.shape}, {len(names)} joints")
    if t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise RoutineError(f"{where}: {path.name}: t must start after 0 and increase")
    rate_max = limits.get("routine_sample_rate_max_hz")
    if len(t) > 1 and np.min(np.diff(t)) < 1.0 / rate_max - TIME_EPS:
        raise RoutineError(f"{where}: {path.name}: samples closer than 1/routine_sample_rate_max_hz ({rate_max:g} Hz)")
    idx = [_upper_index(n, where) for n in names]
    points = []
    for i in range(len(t)):
        goal = {j: _check_angle(j, q[i, c], f"{where}: sample {i + 1}") for c, j in enumerate(idx)}
        points.append((float(t[i]), goal, "to the start of the motion" if i == 0 else ("playing the motion" if i == 1 else ""),
                       "smooth" if i == 0 else "linear"))
    return points


def load_routine(name_or_dir: str | Path, root: Optional[Path] = None) -> Routine:
    """``routine.json`` validated (schema, joints, limits, timing) and its upper-body points built."""
    d = find(name_or_dir, root)
    path = d / "routine.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RoutineError(f"{path}: {e.msg} (line {e.lineno})") from None
    try:
        Draft202012Validator(SCHEMA).validate(data)
    except ValidationError as e:
        where = ".".join(str(p) for p in e.absolute_path) or "routine.json"
        raise RoutineError(f"{path}: {where}: {e.message}") from None
    tracks = data["tracks"]
    upper = tracks.get("upper_body")
    points = []
    if upper is not None:
        points = (_keyframe_points(upper["keyframes"], str(path)) if "keyframes" in upper
                  else _sample_points(d / upper["samples"], str(path)))
    routine = Routine(d.name, d, data.get("notes", ""), tracks, points)
    _check_tracks(routine, str(path))
    return routine


def _check_tracks(r: Routine, where: str) -> None:
    times = {round(t, 6) for t, *_ in r.points} | {0.0}
    on_point = (lambda t: round(t, 6) in times) if r.points else (lambda t: True)
    for i, w in enumerate(r.tracks.get("base", [])):
        at = f"{where}: base window {i + 1}"
        if w["t1"] <= w["t0"] + TIME_EPS:
            raise RoutineError(f"{at}: t1 must come after t0")
        v = np.array([w.get("vx", 0.0), w.get("vy", 0.0), w.get("vyaw", 0.0)])
        over = np.abs(v) > BASE_VEL_MAX + 1e-9
        if over.any():
            names = [n for n, o in zip(("vx", "vy", "vyaw"), over) if o]
            raise RoutineError(f"{at}: {', '.join(names)} over the base ceilings "
                               f"(base_vx_max, base_vy_max, base_vyaw_max: {BASE_VEL_MAX.tolist()})")
        if not (on_point(w["t0"]) and on_point(w["t1"])):
            raise RoutineError(f"{at}: t0 and t1 must be upper-body keyframe times ({sorted(times)}): "
                               "a smooth segment is never split")
    for i, o in enumerate(r.tracks.get("onboard", [])):
        at = f"{where}: onboard {i + 1}"
        if not on_point(o["t"]):
            raise RoutineError(f"{at}: t must be an upper-body keyframe time ({sorted(times)})")
        if not (limits.get("gesture_seconds_min") <= o["seconds"] <= limits.get("gesture_seconds_max")):
            raise RoutineError(f"{at}: seconds must be within gesture_seconds_min..max")
        for w in r.tracks.get("base", []):
            if w["t0"] < o["t"] < w["t1"]:
                raise RoutineError(f"{at}: a gesture cannot start inside a base window (the base would keep moving)")
    total = sum(s.duration for s in compile_segments(r))
    if total > limits.get("routine_duration_max_s"):
        raise RoutineError(f"{where}: {total:.1f} s long; at most routine_duration_max_s "
                           f"({limits.get('routine_duration_max_s'):g} s)")


def _gesture(o: dict) -> list[Segment]:
    ramp = limits.get("gesture_ramp_s")
    label = o.get("label") or o["call"]
    return [Segment(STAND, ramp, weight=lambda a: 1.0 - a, label="handing the arms to the onboard controller"),
            Segment(STAND, float(o["seconds"]), weight=lambda a: 0.0, command=(o["call"], dict(o.get("args", {}))),
                    label=f"{label} (onboard)"),
            Segment(STAND, ramp, weight=lambda a: a, label="taking the arms back")]


def compile_segments(r: Routine) -> list[Segment]:
    """The timeline as the segments every tool plays. Each interval between consecutive edges (keyframe
    or sample times, base edges, gesture times) is one segment: it reaches the pose due at its end (or
    holds when none is), with the base velocity of the window it lies in; a gesture is inserted at its
    time and the timeline goes on after it."""
    goals = {round(t, 6): (goal, label, ease) for t, goal, label, ease in r.points}
    base = r.tracks.get("base", [])
    onboard = sorted(r.tracks.get("onboard", []), key=lambda o: o["t"])
    edges = sorted({0.0} | set(goals) | {round(w["t0"], 6) for w in base} | {round(w["t1"], 6) for w in base}
                   | {round(o["t"], 6) for o in onboard})
    out: list[Segment] = []
    for a, b in zip(edges, edges[1:]):
        out += [s for o in onboard if round(o["t"], 6) == a for s in _gesture(o)]
        goal, label, ease = goals.get(b, ({}, "", "smooth"))
        window = next((w for w in base if w["t0"] - TIME_EPS <= a and b <= w["t1"] + TIME_EPS), None)
        vel = None if window is None else (window.get("vx", 0.0), window.get("vy", 0.0), window.get("vyaw", 0.0))
        if window is not None and window.get("label"):
            label = window["label"]                              # a named step of the base says what is happening
        out.append(Segment(dict(goal), b - a, label=label, base=vel, ease=ease))
    last = edges[-1]
    out += [s for o in onboard if round(o["t"], 6) == last for s in _gesture(o)]
    return out


def run_spec(name_or_dir: str | Path, root: Optional[Path] = None) -> dict[str, Any]:
    """What ``cli.run_once`` needs to play a routine and record it under ``routines/<name>/runs/``:
    a task-shaped dict with the task defaults and ``routine`` set (no instruction, no model)."""
    from g1.agent.task import SCHEMA as TASK_SCHEMA
    r = load_routine(name_or_dir, root)
    spec: dict[str, Any] = {k: v.get("default") for k, v in TASK_SCHEMA["properties"].items()}
    duration = sum(s.duration for s in compile_segments(r))
    spec.update(instruction=None, routine=r.name, routine_dir=str(r.dir), dir=r.dir, name=r.name, notes=r.notes,
                max_time_s=max(limits.get("max_time_s"), duration + 4 * limits.get("bookend_ramp_max_s")))
    return spec
