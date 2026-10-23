"""A task: a folder with what to do and its context, no code. It is what ``g1 run`` runs.

    tasks/<name>/
        task.json       an instruction (the model decides) or a tools chain (no model),
                        the scene, safety notes, the budget
        runs/           every run of this task (git-ignored), each with its trace.npz
        results.jsonl   one line per run: directory, outcome, decisions, tokens, model

    {"instruction": "find the mug and stop in front of it", "scene": "room", "objects": ["mug@1.5,1.2"]}
    {"tools": "tpose,look:30"}
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jsonschema import Draft202012Validator, ValidationError

from g1.core import limits

ROOT = limits.ROOT
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "oneOf": [{"required": ["instruction"], "not": {"required": ["tools"]}},
              {"required": ["tools"], "not": {"required": ["instruction"]}}],
    "properties": {
        "instruction": {"type": "string", "minLength": 1, "description": "what the model is asked to do"},
        "tools": {"type": "string", "minLength": 1, "description": "a fixed chain instead, e.g. tpose,look:30"},
        "scene": {"type": "string", "enum": ["none", "room"], "default": "none", "description": "the sim scene"},
        "objects": {"type": "array", "items": {"type": "string"}, "default": [],
                    "description": "sim objects to place, name@x,y[,z] (needs scene room)"},
        "start": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "default": None,
                  "description": "sim only: where the twin starts, [x, y, yaw_deg] in the room "
                                 "(default the origin facing +x); the robot starts wherever it stands"},
        "safety_notes": {"type": "array", "items": {"type": "string", "minLength": 1}, "default": [],
                         "description": "persistent physical facts the camera cannot see, added to the prompt"},
        "max_decisions": {"type": "integer", "minimum": 1, "default": int(limits.get("max_decisions"))},
        "max_time_s": {"type": "number", "minimum": 1, "default": limits.get("max_time_s")},
        "camera_size": {"type": "string", "pattern": "^[0-9]+x[0-9]+$", "default": None,
                        "description": "the sim head camera, HxW (default: twin_camera_height_px x twin_camera_width_px in "
                                       "configs/limits.json; the robot streams 720x1280)"},
        "notes": {"type": "string", "default": "", "description": "for people; the model never sees it"},
    },
}


def find(name_or_dir: str | Path) -> Path:
    """``tasks/find_the_mug``, a path, or just ``find_the_mug``."""
    p = Path(name_or_dir)
    if (p / "task.json").is_file():
        return p
    if (ROOT / "tasks" / p / "task.json").is_file():
        return ROOT / "tasks" / p
    known = sorted(d.name for d in (ROOT / "tasks").iterdir() if (d / "task.json").is_file())
    raise ValueError(f"no task {str(name_or_dir)!r}; tasks: {', '.join(known)}")


def load_task(directory: Path | str) -> dict:
    """``task.json`` validated and defaulted."""
    d = Path(directory)
    path = d / "task.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"not a task folder (no task.json): {d}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: {e.msg} (line {e.lineno})") from None
    try:
        Draft202012Validator(SCHEMA).validate(data)
    except ValidationError as e:
        where = ".".join(str(p) for p in e.absolute_path) or "task.json"
        message = "needs exactly one of instruction or tools" if e.validator == "oneOf" else e.message
        raise ValueError(f"{path}: {where}: {message}") from None
    task: dict[str, Any] = {k: spec.get("default") for k, spec in SCHEMA["properties"].items()}
    task.update(data)
    if task["objects"] and task["scene"] != "room":
        raise ValueError(f"{path}: objects need \"scene\": \"room\"")
    task["dir"] = d
    task["name"] = d.name
    if task["camera_size"]:
        task["camera_size"] = tuple(int(v) for v in task["camera_size"].split("x"))
    return task


def run_dir(task: dict, env_name: str) -> Path:
    """A fresh directory for a run without the agent's recorder (a tools chain)."""
    from g1.agent.episode import slug
    d = task["dir"] / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{env_name}_{slug(task['tools'])}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_status(run: Path, ok: bool, env_name: str) -> None:
    """A tools run's status.json: its checks are the outcome (the agent's recorder writes its own)."""
    path = run / "status.json"
    if not path.exists():
        path.write_text(json.dumps({"outcome": "success" if ok else "failed", "outcome_source": "checks",
                                    "env": env_name, "ended": datetime.now().isoformat(timespec="seconds")}),
                        encoding="utf-8")


OUTCOMES = ("success", "failed", "unreviewed", "give_up", "interrupted")


def label_run(run: Path, outcome: str) -> Path:
    """The human verdict on a finished run: written into status.json (and episode.json,
    events.jsonl when the agent recorded it) and the directory renamed ``<name>_<outcome>``."""
    task_status = "completed" if outcome == "success" else "failed"
    verdict = {"outcome": outcome, "human_outcome": outcome, "outcome_source": "human", "task_status": task_status}
    for name in ("status.json", "episode.json"):
        path = run / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(verdict)
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    events = run / "events.jsonl"
    if events.exists():
        with events.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at_s": datetime.now().timestamp(), "event": "human_evaluation",
                                "outcome": outcome, "source": "terminal"}) + "\n")
    stem, _, last = run.name.rpartition("_")
    base = stem if last in OUTCOMES else run.name
    dest = run.with_name(f"{base}_{outcome}")
    run.rename(dest)
    return dest


def record_result(task: dict, run: Path, *, ok: Optional[bool] = None, env_name: str = "") -> Optional[dict]:
    """Append what a finished run amounts to, from its status.json (``ok``: a tools run's
    checks, written first if it has none)."""
    status_path = run / "status.json"
    if ok is not None:
        write_status(run, ok, env_name)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    row = {"run": run.name, "outcome": status.get("outcome"), "task_status": status.get("task_status"),
           "outcome_source": status.get("outcome_source"), "decisions": status.get("decisions"),
           "steps": status.get("steps"), "model": status.get("model"), "env": status.get("env"),
           "tokens": (status.get("usage") or {}).get("tokens"), "elapsed_s": status.get("elapsed_s"),
           "ended": status.get("ended")}
    with (task["dir"] / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
