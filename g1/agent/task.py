"""A task: a folder with what to do and its context, no code. It is what ``g1 run`` runs.

    tasks/<name>/
        task.json       an instruction (the model decides) or a tools chain (no model),
                        the scene, safety notes, the budget
        runs/           every run of this task (git-ignored): summary.json, config.json,
                        step files, trace.npz (g1/agent/episode.py)
        results.jsonl   one line per run: outcome, end reason, decisions, tokens, health

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
        "success": {"type": ["object", "null"], "default": None, "additionalProperties": False,
                    "description": "sim only: what counts as done, scored by the twin from its truth (g1/twin/core.py "
                                   "Evaluator); a run of the task with --sim is labelled from it. The model never sees it",
                    "properties": {"point_at": {"type": "string", "minLength": 1},
                                   "within_deg": {"type": "number", "exclusiveMinimum": 0},
                                   "for_s": {"type": "number", "minimum": 0},
                                   "feet_drift_max_m": {"type": "number", "minimum": 0},
                                   "feet_yaw_drift_max_deg": {"type": "number", "minimum": 0},
                                   "no_contact": {"type": "array", "items": {"type": "string", "minLength": 1}}},
                    "dependentRequired": {"point_at": ["within_deg", "for_s"]}},
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


def code_snapshot(root: Path = ROOT) -> dict:
    """The code this run is made of, restorable: ``git checkout <snapshot>``. A clean tree is
    HEAD; otherwise the working tree (tracked edits and new files, never git-ignored ones:
    runs, assets, .env) becomes a commit on HEAD kept by ``refs/g1-runs/<sha>``, built on a
    temporary index so the branch, the real index and the files are untouched. The refs stay
    local unless pushed (``git push origin 'refs/g1-runs/*'``)."""
    import os
    import subprocess
    import tempfile

    def git(*args: str, env: Optional[dict] = None) -> str:
        return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
    try:
        head = git("rev-parse", "HEAD")
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
            git("read-tree", "HEAD", env=env)
            git("add", "-A", env=env)
            tree = git("write-tree", env=env)
        if tree == git("rev-parse", "HEAD^{tree}"):
            return {"head": head, "snapshot": head, "ref": None}
        sha = git("commit-tree", tree, "-p", head, "-m", "g1 run: the working tree this run used")
        ref = f"refs/g1-runs/{sha}"
        git("update-ref", ref, sha)
        return {"head": head, "snapshot": sha, "ref": ref}
    except (OSError, subprocess.SubprocessError) as e:
        detail = getattr(e, "stderr", "") or str(e)
        return {"head": None, "snapshot": None, "ref": None, "error": detail.strip()[:300]}


def versions() -> dict:
    import platform
    from importlib import metadata
    out = {"python": platform.python_version()}
    for name in ("numpy", "mujoco", "onnxruntime", "unitree_sdk2py", "jsonschema", "av"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_config(task: dict, *, env_name: str, mode: str, settings: Any, code: dict) -> dict:
    """config.json: how this run was set up. The task as loaded, the env and mode, the camera
    settings, every limit, the code snapshot, the versions. Never a .env value (secrets, the
    robot's address, the interface). The agent adds the model, the tools and the prompt."""
    t = {k: (list(v) if isinstance(v, tuple) else v) for k, v in task.items() if k != "dir"}
    t["dir"] = str(task["dir"].relative_to(ROOT)) if task["dir"].is_relative_to(ROOT) else str(task["dir"])
    return {"task": t, "env": env_name, "mode": mode,
            "camera": {"use": getattr(settings, "camera", None), "source": getattr(settings, "camera_source", None),
                       "view": getattr(settings, "view", False)},
            "limits": {name: spec["value"] for name, spec in limits.LIMITS.items()},
            "code": code, "versions": versions()}


def write_config(run: Path, config: dict) -> None:
    from g1.agent.episode import save_json
    save_json(run / "config.json", config)


def write_status(run: Path, ok: bool, *, task: dict, env_name: str, mode: Optional[str], started: str,
                 end_reason: str) -> None:
    """A tools run's summary.json: its checks are the outcome (the agent's recorder writes its own)."""
    from g1.agent.episode import new_summary, save_json
    path = run / "summary.json"
    if path.exists():
        return
    s = new_summary(run.name, task=task["name"], instruction=task["tools"], env=env_name, mode=mode, model=None)
    ended = datetime.now()
    s.update(started=started, ended=ended.isoformat(timespec="seconds"),
             duration_s=round((ended - datetime.fromisoformat(started)).total_seconds(), 3),
             outcome="success" if ok else "failed", outcome_source="checks", end_reason=end_reason,
             decisions=None, usage=None)
    save_json(path, s)


def update_summary(folder: Path, **values) -> None:
    from g1.agent.episode import save_json
    path = folder / "summary.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(values)
        save_json(path, data)


OUTCOMES = ("success", "failed", "unreviewed", "give_up", "interrupted")


def label_run(run: Path, outcome: str, source: str = "human") -> Path:
    """The verdict on a finished run (a human's, or the twin's from the task's ``success``):
    written into summary.json and the directory renamed ``<name>_<outcome>``."""
    stem, _, last = run.name.rpartition("_")
    base = stem if last in OUTCOMES else run.name
    dest = run.with_name(f"{base}_{outcome}")
    run.rename(dest)
    update_summary(dest, run=dest.name, outcome=outcome, outcome_source=source)
    return dest


def run_metrics(run: Path, s: dict) -> dict:
    """What a run amounts to, task and execution apart: the twin's verdict and measurements,
    a false "done", and the infrastructure (provider failures, late ticks) that is not the task's."""
    health = s.get("health") or {}
    evaluation = (health.get("twin") or {}).get("evaluation") or {}
    measured = evaluation.get("measured") or {}
    unsettled = 0
    for step in sorted(run.glob("step_*.json")):
        try:
            result = json.loads(step.read_text(encoding="utf-8")).get("result") or {}
        except (OSError, json.JSONDecodeError):
            continue
        settle = (result.get("execution_feedback") or {}).get("settle") or {}
        unsettled += settle.get("settled") is False
    return {"twin_passed": evaluation.get("passed"), "twin_failed": evaluation.get("failed"),
            "false_done": s.get("false_done"),
            "pointing_within_s": measured.get("pointing_within_s"),
            "pointing_error_min_deg": measured.get("pointing_error_min_deg"),
            "feet_drift_max_m": measured.get("feet_drift_max_m"),
            "feet_yaw_drift_max_deg": measured.get("feet_yaw_drift_max_deg"),
            "contacts": sorted(measured.get("contacts") or {}), "unsettled": unsettled,
            "model_calls": (s.get("usage") or {}).get("calls"),
            "failed_calls": (s.get("usage") or {}).get("failed_calls"),
            "timing": health.get("timing"), "overrun_fraction": health.get("overrun_fraction"),
            "max_tick_gap_s": health.get("max_tick_gap_s"), "load_avg": health.get("load_avg")}


def record_result(task: dict, run: Path) -> Optional[dict]:
    """Append what a finished run amounts to, from its summary.json: one row per run, the
    fields to compare runs on."""
    try:
        s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    health = s.get("health") or {}
    row = {"run": run.name, "env": s.get("env"), "mode": s.get("mode"), "model": s.get("model"),
           "outcome": s.get("outcome"), "outcome_source": s.get("outcome_source"),
           "model_outcome": s.get("model_outcome"), "end_reason": s.get("end_reason"),
           "decisions": (s.get("decisions") or {}).get("used"), "tokens": (s.get("usage") or {}).get("tokens"),
           "duration_s": s.get("duration_s"), "ended": s.get("ended"),
           "joint_check": health.get("joint_check"), "violations": health.get("violations"),
           "overruns": health.get("overruns"), "fell": health.get("fell"), **run_metrics(run, s)}
    with (task["dir"] / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
