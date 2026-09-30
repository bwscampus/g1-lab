"""A task: a folder with an instruction and its context, no code.

    tasks/<name>/
        task.json       the instruction, the scene, safety notes, a demonstration, the budget
        demo/           optional: a demo.json bundle or images the task.json refers to
        runs/           every run of this task (git-ignored)
        results.jsonl   one line per run: directory, outcome, decisions, tokens, model

``g1 task run DIR`` turns the folder into the same ``g1 run --tools search``
invocation you could type by hand (flags after DIR override the file), records
under the task's ``runs/`` and appends to ``results.jsonl``. ``g1 task eval DIR
-n N`` runs it N times and prints GPT-Policy's two numbers: success rate and
decisions to success. ``g1 task show DIR`` prints what the model will be given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from jsonschema import Draft202012Validator, ValidationError

from g1.core import limits

ROOT = limits.ROOT
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["instruction"],
    "properties": {
        "instruction": {"type": "string", "minLength": 1, "description": "what the model is asked to do"},
        "scene": {"type": "string", "enum": ["none", "room"], "default": "none", "description": "the sim scene"},
        "objects": {"type": "array", "items": {"type": "string"}, "default": [],
                    "description": "sim objects to place, name@x,y[,z] (needs scene room)"},
        "safety_notes": {"type": "array", "items": {"type": "string", "minLength": 1}, "default": [],
                         "description": "persistent physical facts the camera cannot see, added to the prompt"},
        "demo": {"type": ["string", "null"], "default": None,
                 "description": "a demonstration for turn 0: a video, a runs/<dir>, or a demo.json, relative to the task"},
        "demo_mode": {"type": ["string", "null"], "enum": ["video", "video+action", None], "default": None},
        "refs": {"type": "array", "items": {"type": "string"}, "default": [],
                 "description": "reference images (a photo of the goal), relative to the task"},
        "success": {"type": "string", "enum": ["human"], "default": "human",
                    "description": "who decides: the human verdict after the run"},
        "max_decisions": {"type": "integer", "minimum": 1, "default": int(limits.get("max_decisions"))},
        "max_time_s": {"type": "number", "minimum": 1, "default": 600.0},
        "notes": {"type": "string", "default": "", "description": "for people; the model never sees it"},
    },
}


def load_task(directory: Path | str) -> dict:
    """``task.json`` validated and defaulted; paths resolved against the folder."""
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
        raise ValueError(f"{path}: {where}: {e.message}") from None
    task: dict[str, Any] = {k: spec.get("default") for k, spec in SCHEMA["properties"].items()}
    task.update(data)
    task["dir"] = d
    task["name"] = d.name
    for key in ("demo",):
        if task[key]:
            task[key] = str((d / task[key]).resolve())
    task["refs"] = [str((d / r).resolve()) for r in task["refs"]]
    return task


def run_flags(task: dict) -> list[str]:
    """The ``g1 run`` flags this task stands for."""
    flags = ["--tools", "search", "--instruction", task["instruction"], "--log", str(task["dir"] / "runs"),
             "--max-decisions", str(task["max_decisions"]), "--max-time", str(task["max_time_s"])]
    if task["scene"] != "none":
        flags += ["--scene", task["scene"]]
    if task["objects"]:
        flags += ["--sim-objects", *task["objects"]]
    for note in task["safety_notes"]:
        flags += ["--safety-note", note]
    if task["demo"]:
        flags += ["--demo", task["demo"]]
        if task["demo_mode"]:
            flags += ["--demo-mode", task["demo_mode"]]
    for ref in task["refs"]:
        flags += ["--ref", ref]
    return flags


def results(task: dict) -> list[dict]:
    path = task["dir"] / "results.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_result(task: dict, run_dir: Path) -> Optional[dict]:
    """Append what a finished run under ``runs/`` amounts to."""
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    row = {"run": run_dir.name, "outcome": status.get("outcome"), "task_status": status.get("task_status"),
           "outcome_source": status.get("outcome_source"), "decisions": status.get("decisions"),
           "steps": status.get("steps"), "model": status.get("model"), "env": status.get("env"),
           "tokens": (status.get("usage") or {}).get("tokens"), "elapsed_s": status.get("elapsed_s"),
           "ended": status.get("ended")}
    with (task["dir"] / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def summarize(rows: list[dict]) -> str:
    """GPT-Policy's two metrics: success rate, and decisions per successful run."""
    if not rows:
        return "no runs yet"
    successes = [r for r in rows if r.get("outcome") == "success"]
    reviewed = [r for r in rows if r.get("outcome_source") == "human"]
    lines = [f"{len(rows)} run(s), {len(successes)} success ({len(successes) / len(rows):.0%})"
             + (f"; {len(rows) - len(reviewed)} without a human verdict" if len(reviewed) < len(rows) else "")]
    decisions = [r["decisions"] for r in successes if isinstance(r.get("decisions"), int)]
    if decisions:
        lines.append(f"decisions to success: mean {sum(decisions) / len(decisions):.1f}, min {min(decisions)}, max {max(decisions)}")
    tokens = [sum(v for v in (r.get("tokens") or {}).values() if isinstance(v, int)) for r in rows]
    if any(tokens):
        lines.append(f"tokens per run: mean {sum(tokens) / len(tokens):.0f}")
    return "\n".join(lines)


def _run_once(task: dict, extra: list[str]) -> tuple[int, Optional[dict]]:
    from g1.cli import run_main, run_parser
    runs = task["dir"] / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in runs.iterdir()}
    code = run_main(run_flags(task) + extra, run_parser("g1 task run"))
    new = [p for p in runs.iterdir() if p.name not in before and p.is_dir()]
    row = record_result(task, max(new, key=lambda p: p.stat().st_mtime)) if new else None
    return code, row


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="g1 task", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "show", "eval"))
    p.add_argument("dir", help="the task folder, e.g. tasks/find_the_mug")
    p.add_argument("-n", type=int, default=3, help="eval: how many runs (default 3)")
    p.add_argument("--no-demo", action="store_true", help="eval: run without the task's demonstration")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="g1 run flags that override the task, e.g. --env sim --headless")
    args = p.parse_args(argv)
    try:
        task = load_task(args.dir)
    except ValueError as e:
        p.error(str(e))
    if args.no_demo:
        task["demo"] = None
    if args.command == "show":
        print(f"task {task['name']}: {task['instruction']!r}")
        print("g1 run " + " ".join(f'"{f}"' if " " in f else f for f in run_flags(task)))
        if task["notes"]:
            print(f"notes: {task['notes']}")
        print(summarize(results(task)))
        return 0
    if args.command == "run":
        code, row = _run_once(task, args.rest)
        if row is not None:
            print(f"result: {row['outcome']} ({row['outcome_source']}), {row['decisions']} decision(s) -> results.jsonl")
        return code
    rows = []
    for i in range(args.n):
        print(f"\n== {task['name']} run {i + 1}/{args.n} ==")
        code, row = _run_once(task, args.rest)
        if row is not None:
            rows.append(row)
    print(f"\n{task['name']}: this evaluation\n{summarize(rows)}\n\nall runs\n{summarize(results(task))}")
    return 0
