"""``g1 bench SUITE``: a suite of tasks on the twin, each run ``repeat`` times, scored by the
twin (each task's ``success``) and reported in one table.

    suites/<name>/
        suite.json      {"tasks": ["point_at_the_laptop"], "repeat": 10, "notes": "..."}
        runs/<ts>/report.json   one per bench: every run's row, the totals, the code snapshot

Every run is ``g1 run TASK --sim`` (its records in the task's runs/ as usual), headless:
nobody watches ten runs. One process runs them all, so the code, configs/model.json and the
limits are the same for each; the report records the code snapshot they share.

The table keeps the task apart from the execution: success as the twin judged it, a false
"done", decisions and time; then falls, forbidden contacts, unsettled results, joint
violations, and the infrastructure (provider failures, late ticks) that is not the task's.
A run that failed for the infrastructure (the model's provider, the camera, a loaded machine)
is not counted in the task's success rate; the table says how many there were.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

from jsonschema import Draft202012Validator, ValidationError

from g1.core import limits

ROOT = limits.ROOT
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["tasks"],
    "properties": {
        "tasks": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1,
                  "description": "task names (folders under tasks/) or paths"},
        "repeat": {"type": "integer", "minimum": 1, "default": 1, "description": "runs of each task"},
        "notes": {"type": "string", "default": "", "description": "for people"},
    },
}
# end reasons that are the infrastructure's, not the task's (g1/agent/agent.py)
INFRA_END = ("no_frame", "step_timeout", "model_overloaded", "quota", "model_error")


def find(name_or_dir: str | Path) -> Path:
    p = Path(name_or_dir)
    if (p / "suite.json").is_file():
        return p
    if (ROOT / "suites" / p / "suite.json").is_file():
        return ROOT / "suites" / p
    root = ROOT / "suites"
    known = sorted(d.name for d in root.iterdir() if (d / "suite.json").is_file()) if root.is_dir() else []
    raise ValueError(f"no suite {str(name_or_dir)!r}; suites: {', '.join(known) or 'none'}")


def load_suite(directory: Path | str) -> dict:
    d = Path(directory)
    path = d / "suite.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator(SCHEMA).validate(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: {e.msg} (line {e.lineno})") from None
    except ValidationError as e:
        raise ValueError(f"{path}: {'.'.join(str(p) for p in e.absolute_path) or 'suite.json'}: {e.message}") from None
    suite = {k: spec.get("default") for k, spec in SCHEMA["properties"].items()}
    suite.update(data, dir=d, name=d.name)
    return suite


def row(task: dict, folder: Optional[Path], code: int) -> dict:
    """One run as the report counts it, from its summary.json."""
    from g1.agent.task import run_metrics
    if folder is None:
        return {"task": task["name"], "run": None, "outcome": None, "infra": True, "exit": code}
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    m = run_metrics(folder, s)
    health = s.get("health") or {}
    infra = s.get("end_reason") in INFRA_END or m["timing"] == "degraded"
    return {"task": task["name"], "run": folder.name, "outcome": s.get("outcome"),
            "outcome_source": s.get("outcome_source"), "model_outcome": s.get("model_outcome"),
            "end_reason": s.get("end_reason"), "decisions": (s.get("decisions") or {}).get("used"),
            "duration_s": s.get("duration_s"), "fell": health.get("fell"), "violations": health.get("violations"),
            "infra": infra, "exit": code, **m}


def _median(values) -> Optional[float]:
    v = [x for x in values if isinstance(x, (int, float))]
    return round(statistics.median(v), 3) if v else None


def totals(rows: list[dict]) -> dict:
    """Per task: the task's numbers over the runs the infrastructure did not spoil, and the
    execution and infrastructure counts over every run."""
    out = {}
    for name in dict.fromkeys(r["task"] for r in rows):
        rs = [r for r in rows if r["task"] == name]
        clean = [r for r in rs if not r["infra"]]
        scored = [r for r in clean if r.get("twin_passed") is not None]
        out[name] = {
            "runs": len(rs), "clean": len(clean), "scored": len(scored),
            "success": sum(bool(r["twin_passed"]) for r in scored),
            "false_done": sum(bool(r.get("false_done")) for r in scored),
            "decisions_median": _median(r.get("decisions") for r in clean),
            "decisions_max": max((r["decisions"] for r in clean if r.get("decisions") is not None), default=None),
            "duration_median_s": _median(r.get("duration_s") for r in clean),
            "pointing_error_min_median_deg": _median(r.get("pointing_error_min_deg") for r in clean),
            "feet_drift_max_m": max((r["feet_drift_max_m"] for r in clean if r.get("feet_drift_max_m") is not None),
                                    default=None),
            "falls": sum(bool(r.get("fell")) for r in rs),
            "contact_runs": sum(bool(r.get("contacts")) for r in rs),
            "unsettled": sum(r.get("unsettled") or 0 for r in rs),
            "violations": sum(r.get("violations") or 0 for r in rs),
            "failed_model_calls": sum(r.get("failed_calls") or 0 for r in rs),
            "timing_degraded": sum(r.get("timing") == "degraded" for r in rs),
            "infra_failed": len(rs) - len(clean),
        }
    return out


def table(tot: dict) -> str:
    rate = lambda t: f"{t['success']}/{t['scored']}" if t["scored"] else "-"
    show = lambda v: "-" if v is None else f"{v:g}"
    head = (f"{'task':<30} {'success':>8} {'false done':>10} {'decisions':>9} {'time s':>7} {'point err':>9} "
            f"{'feet m':>7} | {'falls':>5} {'contact':>7} {'unsettled':>9} {'viol':>5} | {'model fail':>10} "
            f"{'late':>4} {'infra':>5}")
    lines = [head, "-" * len(head)]
    for name, t in tot.items():
        lines.append(f"{name:<30} {rate(t):>8} {t['false_done']:>10} {show(t['decisions_median']):>9} "
                     f"{show(t['duration_median_s']):>7} {show(t['pointing_error_min_median_deg']):>9} "
                     f"{show(t['feet_drift_max_m']):>7} | {t['falls']:>5} {t['contact_runs']:>7} {t['unsettled']:>9} "
                     f"{t['violations']:>5} | {t['failed_model_calls']:>10} {t['timing_degraded']:>4} "
                     f"{t['infra_failed']:>5}")
    lines.append("success: twin-judged, over the runs the infrastructure did not spoil; decisions, time, point err "
                 "(best per run): medians; feet m: the most the feet moved. Left of the bar the task, then the "
                 "execution, then the infrastructure (failed model calls, late-tick runs, runs it spoiled).")
    return "\n".join(lines)


def run_suite(suite: dict, run=None) -> dict:
    """Every task ``repeat`` times; the report. ``run(task) -> (exit code, folder)`` is
    injectable for tests (default: ``cli.run_once`` on the twin, headless)."""
    from g1.agent.task import code_snapshot, find as find_task, load_task
    from g1.cli import run_once
    run = run or (lambda task: run_once(task, sim=True, mode=None, headless=True))
    tasks = [load_task(find_task(t)) for t in suite["tasks"]]
    unscored = [t["name"] for t in tasks if not t.get("success")]
    if unscored:
        print(f"warning: no success criteria in {', '.join(unscored)}: their runs stay unreviewed")
    snapshot = code_snapshot()
    started = datetime.now()
    rows = []
    for task in tasks:
        for i in range(suite["repeat"]):
            print(f"\n== bench {suite['name']}: {task['name']} {i + 1}/{suite['repeat']} ==")
            code, folder = run(task)
            rows.append(row(task, folder, code))
    report = {"suite": suite["name"], "started": started.isoformat(timespec="seconds"),
              "ended": datetime.now().isoformat(timespec="seconds"), "repeat": suite["repeat"],
              "code": snapshot, "model": _model(), "totals": totals(rows), "runs": rows}
    out = suite["dir"] / "runs" / f"{started:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    from g1.agent.episode import save_json
    save_json(out / "report.json", report)
    report["path"] = out / "report.json"
    return report


def _model() -> dict:
    from g1 import vlm
    return {"provider": vlm.PROVIDER, "model": vlm.DEFAULT_MODEL, "reasoning_effort": vlm.REASONING_EFFORT,
            "max_tokens": vlm.MAX_TOKENS, "temperature": vlm.TEMPERATURE}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="g1 bench", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", help="a folder under suites/ (e.g. stationary) or its path")
    args = p.parse_args(argv)
    try:
        suite = load_suite(find(args.suite))
    except ValueError as e:
        p.error(str(e))
    report = run_suite(suite)
    print("\n" + table(report["totals"]))
    shown = report["path"].relative_to(ROOT) if report["path"].is_relative_to(ROOT) else report["path"]
    print(f"\nreport -> {shown}")
    return 0
