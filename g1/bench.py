"""``g1 bench SUITE``: a suite of tasks on the twin, each run ``repeat`` times, scored by the
twin (each task's ``success``) and reported in one table.

    suites/<name>/
        suite.json      {"tasks": ["point_at_the_laptop"], "repeat": 10, "notes": "..."}
        runs/<ts>/report.json   one per bench: every run's row, the totals, the code snapshot
        runs/<ts>/run_NN_<task>.log   each run's own output

Every run is ``g1 run TASK --sim`` (its records in the task's runs/ as usual), headless:
nobody watches ten runs. Several run at once, each in its own process with its own twin and
DDS domain: as many as free memory (bench_run_mb each) and the performance cores allow, at most
bench_parallel_max (``workers``). They start from one checkout, so the code, configs/model.json
and the limits are the same for each; the report records the code snapshot they share and the
input and output tokens (and the cost, where the provider reports it) for estimating the cost.

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


def think_stats(seconds: list[float], reasoning: list[int]) -> dict:
    """Model time per call: count, median, 90th percentile and longest, and the most reasoning tokens."""
    if not seconds:
        return {"calls": 0, "think_median_s": None, "think_p90_s": None, "think_max_s": None,
                "reasoning_tokens_max_call": None}
    xs = sorted(seconds)
    p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
    return {"calls": len(xs), "think_median_s": round(statistics.median(xs), 2), "think_p90_s": round(p90, 2),
            "think_max_s": round(xs[-1], 2), "reasoning_tokens_max_call": max(reasoning) if reasoning else None}


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
            # what the model cost, over every run (a spoiled run is paid for too)
            "input_tokens": sum(r.get("input_tokens") or 0 for r in rs),
            "cached_input_tokens": sum(r.get("cached_input_tokens") or 0 for r in rs),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in rs),
            "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in rs),
            "cost_usd": (round(sum(r.get("cost_usd") or 0.0 for r in rs), 6)
                         if any(r.get("cost_usd") is not None for r in rs) else None),
            # how long the model takes per call (request to tool call), over every call of every run
            **think_stats([t for r in rs for t in r.get("think_s") or []],
                          [k for r in rs for k in r.get("reasoning_tokens_per_call") or []]),
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
    lines.append("")
    head = f"{'tokens (all runs)':<30} {'input':>12} {'cached':>12} {'output':>12} {'reasoning':>12} {'cost USD':>10}"
    lines += [head, "-" * len(head)]
    for name, t in tot.items():
        cost = "-" if t.get("cost_usd") is None else f"{t['cost_usd']:.4f}"
        lines.append(f"{name:<30} {t.get('input_tokens', 0):>12,} {t.get('cached_input_tokens', 0):>12,} "
                     f"{t.get('output_tokens', 0):>12,} {t.get('reasoning_tokens', 0):>12,} {cost:>10}")
    lines.append("")
    head = f"{'model time per call':<30} {'calls':>8} {'median s':>9} {'p90 s':>8} {'max s':>8} {'reasoning tok max':>18}"
    lines += [head, "-" * len(head)]
    for name, t in tot.items():
        lines.append(f"{name:<30} {t.get('calls', 0):>8} {show(t.get('think_median_s')):>9} {show(t.get('think_p90_s')):>8} "
                     f"{show(t.get('think_max_s')):>8} {show(t.get('reasoning_tokens_max_call')):>18}")
    lines.append("input includes images and the cached part; output includes reasoning. Cost as the provider "
                 "reported it (OpenRouter does; - where it did not): multiply the tokens by the model's prices otherwise.")
    return "\n".join(lines)


def available_mb() -> Optional[float]:
    """Memory the system could give a new process now (free, inactive and purgeable pages), in MB;
    None where it cannot be read."""
    import re
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        try:                                        # Linux
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
        except OSError:
            return None
        return None
    page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size of" in out else 4096
    pages = sum(int(m.group(2)) for m in re.finditer(r"Pages (free|inactive|speculative|purgeable):\s+(\d+)", out))
    return pages * page / 2**20


def performance_cores() -> int:
    """The cores a real-time run should count on: the performance cores on Apple silicon, else half."""
    import os
    import subprocess
    try:
        n = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True,
                               timeout=5).stdout.strip())
        if n > 0:
            return n
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def workers(runs: int) -> tuple[int, str]:
    """How many runs at once: what memory allows (bench_run_mb each), what the performance cores
    allow (one kept for the system), at most bench_parallel_max; and why."""
    cap = int(limits.get("bench_parallel_max"))
    per = limits.get("bench_run_mb")
    mem = available_mb()
    by_mem = cap if mem is None else max(1, int(mem // per))
    by_cpu = max(1, performance_cores() - 1)
    n = max(1, min(runs, cap, by_mem, by_cpu))
    why = (f"{n} at once: {'?' if mem is None else f'{mem:.0f}'} MB available / {per:g} MB a run -> {by_mem}, "
           f"{performance_cores()} performance cores -> {by_cpu}, bench_parallel_max {cap}")
    return n, why


def _run_child(task_dir: str, log: str, results) -> None:
    """One run in a fresh process of its own (no DDS or MuJoCo state left from another run), its
    output in ``log`` (parallel runs would interleave); (log, exit code, folder) onto ``results``."""
    import os
    import sys
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = sys.stderr = open(1, "w", buffering=1, closefd=False)
    code, folder = 1, None
    try:
        from g1.agent.task import load_task
        from g1.cli import run_once
        code, folder = run_once(load_task(task_dir), sim=True, mode=None, headless=True)
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        results.put((log, code, None if folder is None else str(folder)))


def run_suite(suite: dict, run=None, parallel: Optional[int] = None) -> dict:
    """Every task ``repeat`` times, ``parallel`` at once (default: ``workers``), each in its own
    process with its own twin; the report. ``run(task) -> (exit code, folder)`` is injectable for
    tests and runs one at a time in this process."""
    from g1.agent.task import code_snapshot, find as find_task, load_task
    tasks = [load_task(find_task(t)) for t in suite["tasks"]]
    unscored = [t["name"] for t in tasks if not t.get("success")]
    if unscored:
        print(f"warning: no success criteria in {', '.join(unscored)}: their runs stay unreviewed")
    snapshot = code_snapshot()
    started = datetime.now()
    out = suite["dir"] / "runs" / f"{started:%Y%m%d-%H%M%S}"
    k = 1
    while out.exists():                        # two benches in the same second
        k += 1
        out = suite["dir"] / "runs" / f"{started:%Y%m%d-%H%M%S}-{k}"
    out.mkdir(parents=True)
    jobs = [(task, i) for task in tasks for i in range(suite["repeat"])]
    rows: list[Optional[dict]] = [None] * len(jobs)
    if run is not None:
        n, why = 1, "1 at once (an injected run)"
    elif parallel is not None:
        n, why = max(1, parallel), f"{max(1, parallel)} at once (asked)"
    else:
        n, why = workers(len(jobs))
    print(f"bench {suite['name']}: {len(jobs)} runs, {why}")
    if run is not None:
        for k, (task, i) in enumerate(jobs):
            print(f"\n== bench {suite['name']}: {task['name']} {i + 1}/{suite['repeat']} ==")
            code, folder = run(task)
            rows[k] = row(task, folder, code)
    else:
        import multiprocessing
        import queue
        ctx = multiprocessing.get_context("spawn")
        results = ctx.Queue()
        pending = list(enumerate(jobs))
        running: dict[str, tuple] = {}                # log -> (process, k, task, i)
        while pending or running:
            while pending and len(running) < n:      # a fresh process per run, at most n at once
                k, (task, i) = pending.pop(0)
                log = str(out / f"run_{k + 1:02d}_{task['name']}.log")
                proc = ctx.Process(target=_run_child, args=(str(task["dir"]), log, results))
                proc.start()
                running[log] = (proc, k, task, i)
            try:
                log, code, folder = results.get(timeout=5.0)
            except queue.Empty:                       # look for a run that died without a word
                for log, (proc, k, task, i) in list(running.items()):
                    if not proc.is_alive():
                        running.pop(log)
                        print(f"{task['name']} {i + 1}/{suite['repeat']}: died (exit {proc.exitcode}); see {log}")
                        rows[k] = row(task, None, 1)
                continue
            proc, k, task, i = running.pop(log)
            proc.join()
            rows[k] = row(task, None if folder is None else Path(folder), code)
            r = rows[k]
            print(f"{task['name']} {i + 1}/{suite['repeat']}: {r['outcome'] or 'no record (see the log)'}"
                  + (f" ({r.get('end_reason')}, {r.get('decisions')} decisions)" if r["run"] else "")
                  + f"  log {Path(log).name}")
    report = {"suite": suite["name"], "started": started.isoformat(timespec="seconds"),
              "ended": datetime.now().isoformat(timespec="seconds"), "repeat": suite["repeat"], "parallel": n,
              "code": snapshot, "model": _model(), "totals": totals(rows), "runs": rows}
    from g1.agent.episode import save_json
    save_json(out / "report.json", report)
    report["path"] = out / "report.json"
    return report


def _model() -> dict:
    from g1 import vlm
    return {"provider": vlm.PROVIDER, "model": vlm.DEFAULT_MODEL, "reasoning_effort": vlm.REASONING_EFFORT,
            "max_tokens": vlm.MAX_TOKENS, "temperature": vlm.TEMPERATURE, "agent": dict(vlm.AGENT_SETTINGS)}


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
