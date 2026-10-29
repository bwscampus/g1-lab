"""g1 bench: a suite of tasks, repeated, one report; the task's numbers apart from the infrastructure's."""
import json

import pytest

from g1 import bench


def summary(folder, outcome, passed, *, end_reason="done", timing="ok", model_outcome="success", decisions=4):
    folder.mkdir(parents=True)
    s = {"outcome": outcome, "outcome_source": "twin", "model_outcome": model_outcome, "end_reason": end_reason,
         "decisions": {"used": decisions}, "duration_s": 30.0, "false_done": model_outcome == "success" and not passed,
         "usage": {"calls": decisions, "failed_calls": 1, "cost_usd": 0.01,
                   "tokens": {"prompt_tokens": 1000, "completion_tokens": 100, "reasoning_tokens": 40,
                              "cached_tokens": 200}},
         "health": {"fell": False, "violations": 0, "timing": timing, "overrun_fraction": 0.0,
                    "twin": {"evaluation": {"passed": passed, "failed": [] if passed else ["point_at"],
                                            "measured": {"pointing_error_min_deg": 5.0 if passed else 25.0,
                                                         "feet_drift_max_m": 0.01, "contacts": {}}}}}}
    (folder / "summary.json").write_text(json.dumps(s))
    return folder


def test_a_suite_runs_each_task_repeat_times_and_reports(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks" / "p"
    tasks.mkdir(parents=True)
    (tasks / "task.json").write_text(json.dumps({"instruction": "point at the laptop", "scene": "room",
                                                 "success": {"feet_drift_max_m": 0.05}}))
    d = tmp_path / "suites" / "s"
    d.mkdir(parents=True)
    (d / "suite.json").write_text(json.dumps({"tasks": [str(tasks)], "repeat": 4}))
    runs = iter([summary(tmp_path / "r1", "success", True),
                 summary(tmp_path / "r2", "failed", False),                       # a false done
                 summary(tmp_path / "r3", "failed", False, end_reason="model_overloaded", model_outcome=None),
                 summary(tmp_path / "r4", "success", True, timing="degraded")])
    report = bench.run_suite(bench.load_suite(d), run=lambda task: (0, next(runs)))
    t = report["totals"]["p"]
    assert (t["runs"], t["clean"], t["success"], t["false_done"], t["infra_failed"]) == (4, 2, 1, 1, 2)
    assert t["timing_degraded"] == 1 and t["failed_model_calls"] == 4 and t["pointing_error_min_median_deg"] == 15.0
    # tokens and cost over every run, spoiled ones too: they were paid for
    assert (t["input_tokens"], t["cached_input_tokens"], t["output_tokens"], t["reasoning_tokens"]) == (4000, 800, 400, 160)
    assert t["cost_usd"] == pytest.approx(0.04) and report["runs"][0]["input_tokens"] == 1000
    assert "4,000" in bench.table(report["totals"]) and "0.0400" in bench.table(report["totals"])
    saved = json.loads(report["path"].read_text())
    assert saved["code"]["snapshot"] and saved["model"]["provider"] and len(saved["runs"]) == 4
    assert "1/2" in bench.table(report["totals"])


def test_a_bad_suite_is_refused(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "suite.json").write_text(json.dumps({"tasks": [], "repeat": 0}))
    with pytest.raises(ValueError, match="suite.json"):
        bench.load_suite(d)
    with pytest.raises(ValueError, match="no suite"):
        bench.find("no_such_suite")


def test_parallel_runs_fit_memory_cores_and_the_cap(monkeypatch):
    from g1.core import limits
    per, cap = limits.get("bench_run_mb"), limits.get("bench_parallel_max")
    monkeypatch.setattr(bench, "performance_cores", lambda: 4)
    monkeypatch.setattr(bench, "available_mb", lambda: per * 10)
    assert bench.workers(5)[0] == min(cap, 3)                        # 4 performance cores, one kept free
    assert bench.workers(1)[0] == 1                                  # never more than the runs
    monkeypatch.setattr(bench, "available_mb", lambda: per * 1.5)
    n, why = bench.workers(5)
    assert n == 1 and "MB available" in why                          # only one fits
    monkeypatch.setattr(bench, "available_mb", lambda: None)
    assert bench.workers(5)[0] == min(cap, 3)


def test_model_time_per_call_is_summarised():
    t = bench.think_stats([1.0, 2.0, 3.0, 4.0, 50.0], [10, 20, 30, 40, 9000])
    assert (t["calls"], t["think_median_s"], t["think_max_s"], t["reasoning_tokens_max_call"]) == (5, 3.0, 50.0, 9000)
    assert t["think_p90_s"] == 50.0 and bench.think_stats([], [])["think_median_s"] is None
