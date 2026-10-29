"""g1 bench: a suite of tasks, repeated, one report; the task's numbers apart from the infrastructure's."""
import json

import pytest

from g1 import bench


def summary(folder, outcome, passed, *, end_reason="done", timing="ok", model_outcome="success", decisions=4):
    folder.mkdir(parents=True)
    s = {"outcome": outcome, "outcome_source": "twin", "model_outcome": model_outcome, "end_reason": end_reason,
         "decisions": {"used": decisions}, "duration_s": 30.0, "false_done": model_outcome == "success" and not passed,
         "usage": {"calls": decisions, "failed_calls": 1},
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
