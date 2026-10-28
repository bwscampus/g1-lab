import json

import numpy as np
import pytest

from g1.agent.episode import EpisodeWriter, load_episode


def record(step, name, args, status="completed"):
    return {"step": step, "decision": step - 1, "attempt": 0, "t_start": 0.0, "t_end": 1.0,
            "frame": {"image": f"step_{step:04d}.png", "seq": step, "stamp": 0.0, "age_s": 0.02, "shape": [8, 8, 3]},
            "query": [{"role": "user", "content": "{}"}],
            "response": {"status": "completed", "decision": {"name": name, "arguments": {**args, "note": "n"}}},
            "action": {"tool": name, "args": {**args, "note": "n"}, "t_start": 0.0, "t_end": 1.0},
            "result": {"status": status}, "sdk_calls": []}


def writer(tmp_path, instruction="g", **kw):
    return EpisodeWriter(tmp_path, env="sim", instruction=instruction, model="m", task="t", mode="gantry",
                         config={"task": {"name": "t"}}, max_decisions=5, **kw)


def test_round_trip_is_lossless(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    w = writer(tmp_path, "Find the Mug!", threaded=False)
    assert w.dir.name.endswith("_sim_find-the-mug")
    rec = record(1, "turn", {"angle_deg": 45.0})
    w.step(rec, img)
    summary = json.loads((w.dir / "summary.json").read_text())
    assert summary["steps"] == 1 and summary["outcome"] is None          # rewritten per step
    rec["result"] = {"status": "completed", "duration_s": 2.5}
    w.step(rec)                                                          # rewritten in place, the PNG kept
    w.close("completed", end_reason="done")
    assert w.dir.name.endswith("_sim_find-the-mug_unreviewed")          # a model conclusion, no human verdict
    summary, steps = load_episode(w.dir)
    assert summary["run"] == w.dir.name and summary["ended"] and summary["end_reason"] == "done" and len(steps) == 1
    assert summary["outcome"] == "unreviewed" and summary["outcome_source"] == "unreviewed"
    assert summary["model_outcome"] == "success" and summary["mode"] == "gantry"
    assert np.array_equal(steps[0]["image"], img)                       # exact RGB matrix back
    assert steps[0]["image"][0, 0].tolist() == img[0, 0].tolist()       # channel order preserved
    assert steps[0]["frame"]["image"] == "step_0001.png" and steps[0]["result"]["duration_s"] == 2.5
    assert json.loads((w.dir / "config.json").read_text()) == {"task": {"name": "t"}}


def test_threaded_writer_flushes_on_close(tmp_path):
    img = np.zeros((8, 8, 3), np.uint8)
    w = writer(tmp_path)
    for i in range(1, 6):
        w.step(record(i, "walk_forward", {"distance_m": 0.5}), img)
    assert w.close("budget_exhausted", human="failed").name.endswith("_failed")   # the human's label wins
    summary, steps = load_episode(w.dir, images=False)
    assert len(steps) == 5 and summary["steps"] == 5 and summary["outcome_source"] == "human"
    assert all(s["image"] is None for s in steps)


def test_usage_totals_and_the_file_set(tmp_path):
    w = writer(tmp_path, threaded=False)
    w.model_call({"status": "completed", "elapsed_s": 1.5, "usage": {"prompt_tokens": 4, "completion_tokens": 1}})
    w.model_call({"status": "failed", "elapsed_s": 0.5, "usage": None})
    w.update(tools_used=["turn"], closing={"summary": "s"})
    d = w.close("failed", error="boom", end_reason="model_error")
    assert d.name.endswith("_failed")
    s = json.loads((d / "summary.json").read_text())
    assert s["usage"] == {"calls": 2, "failed_calls": 1, "tokens": {"prompt_tokens": 4, "completion_tokens": 1},
                          "model_s": 2.0}
    assert s["error"] == "boom" and s["end_reason"] == "model_error" and s["outcome_source"] == "runtime"
    assert s["tools_used"] == ["turn"] and s["closing"] == {"summary": "s"} and s["duration_s"] >= 0
    assert {p.name for p in d.iterdir()} == {"summary.json", "config.json"}
    # a finished run is never overwritten by the rename
    w2 = writer(tmp_path / "other", threaded=False)
    w2.dir.with_name(w2.dir.name + "_failed").mkdir()
    with pytest.raises(FileExistsError):
        w2.close("failed")
