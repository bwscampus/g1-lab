import json

import numpy as np
import pytest

from episode import EpisodeWriter, StepRecord, chain_of, load_episode, _main


def record(step, name, args, status="completed"):
    return StepRecord(step, {"policy_start": 0.0, "policy_end": 1.0, "wall_start": 0.0, "wall_end": 1.0,
                             "clock_start": 0.0, "clock_end": 1.0, "think_wall": 0.5},
                      [0.0] * 29, [0.1] * 29, [0.0] * 29, [0.1] * 29,
                      {"seq": step, "stamp": 0.0, "age": 0.02, "image": None, "shape": [8, 8, 3]},
                      {"cmd_start": [0, 0, 0], "cmd_end": [0.5, 0, 0], "env_start": None, "env_end": None},
                      {"name": name, "arguments": {**args, "note": "n"}, "raw": "{}", "latency": 0.5},
                      {"name": name, "args": {**args, "note": "n"}, "duration": 2.5, "chunk": 1, "chunks": 1,
                       "needs_base": name != "hold"},
                      {"status": status, "duration": 1.0})


def test_round_trip_is_lossless(tmp_path):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    w = EpisodeWriter(tmp_path, env="sim", goal="Find the Mug!", model="m", skills=["turn"], threaded=False)
    assert w.dir.name.endswith("_sim_find-the-mug")
    w.write_step(record(1, "turn", {"angle_deg": 45.0}), img)
    meta = json.loads((w.dir / "episode.json").read_text())
    assert meta["steps"] == 1 and meta["result"] is None            # rewritten per step
    w.finish("completed")
    w.close("completed")
    assert w.dir.name.endswith("_sim_find-the-mug_unreviewed")      # a model conclusion, no human verdict
    meta, steps = load_episode(w.dir)
    assert meta["result"] == "completed" and meta["ended"] and len(steps) == 1
    assert meta["outcome"] == "unreviewed" and meta["outcome_source"] == "unreviewed"
    assert np.array_equal(steps[0].image, img)                       # exact RGB matrix back
    assert steps[0].image[0, 0].tolist() == img[0, 0].tolist()       # channel order preserved
    assert steps[0].frame["image"] == "step_0001.png" and steps[0].skill["args"] == {"angle_deg": 45.0, "note": "n"}


def test_threaded_writer_flushes_on_close(tmp_path):
    pytest.importorskip("cv2")
    img = np.zeros((8, 8, 3), np.uint8)
    w = EpisodeWriter(tmp_path, env="sim", goal="g", model="m", skills=[])
    for i in range(1, 6):
        w.write_step(record(i, "walk_forward", {"distance_m": 0.5}), img)
    for t in np.arange(0, 1.0, 0.02):
        w.state(t, np.zeros(29), np.zeros(29))
    w.finish("budget_exhausted")
    assert w.close("budget_exhausted", human="failed").name.endswith("_failed")   # the human's label wins
    meta, steps = load_episode(w.dir, images=False)
    assert len(steps) == 5 and meta["steps"] == 5 and meta["result"] == "budget_exhausted"
    assert all(s.image is None for s in steps)
    assert len((w.dir / "states.jsonl").read_text().splitlines()) == 20      # 20 Hz from a 50 Hz tick


def test_chain_of_and_cli(tmp_path, capsys):
    pytest.importorskip("cv2")
    w = EpisodeWriter(tmp_path, env="sim", goal="g", model="m", skills=[], threaded=False)
    w.write_step(record(1, "turn", {"angle_deg": 45.0}), None)
    w.write_step(record(2, "walk_forward", {"distance_m": 0.5}), None)
    w.write_step(record(3, "hold", {"seconds": 1.0}, status="rejected"), None)
    w.write_step(record(4, "done", {}, status="done"), None)
    w.finish("completed"); w.close("completed", human="failed")
    assert w.dir.name.endswith("_failed")
    _, steps = load_episode(w.dir, images=False)
    assert chain_of(steps) == "turn:angle_deg=45.0,walk_forward:distance_m=0.5"    # notes left out
    assert _main([str(w.dir)]) == 0
    out = capsys.readouterr().out
    assert "replay:  --policy turn:angle_deg=45.0,walk_forward:distance_m=0.5" in out and "4 step(s)" in out
    assert "outcome failed (human)" in out


def test_events_transcript_and_status(tmp_path):
    w = EpisodeWriter(tmp_path, env="sim", goal="g", model="m", skills=[], threaded=False)
    w.event("protocol", {"base_instructions": "SYS", "tools": [], "output_schema": {}})
    w.event("observation", {"step": 0, "input_json": "{}", "images": [{"name": "head", "path": "step_0001.png"}]})
    w.event("model_decision", {"step": 0, "decision": {"name": "turn", "arguments": {}}})
    w.event("tool_error", {"step": 0, "tool": "turn", "error": "tool_rejected: x"})
    w.event("model_retry", {"step": 1, "retry": 1, "delay_s": 2.0})
    w.usage({"model": "m", "elapsed_s": 1.5, "status": "completed", "usage": {"prompt_tokens": 4, "completion_tokens": 1}})
    w.usage({"model": "m", "elapsed_s": 0.5, "status": "failed", "usage": None})
    with pytest.raises(ValueError):
        w.event("human_evaluation", {"outcome": "maybe"})
    d = w.close("failed", error="boom")
    assert d.name.endswith("_failed")
    events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["run_started", "protocol", "observation", "model_decision", "tool_error",
                                            "model_retry", "run_finished"]
    assert all("at_s" in e for e in events)
    assert events[-1] == {**events[-1], "status": "failed", "task_status": "failed", "outcome": "failed",
                          "model_outcome": None, "human_outcome": None, "outcome_source": "runtime", "error": "boom"}
    tr = json.loads((d / "transcript.json").read_text())
    assert [m["role"] for m in tr] == ["system", "user", "assistant", "tool", "user"]
    assert tr[0]["content"] == "SYS" and tr[1]["images"][0]["path"] == "step_0001.png" and tr[2]["tool_call"]["name"] == "turn"
    assert json.loads((d / "protocol.json").read_text())["base_instructions"] == "SYS"
    status = json.loads((d / "status.json").read_text())
    assert status["error"] == "boom" and status["usage"]["calls"] == 2 and status["usage"]["failed_calls"] == 1
    assert status["usage"]["tokens"] == {"prompt_tokens": 4, "completion_tokens": 1} and status["usage"]["usage_missing_calls"] == 1
    usage = [json.loads(l) for l in (d / "usage.jsonl").read_text().splitlines()]
    assert usage[1]["call"] == 2 and usage[0]["cost_unavailable_reason"] == "model_price_unknown"
    assert json.loads((d / "config.json").read_text())["goal"] == "g"
    with pytest.raises(FileExistsError):
        w2 = EpisodeWriter(tmp_path, env="sim", goal="g", model="m", skills=[], threaded=False)
        w2.dir.rename(w2.dir.with_name(w2.dir.name + "_x")); w2.dir = w2.dir.with_name(w2.dir.name + "_x")
        w2.dir.with_name(w2.dir.name[:-2] + "_failed").mkdir(); w2.close("failed")
