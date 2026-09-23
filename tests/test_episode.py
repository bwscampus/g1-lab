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
                      {"scene": "a room", "action": name, "args": args, "raw": "{}", "latency": 0.5},
                      {"name": name, "args": args, "max_duration": 2.5, "kind": "motion", "needs_base": name != "wave"},
                      {"status": status, "duration": 1.0})


def test_round_trip_is_lossless(tmp_path):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    w = EpisodeWriter(tmp_path, env="check", goal="Find the Mug!", model="m", skills=["turn"], threaded=False)
    assert w.dir.name.endswith("_check_find-the-mug")
    w.write_step(record(1, "turn", {"angle_deg": 45.0}), img)
    meta = json.loads((w.dir / "episode.json").read_text())
    assert meta["steps"] == 1 and meta["result"] is None            # rewritten per step
    w.finish("found")
    w.close()
    meta, steps = load_episode(w.dir)
    assert meta["result"] == "found" and meta["ended"] and len(steps) == 1
    assert np.array_equal(steps[0].image, img)                       # exact RGB matrix back
    assert steps[0].image[0, 0].tolist() == img[0, 0].tolist()       # channel order preserved
    assert steps[0].frame["image"] == "step_0001.png" and steps[0].skill["args"] == {"angle_deg": 45.0}


def test_threaded_writer_flushes_on_close(tmp_path):
    pytest.importorskip("cv2")
    img = np.zeros((8, 8, 3), np.uint8)
    w = EpisodeWriter(tmp_path, env="sim", goal="g", model="m", skills=[])
    for i in range(1, 6):
        w.write_step(record(i, "walk_forward", {"distance_m": 0.5}), img)
    w.finish("max_steps")
    w.close()
    meta, steps = load_episode(w.dir, images=False)
    assert len(steps) == 5 and meta["steps"] == 5 and meta["result"] == "max_steps"
    assert all(s.image is None for s in steps)


def test_chain_of_and_cli(tmp_path, capsys):
    pytest.importorskip("cv2")
    w = EpisodeWriter(tmp_path, env="check", goal="g", model="m", skills=[], threaded=False)
    w.write_step(record(1, "turn", {"angle_deg": 45.0}), None)
    w.write_step(record(2, "walk_forward", {"distance_m": 0.5}), None)
    w.write_step(record(3, "hold", {"seconds": 1.0}, status="timeout"), None)
    w.write_step(record(4, "done", {"found": True}, status="done"), None)
    w.finish("found"); w.close()
    _, steps = load_episode(w.dir, images=False)
    assert chain_of(steps) == "turn:angle_deg=45.0,walk_forward:distance_m=0.5"
    assert _main([str(w.dir)]) == 0
    out = capsys.readouterr().out
    assert "replay:  --policy turn:angle_deg=45.0,walk_forward:distance_m=0.5" in out and "4 step(s)" in out
