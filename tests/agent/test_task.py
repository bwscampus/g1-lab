"""A task folder is an instruction (the model decides) or a tools chain, plus context; `g1 run` runs it."""
import json

import pytest

from g1.agent.task import find, load_task, record_result
from g1.core import limits

ROOT = limits.ROOT


def task_dir(tmp_path, name="find_the_mug", **data):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps(data))
    return d


def test_load_task_defaults_and_validates(tmp_path):
    d = task_dir(tmp_path, instruction="find the mug")
    t = load_task(d)
    assert t["instruction"] == "find the mug" and t["tools"] is None and t["scene"] == "none" and t["safety_notes"] == []
    assert t["max_decisions"] == limits.get("max_decisions") and t["dir"] == d and t["name"] == "find_the_mug"
    t = load_task(task_dir(tmp_path, "b", tools="tpose,look:30", scene="room", objects=["mug@1.5,1.2"], camera_size="720x1280"))
    assert t["tools"] == "tpose,look:30" and t["instruction"] is None and t["camera_size"] == (720, 1280)
    for bad, match in [({}, "exactly one of instruction or tools"), ({"instruction": "x", "tools": "tpose"}, "exactly one"),
                       ({"instruction": ""}, "exactly one|instruction"), ({"instruction": "x", "scene": "moon"}, "scene"),
                       ({"instruction": "x", "demo": "walk.mp4"}, "demo"), ({"instruction": "x", "max_decisions": 0}, "max_decisions"),
                       ({"tools": "tpose", "camera_size": "big"}, "camera_size"),
                       ({"tools": "tpose", "objects": ["mug@1,0"]}, "objects need")]:
        (d / "task.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=match):
            load_task(d)
    (d / "task.json").write_text("{")
    with pytest.raises(ValueError, match="task.json"):
        load_task(d)
    with pytest.raises(ValueError, match="no task.json"):
        load_task(tmp_path / "nowhere")


def test_find_takes_a_name_or_a_path(tmp_path):
    assert find("tpose") == ROOT / "tasks" / "tpose"
    assert find(ROOT / "tasks" / "find_the_mug") == ROOT / "tasks" / "find_the_mug"
    d = task_dir(tmp_path, "mine", tools="tpose")
    assert find(d) == d
    with pytest.raises(ValueError, match="tasks: .*tpose"):
        find("no_such_task")


def test_the_shipped_tasks_load():
    assert load_task(find("tpose"))["tools"] == "tpose"
    t = load_task(find("find_the_mug"))
    assert t["instruction"] and t["scene"] == "room" and t["objects"] == ["mug@1.5,1.2"]
    t = load_task(find("point_at_the_mug"))
    assert t["instruction"] and t["scene"] == "room" and t["objects"] == ["mug@1.0,1.5"]
    t = load_task(find("sit_in_the_chair"))
    assert t["instruction"] and t["objects"] == ["chair@1.5,0"]


def test_record_result_from_an_agent_status_or_the_checks(tmp_path):
    t = load_task(task_dir(tmp_path, instruction="find the mug"))
    run_dir = t["dir"] / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"outcome": "success", "task_status": "completed", "outcome_source": "human",
                                                     "decisions": 4, "steps": 6, "model": "m", "env": "sim",
                                                     "usage": {"tokens": {"prompt_tokens": 100}}, "elapsed_s": 30.0, "ended": "t"}))
    row = record_result(t, run_dir)
    assert row["outcome"] == "success" and row["decisions"] == 4 and row["tokens"] == {"prompt_tokens": 100}
    chain = t["dir"] / "runs" / "r2"
    chain.mkdir()
    row = record_result(t, chain, ok=False, env_name="sim")                    # a tools run: the checks decide
    assert row["outcome"] == "failed" and row["outcome_source"] == "checks"
    lines = (t["dir"] / "results.jsonl").read_text().splitlines()
    assert [json.loads(l)["run"] for l in lines] == ["r1", "r2"]
    assert record_result(t, t["dir"] / "runs" / "missing") is None
