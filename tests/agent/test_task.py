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
    t = load_task(find("point_at_the_laptop"))
    assert t["scene"] == "room" and t["objects"] == ["laptop@1.75,-1.5,0.77"] and t["start"] == [1.2, -0.95, 0]
    t = load_task(find("find_and_point_at_the_laptop"))
    assert "laptop" in t["instruction"] and t["start"] is None
    with pytest.raises(ValueError, match="no task 'point_at_the_mug'"):
        find("point_at_the_mug")                                             # the point demo's target is the laptop
    # every shipped task loads and places only objects the room can build
    from g1.envs.scene import parse_object
    for d in sorted((ROOT / "tasks").iterdir()):
        if (d / "task.json").is_file():
            t = load_task(d)
            for o in t["objects"]:
                parse_object(o)
            assert t["start"] is None or len(t["start"]) == 3
    t = load_task(find("sit_in_the_chair"))
    assert t["instruction"] and t["objects"] == ["chair@1.5,0"]


def test_record_result_from_a_summary(tmp_path):
    t = load_task(task_dir(tmp_path, instruction="find the mug"))
    run_dir = t["dir"] / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({
        "outcome": "success", "outcome_source": "human", "model_outcome": "success", "end_reason": "done",
        "decisions": {"used": 4, "max": 12}, "model": "m", "env": "sim", "mode": "gantry",
        "usage": {"tokens": {"prompt_tokens": 100}}, "duration_s": 30.0, "ended": "t",
        "health": {"joint_check": "pass", "violations": 0, "overruns": 3, "fell": False}}))
    row = record_result(t, run_dir)
    assert row["outcome"] == "success" and row["decisions"] == 4 and row["tokens"] == {"prompt_tokens": 100}
    assert row["end_reason"] == "done" and row["mode"] == "gantry" and row["overruns"] == 3 and row["fell"] is False
    assert [json.loads(l)["run"] for l in (t["dir"] / "results.jsonl").read_text().splitlines()] == ["r1"]
    assert record_result(t, t["dir"] / "runs" / "missing") is None


def test_code_snapshot_restores_the_working_tree_untouched(tmp_path, no_code_snapshot):
    """A dirty tree becomes a commit kept by a ref; the branch, index and files stay as they were."""
    import subprocess
    code_snapshot = no_code_snapshot                     # the real one (conftest stubs it elsewhere)

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("runs/\n")
    (tmp_path / "a.py").write_text("one\n")
    git("add", "-A")
    git("commit", "-qm", "first")
    head = git("rev-parse", "HEAD")
    clean = code_snapshot(tmp_path)
    assert clean == {"head": head, "snapshot": head, "ref": None}       # clean: HEAD is the code
    (tmp_path / "a.py").write_text("two\n")                          # a tracked edit
    (tmp_path / "new.py").write_text("new\n")                        # a new file
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "x.json").write_text("{}")                   # ignored: never in the snapshot
    status = git("status", "--porcelain")
    snap = code_snapshot(tmp_path)
    assert snap["head"] == head and snap["snapshot"] != head and snap["ref"] == f"refs/g1-runs/{snap['snapshot']}"
    assert git("show", f"{snap['snapshot']}:a.py") == "two" and git("show", f"{snap['snapshot']}:new.py") == "new"
    assert "runs/x.json" not in git("ls-tree", "-r", "--name-only", snap["snapshot"])
    assert git("rev-parse", snap["ref"]) == snap["snapshot"] and git("rev-parse", f"{snap['snapshot']}^") == head
    assert git("rev-parse", "HEAD") == head and git("status", "--porcelain") == status   # nothing else moved
    assert code_snapshot(tmp_path / "runs" / "nope")["snapshot"] is None                 # no repo: recorded, not fatal
