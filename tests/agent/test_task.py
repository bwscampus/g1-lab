"""A task folder is an instruction plus context; `g1 task` runs, shows and scores it."""
import json

import pytest

from g1.agent.task import load_task, record_result, results, run_flags, summarize
from g1.cli import main
from g1.core import limits

ROOT = limits.ROOT


def task_dir(tmp_path, **data):
    d = tmp_path / "find_the_mug"
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"instruction": "find the mug", **data}))
    return d


def test_load_task_defaults_validates_and_resolves(tmp_path):
    d = task_dir(tmp_path)
    t = load_task(d)
    assert t["instruction"] == "find the mug" and t["scene"] == "none" and t["safety_notes"] == []
    assert t["max_decisions"] == limits.get("max_decisions") and t["demo"] is None and t["dir"] == d
    assert run_flags(t)[:4] == ["--tools", "search", "--instruction", "find the mug"] and "--log" in run_flags(t)
    (d / "walk.mp4").write_bytes(b"x")
    d2 = task_dir(tmp_path / "b", scene="room", objects=["mug@1.5,1.2"], safety_notes=["a wall behind"],
                  demo="../../find_the_mug/walk.mp4", refs=[], max_decisions=5, notes="for people")
    f = run_flags(load_task(d2))
    assert f[f.index("--scene") + 1] == "room" and f[f.index("--sim-objects") + 1] == "mug@1.5,1.2"
    assert f[f.index("--safety-note") + 1] == "a wall behind" and f[f.index("--demo") + 1].endswith("find_the_mug/walk.mp4")
    assert f[f.index("--max-decisions") + 1] == "5"
    for bad, match in [({"instruction": ""}, "instruction"), ({"scene": "moon"}, "scene"), ({"speed": 3}, "speed"),
                       ({"max_decisions": 0}, "max_decisions")]:
        (d / "task.json").write_text(json.dumps({"instruction": "x", **bad}))
        with pytest.raises(ValueError, match=match):
            load_task(d)
    (d / "task.json").write_text("{")
    with pytest.raises(ValueError, match="task.json"):
        load_task(d)
    with pytest.raises(ValueError, match="no task.json"):
        load_task(tmp_path / "nowhere")


def test_results_and_summary(tmp_path):
    d = task_dir(tmp_path)
    t = load_task(d)
    assert results(t) == [] and summarize([]) == "no runs yet"
    run_dir = d / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"outcome": "success", "task_status": "completed", "outcome_source": "human",
                                                     "decisions": 4, "steps": 6, "model": "m", "env": "sim",
                                                     "usage": {"tokens": {"prompt_tokens": 100, "completion_tokens": 20}},
                                                     "elapsed_s": 30.0, "ended": "t"}))
    row = record_result(t, run_dir)
    assert row["outcome"] == "success" and row["decisions"] == 4 and results(t) == [row]
    (run_dir / "status.json").write_text(json.dumps({"outcome": "unreviewed", "task_status": "unreviewed",
                                                     "outcome_source": "unreviewed", "decisions": 7}))
    record_result(t, run_dir)
    text = summarize(results(t))
    assert "2 run(s), 1 success (50%)" in text and "1 without a human verdict" in text
    assert "decisions to success: mean 4.0" in text and "tokens per run: mean 60" in text
    assert record_result(t, tmp_path / "missing") is None


def test_task_cli_show_and_run(tmp_path, capsys, monkeypatch):
    d = task_dir(tmp_path, notes="a note for people")
    assert main(["task", "show", str(d)]) == 0
    out = capsys.readouterr().out
    assert "find the mug" in out and "--tools search" in out and "a note for people" in out and "no runs yet" in out
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        main(["task", "run", str(d), "--env", "sim", "--headless"])      # the same code path as g1 run: needs the key
    assert "HF_TOKEN" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["task", "show", str(tmp_path / "nope")])
    assert "no task.json" in capsys.readouterr().err


def test_task_run_records_a_result(tmp_path, capsys, monkeypatch):
    """The whole path with a scripted decider: the run lands under the task's runs/ and in results.jsonl."""
    from g1.agent import agent as agent_module
    from tests.agent.test_agent import Sequence
    d = task_dir(tmp_path, max_decisions=3, max_time_s=60)
    dec = Sequence([("hold", {"seconds": 0.3}), ("done", {"summary": "s", "hindsight": "h"})])
    monkeypatch.setattr(agent_module.VLMDecider, "from_env", classmethod(lambda cls, *a, **k: dec))
    assert main(["task", "run", str(d), "--env", "sim", "--headless", "--no-verdict"]) == 0
    out = capsys.readouterr().out
    assert "result: unreviewed" in out and "2 decision(s)" in out
    rows = results(load_task(d))
    assert len(rows) == 1 and rows[0]["decisions"] == 2 and rows[0]["model"] == "sequence"
    assert (d / "runs" / rows[0]["run"] / "episode.json").is_file() and rows[0]["run"].endswith("_unreviewed")
    assert main(["task", "show", str(d)]) == 0 and "1 run(s), 0 success" in capsys.readouterr().out


def test_template_task_and_scaffolder(tmp_path, monkeypatch, capsys):
    t = load_task(ROOT / "tasks" / "_template")
    assert t["instruction"] and t["notes"]
    t = load_task(ROOT / "tasks" / "find_the_mug")
    assert t["scene"] == "room" and t["objects"] and t["safety_notes"] == []      # the room's notes come from --scene
    import g1.cli as cli
    root = tmp_path / "repo"
    (root / "tasks").mkdir(parents=True)
    import shutil
    shutil.copytree(ROOT / "tasks" / "_template", root / "tasks" / "_template")
    monkeypatch.setattr(cli, "ROOT", root)
    assert main(["new", "task", "wave_at_me"]) == 0
    assert (root / "tasks" / "wave_at_me" / "task.json").is_file() and "wave_at_me" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["new", "task", "wave_at_me"])
