"""In-context demos: a successful run promoted into its task's demos/, shown to the model at the
head of every request of the next run of that task, never carrying host truth."""
import json

import pytest

from g1.agent import agent as agent_module
from g1.agent.decider import VLMDecider
from g1.agent.demos import DemoError, load_demos, promote, render
from g1.agent.task import label_run, load_task, update_summary
from g1.vlm import ToolCall
from tests.agent.test_agent import Scripted
from tests.doubles import scripted_sim

SECRETS = ("SECRET_SUMMARY", "SECRET_HINDSIGHT", "base_measured", "base_error", "within_deg", "evaluation",
           "criteria", '"point_at"', "TRUTH_POSE")
PLAN = [("look", {"yaw_deg": -20.0, "seconds": 1.0, "note": "the laptop is at the right edge; turning to centre it"}),
        ("hold", {"seconds": 0.3, "note": "holding still to look again"}),
        ("done", {"summary": "SECRET_SUMMARY", "hindsight": "SECRET_HINDSIGHT"})]


class FakeClient:
    """Stands in for VLMClient: scripted tool calls, every request kept as sent."""
    model, provider, stream, reasoning_effort, max_tokens = "fake", "openrouter", True, "low", None
    last_usage, response_mode, last_elapsed, last_chunks = {}, "tool_calls", 0.0, []

    def __init__(self, plan):
        self.plan, self.sent = list(plan), []

    def complete(self, messages, *, tools, max_tokens=None):
        self.sent.append(messages)
        name, args = self.plan[min(len(self.sent) - 1, len(self.plan) - 1)]
        return ToolCall(id=f"call_{len(self.sent)}", name=name, arguments=json.dumps(args))


def task_folder(tmp_path, name="find_it"):
    d = tmp_path / "tasks" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({"instruction": "find the laptop and point at it", "modes": ["gantry", "standing"]}))
    return load_task(d)


def recorded_run(tmp_path, monkeypatch, plan=PLAN, task=None):
    """A real agent run on the in-process twin, the real decider asking a scripted client."""
    from g1.cli import run_task
    task = task or task_folder(tmp_path)
    client = FakeClient(plan)
    made = []

    def from_env(cls, *a, demos=None, **k):
        made.append(VLMDecider(client, demos=demos))
        return made[-1]
    monkeypatch.setattr(agent_module.VLMDecider, "from_env", classmethod(from_env))
    assert run_task(task, sim=True, mode=None, env=scripted_sim(Scripted())) == 0
    (folder,) = sorted((task["dir"] / "runs").iterdir(), key=lambda p: p.stat().st_mtime)[-1:]
    return task, folder, client, made[-1]


def as_twin_success(folder):
    """What the twin's verdict leaves on a run: its evaluation (truth) in summary.json, then the label."""
    update_summary(folder, health={"twin": {"base_pose": "TRUTH_POSE", "evaluation": {
        "passed": True, "criteria": {"point_at": "laptop", "within_deg": 10}, "measured": {"miss": 1}}}})
    return label_run(folder, "success", source="twin")


def test_promote_keeps_what_the_model_lived_and_drops_host_truth(tmp_path, monkeypatch):
    task, folder, _, _ = recorded_run(tmp_path, monkeypatch)
    raw = (folder / "step_0001.json").read_text()
    assert "base_measured_pose" in raw                                  # the record has host truth...
    dest = promote(as_twin_success(folder))
    assert dest.parent == task["dir"] / "demos"
    demo = json.loads((dest / "demo.json").read_text())
    text = json.dumps(demo)
    assert not [s for s in SECRETS if s in text], text                  # ...the demo has none
    assert [s["tool"] for s in demo["steps"]] == ["look", "hold", "done"]
    assert demo["steps"][0]["arguments"]["note"].startswith("the laptop is at the right edge")   # its own words stay
    assert demo["steps"][0]["result"]["result"]["status"] == "completed"                          # what it was shown next
    assert demo["steps"][-1]["result"] == {"status": "done"} and demo["steps"][-1]["arguments"] == {}
    assert demo["env"] == "sim" and demo["outcome_source"] == "twin" and demo["decisions"] == 3
    images = [s["image"] for s in demo["steps"] if s["image"]]
    assert images and all((dest / i).is_file() for i in images)


def test_only_a_success_judged_by_the_twin_or_a_person_becomes_a_demo(tmp_path, monkeypatch):
    task, folder, _, _ = recorded_run(tmp_path, monkeypatch)
    with pytest.raises(DemoError, match="unreviewed"):
        promote(folder)                                                  # the model said done: not enough
    failed = label_run(folder, "failed")
    with pytest.raises(DemoError, match="failed"):
        promote(failed)
    assert promote(label_run(failed, "success")).is_dir()              # a person's success
    with pytest.raises(DemoError, match="summary.json"):
        promote(tmp_path / "nowhere")


def demo_file(task, run, env, decisions):
    d = task["dir"] / "demos" / run
    d.mkdir(parents=True)
    steps = [{"turn": i, "tool": "hold", "arguments": {"seconds": 0.3, "note": "n"}, "result": {"status": "ok"},
              "image": None} for i in range(decisions)]
    (d / "demo.json").write_text(json.dumps({"run": run, "task": task["name"], "instruction": "i", "env": env,
                                             "outcome_source": "twin", "decisions": decisions, "steps": steps}))


def test_demos_are_picked_from_the_same_env_fewest_decisions_then_newest(tmp_path):
    task = task_folder(tmp_path)
    for run, env, n in [("20261101-090000_sim_a_success", "sim", 5), ("20261101-100000_sim_b_success", "sim", 3),
                        ("20261102-100000_robot_c_success", "robot", 2), ("20261103-100000_sim_d_success", "sim", 3)]:
        demo_file(task, run, env, n)
    assert [d["run"][:15] for d in load_demos(task, "sim", 2)] == ["20261103-100000", "20261101-100000"]
    assert [d["run"][:15] for d in load_demos(task, "robot", 5)] == ["20261102-100000"]   # never the twin's on the robot
    assert load_demos(task, "sim", 0) == [] and load_demos(task_folder(tmp_path, "empty"), "sim", 2) == []


def test_render_spreads_the_images_and_ends_with_the_current_run(tmp_path):
    task = task_folder(tmp_path)
    demo_file(task, "20261101-090000_sim_a_success", "sim", 6)
    (demo,) = load_demos(task, "sim", 1)
    for s in demo["steps"]:
        s["image"] = f"step_{s['turn']:04d}.png"
    parts = []
    blocks, text = render([demo], images_max=2, image_width=320, read_image=lambda p: p.name,
                          image_part=lambda img, w: parts.append((img, w)) or {"type": "image_url", "image_url": img})
    assert parts == [("step_0000.png", 320), ("step_0005.png", 320)]     # the first and the last, at 320 px
    assert blocks[0]["text"].startswith("Earlier successful runs of this task") and blocks[-1]["text"] == "Now the current run:"
    assert text.count("turn ") == 6 + 2 and "hold(" in text and render([], images_max=2, image_width=320) == ([], "")


def test_the_next_run_is_shown_the_demos_in_every_request_and_records_them_once(tmp_path, monkeypatch):
    task, first, _, _ = recorded_run(tmp_path, monkeypatch)
    promote(as_twin_success(first))
    task, second, client, decider = recorded_run(tmp_path, monkeypatch, task=task)
    assert len(client.sent) == 3
    heads = [next(m for m in msgs if m["role"] == "user")["content"] for msgs in client.sent]
    n = len(decider._demo_blocks)
    assert n and all(h[:n] == heads[0][:n] for h in heads)              # the same prefix every request (cacheable)
    assert heads[0][0]["text"].startswith("Earlier successful runs") and any(b.get("type") == "image_url" for b in heads[0][:n])
    sent = json.dumps(client.sent)
    assert not [s for s in SECRETS if s in sent]                        # host truth never reaches the model
    assert "the laptop is at the right edge" in sent                    # its own earlier words do
    config = json.loads((second / "config.json").read_text())
    assert config["demos"]["ids"] == [first.parent.parent.joinpath("demos").iterdir().__next__().name]
    assert "turning to centre it" in config["demos"]["text"]
    step = json.loads((second / "step_0002.json").read_text())
    head = next(m for m in step["query"] if m["role"] == "user")["content"]
    assert head[0] == {"type": "demos", "ids": config["demos"]["ids"]}  # one marker per step file, not the demo
    assert "Earlier successful runs" not in json.dumps(step["query"])


def test_demos_max_zero_shows_none(tmp_path, monkeypatch):
    from g1.vlm import AGENT_SETTINGS
    task, first, _, _ = recorded_run(tmp_path, monkeypatch)
    promote(as_twin_success(first))
    monkeypatch.setitem(AGENT_SETTINGS, "demos_max", 0)
    task, second, client, decider = recorded_run(tmp_path, monkeypatch, task=task)
    assert decider.demos == [] and "Earlier successful runs" not in json.dumps(client.sent)
    assert json.loads((second / "config.json").read_text())["demos"] == {"ids": [], "text": ""}
