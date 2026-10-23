"""The g1 command, and: an interrupted run returns the robot to a safe state, and nothing can cut that short."""
import json
import os
import signal

import numpy as np
import pytest

from g1.agent.agent import Agent
from g1.agent.decider import Decider
from g1.agent.episode import EpisodeWriter
from g1.cli import RAMP, RELEASED, TO_STAND, main, run
from g1.core.action import Runnable
from g1.core.config import CONTROL_DT, STAND_Q, UPPER_BODY
from g1.tools import TOOLS, Chain
from tests.agent.test_agent import Scripted, Sequence
from tests.doubles import scripted_sim, sim_env

JOINTS = sorted(UPPER_BODY)


class Interrupting(Runnable):
    """Wraps a program and raises ``exc`` once ``t`` reaches ``at``."""

    def __init__(self, inner, at, exc=KeyboardInterrupt):
        self.inner, self.at, self.exc = inner, at, exc
        self.joints = inner.joints
        self.name = "interrupting"
        self.n_before = 0

    def reset(self, obs):
        self.inner.reset(obs)

    def step(self, t, obs):
        if t >= self.at - 1e-9:
            raise self.exc("test")
        self.n_before += 1
        return self.inner.step(t, obs)


def env_with(program, **kw):
    env = scripted_sim(Scripted())
    ok = run(program, env, **kw)
    return env, ok


def returned(env, n_before):
    """The actions the safe return sent, and checks every return must pass."""
    after = env.actions[n_before:]
    assert len(after) == round((TO_STAND + RAMP + RELEASED) / CONTROL_DT)
    assert all(a.base is None and a.command is None for a in after)
    qs = np.array([a.q[JOINTS] for a in after])
    assert np.allclose(qs[-1], STAND_Q[JOINTS], atol=1e-9)
    assert after[-1].weight == pytest.approx(0.0, abs=1e-9)
    # no jump at the seam and no jump inside: a continuous 50 Hz command stream
    before = env.actions[n_before - 1].q[JOINTS]
    steps = np.abs(np.diff(np.vstack([before, qs]), axis=0)) / CONTROL_DT
    assert steps.max() < 4.0
    assert env.violations == []
    return after


def test_ctrl_c_returns_to_stand_and_hands_back():
    pol = Interrupting(Chain(TOOLS["tpose"](hold_s=2.0, rise_s=2.0)), at=3.5)   # mid-raise, weight 1
    env, ok = env_with(pol)
    assert ok is False
    after = returned(env, pol.n_before)
    assert after[0].weight == pytest.approx(1.0) and after[len(after) // 2].weight == pytest.approx(1.0)
    assert env.actions[pol.n_before - 1].q[16] > 0.5                                # it was on its way up
    t_stand = round(TO_STAND / CONTROL_DT)
    assert np.allclose(after[t_stand - 1].q[JOINTS], STAND_Q[JOINTS], atol=1e-3)    # at stand after TO_STAND (eased)


def test_interrupt_mid_takeover_never_raises_the_weight():
    pol = Interrupting(Chain(TOOLS["hold"](seconds=1.0)), at=1.0)                 # takeover ramp: weight 0.5
    env, ok = env_with(pol)
    w0 = env.actions[pol.n_before - 1].weight
    assert 0.3 < w0 < 0.7 and ok is False
    after = returned(env, pol.n_before)
    assert max(a.weight for a in after) <= w0 + 1e-9


def test_sigint_during_the_return_is_ignored(capsys):
    pol = Interrupting(Chain(TOOLS["tpose"](hold_s=2.0, rise_s=2.0)), at=3.5)
    env = scripted_sim(Scripted())
    fired = []
    original = env.step

    def step(action):
        if len(env.actions) == pol.n_before + 10 and not fired:          # inside the return
            fired.append(True)
            os.kill(os.getpid(), signal.SIGINT)
        return original(action)

    env.step = step
    assert run(pol, env) is False and fired
    returned(env, pol.n_before)                                            # the stream ran to the end
    assert "interrupt ignored" in capsys.readouterr().out


def test_errors_and_max_time_also_return():
    pol = Interrupting(Chain(TOOLS["hold"](seconds=3.0)), at=3.5, exc=RuntimeError)
    env = scripted_sim(Scripted())
    with pytest.raises(RuntimeError, match="test"):
        run(pol, env)
    returned(env, pol.n_before)
    pol = Chain(TOOLS["hold"](seconds=30.0))
    env = scripted_sim(Scripted())
    assert run(pol, env, max_time=6.0) is False
    n = round(6.0 / CONTROL_DT) + 1
    returned(env, n)


def test_agent_records_the_interruption(tmp_path):
    class Boom(Decider):
        model = "boom"

        def __init__(self):
            super().__init__(threaded=False)

        def decide(self, turn):
            raise KeyboardInterrupt("test")

    dec = Boom()
    rec = EpisodeWriter(tmp_path, env="sim", instruction="g", model="boom", tools=list(TOOLS), threaded=False)
    agent = Agent("g", dec, [TOOLS["hold"], TOOLS["done"]], recorder=rec, verdict=lambda: None)
    env = scripted_sim(Scripted())
    assert run(agent, env) is False
    agent.close()
    assert rec.dir.name.endswith("_interrupted")
    import json
    kinds = [json.loads(l)["event"] for l in (rec.dir / "events.jsonl").read_text().splitlines()]
    i = kinds.index("interrupted")
    assert kinds[i:i + 3] == ["interrupted", "return_home_started", "return_home"]
    assert kinds[-1] == "run_finished" and json.loads((rec.dir / "status.json").read_text())["state"] == "interrupted"
    assert env.actions[-1].weight == pytest.approx(0.0) and env.violations == []


# -- the command --------------------------------------------------------------------------

def task_dir(tmp_path, name="t", **spec):
    d = tmp_path / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps(spec))
    return d


def test_g1_has_five_commands_and_run_three_flags(capsys):
    assert main([]) == 0 and "g1 run TASK --sim" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["fly"])
    assert "unknown command" in capsys.readouterr().err
    for gone in ("task", "tools", "camera", "decide", "episode", "demo", "scene", "new"):
        with pytest.raises(SystemExit):
            main([gone])
    with pytest.raises(SystemExit):
        main(["run", "tpose", "--help"])
    text = capsys.readouterr().out
    assert all(f in text for f in ("--sim", "--mode", "--view")) and "--viewer" not in text and "--walk" not in text
    assert "--tools" not in text and "--headless" not in text


def test_the_robot_needs_a_mode_and_view_is_the_robots_camera(capsys):
    with pytest.raises(SystemExit, match="needs --mode"):
        main(["run", "tpose"])                                        # before anything touches the network
    with pytest.raises(SystemExit):
        main(["run", "tpose", "--sim", "--view"])
    assert "--sim already shows the twin" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["run", "no_such_task", "--sim"])
    assert "tasks: " in capsys.readouterr().err


def test_a_tools_task_runs_records_and_traces(tmp_path, capsys):
    from g1.agent.task import load_task
    from g1.cli import run_task
    task = load_task(task_dir(tmp_path, tools="hold:0.3"))
    assert run_task(task, sim=True, mode=None, env=sim_env()) == 0
    (run_dir,) = (task["dir"] / "runs").iterdir()
    assert (run_dir / "trace.npz").stat().st_size > 0 and json.loads((run_dir / "status.json").read_text())["outcome"] == "success"
    row = json.loads((task["dir"] / "results.jsonl").read_text())
    assert row["run"] == run_dir.name and row["outcome_source"] == "checks"
    assert "result: success" in capsys.readouterr().out


def test_gantry_never_walks(tmp_path, capsys):
    from g1.agent.task import load_task
    from g1.cli import run_task
    task = load_task(task_dir(tmp_path, tools="walk_forward:0.3"))
    with pytest.raises(SystemExit, match="never walks"):
        run_task(task, sim=True, mode="gantry", env=sim_env(mode="gantry"))
    assert not (task["dir"] / "runs").exists() or not any((task["dir"] / "runs").iterdir())
    assert sim_env(mode="gantry").can_walk is False and sim_env().can_walk is True


def test_an_instruction_task_runs_the_agent(tmp_path, monkeypatch, capsys):
    from g1.agent import agent as agent_module
    from g1.agent.task import load_task
    from g1.cli import run_task
    from tests.agent.test_agent import Sequence
    monkeypatch.setattr(agent_module, "ask_save", lambda: pytest.fail("a sim run asks nothing"))
    monkeypatch.setattr(agent_module, "ask_verdict", lambda: pytest.fail("a sim run asks nothing"))
    dec = Sequence([("done", {"summary": "s", "hindsight": "h"})])
    monkeypatch.setattr(agent_module.VLMDecider, "from_env", classmethod(lambda cls, *a, **k: dec))
    task = load_task(task_dir(tmp_path, instruction="find it", safety_notes=["a wall behind"]))
    assert run_task(task, sim=True, mode=None, env=scripted_sim(Scripted())) == 0
    (run_dir,) = (task["dir"] / "runs").iterdir()
    assert run_dir.name.endswith("_unreviewed") and (run_dir / "trace.npz").exists() and (run_dir / "episode.json").exists()
    assert json.loads((task["dir"] / "results.jsonl").read_text())["run"] == run_dir.name
    assert "a wall behind" in dec.context.instructions


def test_agent_records_sdk_calls_and_verbose_health(tmp_path, capsys):
    """Every SDK call an env makes lands in events.jsonl; -v prints the health line."""
    import json
    from g1.tools import TOOLS, menu
    dec = Sequence([("hold", {"seconds": 0.3}), ("done", {"summary": "s", "hindsight": "h"})])
    rec = EpisodeWriter(tmp_path, env="sim", instruction="g", model="sequence", tools=list(TOOLS), threaded=False)
    agent = Agent("g", dec, menu(True), recorder=rec, verdict=lambda: None)
    env = scripted_sim(Scripted())
    env.verbose = True
    original = env.step

    def step(action):                                                   # an env that talks to an SDK
        if len(env.actions) == 5:
            env.sdk_log.append({"at_s": 0.0, "client": "LocoClient", "name": "StopMove", "args": [], "kwargs": {},
                                "code": 3104, "elapsed_s": 0.5})
        return original(action)
    env.step = step
    assert run(agent, env, max_time=60) is True
    agent.close()
    events = [json.loads(l) for l in (rec.dir / "events.jsonl").read_text().splitlines()]
    calls = [e for e in events if e["event"] == "sdk_call"]
    assert [(c["client"], c["name"]) for c in calls[:2]] == [("AudioClient", "GetVolume"), ("LocoClient", "GetFsmId")]
    calls = calls[2:]                                                    # after the bring-up: the injected failure
    assert len(calls) == 1 and calls[0]["name"] == "StopMove" and calls[0]["code"] == 3104 and "step" in calls[0]
    out = capsys.readouterr().out
    assert "t=   1.0  lowstate " in out and "ticks 50 " in out      # the robot's health line, on the twin's clock


def test_view_serves_the_head_camera_while_the_run_lasts(capsys):
    import urllib.request
    from g1.tools.control import Hold
    seen = {}

    class Peek(Chain):
        def step(self, t, obs):
            if "url" not in seen and t > 0.5:
                url = next(l.split("view: ")[1] for l in capsys.readouterr().out.splitlines() if l.startswith("view: "))
                seen["url"] = url
                seen["frame"] = urllib.request.urlopen(url + "frame", timeout=2).read()[:2]
            return super().step(t, obs)
    env = sim_env(view=True)
    assert run(Peek(Hold(seconds=1.0), pause=0.0), env) is True       # a chain alone opens no camera; --view does
    assert seen["frame"] == b"\xff\xd8"                                 # a JPEG, served mid-run
    import urllib.error
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(seen["url"], timeout=1)                   # gone once the run ended


def test_a_robot_run_asks_to_save_then_for_the_verdict(tmp_path, monkeypatch, capsys):
    from g1.agent import agent as agent_module
    from g1.agent.task import load_task
    from g1.cli import run_task
    task = load_task(task_dir(tmp_path, tools="hold:0.3"))
    asked = []
    monkeypatch.setattr(agent_module, "ask_save", lambda: asked.append("save") or False)
    monkeypatch.setattr(agent_module, "ask_verdict", lambda: asked.append("verdict") or "success")
    # the in-process twin stands in for the robot: sim=False is the robot's path after the run
    assert run_task(task, sim=False, mode="standing", env=sim_env(mode="standing")) == 0
    assert asked == ["save"] and not any((task["dir"] / "runs").iterdir())    # not saved: gone, never asked the verdict
    assert not (task["dir"] / "results.jsonl").exists() and "run not saved" in capsys.readouterr().out
    monkeypatch.setattr(agent_module, "ask_save", lambda: asked.append("save") or True)
    monkeypatch.setattr(agent_module, "ask_verdict", lambda: asked.append("verdict") or "failed")
    assert run_task(task, sim=False, mode="standing", env=sim_env(mode="standing")) == 0
    (run_dir,) = (task["dir"] / "runs").iterdir()
    assert asked[1:] == ["save", "verdict"] and run_dir.name.endswith("_robot_hold-0-3_failed")
    status = json.loads((run_dir / "status.json").read_text())
    assert status["outcome"] == "failed" and status["outcome_source"] == "human"     # the human, not the checks
    assert json.loads((task["dir"] / "results.jsonl").read_text())["outcome_source"] == "human"


def test_label_run_marks_an_agent_record(tmp_path):
    from g1.agent.task import label_run
    run = tmp_path / "20261019-120000_robot_find-it_unreviewed"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"outcome": "unreviewed", "outcome_source": "unreviewed"}))
    (run / "episode.json").write_text(json.dumps({"outcome": "unreviewed"}))
    (run / "events.jsonl").write_text("")
    out = label_run(run, "success")
    assert out.name == "20261019-120000_robot_find-it_success" and not run.exists()
    assert json.loads((out / "status.json").read_text())["human_outcome"] == "success"
    assert json.loads((out / "episode.json").read_text())["outcome"] == "success"
    assert json.loads((out / "events.jsonl").read_text())["event"] == "human_evaluation"


def test_g1_limits_prints_every_limit_as_one_table(capsys):
    from g1.core import limits
    assert main(["limits"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in limits.NAMES) and f"{len(limits.NAMES)} limits" in out
    assert main(["limits", "guess"]) == 0
    out = capsys.readouterr().out
    assert "with source 'guess'" in out and "control_dt_s" not in out            # control_dt_s is a robot fact
    assert main(["limits", "vibes"]) == 2
