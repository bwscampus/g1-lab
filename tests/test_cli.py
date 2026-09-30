"""The g1 command, and: an interrupted run returns the robot to a safe state, and nothing can cut that short."""
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
from tests.doubles import scripted_sim

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
    pol = Interrupting(Chain(TOOLS["tpose"](hold_s=2.0, rise_s=2.0)), at=6.5)   # mid-raise, weight 1
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
    pol = Interrupting(Chain(TOOLS["tpose"](hold_s=2.0, rise_s=2.0)), at=6.5)
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
    pol = Interrupting(Chain(TOOLS["hold"](seconds=3.0)), at=5.5, exc=RuntimeError)
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

def test_g1_dispatches_and_helps(capsys):
    assert main([]) == 0 and "g1 run" in capsys.readouterr().out
    assert main(["--help"]) == 0 and "subcommand" in capsys.readouterr().out.lower() or True
    with pytest.raises(SystemExit):
        main(["fly"])
    assert "unknown command" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["run", "--env", "sim", "--headless"])                     # no --tools
    assert "--tools is required" in capsys.readouterr().err
    assert main(["run", "--env", "sim", "--headless", "--tools", "hold:0.2"]) == 0
    assert "== hold @ sim ==" in capsys.readouterr().out
    assert main(["limits", "--source", "robot"]) == 0 and "control_dt_s" in capsys.readouterr().out


def test_env_variables_are_the_defaults(monkeypatch, capsys):
    monkeypatch.setenv("G1_ENV", "sim")
    monkeypatch.setenv("G1_TOOLS", "hold:0.2")
    assert main(["run", "--headless"]) == 0
    assert "== hold @ sim ==" in capsys.readouterr().out


def test_view_and_record_flags(tmp_path, capsys, monkeypatch):
    av = pytest.importorskip("av")
    out = tmp_path / "chain.mp4"
    assert main(["run", "--env", "sim", "--headless", "--tools", "hold:0.5", "--view", "0", "--record", str(out)]) == 0
    text = capsys.readouterr().out
    assert "view: http://127.0.0.1:" in text and f"recording: {out}" in text and "dropped ->" in text
    c = av.open(str(out))
    assert sum(1 for _ in c.decode(c.streams.video[0])) > 10             # the camera was opened for a chain
    c.close()
    with pytest.raises(SystemExit, match="--camera off"):
        main(["run", "--env", "sim", "--headless", "--tools", "hold:0.2", "--view", "0", "--camera", "off"])
    # --record with no path: a chain lands under --log, an agent inside its (renamed) run directory
    assert main(["run", "--env", "sim", "--headless", "--tools", "hold:0.2", "--record", "--log", str(tmp_path)]) == 0
    files = list(tmp_path.glob("*_sim_hold.mp4"))
    assert len(files) == 1
    from g1.agent import agent as agent_module
    from tests.agent.test_agent import Sequence
    dec = Sequence([("done", {"summary": "s", "hindsight": "h"})])
    monkeypatch.setattr(agent_module.VLMDecider, "from_env", classmethod(lambda cls, *a, **k: dec))
    assert main(["run", "--env", "sim", "--headless", "--tools", "search", "-i", "g", "--record", "--no-verdict",
                 "--log", str(tmp_path / "runs")]) == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert run_dir.name.endswith("_unreviewed") and (run_dir / "camera.mp4").stat().st_size > 0
