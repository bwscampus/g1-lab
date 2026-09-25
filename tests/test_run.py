"""An interrupted run returns the robot to a safe state, and nothing can cut that short."""
import os
import signal

import numpy as np
import pytest

from agent import Agent
from config import CONTROL_DT, STAND_Q, UPPER_BODY
from decider import Decider
from episode import EpisodeWriter
from policy import Policy
from routines import Routine
from run import RAMP, RELEASED, TO_STAND, run
from skills import SKILLS
from tests.doubles import scripted_sim
from tests.test_agent import Scripted, Sequence

JOINTS = sorted(UPPER_BODY)


class Interrupting(Policy):
    """Wraps a policy and raises ``exc`` once ``t`` reaches ``at``."""

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


def env_with(policy, **kw):
    env = scripted_sim(Scripted())
    ok = run(policy, env, **kw)
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
    pol = Interrupting(Routine(SKILLS["tpose"](hold=2.0, rise=2.0)), at=6.5)     # mid-raise, weight 1
    env, ok = env_with(pol)
    assert ok is False
    after = returned(env, pol.n_before)
    assert after[0].weight == pytest.approx(1.0) and after[len(after) // 2].weight == pytest.approx(1.0)
    assert env.actions[pol.n_before - 1].q[16] > 0.5                                # it was on its way up
    t_stand = round(TO_STAND / CONTROL_DT)
    assert np.allclose(after[t_stand - 1].q[JOINTS], STAND_Q[JOINTS], atol=1e-3)    # at stand after TO_STAND (eased)


def test_interrupt_mid_takeover_never_raises_the_weight():
    pol = Interrupting(Routine(SKILLS["hold"](seconds=1.0)), at=1.0)               # takeover ramp: weight 0.5
    env, ok = env_with(pol)
    w0 = env.actions[pol.n_before - 1].weight
    assert 0.3 < w0 < 0.7 and ok is False
    after = returned(env, pol.n_before)
    assert max(a.weight for a in after) <= w0 + 1e-9


def test_sigint_during_the_return_is_ignored(capsys):
    class Sim(type(scripted_sim(Scripted()))):
        pass

    pol = Interrupting(Routine(SKILLS["tpose"](hold=2.0, rise=2.0)), at=6.5)
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
    pol = Interrupting(Routine(SKILLS["hold"](seconds=3.0)), at=5.5, exc=RuntimeError)
    env = scripted_sim(Scripted())
    with pytest.raises(RuntimeError, match="test"):
        run(pol, env)
    returned(env, pol.n_before)
    pol = Routine(SKILLS["hold"](seconds=30.0))
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
    rec = EpisodeWriter(tmp_path, env="sim", goal="g", model="boom", skills=list(SKILLS), threaded=False)
    agent = Agent("g", dec, [SKILLS["hold"], SKILLS["done"]], recorder=rec, verdict=lambda: None)
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
