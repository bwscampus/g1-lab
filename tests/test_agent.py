import math
import time

import numpy as np
import pytest

from agent import AGENTS, Agent, build_replay
from camera import ClockedCamera
from config import CONTROL_DT, STAND_Q
from decider import Decider, Decision
from episode import EpisodeWriter, chain_of, load_episode
from run import build_parser, main, run
from skills import SKILLS, STEP_MAX, menu
from tests.doubles import RedBallDecider, scripted_sim, sim_env


def solid(color=(0, 0, 0), h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def red_square(cx, cy, side):
    img = solid()
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    img[max(0, y0):y0 + side, max(0, x0):x0 + side] = (230, 20, 20)
    return img


class Scripted(ClockedCamera):
    """Black, then a small centred ball, then a looming one."""

    def __init__(self, appear=12.0, loom=16.0, fps=10.0):
        super().__init__(fps)
        self.appear, self.loom = appear, loom

    def image(self, i):
        t = i / self.fps
        if t < self.appear:
            return solid()
        if t < self.loom:
            return red_square(32, 24, 5)
        return red_square(32, 30, 14)


class NoFrames(ClockedCamera):
    def image(self, i):
        return None


def check(cam=None, *extra):
    return scripted_sim(cam or Scripted(), *extra)


class Sequence(Decider):
    """Inline decider that returns the given (action, args) per call."""

    model = "sequence"

    def __init__(self, plan):
        super().__init__(threaded=False)
        self.plan = list(plan)
        self.i = 0

    def decide(self, ctx):
        action, args = self.plan[min(self.i, len(self.plan) - 1)]
        self.i += 1
        return Decision.from_json({"scene": f"call {self.i}", "path_clear": True, "found": False,
                                   "action": action, "args": args, "reason": "scripted"}, ctx)


def test_search_finds_the_ball_and_records(tmp_path):
    pytest.importorskip("cv2")
    env = check()
    dec = RedBallDecider()
    rec = EpisodeWriter(tmp_path, env="sim", goal="find the red ball", model=dec.model,
                        skills=list(SKILLS), threaded=False)
    agent = Agent("find the red ball", dec, menu(True), recorder=rec)
    assert run(agent, env, max_time=120) is True
    agent.close()
    assert agent.result == "found" and env.violations == []
    names = [s.skill["name"] for s in agent.steps]
    assert names == ["turn", "turn", "turn", "walk_forward", "done"]
    assert [s.outcome["status"] for s in agent.steps] == ["completed"] * 4 + ["done"]
    for a, b in zip(agent.steps, agent.steps[1:]):
        assert a.cmd_end == b.cmd_start                       # seeded from the last command
        assert a.base_pose["cmd_end"] == b.base_pose["cmd_start"]
        assert b.step == a.step + 1 and b.t["policy_start"] > a.t["policy_end"]
    assert agent.steps[0].frame["shape"] == [48, 64, 3] and agent.steps[0].t["think_wall"] is not None
    assert env.base_path == pytest.approx(0.5) and env.base_pose()[2] > 2.0   # 3 x 45 deg, then 0.5 m
    meta, steps = load_episode(rec.dir)
    assert meta["result"] == "found" and meta["steps"] == 5 and len(steps) == 5
    assert np.array_equal(steps[3].image, red_square(32, 24, 5))              # the frame it walked on
    assert steps[3].decision["action"] == "walk_forward" and steps[3].decision["model"] == "red-ball-rules"
    assert steps[0].base_pose["env_start"] == [0.0, 0.0, 0.0]
    assert math.hypot(*steps[4].base_pose["env_end"][:2]) == pytest.approx(0.5, abs=1e-6)
    # the base is never commanded while the agent is thinking, so StopMove lands before each frame
    idx = [i for i, a in enumerate(env.actions) if a.base is not None]
    assert all(env.actions[i].base is None for i in range(idx[-1] + 1, len(env.actions)))


def test_long_skill_runs_as_three_second_chunks(tmp_path):
    """One model call, one continuous motion, a record every STEP_MAX seconds."""
    pytest.importorskip("cv2")
    env = check()
    dec = Sequence([("tpose", {}), ("done", {"found": True})])     # tpose is 11 s
    rec = EpisodeWriter(tmp_path, env="sim", goal="g", model=dec.model, skills=list(SKILLS),
                        threaded=False)
    agent = Agent("g", dec, menu(True), recorder=rec)
    assert run(agent, env, max_time=120) is True
    agent.close()
    tpose = [s for s in agent.steps if s.skill["name"] == "tpose"]
    assert len(tpose) == 4 and dec.requests == 2                   # 11 s / 3 s, and only two calls
    assert [s.skill["chunk"] for s in tpose] == [1, 2, 3, 4]
    assert all(s.skill["chunks"] == 4 for s in tpose)
    assert [s.outcome["status"] for s in tpose] == ["running", "running", "running", "completed"]
    for s in tpose[:3]:
        assert s.outcome["duration"] == pytest.approx(STEP_MAX, abs=CONTROL_DT)
    assert all(s.decision["action"] == "tpose" for s in tpose)     # the same decision throughout
    for a, b in zip(tpose, tpose[1:]):
        assert a.cmd_end == b.cmd_start and a.t["policy_end"] == b.t["policy_start"]
    assert tpose[1].t["think_wall"] is None                        # only the first chunk thought
    # the arms never pause at a chunk boundary: the joint keeps moving through it
    arm = [s.q_end[16] for s in tpose]
    assert arm[0] < arm[1] and max(arm) > 1.5
    _, steps = load_episode(rec.dir, images=False)
    assert chain_of(steps) == "tpose:hold=5.0:rise=3.0"             # one entry for the whole skill


def test_search_without_base_skills_looks_instead():
    env = check()
    agent = Agent("find the red ball", RedBallDecider(), menu(False), max_steps=6)
    assert run(agent, env, max_time=120) is True
    assert env.violations == [] and env.base_ticks == 0
    assert agent.steps[0].skill["name"] == "look" and agent.result in ("max_steps", "found")


class Never(Decider):
    model = "never"

    def decide(self, ctx):
        time.sleep(30)
        return None


def test_decision_timeout_then_error():
    env = check()
    agent = Agent("g", Never(), menu(True), step_timeout=0.05, max_failures=2)
    assert run(agent, env, max_time=120) is True
    agent.close()
    assert agent.result == "error"
    assert [s.outcome["status"] for s in agent.steps] == ["timeout", "timeout"]
    assert all(s.decision is None for s in agent.steps)


class FlakyThenOk(Decider):
    model = "flaky"

    def __init__(self):
        super().__init__(threaded=False)
        self.n = 0

    def decide(self, ctx):
        self.n += 1
        if self.n == 1:
            raise ValueError("unknown action 'fly'")
        assert "rejected" in ctx.note
        return Decision.from_json({"action": "done", "args": {"found": False}, "scene": "s"}, ctx)


def test_invalid_reply_is_retried_with_a_note():
    env = check()
    agent = Agent("g", FlakyThenOk(), menu(True))
    assert run(agent, env, max_time=120) is True
    assert agent.result == "not_found" and agent.steps[-1].decision["retries"] == 1


def test_max_steps_and_no_frame():
    env = check()
    agent = Agent("g", Sequence([("hold", {"seconds": 0.5})]), menu(True), max_steps=2)
    assert run(agent, env, max_time=120) is True
    assert agent.result == "max_steps" and len(agent.steps) == 2
    env = check(NoFrames())
    agent = Agent("g", Sequence([("hold", {"seconds": 0.5})]), menu(True), max_failures=1)
    assert run(agent, env, max_time=120) is True
    assert agent.result == "error" and agent.steps[0].outcome["status"] == "no_frame"


def test_replay_reproduces_a_recorded_run(tmp_path):
    pytest.importorskip("cv2")
    env = check()
    rec = EpisodeWriter(tmp_path, env="sim", goal="g", model="rules", skills=list(SKILLS), threaded=False)
    agent = Agent("g", RedBallDecider(), menu(True), recorder=rec)
    run(agent, env, max_time=120)
    agent.close()
    args = build_parser().parse_args(["--env", "sim", "--headless", "--policy", "replay",
                                      "--episode", str(rec.dir)])
    policy = build_replay(args, True)
    assert policy.name.startswith("replay:")
    assert [s.name for s in policy.skills] == ["turn", "turn", "turn", "walk_forward"]
    env2 = sim_env()
    assert run(policy, env2) is True and env2.violations == []
    assert env2.base_path == pytest.approx(env.base_path)
    assert env2.base_pose()[2] == pytest.approx(env.base_pose()[2])
    with pytest.raises(ValueError):
        build_replay(build_parser().parse_args(["--env", "sim", "--policy", "replay"]), True)


def test_cli_search_needs_goal_and_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("G1_VISION_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["--env", "sim", "--headless", "--policy", "search", "--no-log"])
    assert "needs --goal" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--env", "sim", "--headless", "--policy", "search", "--goal", "x", "--no-log"])
    assert "HF_TOKEN" in capsys.readouterr().err
    assert main(["--list"]) == 0
    assert "agents:   replay, search" in capsys.readouterr().out
    assert set(AGENTS) == {"search", "replay"}
