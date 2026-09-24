import json
import math
import time

import numpy as np
import pytest

from agent import AGENTS, Agent, build_replay, dry_run
from camera import ClockedCamera
from config import CONTROL_DT, STAND_Q
from decider import Decider, ProtocolError
from episode import EpisodeWriter, chain_of, load_episode
from hf import Overloaded, QuotaExceeded
from run import build_parser, main, run
from skills import SKILLS, STEP_MAX, menu
from tests.doubles import RedBallDecider, scripted_sim, select, sim_env


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
    """Inline decider that returns the given (name, arguments) per call and
    keeps every turn it was asked."""

    model = "sequence"

    def __init__(self, plan):
        super().__init__(threaded=False)
        self.plan = list(plan)
        self.i = 0
        self.turns = []

    def decide(self, turn):
        self.turns.append(turn)
        name, args = self.plan[min(self.i, len(self.plan) - 1)]
        self.i += 1
        if name == "raw":
            raise ProtocolError("invalid selection: " + args, raw=args)
        if isinstance(name, type) and issubclass(name, Exception):
            raise name(429, args) if issubclass(name, Exception) and name is not RuntimeError else name(args)
        args = dict(args)
        if name not in ("done", "give_up"):
            args.setdefault("note", "scripted")
        return select(self, turn, name, args)


def observations(dec):
    return [json.loads(t.observation) for t in dec.turns]


def recorder(tmp_path, dec, goal="g"):
    return EpisodeWriter(tmp_path, env="sim", goal=goal, model=dec.model, skills=list(SKILLS), threaded=False)


def test_search_finds_the_ball_and_records(tmp_path):
    pytest.importorskip("cv2")
    env = check()
    dec = RedBallDecider()
    rec = recorder(tmp_path, dec, "find the red ball")
    agent = Agent("find the red ball", dec, menu(True), recorder=rec)
    assert run(agent, env, max_time=120) is True
    agent.close()
    assert agent.result == "completed" and env.violations == []
    names = [s.skill["name"] for s in agent.steps]
    assert names[-2:] == ["walk_forward", "done"] and set(names[:-2]) == {"turn"}
    assert [s.outcome["status"] for s in agent.steps] == ["completed"] * (len(names) - 1) + ["done"]
    for a, b in zip(agent.steps, agent.steps[1:]):
        assert a.cmd_end == b.cmd_start                       # seeded from the last command
        assert a.base_pose["cmd_end"] == b.base_pose["cmd_start"]
        assert b.step == a.step + 1 and b.t["policy_start"] > a.t["policy_end"]
    assert agent.steps[0].frame["shape"] == [48, 64, 3] and agent.steps[0].t["think_wall"] is not None
    assert env.base_path == pytest.approx(0.5) and env.base_pose()[2] > 0.7
    meta, steps = load_episode(rec.dir)
    assert rec.dir.name.endswith("_unreviewed")                              # no human verdict
    assert meta["result"] == "completed" and meta["outcome"] == "unreviewed" and meta["model_outcome"] == "success"
    assert meta["steps"] == len(names) and len(steps) == len(names)
    assert np.array_equal(steps[-2].image, red_square(32, 24, 5))             # the frame it walked on
    assert steps[-2].decision["name"] == "walk_forward" and steps[-2].decision["model"] == "red-ball-rules"
    assert "floor clear" in steps[-2].decision["arguments"]["note"]
    assert steps[0].base_pose["env_start"] == [0.0, 0.0, 0.0]
    assert math.hypot(*steps[-1].base_pose["env_end"][:2]) == pytest.approx(0.5, abs=1e-6)
    # the base is never commanded while the agent is thinking, so StopMove lands before each frame
    idx = [i for i, a in enumerate(env.actions) if a.base is not None]
    assert all(env.actions[i].base is None for i in range(idx[-1] + 1, len(env.actions)))


def test_observation_and_feedback_follow_the_contract():
    pytest.importorskip("cv2")
    env = check()
    dec = Sequence([("turn", {"angle_deg": 30}), ("walk_forward", {"distance_m": 0.3}),
                    ("done", {"summary": "s", "hindsight": "h"})])
    agent = Agent("find it", dec, menu(True), safety_notes=["a wall behind"], max_decisions=10)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    assert [x["extra"]["env_step"] for x in o] == [0, 1, 2]
    assert o[0]["instruction"] == "find it" and "previous_result" not in o[0]
    assert o[0]["images"] == [{"name": "head", "width": 64, "height": 48, "captured_age_s": o[0]["images"][0]["captured_age_s"]}]
    st = o[0]["state"]
    assert len(st["joint_pos"]) == 29 and len(st["joint_vel"]) == 29 and len(st["joint_torque"]) == 29
    assert st["base_pose_cmd"] == [0.0, 0.0, 0.0] and st["base_pose_env"] == [0.0, 0.0, 0.0]
    assert o[0]["extra"] == {"env_step": 0, "decisions_left": 10, "can_walk": True}
    prev = o[1]["previous_result"]
    fb = prev["result"]["execution_feedback"]
    assert prev["tool"] == "turn" and prev["result"]["status"] == "completed"
    assert fb["base_target_pose"][2] == pytest.approx(30.0, abs=1e-6)
    assert fb["base_measured_source"] == "env" and max(abs(v) for v in fb["base_error"]) < 1e-6
    assert fb["max_joint_residual_rad"] < 0.05 and len(fb["joint_residual_rad"]) == 29
    assert fb["joint_residual_rad"][0] is None                     # legs are not commanded
    assert "motion_progress" not in fb                              # recorded, hidden from the model
    assert fb["settle"]["settled"] is True and fb["settle"]["max_velocity_rad_s"] <= 0.05
    assert "required_samples" not in fb["settle"]                   # bookkeeping dropped once settled
    assert o[2]["previous_result"]["tool"] == "walk_forward"
    assert o[2]["state"]["base_pose_cmd"][0] == pytest.approx(0.3 * math.cos(math.radians(30)), abs=1e-6)
    # the context the decider was started with
    ctx = dec.context
    assert "a wall behind" in ctx.instructions and "Robot skill catalog:" in ctx.instructions
    assert [t["function"]["name"] for t in ctx.tools] == [s.name for s in menu(True)]
    assert ctx.output_schema["properties"]["name"]["enum"] == [s.name for s in menu(True)]
    assert dec.turns[0].content == () and agent.result == "completed"


def test_long_skill_runs_as_three_second_chunks(tmp_path):
    """One model call, one continuous motion, a record every STEP_MAX seconds."""
    pytest.importorskip("cv2")
    env = check()
    dec = Sequence([("tpose", {}), ("done", {"summary": "s", "hindsight": ""})])     # tpose is 11 s
    rec = recorder(tmp_path, dec)
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
    assert all(s.decision["name"] == "tpose" for s in tpose)       # the same decision throughout
    for a, b in zip(tpose, tpose[1:]):
        assert a.cmd_end == b.cmd_start and a.t["policy_end"] == b.t["policy_start"]
    assert tpose[1].t["think_wall"] is None                        # only the first chunk thought
    # the arms never pause at a chunk boundary: the joint keeps moving through it
    arm = [s.q_end[16] for s in tpose]
    assert arm[0] < arm[1] and max(arm) > 1.5
    _, steps = load_episode(rec.dir, images=False)
    assert chain_of(steps) == "tpose:hold=5.0:rise=3.0"             # one entry for the whole skill, no note


def test_search_without_base_skills_looks_instead():
    env = check()
    agent = Agent("find the red ball", RedBallDecider(), menu(False), max_decisions=4, can_walk=False)
    assert run(agent, env, max_time=120) is True
    assert env.violations == [] and env.base_ticks == 0
    assert agent.steps[0].skill["name"] == "look" and agent.result in ("budget_exhausted", "completed")
    assert "no walking skills" in agent.context.instructions


def test_rejected_and_invalid_replies_are_feedback_and_cost_a_decision():
    dec = Sequence([("raw", "not json at all"),                              # unparseable
                    ("walk_forward", {"distance_m": 0.2}),                  # base skill, but cannot walk
                    ("hold", {"seconds": 0.3}),
                    ("give_up", {"reason": "r", "hindsight": "h"})])
    env = check()
    agent = Agent("g", dec, menu(True), can_walk=False, max_decisions=10)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    assert [x["extra"]["env_step"] for x in o] == [0, 1, 2, 3]
    assert o[1]["previous_result"] == {"tool": None, "error": "invalid_selection: invalid selection: not json at all"}
    assert o[2]["previous_result"]["tool"] == "walk_forward" and "tool_rejected" in o[2]["previous_result"]["error"]
    assert o[3]["previous_result"]["tool"] == "hold" and o[3]["previous_result"]["result"]["status"] == "completed"
    assert [s.outcome["status"] for s in agent.steps] == ["rejected", "rejected", "completed", "give_up"]
    assert agent.steps[0].decision["raw"] == "not json at all" and agent.steps[0].skill is None
    assert agent.result == "give_up" and env.base_ticks == 0


def test_check_dry_runs_without_moving():
    dec = Sequence([("check", {"skill": "walk_forward", "arguments": {"distance_m": 1.0}}),
                    ("check", {"skill": "turn", "arguments": {"angle_deg": "fast"}}),
                    ("done", {"summary": "s", "hindsight": ""})])
    env = check()
    agent = Agent("g", dec, menu(True), max_decisions=10)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    r = o[1]["previous_result"]
    assert r["tool"] == "check" and r["result"]["status"] == "ok" and r["result"]["duration_s"] == pytest.approx(5.0)
    assert r["result"]["base_delta"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6) and r["result"]["violations"] == []
    assert "tool_rejected" in o[2]["previous_result"]["error"]      # bad inner arguments
    assert env.base_ticks == 0 and [s.outcome["status"] for s in agent.steps] == ["checked", "rejected", "done"]
    verdict = dry_run(SKILLS["turn"](angle_deg=90, note="n"), STAND_Q.copy(), agent.joints)
    assert verdict["status"] == "ok" and verdict["base_delta"][2] == pytest.approx(90.0, abs=1e-6)


def test_unsettled_pose_is_reported():
    dec = Sequence([("look", {"yaw_deg": 40}), ("done", {"summary": "s", "hindsight": ""})])
    env = check()
    agent = Agent("g", dec, menu(True), settle_tol=1e-9, settle_samples=5, settle_timeout=1.0)
    assert run(agent, env, max_time=120) is True
    settle = observations(dec)[1]["previous_result"]["result"]["execution_feedback"]["settle"]
    assert settle["settled"] is False and settle["observed_s"] == pytest.approx(1.0, abs=CONTROL_DT)
    assert settle["required_samples"] == 5 and settle["max_position_error_rad"] > 1e-9


class Never(Decider):
    model = "never"

    def decide(self, turn):
        time.sleep(30)
        return None


def test_decision_timeout_fails_the_run():
    env = check()
    agent = Agent("g", Never(), menu(True), step_timeout=0.05)
    assert run(agent, env, max_time=120) is True
    agent.close()
    assert agent.result == "failed" and "timeout" in agent.error and agent.steps == []


def test_overload_retries_reobserve_then_quota_fails():
    class Flaky(Decider):
        model = "flaky"

        def __init__(self, errors):
            super().__init__(threaded=False)
            self.faults = list(errors)
            self.turns = []

        def decide(self, turn):
            self.turns.append(turn)
            if self.faults:
                raise self.faults.pop(0)
            return select(self, turn, "done", {"summary": "s", "hindsight": ""})

    env = check()
    dec = Flaky([Overloaded(429, "busy"), Overloaded(503, "overloaded")])
    agent = Agent("g", dec, menu(True), retry_delays=(0.01, 0.01, 0.01))
    assert run(agent, env, max_time=120) is True
    assert agent.result == "completed"
    o = observations(dec)
    assert [x["extra"].get("attempt") for x in o] == [None, 1, 2] and [x["extra"]["env_step"] for x in o] == [0, 0, 0]
    assert dec.turns[0].frame_seq != dec.turns[1].frame_seq       # every retry re-observes
    env = check()
    dec = Flaky([Overloaded(429, "busy")] * 3)
    agent = Agent("g", dec, menu(True), retry_delays=(0.01, 0.01))
    assert run(agent, env, max_time=120) is True
    assert agent.result == "failed" and "retry_limit" in agent.error
    env = check()
    agent = Agent("g", Flaky([QuotaExceeded(402, "no credits")]), menu(True))
    assert run(agent, env, max_time=120) is True
    assert agent.result == "failed" and "quota" in agent.error


def test_budget_and_no_frame():
    env = check()
    dec = Sequence([("hold", {"seconds": 0.5})])
    agent = Agent("g", dec, menu(True), max_decisions=2)
    assert run(agent, env, max_time=120) is True
    assert agent.result == "budget_exhausted" and len(agent.steps) == 2 and dec.requests == 2
    env = check(NoFrames())
    dec = Sequence([("hold", {"seconds": 0.5})])
    agent = Agent("g", dec, menu(True), max_failures=2, frame_timeout=0.5)
    assert run(agent, env, max_time=120) is True
    assert agent.result == "failed" and agent.error == "no_frame"
    o = observations(dec)                                            # observed once without an image
    assert len(o) == 1 and o[0]["images"] == [] and o[0]["previous_result"]["error"].startswith("frame_unavailable")
    assert agent.steps[-1].outcome["status"] == "no_frame"


def test_run_files_and_human_verdict(tmp_path):
    pytest.importorskip("cv2")
    env = check()
    dec = Sequence([("turn", {"angle_deg": 20}), ("raw", "junk"), ("done", {"summary": "s", "hindsight": "h"})])
    rec = recorder(tmp_path, dec)
    agent = Agent("g", dec, menu(True), recorder=rec, verdict=lambda: "success")
    assert run(agent, env, max_time=120) is True
    agent.close()
    d = rec.dir
    assert d.name.endswith("_success")
    names = {p.name for p in d.iterdir()}
    assert {"config.json", "events.jsonl", "transcript.json", "protocol.json", "states.jsonl", "usage.jsonl",
            "usage.json", "status.json", "episode.json", "step_0001.json", "step_0001.png"} <= names
    events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[:3] == ["run_started", "protocol", "observation"]
    assert kinds.count("observation") == 3 and kinds.count("model_decision") == 2 and kinds.count("tool_error") == 1
    assert {"decision_timing", "tool_timing", "execution_result", "terminal", "return_home", "execution_finished",
            "human_evaluation", "run_finished"} <= set(kinds)
    obs = next(e for e in events if e["event"] == "observation")
    assert json.loads(obs["input_json"])["instruction"] == "g" and obs["images"][0]["path"] == "step_0001.png"
    res = next(e for e in events if e["event"] == "execution_result")
    assert "motion_progress" in res["result"]["execution_feedback"]        # the full result is recorded
    fin = events[-1]
    assert fin["outcome"] == "success" and fin["outcome_source"] == "human" and fin["model_outcome"] == "success"
    tr = json.loads((d / "transcript.json").read_text())
    assert [m["role"] for m in tr][:4] == ["system", "user", "assistant", "tool"]
    assert tr[0]["content"] == agent.context.instructions
    proto = json.loads((d / "protocol.json").read_text())
    assert proto["output_schema"] == agent.context.output_schema and proto["menu"] == [s.name for s in menu(True)]
    status = json.loads((d / "status.json").read_text())
    assert status["state"] == "completed" and status["human_outcome"] == "success" and status["usage"]["calls"] == 3
    states = [json.loads(l) for l in (d / "states.jsonl").read_text().splitlines()]
    assert len(states) > 20 and len(states[0]["q"]) == 29
    assert (states[-1]["t"] - states[0]["t"]) / (len(states) - 1) == pytest.approx(0.05, abs=0.002)   # 20 Hz
    usage = [json.loads(l) for l in (d / "usage.jsonl").read_text().splitlines()]
    assert [u["status"] for u in usage] == ["completed", "failed", "completed"] and usage[0]["model"] == "sequence"
    meta, steps = load_episode(d, images=False)
    assert [s.outcome["status"] for s in steps] == ["completed", "rejected", "done"]


def test_give_up_and_skipped_verdict(tmp_path):
    env = check()
    dec = Sequence([("give_up", {"reason": "r", "hindsight": "h"})])
    rec = recorder(tmp_path, dec)
    agent = Agent("g", dec, menu(True), recorder=rec, verdict=lambda: None)
    assert run(agent, env, max_time=120) is True
    agent.close()
    assert agent.result == "give_up" and rec.dir.name.endswith("_unreviewed")
    events = [json.loads(l) for l in (rec.dir / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "human_evaluation_skipped" for e in events)
    assert events[-1]["model_outcome"] == "give_up" and events[-1]["status"] == "give_up"


def test_replay_reproduces_a_recorded_run(tmp_path):
    pytest.importorskip("cv2")
    env = check()
    rec = recorder(tmp_path, RedBallDecider())
    agent = Agent("g", RedBallDecider(), menu(True), recorder=rec)
    run(agent, env, max_time=120)
    agent.close()
    args = build_parser().parse_args(["--env", "sim", "--headless", "--policy", "replay",
                                      "--episode", str(rec.dir)])
    policy = build_replay(args, True)
    assert policy.name.startswith("replay:")
    names = [s.name for s in policy.skills]
    assert names[-1] == "walk_forward" and set(names[:-1]) == {"turn"}
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
    out = capsys.readouterr().out
    assert "agents:   replay, search" in out and "check(skill: enum" in out
    assert set(AGENTS) == {"search", "replay"}
    with pytest.raises(SystemExit):
        main(["--list", "--skills", "nope.json"])
    assert "skill catalog not found" in capsys.readouterr().err
