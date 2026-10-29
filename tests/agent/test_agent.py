import json
import math
import time

import numpy as np
import pytest

from g1.agent.agent import Agent
from g1.agent.decider import Decider, ProtocolError
from g1.agent.episode import EpisodeWriter, load_episode
from g1.camera import ClockedCamera
from g1.cli import run
from g1.core.config import CONTROL_DT, STAND_Q
from g1.tools import TOOLS, menu
from g1.tools.control import dry_run
from g1.vlm import Overloaded, QuotaExceeded
from tests.doubles import RedBallDecider, scripted_sim, select


def solid(color=(0, 0, 0), h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def red_square(cx, cy, side):
    img = solid()
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    img[max(0, y0):y0 + side, max(0, x0):x0 + side] = (230, 20, 20)
    return img


class Scripted(ClockedCamera):
    """Black, then a small centred ball, then a looming one. The times follow
    the agent's timeline: a 2 s takeover, then ~2.5 s per decision."""

    def __init__(self, appear=9.0, loom=13.0, fps=10.0):
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


def check(cam=None, **settings):
    return scripted_sim(cam or Scripted(), **settings)


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


def recorder(tmp_path, dec, instruction="g"):
    return EpisodeWriter(tmp_path, env="sim", instruction=instruction, model=dec.model, config={},
                         task="t", mode="standing", max_decisions=30, threaded=False)


def chosen(s):
    """The tool a step's response chose (None for an invalid reply)."""
    return ((s["response"] or {}).get("decision") or {}).get("name")


def status(s):
    return (s["result"] or {}).get("status")


def test_search_finds_the_ball_and_records(tmp_path):
    env = check()
    dec = RedBallDecider()
    rec = recorder(tmp_path, dec, "find the red ball")
    agent = Agent("find the red ball", dec, menu(True), recorder=rec)
    assert run(agent, env, max_time=120) is True
    agent.close()
    assert agent.result == "completed" and env.violations == []
    names = [chosen(s) for s in agent.steps]
    assert names[-1] == "done" and set(names[:-1]) == {"move"}
    assert agent.steps[-2]["action"]["args"]["dx_m"] == 0.5
    assert all(s["action"]["args"]["dx_m"] == 0 for s in agent.steps[:-2])
    assert [status(s) for s in agent.steps] == ["completed"] * (len(names) - 1) + ["done"]
    for a, b in zip(agent.steps, agent.steps[1:]):
        if b["action"] is not None:
            assert a["action"]["cmd_end"] == b["action"]["cmd_start"]          # seeded from the last command
            assert a["action"]["base_cmd_end"] == b["action"]["base_cmd_start"]
        assert b["step"] == a["step"] + 1 and b["t_start"] >= a["t_end"]
    first = agent.steps[0]
    assert first["frame"]["shape"] == [48, 64, 3] and first["response"]["decide_s"] is not None
    assert first["t_start"] <= first["action"]["t_start"] < first["action"]["t_end"] <= first["t_end"]
    assert first["action"]["tick_end"] - first["action"]["tick_start"] == round(first["action"]["duration_s"] / CONTROL_DT)
    assert env.base_path == pytest.approx(0.5) and env.base_pose()[2] > 0.7
    summary, steps = load_episode(rec.dir)
    assert rec.dir.name.endswith("_unreviewed")                              # no human verdict
    assert summary["end_reason"] == "done" and summary["outcome"] == "unreviewed" and summary["model_outcome"] == "success"
    assert summary["steps"] == len(names) and len(steps) == len(names) and summary["tools_used"] == names
    assert np.array_equal(steps[-2]["image"], red_square(32, 24, 5))         # the frame it walked on
    assert chosen(steps[-2]) == "move" and "floor clear" in steps[-2]["response"]["decision"]["arguments"]["note"]
    fb = steps[-2]["result"]["execution_feedback"]                           # the full result, truth included
    assert math.hypot(*fb["base_measured_pose"][:2]) == pytest.approx(0.5, abs=0.15)
    # the base is never commanded while the agent is thinking, so StopMove lands before each frame
    idx = [i for i, a in enumerate(env.actions) if a.base is not None]
    assert all(env.actions[i].base is None for i in range(idx[-1] + 1, len(env.actions)))


def test_observation_and_feedback_follow_the_contract():
    env = check()
    dec = Sequence([("move", {"dyaw_deg": 30}), ("move", {"dx_m": 0.3}),
                    ("done", {"summary": "s", "hindsight": "h"})])
    agent = Agent("find it", dec, menu(True), safety_notes=["never walk backward"], max_decisions=10)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    assert [x["extra"]["env_step"] for x in o] == [0, 1, 2]
    assert o[0]["instruction"] == "find it" and "previous_result" not in o[0]
    assert o[0]["images"] == [{"name": "head", "width": 64, "height": 48, "captured_age_s": o[0]["images"][0]["captured_age_s"]}]
    st = o[0]["state"]
    assert len(st["joint_pos"]) == 29 and len(st["joint_vel"]) == 29 and len(st["joint_torque"]) == 29
    assert st["base_pose_cmd"] == [0.0, 0.0, 0.0]
    assert set(st) == {"joint_pos", "joint_vel", "joint_torque", "waist_yaw_deg", "base_pose_cmd"}   # never the twin's truth
    assert o[0]["extra"] == {"env_step": 0, "decisions_left": 10, "can_walk": True}
    prev = o[1]["previous_result"]
    fb = prev["result"]["execution_feedback"]
    assert prev["tool"] == "move" and prev["result"]["status"] == "completed"
    assert fb["base_target_pose"][2] == pytest.approx(30.0, abs=1e-6)
    assert fb["base_commanded_delta"] == pytest.approx([0.0, 0.0, 30.0], abs=1e-6)
    assert not {"base_measured_pose", "base_measured_source", "base_error"} & set(fb)   # the twin's truth stays in the record
    assert fb["max_joint_residual_rad"] < 0.05 and len(fb["joint_residual_rad"]) == 29
    assert fb["joint_residual_rad"][0] is None                     # legs are not commanded
    assert "motion_progress" not in fb                              # recorded, hidden from the model
    # after a walk the waist at arm_sdk's kp 60 / kd 1.5 is underdamped and may not settle in 3 s: reported, either way
    st = fb["settle"]
    if st["settled"]:
        assert st["max_velocity_rad_s"] <= 0.05 and "required_samples" not in st   # bookkeeping dropped once settled
    else:
        assert st["observed_s"] == pytest.approx(3.0, abs=0.05) and st["required_samples"] == 10   # the 3 s timeout
    assert o[2]["previous_result"]["tool"] == "move"
    assert o[2]["state"]["base_pose_cmd"][0] == pytest.approx(0.3 * math.cos(math.radians(30)), abs=1e-6)
    # the context the decider was started with
    ctx = dec.context
    assert "never walk backward" in ctx.instructions and "Call exactly one tool per turn" in ctx.instructions
    assert [t["function"]["name"] for t in ctx.tools] == [s.name for s in menu(True)]
    assert agent.result == "completed"


def test_a_long_tool_is_one_step(tmp_path):
    """One model call, one continuous motion, one step; the ticks in between are trace.npz."""
    env = check()
    dec = Sequence([("tpose", {}), ("done", {"summary": "s", "hindsight": ""})])     # tpose is 11 s
    rec = recorder(tmp_path, dec)
    agent = Agent("g", dec, menu(True) + [TOOLS["tpose"]], recorder=rec)               # a preset, offered here
    assert run(agent, env, max_time=120) is True
    agent.close()
    tpose = agent.steps[0]
    assert len(agent.steps) == 2 and dec.requests == 2 and chosen(tpose) == "tpose"
    a = tpose["action"]
    assert a["duration_s"] == pytest.approx(11.0, abs=2 * CONTROL_DT) and a["planned_s"] == pytest.approx(11.0)
    assert env.q_max[16] > 1.4 and a["q_end"][16] < 0.5            # raised (1.57, sagging at kp 60), then lowered
    assert {k: v for k, v in a["args"].items() if k != "note"} == {"hold_s": 5.0, "rise_s": 3.0}
    assert status(tpose) == "completed" and tpose["t_end"] >= a["t_end"]   # the result comes after the settle


def test_search_without_base_skills_looks_instead():
    env = check()
    agent = Agent("find the red ball", RedBallDecider(), menu(False), max_decisions=4, can_walk=False)
    assert run(agent, env, max_time=120) is True
    assert env.violations == [] and env.base_ticks == 0
    assert chosen(agent.steps[0]) == "arm_path" and agent.result in ("budget_exhausted", "completed")
    assert "no walking tools" in agent.context.instructions


def test_rejected_and_invalid_replies_are_feedback_and_cost_a_decision():
    dec = Sequence([("raw", "not json at all"),                              # unparseable
                    ("move", {"dx_m": 0.2}),                                # base tool, but cannot walk
                    ("hold", {"seconds": 0.3}),
                    ("give_up", {"reason": "r", "hindsight": "h"})])
    env = check()
    agent = Agent("g", dec, menu(True), can_walk=False, max_decisions=10)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    assert [x["extra"]["env_step"] for x in o] == [0, 1, 2, 3]
    assert o[1]["previous_result"] == {"tool": None, "error": "invalid_selection: invalid selection: not json at all"}
    assert o[2]["previous_result"]["tool"] == "move" and "tool_rejected" in o[2]["previous_result"]["error"]
    assert o[3]["previous_result"]["tool"] == "hold" and o[3]["previous_result"]["result"]["status"] == "completed"
    assert [status(s) for s in agent.steps] == ["rejected", "rejected", "completed", "give_up"]
    assert agent.steps[0]["response"]["text"] == "not json at all" and agent.steps[0]["action"] is None
    assert agent.steps[1]["result"]["tool"] == "move" and agent.steps[1]["action"] is None
    assert agent.result == "give_up" and env.base_ticks == 0


def test_check_dry_runs_without_moving():
    dec = Sequence([("check", {"tool": "move", "arguments": {"dx_m": 1.0}}),
                    ("check", {"tool": "move", "arguments": {"dyaw_deg": "fast"}}),
                    ("done", {"summary": "s", "hindsight": ""})])
    env = check()
    agent = Agent("g", dec, menu(True), max_decisions=10)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    r = o[1]["previous_result"]
    assert r["tool"] == "check" and r["result"]["status"] == "ok" and r["result"]["duration_s"] == pytest.approx(5.0)
    assert r["result"]["base_delta"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6) and r["result"]["violations"] == []
    assert "tool_rejected" in o[2]["previous_result"]["error"]      # bad inner arguments
    assert env.base_ticks == 0 and [status(s) for s in agent.steps] == ["ok", "rejected", "done"]
    assert agent.steps[0]["result"]["base_delta"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)   # the answer, in full
    verdict = dry_run(TOOLS["turn"](angle_deg=90, note="n"), STAND_Q.copy(), agent.joints)
    assert verdict["status"] == "ok" and verdict["base_delta"][2] == pytest.approx(90.0, abs=1e-6)


def test_too_fast_or_out_of_reach_motion_is_not_executed():
    """Their IK check before submission: a violating plan is rejected before anything moves."""
    fast = {"waypoints": [{"joints": {"left_elbow": 2.0}, "seconds": 0.5}, {"joints": {"left_elbow": -1.0}, "seconds": 0.5}]}
    dec = Sequence([("arm_path", fast), ("done", {"summary": "s", "hindsight": ""})])
    env = check()
    agent = Agent("g", dec, menu(True), max_decisions=5)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    err = o[1]["previous_result"]
    assert err["tool"] == "arm_path" and err["error"] == "motion_not_executed" and err["n_violations"] > 0
    assert "velocity" in err["violations"][0] and err["requested_motion"] == fast
    assert status(agent.steps[0]) == "rejected" and env.cmd_max[18] == pytest.approx(1.28, abs=0.01)   # the observed pose, held
    assert env.violations == []


def test_gestures_need_the_onboard_controller(capsys):
    dec = Sequence([("wave_hand", {}), ("done", {"summary": "s", "hindsight": ""})])
    env = check()
    agent = Agent("g", dec, menu(True, True), max_decisions=5, has_loco=False)
    assert run(agent, env, max_time=120) is True
    assert "no onboard controller" in observations(dec)[1]["previous_result"]["error"] and agent.result == "completed"
    env = check()
    agent = Agent("g", Sequence([("wave_hand", {"seconds": 1.0}), ("done", {"summary": "s", "hindsight": ""})]),
                  menu(True, True), max_decisions=5, has_loco=True)
    assert run(agent, env, max_time=120) is True                      # the twin plays the onboard gesture
    assert agent.result == "completed" and any(c["api"] == 7106 for c in env.twin.services.log)
    agent.close()


def test_seated_robot_walks_no_more():
    dec = Sequence([("sit", {}), ("move", {"dx_m": 0.2}), ("wave_hand", {}), ("hold", {"seconds": 0.3}),
                    ("done", {"summary": "s", "hindsight": ""})])
    env = check(mode="standing", scene="room", sim_objects=[("chair", -0.37, 0.0, None)])
    agent = Agent("sit", dec, menu(True, True), max_decisions=8, has_loco=True)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    assert "seated" in o[2]["previous_result"]["error"] and "seated" in o[3]["previous_result"]["error"]
    assert agent.result == "completed" and "previous_result" in o[1] and "error" not in o[1]["previous_result"]
    assert env.base_ticks == 0 and "seated on chair1" in env.twin.summary()
    agent.close()


def test_unsettled_pose_is_reported():
    dec = Sequence([("arm_path", {"waypoints": [{"joints": {"waist_yaw": 0.7}, "seconds": 1.5}]}),
                    ("done", {"summary": "s", "hindsight": ""})])
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
    assert agent.result == "failed" and "timeout" in agent.error and agent.end_reason == "step_timeout"
    assert [status(s) for s in agent.steps] == ["timeout"] and agent.steps[0]["response"]["status"] == "timeout"


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
    assert [s["attempt"] for s in agent.steps] == [0, 1, 2] and [status(s) for s in agent.steps] == ["retry", "retry", "done"]
    assert "busy" in agent.steps[0]["response"]["error"]          # each retry is a step of its own
    env = check()
    dec = Flaky([Overloaded(429, "busy")] * 3)
    agent = Agent("g", dec, menu(True), retry_delays=(0.01, 0.01))
    assert run(agent, env, max_time=120) is True
    assert agent.result == "failed" and "retry_limit" in agent.error and agent.end_reason == "model_overloaded"
    env = check()
    agent = Agent("g", Flaky([QuotaExceeded(402, "no credits")]), menu(True))
    assert run(agent, env, max_time=120) is True
    assert agent.result == "failed" and "quota" in agent.error and agent.end_reason == "quota"


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
    assert agent.end_reason == "no_frame" and agent.steps[0]["frame"] is None
    assert agent.steps[0]["frame_error"].startswith("frame_unavailable")


def test_run_files_and_human_verdict(tmp_path):
    env = check()
    dec = Sequence([("move", {"dyaw_deg": 20}), ("raw", "junk"), ("done", {"summary": "s", "hindsight": "h"})])
    rec = recorder(tmp_path, dec)
    agent = Agent("g", dec, menu(True), recorder=rec, verdict=lambda: "success")
    assert run(agent, env, max_time=120) is True
    agent.close()
    d = rec.dir
    assert d.name.endswith("_success")
    names = {p.name for p in d.iterdir()}
    # exactly these: a new file in a run is a deliberate change (the env's trace.npz is off here)
    steps = {f"step_{i:04d}.{x}" for i in (1, 2, 3) for x in ("json", "png")}
    assert names == {"summary.json", "config.json"} | steps, names
    summary, steps = load_episode(d, images=False)
    assert [status(s) for s in steps] == ["completed", "rejected", "done"]
    assert summary["outcome"] == "success" and summary["outcome_source"] == "human" and summary["model_outcome"] == "success"
    assert summary["usage"]["calls"] == 3 and summary["usage"]["failed_calls"] == 1
    assert summary["decisions"] == {"used": 3, "max": 30} and summary["tools_used"] == ["move", "(invalid)", "done"]
    assert summary["closing"] == {"summary": "s", "hindsight": "h"}
    first = steps[0]
    # the query: the observation exactly as sent (the system prompt is config.json's)
    assert first["query"][-1]["content"] == dec.turns[0].observation
    assert json.loads(first["query"][-1]["content"])["instruction"] == "g"
    assert first["response"]["status"] == "completed" and first["response"]["decision"]["name"] == "move"
    assert first["action"]["tool"] == "move" and first["action"]["dry_run"]["status"] == "ok"
    assert "motion_progress" in first["result"]["execution_feedback"]        # the full result is recorded
    assert steps[1]["response"]["status"] == "failed" and steps[1]["response"]["text"] == "junk"
    sdk = summary["sdk_outside_steps"]                                       # the bring-up's GetVolume, GetFsmId
    assert all("name" in c and "code" in c for c in sdk)


def test_give_up_and_skipped_verdict(tmp_path):
    env = check()
    dec = Sequence([("give_up", {"reason": "r", "hindsight": "h"})])
    rec = recorder(tmp_path, dec)
    agent = Agent("g", dec, menu(True), recorder=rec, verdict=lambda: None)
    assert run(agent, env, max_time=120) is True
    agent.close()
    assert agent.result == "give_up" and rec.dir.name.endswith("_unreviewed")
    summary = json.loads((rec.dir / "summary.json").read_text())
    assert summary["model_outcome"] == "give_up" and summary["end_reason"] == "give_up"
    assert summary["closing"] == {"reason": "r", "hindsight": "h"} and summary["outcome_source"] == "unreviewed"


@pytest.mark.parametrize("end", [("done", {"summary": "s", "hindsight": ""}), ("give_up", {"reason": "r", "hindsight": ""})])
def test_a_point_stays_up_for_the_next_image_and_is_lowered_at_the_end(end):
    from g1.core.poses import STAND
    from g1.tools.arms import R_ARM
    dec = Sequence([("point", {"down_deg": 20.0, "rise_s": 1.0, "hold_s": 0.5}), ("move", {"dx_m": 0.2}), end])
    env = check(mode="standing")
    agent = Agent("point at it", dec, menu(True), max_decisions=5)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    shoulder = R_ARM[0]
    assert o[1]["state"]["joint_pos"][shoulder] < -1.0                       # the next observation: the arm still raised
    settle = o[1]["previous_result"]["result"]["execution_feedback"]["settle"]
    assert settle["settled"] is True and "raised right arm" in settle["position_excludes"]   # its sag is expected
    assert "lower_arm" in o[2]["previous_result"]["error"] and env.base_ticks == 0             # no walking while pointing
    assert agent._cmd[R_ARM] == pytest.approx([STAND[j] for j in R_ARM])                     # lowered before the hand-back
    agent.close()


def test_lower_arm_ends_the_point():
    dec = Sequence([("point", {"down_deg": 20.0, "rise_s": 1.0, "hold_s": 0.5}), ("lower_arm", {"seconds": 1.0}),
                    ("move", {"dyaw_deg": 10}), ("done", {"summary": "s", "hindsight": ""})])
    env = check(mode="standing")
    agent = Agent("point at it", dec, menu(True), max_decisions=6)
    assert run(agent, env, max_time=120) is True
    o = observations(dec)
    assert "error" not in o[3]["previous_result"] and env.base_ticks > 0      # walking allowed again
    agent.close()
