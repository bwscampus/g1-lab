import json

import numpy as np
import pytest

from decider import (AgentTurn, Decision, HFDecider, ProtocolError, build_context, instructions, observation,
                     parse_selection)
from hf import HFClient, Overloaded, QuotaExceeded, RequestError, classify
from skills import CATALOG, menu
from tests.doubles import RedBallDecider


def solid(color=(0, 0, 0), h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def ctx(allow_base=True, **kw):
    return build_context(menu(allow_base), can_walk=allow_base, max_decisions=30, **kw)


def obs_text(waist=0.0, step=3, previous=None):
    state = {"joint_pos": [0.0] * 29, "joint_vel": [0.0] * 29, "joint_torque": None, "waist_yaw_deg": waist,
             "base_pose_cmd": [0.3, -0.1, 12.0], "base_pose_env": None}
    return observation("find the red ball", state, [{"name": "head", "width": 64, "height": 48, "captured_age_s": 0.02}],
                       {"env_step": step, "decisions_left": 27, "can_walk": True}, previous)


def turn(image=None, waist=0.0, step=3, request_id=7):
    return AgentTurn(obs_text(waist, step), {"head": solid() if image is None else image},
                     request_id=request_id, step=step, frame_seq=9, frame_stamp=1.5)


def test_observation_is_compact_rounded_json():
    text = observation("g", {"joint_pos": np.array([0.12345678, np.inf]), "n": np.float64(1.0)}, [],
                       {"env_step": 0}, {"tool": "turn", "error": "x"})
    assert text == '{"instruction":"g","images":[],"state":{"joint_pos":[0.123457,null],"n":1.0},"extra":{"env_step":0},"previous_result":{"tool":"turn","error":"x"}}'
    assert "previous_result" not in json.loads(observation("g", {}, [], {}))


def test_decision_parse_validates_like_theirs():
    c = ctx()
    t = turn()
    d = Decision.parse('{"name": "move", "arguments": {"dyaw_deg": "200", "note": "a wall"}}', t, c)
    assert d.name == "move" and d.arguments == {"dx_m": 0.0, "dy_m": 0.0, "dyaw_deg": 180.0, "note": "a wall"} and d.notes
    assert d.request_id == 7 and d.step == 3 and d.frame_seq == 9 and d.frame_stamp == 1.5 and d.note == "a wall"
    assert d.wire == {"name": "move", "arguments": {"dyaw_deg": "200", "note": "a wall"}}
    d = Decision.parse('{"name": "arm_path", "arguments": {"waypoints": [{"joints": {"waist_yaw": 0.3}}], "note": "n"}}', t, c)
    assert d.arguments["waypoints"] == [{"joints": {"waist_yaw": 0.3}}]
    fenced = '```json\n{"name": "done", "arguments": {"summary": "seen", "hindsight": ""}}\n```'
    assert Decision.parse(fenced, t, c).name == "done"
    d = Decision.parse('{"name": "hold", "arguments": {"note": "wait"}}', t, c)
    assert d.arguments == {"seconds": 1.0, "note": "wait"}                 # defaults fill in
    bad = ['{"name": "fly", "arguments": {}}',
           '{"name": "move", "arguments": {"dx_m": "far", "note": "n"}}',
           '{"name": "move", "arguments": {"dyaw_deg": 10}}',                # note required
           '{"name": "move", "arguments": {"dyaw_deg": 10, "note": ""}}',    # note empty
           '{"name": "done", "arguments": []}',
           '{"name": "done", "arguments": {"summary": "s"}}',                # hindsight required
           'sure! {"name": "move", "arguments": {"dyaw_deg": 10, "note": "n"}}',   # a fragment in prose
           '{"name": "move", "arguments": {"dyaw_deg": NaN, "note": "n"}}',
           '{"name": "check", "arguments": {"skill": "done", "arguments": {}, "note": "n"}}',
           '{"name": "turn", "arguments": {"angle_deg": 10, "note": "n"}}',        # a preset, not offered
           '{"name": "arm_path", "arguments": {"waypoints": [{"joints": {"left_elbow": 3.0}}], "note": "n"}}',   # limit
           '{"name": "arm_path", "arguments": {"waypoints": [{"joints": {"nope": 0.1}}], "note": "n"}}']
    for text in bad:
        with pytest.raises(ProtocolError) as e:
            Decision.parse(text, t, c)
        assert e.value.raw == text and str(e.value).startswith("invalid selection")
    with pytest.raises(ProtocolError):
        Decision.parse('{"name": "move", "arguments": {"dx_m": 0.3, "note": "n"}}', t, ctx(False))
    with pytest.raises(ValueError):
        parse_selection("[1, 2]")


def test_context_renders_catalog_and_rules():
    c = ctx(safety_notes=["a table behind the robot"])
    text = c.instructions
    assert text.count("Robot skill catalog:") == 1 and "- move:" in text and "- check:" in text
    assert "- left_elbow: [-1.047, 2.094] rad, stand 1.28" in text and "palms up" in text
    assert "hidden obstacles" in text and "a table behind the robot" in text
    assert "30 decisions" in text and "note" in text and "give_up" in text
    assert json.loads(text.split("Robot skill catalog:\n", 1)[1]) == c.tools
    schema = c.output_schema
    assert schema["required"] == ["name", "arguments"] and schema["additionalProperties"] is False
    strict = next(a for a in schema["properties"]["arguments"]["anyOf"] if "seconds" in a["properties"])
    assert strict["required"] == ["seconds", "note"] and strict["properties"]["seconds"]["anyOf"][1] == {"type": "null"}
    check = next(t for t in c.tools if t["function"]["name"] == "check")["function"]["parameters"]
    assert check["properties"]["skill"]["enum"] == ["move", "arm_path", "hold"]
    no_base = ctx(False).instructions
    assert "- move:" not in no_base.split("Skills:")[1].split("Every movement")[0]
    assert instructions(menu(True), can_walk=True, max_decisions=5) == build_context(menu(True), can_walk=True, max_decisions=5).instructions


def test_red_ball_rules():
    dec = RedBallDecider()
    dec.start(ctx())
    d = dec.decide(turn())
    assert d.name == "move" and d.arguments["dyaw_deg"] == 45.0
    img = solid(); img[20:28, 52:60] = (230, 20, 20)                 # small, far right
    d = dec.decide(turn(img))
    assert d.name == "move" and d.arguments["dyaw_deg"] < 0
    d = dec.decide(turn(img, waist=40.0))         # camera turned 40 left, ball 30 right of it: net left
    assert d.name == "move" and d.arguments["dyaw_deg"] > 0
    img = solid(); img[20:28, 28:36] = (230, 20, 20)                  # small, centred
    d = dec.decide(turn(img))
    assert d.name == "move" and d.arguments["dx_m"] == 0.5
    img = solid(); img[10:38, 18:46] = (230, 20, 20)                  # looming
    d = dec.decide(turn(img))
    assert d.name == "done" and "reached" in d.arguments["summary"]
    dec.start(ctx(False))
    assert dec.decide(turn()).name == "arm_path"
    assert len(dec.calls) == 6


def sse(text, usage=None):
    chunks = [json.dumps({"choices": [{"delta": {"content": text}}]})]
    if usage:
        chunks.append(json.dumps({"choices": [], "usage": usage}))
    return iter(chunks + ["[DONE]"])


def transport_of(replies, calls):
    replies = list(replies)

    def transport(url, headers, body, timeout):
        calls.append(body)
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return sse(r, {"prompt_tokens": 10, "completion_tokens": 3})
    return transport


TURN = '{"name": "move", "arguments": {"dyaw_deg": 45, "note": "searching"}}'
HOLD = '{"name": "hold", "arguments": {"seconds": 1, "note": "waiting"}}'


def test_hf_decider_keeps_the_conversation_and_prunes_images():
    calls = []
    dec = HFDecider(HFClient("m/vl", "hf_x", transport=transport_of([TURN, HOLD, TURN], calls)), live_image_window=2)
    dec.start(ctx())
    d = dec.decide(turn(step=0, request_id=1))
    assert d.name == "move" and d.arguments == {"dx_m": 0.0, "dy_m": 0.0, "dyaw_deg": 45.0, "note": "searching"}
    body = calls[0]
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["schema"] == dec.context.output_schema
    assert body["stream_options"] == {"include_usage": True} and body["max_tokens"] == 400
    msgs = body["messages"]
    assert msgs[0] == {"role": "system", "content": dec.context.instructions} and len(msgs) == 2
    user = msgs[1]["content"]
    assert user[0]["type"] == "text" and json.loads(user[0]["text"])["extra"]["env_step"] == 0
    assert user[1] == {"type": "text", "text": "Camera image: head"}
    assert user[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert dec.last_call is None                                     # set by process(), not decide()
    dec.decide(turn(step=1, request_id=2))
    msgs = calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == TURN                                # its own reply, verbatim
    dec.decide(turn(step=2, request_id=3))
    msgs = calls[2]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    first = msgs[1]["content"]                                       # oldest image pruned, text kept
    assert [b["type"] for b in first] == ["text", "text"] and "omitted" in first[-1]["text"]
    assert json.loads(first[0]["text"])["extra"]["env_step"] == 0
    assert any(b["type"] == "image_url" for b in msgs[3]["content"]) and any(b["type"] == "image_url" for b in msgs[5]["content"])
    tr = dec.transcript()
    assert tr[0]["role"] == "system" and {"type": "image"} in tr[3]["content"]


def test_hf_decider_fresh_turns_and_call_details():
    calls = []
    dec = HFDecider(HFClient("m/vl", "hf_x", transport=transport_of([TURN, HOLD], calls)), fresh_turns=True)
    dec.start(ctx())
    dec.threaded = False
    assert dec.request(turn(step=0, request_id=1)) and dec.request(turn(step=1, request_id=2))
    assert [m["role"] for m in calls[1]["messages"]] == ["system", "user"]
    assert dec.latest().name == "hold" and dec.latest().request_id == 2 and dec.latest().latency >= 0
    call = dec.last_call
    assert call["status"] == "completed" and call["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
    assert call["response_mode"] == "json_schema" and call["provider"] == "huggingface" and call["model"] == "m/vl"


def test_hf_decider_bad_reply_is_a_protocol_error():
    calls = []
    dec = HFDecider(HFClient("m/vl", "hf_x", transport=transport_of(['{"name": "fly", "arguments": {}}'], calls)))
    dec.start(ctx())
    dec.threaded = False
    assert dec.request(turn()) is True
    assert dec.latest() is None and dec.errors == 1 and isinstance(dec.last_error, ProtocolError)
    assert "unknown skill" in str(dec.last_error) and dec.last_call["status"] == "failed"
    assert dec.last_call["error_type"] == "ProtocolError" and not dec.pending
    assert dec._turns[-1]["assistant"]["content"] == '{"name": "fly", "arguments": {}}'   # the mistake stays in history


def test_hf_client_steps_down_structured_output_and_classifies_errors():
    calls = []
    replies = [RequestError(400, "response_format json_schema is not supported"), TURN,
               RequestError(400, "response_format not supported"), TURN, TURN]
    client = HFClient("m/vl", "hf_x", transport=transport_of(replies, calls))
    assert client.complete([], schema={"type": "object"}) == TURN
    assert [c.get("response_format", {}).get("type") for c in calls] == ["json_schema", "json_object"]
    assert client.response_mode == "json_object" and not client.json_schema
    assert client.complete([], schema={"type": "object"}) == TURN
    assert [c.get("response_format", {}).get("type") for c in calls[2:]] == ["json_object", None]
    assert client.response_mode == "none" and client.last_usage == {"prompt_tokens": 10, "completion_tokens": 3}
    assert client.complete([]) == TURN and "response_format" not in calls[-1] and client.calls == 5
    assert isinstance(classify(429, ""), Overloaded) and isinstance(classify(503, ""), Overloaded)
    assert isinstance(classify(500, "The model is overloaded"), Overloaded)
    assert isinstance(classify(402, ""), QuotaExceeded) and isinstance(classify(403, "monthly quota"), QuotaExceeded)
    assert type(classify(500, "boom")) is RequestError
    with pytest.raises(Overloaded):
        HFClient("m", "k", transport=transport_of([Overloaded(429, "x")], [])).complete([])
