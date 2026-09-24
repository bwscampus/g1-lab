import json

import numpy as np
import pytest

from camera import Frame
from config import STAND_Q
from decider import Context, Decision, HFDecider
from hf import HFClient, RequestError
from skills import menu
from tests.doubles import RedBallDecider


def solid(color=(0, 0, 0), h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def ctx(image=None, allow_base=True, waist=0.0, note=""):
    return Context("find the red ball", 3, 30, Frame(solid() if image is None else image, 1.5, 9), STAND_Q,
                   waist, (0.3, -0.1, 12.0), [{"step": 2, "action": "turn", "args": {"angle_deg": 45},
                                              "outcome": "completed", "scene": "a wall"}],
                   menu(allow_base), note)


def test_decision_from_json():
    d = Decision.from_json({"scene": "a mug", "path_clear": False, "found": True, "action": "turn",
                            "args": {"angle_deg": "200"}, "reason": "r"}, ctx(), raw="x")
    assert d.action == "turn" and d.args == {"angle_deg": 180.0} and d.notes and d.path_clear is False
    assert d.step == 3 and d.frame_seq == 9 and d.frame_stamp == 1.5 and d.raw == "x"
    with pytest.raises(ValueError):
        Decision.from_json({"action": "fly", "args": {}}, ctx())
    with pytest.raises(ValueError):
        Decision.from_json({"action": "walk_forward", "args": {"distance_m": "far"}}, ctx())
    with pytest.raises(ValueError):
        Decision.from_json({"action": "walk_forward", "args": {"distance_m": 0.3}}, ctx(allow_base=False))
    with pytest.raises(ValueError):
        Decision.from_json({"action": "done", "args": []}, ctx())
    assert Decision.from_json({"action": "done", "args": {"found": False}}, ctx()).found is False


def test_red_ball_rules():
    dec = RedBallDecider()
    d = dec.decide(ctx())
    assert d.action == "turn" and d.args == {"angle_deg": 45.0} and not d.found
    img = solid(); img[20:28, 52:60] = (230, 20, 20)                 # small, far right
    d = dec.decide(ctx(img))
    assert d.action == "turn" and d.args["angle_deg"] < 0 and d.found
    d = dec.decide(ctx(img, waist=40.0))          # camera turned 40 left, ball 30 right of it: net left
    assert d.action == "turn" and d.args["angle_deg"] > 0
    img = solid(); img[20:28, 28:36] = (230, 20, 20)                  # small, centred
    assert dec.decide(ctx(img)).action == "walk_forward"
    img = solid(); img[10:38, 18:46] = (230, 20, 20)                  # looming
    d = dec.decide(ctx(img))
    assert d.action == "done" and d.args["found"] is True
    d = dec.decide(ctx(allow_base=False))
    assert d.action == "look"
    assert len(dec.calls) == 6


def sse(text):
    return iter([json.dumps({"choices": [{"delta": {"content": text}}]}), "[DONE]"])


def test_hf_decider_messages_and_decode():
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body)
        return sse('{"scene": "a table", "path_clear": true, "found": false, "action": "turn", '
                   '"args": {"angle_deg": 45}, "reason": "searching"}')

    dec = HFDecider(HFClient("m/vl", "hf_x", transport=transport))
    d = dec.decide(ctx(note="previous reply was invalid"))
    assert d.action == "turn" and d.args == {"angle_deg": 45.0} and d.scene == "a table"
    msgs = calls[0]["messages"]
    text = msgs[1]["content"][0]["text"]
    assert "Goal: find the red ball" in text and "Step 3 of 30" in text and "#2 turn" in text
    assert '"name": "walk_forward"' in text and "Note: previous reply was invalid" in text
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert calls[0]["max_tokens"] == 300 and dec.model == "m/vl"


def test_hf_decider_bad_reply_is_counted_not_published():
    def transport(url, headers, body, timeout):
        return sse('{"action": "fly", "args": {}}')

    dec = HFDecider(HFClient("m/vl", "hf_x", transport=transport))
    dec.threaded = False
    assert dec.request(ctx()) is True
    assert dec.latest() is None and dec.errors == 1 and "unknown action" in str(dec.last_error)
    assert not dec.pending
