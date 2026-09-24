import json
import math
import threading
import time

import numpy as np
import pytest

from camera import Frame
from config import CONTROL_DT, STAND_Q, joint_index
from skills import SixSeven
from perception import (DEFAULT_MODEL, Detected, HFPerceiver, Perceiver, Percept, RequestError,
                        VisionQuery, build_perceiver, extract_json)
from tests.doubles import FakePerceiver
from policy import Obs
from routines import POLICIES, Selector
from run import build_parser, main, run
from behaviors import Describe
from vision import bearing

WAIST_YAW = joint_index("waist_yaw")


def solid(color, h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def frame(seq=1, color=(0, 0, 0), stamp=0.0):
    return Frame(solid(color), stamp, seq)


from tests.doubles import sim_env as check_env


# -- data -------------------------------------------------------------------------

def test_percept_from_json_bearing_and_clamps():
    f = frame()
    data = {"summary": "a chair", "path_clear": False,
            "objects": [{"label": "chair", "x": 1.0, "y": 0.5, "width": 0.3, "height": 0.4, "distance_m": "2"},
                        {"label": "wall", "x": 7, "y": -1, "width": "big", "distance_m": "far"},
                        "junk"]}
    p = Percept.from_json(data, f, raw="...")
    assert p.summary == "a chair" and p.path_clear is False and p.frame_seq == 1
    chair, wall = p.objects
    assert chair.bearing == pytest.approx(bearing(1.0, f.image.shape)) and chair.distance_m == 2.0
    assert wall.x == 1.0 and wall.y == 0.0 and wall.width == 0.0 and wall.distance_m is None
    assert p.salient() is chair and p.find("CHAIR") is chair and p.find("dog") is None
    empty = Percept.from_json({}, f)
    assert empty.objects == [] and empty.path_clear is True and empty.salient() is None


def test_extract_json_robust():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! ```json\n{"a": {"b": [1, 2]}}\n``` done') == {"a": {"b": [1, 2]}}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("{not: json}")


# -- worker -------------------------------------------------------------------------

class Gated(Perceiver):
    def __init__(self):
        super().__init__(min_interval=0.0)
        self.gate = threading.Event()
        self.seen = []

    def describe(self, f):
        self.gate.wait(5.0)
        self.seen.append(f.seq)
        return Percept(f"frame {f.seq}", [], True, f.seq, f.stamp)


def wait_until(cond, timeout=2.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.005)
    return True


def test_worker_runs_only_on_request_and_coalesces():
    per = Gated()
    per.start()
    try:
        assert per.request() is False                 # nothing offered yet
        per.offer(frame(1))
        time.sleep(0.05)
        assert per.requests == 0                      # offering alone never runs the model
        assert per.request() is True
        assert wait_until(lambda: per.requests == 1)  # the worker now holds frame 1 (gated)
        assert per.pending
        for s in range(2, 11):
            per.offer(frame(s))
            assert per.request() is False             # in flight: ignored
        per.gate.set()
        assert wait_until(lambda: per.count == 1 and not per.pending)
        assert per.seen == [1] and per.latest().frame_seq == 1
        assert per.request() is True                  # the newest offered frame, 10
        assert wait_until(lambda: per.count == 2)
        assert per.seen == [1, 10] and per.latest().seq == 2
        assert wait_until(lambda: not per.pending)
        assert per.request() is False                 # frame 10 already described
    finally:
        per.stop()


def test_request_min_interval_floor():
    per = FakePerceiver(min_interval=10.0)
    per.offer(frame(1))
    assert per.request() is True
    per.offer(frame(2))
    assert per.request() is False and per.requests == 1


def test_vision_query_polls_at_refresh():
    per = FakePerceiver()
    q = VisionQuery(refresh=1.0)
    f1 = frame(1)
    assert q.poll(per, Obs(STAND_Q), 0.0) is False                      # no frame
    assert q.poll(per, Obs(STAND_Q, f1, 0.0), 0.0) is True
    p = per.latest()
    assert q.poll(per, Obs(STAND_Q, f1, 0.0, p, 0.0), 0.5) is False     # percept already describes f1
    f2 = frame(2)
    assert q.poll(per, Obs(STAND_Q, f2, 0.0, p, 0.1), 0.5) is False     # too soon
    assert q.poll(per, Obs(STAND_Q, f2, 0.0, p, 0.6), 1.0) is True
    assert q.sent == 2


def test_worker_errors_counted_not_raised():
    class Boom(Perceiver):
        def describe(self, f):
            if f.seq == 2:
                raise RuntimeError("model down")
            return Percept("ok", [], True, f.seq, f.stamp)

    per = Boom(min_interval=0.0, threaded=False)
    per.offer(frame(1)); per.request()
    per.offer(frame(2)); per.request()
    assert per.errors == 1 and "model down" in str(per.last_error)
    assert per.latest().frame_seq == 1 and not per.pending
    per.offer(frame(3)); per.request()
    assert per.latest().frame_seq == 3 and per.requests == 3
    assert "1 error(s)" in per.summary()


def test_fake_perceiver_labels_red_blob():
    per = FakePerceiver(label="ball")
    img = solid((0, 0, 0))
    img[:, 48:] = (230, 20, 20)
    per.offer(Frame(img, 1.5, 7))
    assert per.latest() is None and per.request()
    p = per.latest()
    assert p.frame_seq == 7 and p.frame_stamp == 1.5 and p.seq == 1
    d = p.salient()
    assert d.label == "ball" and d.x > 0.5 and d.bearing > 0 and p.path_clear is False
    per.offer(frame(8)); per.request()
    assert per.latest().objects == [] and per.latest().path_clear is True


# -- HF request/stream ------------------------------------------------------------------

def sse(*chunks, done=True):
    lines = [json.dumps({"choices": [{"delta": {"content": c}}]}) for c in chunks]
    if done:
        lines.append("[DONE]")
    return lines


def test_hf_request_and_stream():
    cv2 = pytest.importorskip("cv2")
    import base64
    calls = []

    def transport(url, headers, body, timeout):
        calls.append((url, headers, body))
        return iter(sse("```json\n{\"summary\": \"a chair ahead\", \"path_clear\": false,",
                        " \"objects\": [{\"label\": \"chair\", \"x\": 0.75, \"y\": 0.6,"
                        " \"width\": 0.3, \"height\": 0.5, \"distance_m\": 1.5}]}\n```"))

    deltas = []
    per = HFPerceiver("m/vl", "hf_x", transport=transport, on_text=deltas.append, max_width=32)
    img = solid((200, 30, 30), h=48, w=64)        # red in RGB
    p = per.describe(Frame(img, 2.0, 3))
    url, headers, body = calls[0]
    assert per.model == "m/vl"
    assert url == "https://router.huggingface.co/v1/chat/completions"
    assert headers["Authorization"] == "Bearer hf_x"
    assert body["model"] == "m/vl" and body["stream"] is True
    assert body["response_format"] == {"type": "json_object"}
    data_url = body["messages"][1]["content"][0]["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    jpeg = base64.b64decode(data_url.split(",", 1)[1])
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert bgr.shape[1] == 32                                   # downscaled to max_width
    assert bgr[0, 0, 2] > 150 and bgr[0, 0, 0] < 80             # still red after RGB->BGR->JPEG
    assert len(deltas) == 2 and "".join(deltas) == p.raw
    assert p.summary == "a chair ahead" and p.path_clear is False and p.frame_seq == 3
    assert p.objects[0].label == "chair" and p.objects[0].distance_m == 1.5 and p.objects[0].bearing > 0


def test_hf_drops_json_mode_on_400():
    bodies = []

    def transport(url, headers, body, timeout):
        bodies.append(dict(body))
        if "response_format" in body:
            raise RequestError(400, "response_format is not supported by this provider")
        return iter(sse('{"summary": "empty room", "objects": [], "path_clear": true}'))

    per = HFPerceiver("m/vl", "hf_x", transport=transport)
    p = per.describe(frame())
    assert p.summary == "empty room" and len(bodies) == 2
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    assert per.json_mode is False
    per.describe(frame(2))
    assert len(bodies) == 3 and "response_format" not in bodies[2]


def test_hf_other_errors_propagate():
    def transport(url, headers, body, timeout):
        raise RequestError(401, "bad token")

    per = HFPerceiver("m/vl", "hf_x", transport=transport)
    with pytest.raises(RequestError):
        per.describe(frame())


def test_hf_from_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("G1_VISION_API_KEY", raising=False)
    monkeypatch.delenv("G1_VISION_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        HFPerceiver.from_env()
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    assert HFPerceiver.from_env().model == DEFAULT_MODEL
    monkeypatch.setenv("G1_VISION_MODEL", "a/b:deepinfra")
    assert HFPerceiver.from_env().model == "a/b:deepinfra"
    assert HFPerceiver.from_env("c/d").model == "c/d"


# -- through the check env -------------------------------------------------------------

def test_describe_passes_check_under_noise():
    env = check_env("--camera-noise")
    per = FakePerceiver()
    ages = []
    orig = env.observe

    def observe(q):
        obs = orig(q)
        ages.append((obs.frame_age, obs.percept_age))
        return obs
    env.observe = observe
    p = Describe(duration=3.0, vision_refresh=0.5)
    assert run(p, env, perceiver=per) is True
    assert env.violations == []
    assert 5 <= per.requests <= 8                       # one per refresh, not one per frame
    assert all(pa >= fa for fa, pa in ages if pa != math.inf)   # a percept is never newer than its frame
    assert env.q_min[WAIST_YAW] < 0 or env.q_max[WAIST_YAW] > 0   # it moved


def test_describe_holds_without_percepts():
    env = check_env("--camera-noise")
    assert run(Describe(duration=2.0), env) is True         # no perceiver at all
    assert env.cmd_min[WAIST_YAW] == env.cmd_max[WAIST_YAW] == 0.0
    assert env.q_max[WAIST_YAW] == pytest.approx(0.0, abs=1e-3)


def scripted(x, frame_seq=None):
    d = Detected("thing", x, 0.5, 0.2, 0.2, None, bearing(2 * x - 1, (48, 64, 3)))
    return Percept("a thing", [d], True, frame_seq or 0, 0.0)


def test_describe_uses_yaw_at_capture():
    p = Describe(duration=10.0, ramp=0.0, to_stand=0.0)
    p.reset(Obs(STAND_Q))
    t = 0.0
    # frame 1 arrives with the waist at 0; its percept says the thing is on the right
    p.step(t, Obs(STAND_Q, frame(1), 0.0)); t += CONTROL_DT
    right = scripted(0.9, frame_seq=1); right.seq = 1
    for _ in range(40):
        a = p.step(t, Obs(STAND_Q, frame(1), 0.0, right, 0.0)); t += CONTROL_DT
    turned = a.q[WAIST_YAW]
    assert turned == pytest.approx(-0.6 * bearing(0.8, (48, 64, 3)), abs=1e-6)
    # frame 2 is captured with the waist turned; a percept that still describes
    # frame 1 (centred thing) must aim relative to frame 1's yaw (0), not the current one
    p.step(t, Obs(STAND_Q, frame(2), 0.0, right, 0.0)); t += CONTROL_DT
    centred = scripted(0.5, frame_seq=1); centred.seq = 2
    for _ in range(40):
        a = p.step(t, Obs(STAND_Q, frame(2), 0.0, centred, 0.0)); t += CONTROL_DT
    assert a.q[WAIST_YAW] == pytest.approx(0.0, abs=1e-6)
    # stale percepts are ignored: nothing moves
    stale = scripted(0.9, frame_seq=2); stale.seq = 3
    a = p.step(t, Obs(STAND_Q, frame(2), 0.0, stale, 10.0))
    assert a.q[WAIST_YAW] == pytest.approx(0.0, abs=1e-6)


def test_wave_on_person_selector():
    env = check_env("--camera-noise")
    per = FakePerceiver(label="person")
    p = POLICIES["wave_on_person"]()
    assert p.uses_vision is True
    assert run(p, env, perceiver=per) is True
    assert env.violations == []
    # takeover 5 s -> first idle tick asks the model (served inline) -> the percept is visible
    # on the next tick -> trigger -> sixseven -> handback: one tick more than a frame trigger
    assert env.ticks == round((10.0 + SixSeven().duration) / CONTROL_DT) + 1
    assert per.requests == 1


def test_selector_rules_see_percepts_without_frames():
    fired = []
    p = Selector([(lambda o: fired.append(o.percept.seq) or True, "tpose")], timeout=1.0)
    p.reset(Obs(STAND_Q))
    t = 0.0
    while t < 5.0 - 1e-9:
        p.step(t, Obs(STAND_Q)); t += CONTROL_DT
    per = scripted(0.5); per.seq = 4
    p.step(t, Obs(STAND_Q, None, math.inf, per, 0.0))        # percept only, no frame
    assert fired == [4]


# -- CLI ---------------------------------------------------------------------------------

def test_build_perceiver_modes(monkeypatch, capsys):
    parse = lambda *a: build_parser().parse_args(["--env", "sim", "--policy", "x", *a])
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("G1_VISION_API_KEY", raising=False)
    assert build_perceiver(parse(), POLICIES["look"]()) is None              # no vision needed
    assert build_perceiver(parse(), Describe()) is None                       # auto without a token
    assert "no HF_TOKEN" in capsys.readouterr().out
    assert build_perceiver(parse("--vision", "off"), Describe()) is None
    with pytest.raises(RuntimeError):
        build_perceiver(parse("--vision", "api"), Describe())
    with pytest.raises(SystemExit):
        main(["--env", "sim", "--headless", "--policy", "describe", "--vision", "api"])
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    auto = build_perceiver(parse(), Describe())
    assert isinstance(auto, HFPerceiver) and auto.model == DEFAULT_MODEL
    api = build_perceiver(parse("--vision", "api", "--vision-model", "x/y", "--vision-interval", "5"), Describe())
    assert isinstance(api, HFPerceiver) and api.model == "x/y" and api.min_interval == 5.0


def test_describe_with_fake_perceiver_narrates(capsys):
    env = check_env("--camera-noise")
    assert run(Describe(duration=2.0, vision_refresh=0.5), env, perceiver=FakePerceiver()) is True
    out = capsys.readouterr().out
    assert "[describe] a red ball at" in out and "perception:" in out
