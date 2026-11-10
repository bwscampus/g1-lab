"""g1 start's live voice (g1/voice/): the player paces our voice onto the robot's speaker and stops it at
once; LiveCore turns the realtime model's events into speech, interruptions and tool calls; the process
starts, talks and closes; and only g1/voice/ ever reaches PlayStream."""
import ast
import base64
import json
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from g1.core.sounds import dbfs
from g1.voice import Voice, available
from g1.voice.child import BYTES_PER_S, MicSocket, Player, membership
from g1.voice.openai_realtime import (LiveCore, realtime_tools, say_exactly, session_refresh, session_update)
from g1.vlm import VOICE_SETTINGS

ROOT = Path(__file__).resolve().parents[2]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def tone(seconds: float, rate: int = 16000, amp: int = 8000) -> bytes:
    n = int(seconds * rate)
    return (np.sin(2 * np.pi * 300 * np.arange(n) / rate) * amp).astype("<i2").tobytes()


# -- the player ------------------------------------------------------------------------------------

def make_player(**kw):
    clock, played, halted, events = Clock(), [], [], []
    p = Player(lambda stream, pcm: played.append((stream, len(pcm))) or 0, lambda: halted.append(clock.t) or 0,
               clock=clock, emit=events.append, chunk_s=0.32, ahead_s=0.64, echo_s=1.0, **kw)
    return p, clock, played, halted, events


def test_the_player_paces_a_reply_and_ends_when_it_has_played():
    p, clock, played, halted, events = make_player()
    p.begin()
    p.feed(tone(1.0))
    p.end()
    for _ in range(10):                              # all at once: only ahead_s worth goes out
        p.pump()
    assert [n for _, n in played] == [10240, 10240] and p.playing   # 0.64 s queued ahead: ahead_s
    clock.t = 0.5
    p.pump()
    p.pump()
    assert len(played) == 4 and played[-1][1] == 32000 - 3 * 10240          # two more: the last one short
    assert len({s for s, _ in played}) == 1                                  # one stream for the reply
    clock.t = 0.99
    p.pump()
    assert p.playing                                  # not yet played out
    clock.t = 1.01
    p.pump()
    assert not p.playing and p.speaking and events == [("speaking", True), ("speaking", False)]
    clock.t = 2.1
    assert not p.speaking                             # the echo guard is over
    assert halted == []                               # it ended, nothing stopped


def test_a_stop_cuts_the_reply_and_says_how_much_was_heard():
    p, clock, played, halted, events = make_player()
    p.begin()
    p.feed(tone(2.0))
    p.pump()
    clock.t = 0.4
    assert p.stop() == 320                           # only what was sent can have played (0.32 s)
    assert halted == [0.4] and not p.playing and events[-1] == ("speaking", False)
    p.feed(tone(0.5))
    p.pump()
    assert len(played) == 1                          # nothing more after a stop
    p.begin()
    p.feed(tone(0.4))
    p.pump()
    assert len(played) == 2 and played[1][0] != played[0][0]   # the next reply, a new stream
    assert p.stop() == 0 and p.stop() == 0 and len(halted) == 2  # a second stop does nothing


# -- the model's events ----------------------------------------------------------------------------

class FakePlayer:
    def __init__(self):
        self.playing = self.speaking = False
        self.fed, self.calls = 0, []

    def begin(self):
        self.playing = self.speaking = True
        self.calls.append("begin")

    def feed(self, pcm):
        self.fed += len(pcm)

    def end(self):
        self.calls.append("end")

    def stop(self):
        self.calls.append("stop")
        self.playing = False
        return 700


def make_core(**kw):
    sent, emitted, player = [], [], FakePlayer()
    core = LiveCore(sent.append, player, emitted.append, barge_in=kw.pop("barge_in", "gate"), gate_db=30.0, **kw)
    return core, sent, emitted, player


def audio_delta(item, seconds):
    pcm = tone(seconds, 24000)
    return {"type": "response.output_audio.delta", "item_id": item, "delta": base64.b64encode(pcm).decode()}


def test_the_model_speaks_through_the_player_and_its_words_reach_the_session():
    core, sent, emitted, player = make_core()
    core.on_server({"type": "session.updated"})
    core.on_server(audio_delta("item_1", 0.5))
    core.on_server(audio_delta("item_1", 0.5))
    core.on_server({"type": "response.output_audio.done", "item_id": "item_1"})
    assert player.calls == ["begin", "end"] and abs(player.fed - BYTES_PER_S) < 200    # 1 s at 16 kHz
    core.on_server({"type": "conversation.item.input_audio_transcription.completed", "transcript": " hello  robot "})
    core.on_server({"type": "response.output_audio_transcript.done", "transcript": "Hi there!"})
    core.on_server({"type": "response.function_call_arguments.done", "call_id": "c1", "name": "move",
                    "arguments": '{"move_id": 516}'})
    core.on_server({"type": "response.done", "response": {"status": "completed", "usage": {"total_tokens": 9}}})
    core.on_server({"type": "error", "error": {"code": "response_cancel_not_active", "message": "nothing"}})
    core.on_server({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})
    assert emitted == [("ready",), ("user_said", "hello robot"), ("robot_said", "Hi there!"),
                       ("tool_call", "c1", "move", '{"move_id": 516}'), ("response_done", "completed", {"total_tokens": 9}),
                       ("error", "invalid_request_error: bad")]
    assert sent == []                                # the model's events need no answer


def test_someone_talking_over_the_robot_stops_it_where_they_stopped_hearing():
    core, sent, emitted, player = make_core()
    core.on_server(audio_delta("item_7", 0.5))
    core.on_server({"type": "input_audio_buffer.speech_started", "item_id": "x", "audio_start_ms": 100})
    assert player.calls[-1] == "stop" and emitted[-1] == ("speech_started",)
    assert sent == [{"type": "conversation.item.truncate", "item_id": "item_7", "content_index": 0, "audio_end_ms": 700},
                    {"type": "response.cancel"}]
    sent.clear()
    core.on_server({"type": "input_audio_buffer.speech_started"})     # not speaking: nothing to stop
    assert sent == [] and core.counts["interrupted"] == 1


def test_the_mic_reaches_the_model_unmuted_and_its_own_voice_is_gated():
    core, sent, _, player = make_core()
    core.on_mic(tone(0.32))
    assert sent == []                                # muted at the start (/mic starts off)
    core.on_parent(("mute", False))
    core.on_mic(tone(0.32, amp=8000))                # about -15 dBFS: a person close by
    up = base64.b64decode(sent[-1]["audio"])
    assert sent[-1]["type"] == "input_audio_buffer.append" and abs(len(up) - 0.32 * 48000) < 200   # 24 kHz
    player.speaking = True
    core.on_mic(tone(0.32, amp=300))                 # about -44 dBFS while the robot speaks: its echo
    assert dbfs(base64.b64decode(sent[-1]["audio"])) < -35 and core.counts["gated"] == 1   # sent as silence
    #                                                (but for the filter's tail of the piece before)
    core.on_mic(tone(0.32, amp=8000))                # loud: someone talks over it
    assert any(base64.b64decode(sent[-1]["audio"]))
    open_core, open_sent, _, open_player = make_core(barge_in="open", muted=False)
    open_player.speaking = True
    open_core.on_mic(tone(0.32, amp=300))
    assert any(base64.b64decode(open_sent[-1]["audio"]))          # open: everything goes up
    core.on_parent(("mute", True))
    assert sent[-1] == {"type": "input_audio_buffer.clear"}


def test_what_the_session_sends_becomes_the_right_events():
    core, sent, _, player = make_core()
    core.on_parent(("text", "what can you do"))
    assert sent[0]["item"]["content"] == [{"type": "input_text", "text": "what can you do"}]
    assert sent[1] == {"type": "response.create"}
    sent.clear()
    core.on_parent(("tool_result", "c1", '{"result": "done"}', ("Your camera.", "data:image/jpeg;base64,AA=="), True))
    assert sent[0]["item"] == {"type": "function_call_output", "call_id": "c1", "output": '{"result": "done"}'}
    assert sent[1]["item"]["content"][1] == {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="}
    assert sent[2] == {"type": "response.create"}
    sent.clear()
    core.on_parent(("tool_result", "c2", "{}", None, False))     # ignore: the model stays quiet
    assert len(sent) == 1
    sent.clear()
    core.on_parent(("say", "One moment."))
    assert sent == [say_exactly("One moment.")] and "One moment." in sent[0]["response"]["instructions"]
    sent.clear()
    core.on_parent(("stop",))
    assert sent == [] and "stop" not in player.calls             # nothing playing
    player.playing = True
    core.on_parent(("stop",))
    assert player.calls[-1] == "stop" and sent[-1] == {"type": "response.cancel"}
    core.on_parent(("update", "new rules", []))
    assert sent[-1] == session_refresh("new rules", [])
    core.on_parent(("speaker", False))
    core.on_server(audio_delta("item_9", 0.2))
    assert player.fed == 0                                        # speaker off: the words only


def test_the_session_payload_hears_far_and_speaks_short():
    tools = [{"type": "function", "function": {"name": "move", "description": "a move",
                                               "parameters": {"type": "object", "properties": {}}}}]
    ev = session_update({**VOICE_SETTINGS, "engine": "realtime"}, "be a robot", tools)
    s = ev["session"]
    assert ev["type"] == "session.update" and s["type"] == "realtime" and s["instructions"] == "be a robot"
    assert s["tools"] == realtime_tools(tools) == [{"type": "function", "name": "move", "description": "a move",
                                                    "parameters": {"type": "object", "properties": {}}}]
    assert s["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert s["audio"]["input"]["noise_reduction"] == {"type": VOICE_SETTINGS["noise_reduction"]}
    assert s["audio"]["input"]["turn_detection"]["type"] == VOICE_SETTINGS["turn_detection"]
    assert s["audio"]["input"]["turn_detection"]["interrupt_response"] is True
    assert s["audio"]["output"]["voice"] == VOICE_SETTINGS["voice"]
    assert s["reasoning"] == {"effort": VOICE_SETTINGS["reasoning_effort"]} and "model" not in s
    quiet = session_update({**VOICE_SETTINGS, "noise_reduction": None, "reasoning_effort": "default",
                            "turn_detection": "server_vad"}, "x", [])["session"]
    assert quiet["audio"]["input"]["noise_reduction"] is None and "reasoning" not in quiet
    assert quiet["audio"]["input"]["turn_detection"]["type"] == "server_vad"


# -- the mic socket --------------------------------------------------------------------------------

def test_the_mic_socket_reads_packets():
    mic = MicSocket(group=None, port=0, timeout=0.2)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        out.sendto(tone(0.16), ("127.0.0.1", mic.port))
        assert len(mic.read()) == 5120 and mic.packets == 1
        assert mic.read() is None                    # nothing more: a timeout, not an error
    finally:
        out.close()
        mic.close()
    assert membership("239.168.123.161", "192.168.123.164") == socket.inet_aton("239.168.123.161") + \
        socket.inet_aton("192.168.123.164")


# -- the process -----------------------------------------------------------------------------------

def echo_child(conn, cfg):
    """A stand-in voice process: ready, then echoes what it is told as robot_said until close."""
    conn.send(("ready",))
    while True:
        msg = conn.recv()
        if msg[0] == "close":
            conn.send(("closed", "bye"))
            return
        conn.send(("robot_said", json.dumps(msg)))


def failing_child(conn, cfg):
    conn.send(("closed", "no key"))


def test_the_voice_process_starts_talks_and_closes():
    got = []
    v = Voice("lo0", 1, "rules", [], got.append, target=echo_child, timeout=20)
    v.start()
    assert v.alive and v.cfg["start"]["session"]["instructions"] == "rules" and v.cfg["muted"] is True
    v.text("hello")
    v.tool_result("c1", "{}", respond=False)
    t0 = time.monotonic()
    said = lambda: [json.loads(m[1]) for m in got if m[0] == "robot_said"]
    while len(said()) < 2 and time.monotonic() - t0 < 10:
        time.sleep(0.02)
    assert got[0] == ("ready",) and said() == [["text", "hello"], ["tool_result", "c1", "{}", None, False]]
    v.close()
    assert not v.alive and got[-1][0] == "robot_said"     # a close the session made is not news to it
    with pytest.raises(ConnectionError, match="no key"):
        Voice("lo0", 1, "rules", [], got.append, target=failing_child, timeout=20).start()


def test_the_live_voice_needs_its_keys_and_a_live_engine(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gpt_live, realtime = {**VOICE_SETTINGS, "engine": "gpt-live"}, {**VOICE_SETTINGS, "engine": "realtime"}
    assert "OPENAI_API_KEY" in available(gpt_live)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert available(realtime) is None
    assert "OPENROUTER_API_KEY" in available(gpt_live)            # its backend (Sol) is on OpenRouter
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert available(gpt_live) is None
    assert "unitree" in available({**VOICE_SETTINGS, "engine": "unitree"})


# -- what may reach the speaker --------------------------------------------------------------------

def test_only_the_voice_package_plays_raw_audio():
    """PlayStream and PlayStop are called only from g1/voice/ (the model's own voice), never from a tool,
    the session or an env; a tool asking for one is refused (tests/tools/test_tools.py)."""
    from g1.core.config import AUDIO_METHODS, VOICE_METHODS
    assert not AUDIO_METHODS & VOICE_METHODS
    for path in (ROOT / "g1").rglob("*.py"):
        tree = ast.parse(path.read_text())
        named = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value in VOICE_METHODS}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in VOICE_METHODS}
        if path.parent.name != "voice" and path.name != "config.py":
            assert not named and not attrs, (path, named | attrs)


# -- GPT-Live ---------------------------------------------------------------------------------------

def make_duplex(**kw):
    from g1.voice.openai_live import DuplexCore
    sent, emitted, player, clock = [], [], FakePlayer(), Clock()
    player.end = lambda: player.calls.append("end")
    core = DuplexCore(sent.append, player, emitted.append, gate_db=30.0, hold_s=1.5, clock=clock, **kw)
    return core, sent, emitted, player, clock


def test_gpt_live_starts_at_the_robots_rate_with_client_delegation():
    from g1.voice.openai_live import URL, append, session_start
    ev = session_start(VOICE_SETTINGS, "talk, delegate the rest")
    s = ev["session"]
    assert ev["type"] == "session.start" and URL == "wss://api.openai.com/v1/live/sessions"
    assert s["model"] == VOICE_SETTINGS["live_model"] and s["instructions"] == "talk, delegate the rest"
    assert s["audio"] == {"format": {"type": "audio/pcm", "rate": 16000}, "output": {"voice": VOICE_SETTINGS["voice"]}}
    assert s["delegation"] == {"type": "client"}
    long = append("commentary", "del_1", "word " * 1000)
    assert long["type"] == "session.commentary.append" and long["delegation_id"] == "del_1"
    assert len(long["content"]) <= 1500 and long["content"].endswith("...")      # 500 tokens at most


def test_gpt_live_events_reach_the_player_and_the_session():
    core, sent, emitted, player, clock = make_duplex()
    core.on_server({"type": "session.started", "session": {"id": "s1"}})
    pcm = tone(0.2)                                    # 16 kHz both ways: no resampling
    core.on_server({"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
    core.on_server({"type": "session.output_audio.delta", "delta": base64.b64encode(pcm).decode()})
    assert player.calls == ["begin", "end", "end"] and player.fed == 2 * len(pcm)   # one stream while it plays
    core.on_server({"type": "session.input_transcript.delta", "delta": "can you ", "start_ms": 0, "end_ms": 300})
    core.on_server({"type": "session.output_transcript.delta", "delta": "Sure!"})
    core.on_server({"type": "session.delegation.created", "delegation": {"id": "del_7", "type": "delegation",
                                                                        "target": "client"}})
    core.on_server({"type": "session.thinking.appended", "client_event_id": "g1_1"})
    core.on_server({"type": "error", "error": {"code": "bad_delegation", "message": "unknown id"}})
    core.on_server({"type": "session.closed", "reason": "idle"})
    assert emitted == [("ready",), ("user_delta", "can you "), ("robot_delta", "Sure!"), ("delegation", "del_7"),
                       ("error", "bad_delegation: unknown id"), ("closed", "idle")]
    assert sent == []


def test_gpt_live_hears_the_mic_as_it_comes_and_says_what_the_backend_sends():
    core, sent, _, player, clock = make_duplex()
    core.on_mic(tone(0.16))
    assert sent == []                                  # muted at the start
    core.on_parent(("mute", False))
    assert sent[-1]["type"] == "session.input_audio.unmute"
    core.on_mic(tone(0.16))
    assert sent[-1]["type"] == "session.input_audio.append" and len(base64.b64decode(sent[-1]["audio"])) == 5120
    player.speaking = True
    core.on_mic(tone(0.16, amp=300))                   # its own echo while it speaks: silence
    assert not any(base64.b64decode(sent[-1]["audio"]))
    core.on_parent(("commentary", "del_7", "I bowed."))
    core.on_parent(("thinking", "del_7", "played move 516"))
    assert [(e["type"], e["delegation_id"], e["content"]) for e in sent[-2:]] == [
        ("session.commentary.append", "del_7", "I bowed."), ("session.thinking.append", "del_7", "played move 516")]
    core.on_parent(("text", "what is your name"))
    assert sent[-1]["type"] == "session.instructions.append" and sent[-1]["delegation_id"] is None
    assert "what is your name" in sent[-1]["content"]
    n = len(sent)
    core.on_parent(("tool_result", "c", "{}", None, True))
    core.on_parent(("update", "rules", []))
    assert len(sent) == n                              # the realtime engine's messages: nothing to do


def test_esc_stops_gpt_live_and_drops_what_it_still_says():
    core, sent, _, player, clock = make_duplex()
    player.playing = True
    core.on_parent(("stop",))
    assert player.calls == ["stop"] and sent[-1]["type"] == "session.instructions.append"
    pcm = base64.b64encode(tone(0.1)).decode()
    clock.t = 1.0
    core.on_server({"type": "session.output_audio.delta", "delta": pcm})
    assert player.fed == 0 and core.counts["dropped"] == 1        # its last words are not played
    clock.t = 1.6
    core.on_server({"type": "session.output_audio.delta", "delta": pcm})
    assert player.fed > 0
