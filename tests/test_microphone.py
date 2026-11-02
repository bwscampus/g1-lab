"""The microphone: the robot's speech recognition (rt/audio_msg) as an utterance log, the model's
``heard``, and ``listen``, which waits until someone speaks."""
import json

import pytest

from g1.agent.agent import Agent
from g1.cli import run
from g1.microphone import Microphone, heard_view
from g1.tools import menu
from tests.agent.test_agent import Sequence, check, observations


def said(text, language="en"):
    """A message as the G1 publishes it (measured 2026-10-30)."""
    return json.dumps({"index": 1, "timestamp": 0, "type": 0, "text": text, "angle": 0, "speaker_id": 0,
                       "emotion": "<|EMO_UNKNOWN|>", "confidence": 0.5, "language": f"<|{language}|>",
                       "is_final": False})


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_utterances_are_logged_and_other_languages_and_echoes_dropped():
    clock = Clock()
    mic = Microphone(clock, keep=frozenset({"en"}), echo_guard_s=1.0)
    mic.on_message(said("Point at the mug."))
    mic.on_message(said("奖一快乐。", "zh"))                       # mis-tagged noise
    mic.on_message(json.dumps({"play_state": 1}))                  # the robot speaks: it hears itself
    mic.on_message(said("I cannot find the banana."))
    clock.t = 5.0
    mic.on_message(json.dumps({"play_state": 0}))
    clock.t = 5.5
    mic.on_message(said("echo tail"))                              # within the guard after it stopped
    clock.t = 7.0
    mic.on_message(said("Now the laptop."))
    mic.on_message("not json")
    assert [h.text for h in mic.since(0)] == ["Point at the mug.", "Now the laptop."]
    assert [h.text for h in mic.since(1)] == ["Now the laptop."] and mic.seq == 2
    assert mic.counts == {"messages": 8, "heard": 2, "other_language": 1, "while_speaking": 2, "unreadable": 1}


def test_heard_view_keeps_the_newest_within_its_limits():
    mic = Microphone(Clock(), keep=frozenset({"en"}))
    for text in ("one", "two", "three"):
        mic.on_message(said(text))
    assert heard_view(mic.since(0), 10.0) == [{"text": "one", "s_ago": 10.0}, {"text": "two", "s_ago": 10.0},
                                              {"text": "three", "s_ago": 10.0}]
    assert [h["text"] for h in heard_view(mic.since(0), 10.0, max_chars=8)] == ["two", "three"]
    assert heard_view(mic.since(0), 200.0, max_age_s=120.0) == []


def hearing_run(plan, voices, max_decisions=6):
    dec = Sequence(plan)
    env = check(mode="gantry", hears=True, voices=voices)
    agent = Agent("point at what you are asked to", dec, menu(False, True, hears=True), max_decisions=max_decisions)
    assert run(agent, env, max_time=200) is True
    agent.close()
    return dec, env


def test_listen_ends_when_someone_speaks_and_the_model_hears_it():
    dec, env = hearing_run([("listen", {"seconds": 40}), ("listen", {"seconds": 2}), ("done", {"summary": "s",
                                                                                              "hindsight": ""})],
                           voices=[{"at_s": 8, "text": "Point at the mug."}])
    o = observations(dec)
    assert o[0]["heard"] == []                                               # nothing yet
    first = o[1]["previous_result"]["result"]
    assert first["ended"] == "speech heard (see heard)" and first["duration_s"] < 10   # not the 40 s asked
    assert [h["text"] for h in o[1]["heard"]] == ["Point at the mug."]
    assert o[2]["heard"] == [] and o[2]["previous_result"]["result"]["ended"] == "nothing heard in 2 s"
    assert env.microphone.counts["heard"] == 1


def test_a_task_that_does_not_hear_has_no_heard_and_no_listen():
    dec = Sequence([("done", {"summary": "s", "hindsight": ""})])
    agent = Agent("g", dec, menu(False, True), max_decisions=2)
    assert run(agent, check(mode="gantry"), max_time=60) is True
    agent.close()
    assert "heard" not in observations(dec)[0] and "listen" not in agent.tools
    assert "microphone" not in agent.context.instructions


def test_the_twin_reports_its_own_speech_so_it_is_not_heard():
    # the twin's TtsMaker plays for len/speech_chars_per_s: a voice in that time is the robot hearing itself
    dec, env = hearing_run([("say", {"text": "I am looking for the mug on the table now."}),
                            ("done", {"summary": "s", "hindsight": ""})],
                           voices=[{"at_s": 4.5, "text": "said while it speaks"}])
    assert env.microphone.counts["while_speaking"] == 1 and env.microphone.counts["heard"] == 0
    assert observations(dec)[1]["heard"] == []


@pytest.mark.parametrize("bad, match", [({"voices": [{"at_s": 1, "text": "hi"}]}, "hears"),
                                        ({"hears": True, "tools": "hold:1", "instruction": None}, "instruction")])
def test_task_rules_for_hearing(tmp_path, bad, match):
    from g1.agent.task import load_task
    d = tmp_path / "t"
    d.mkdir()
    spec = {"instruction": "listen", **bad}
    (d / "task.json").write_text(json.dumps({k: v for k, v in spec.items() if v is not None}))
    with pytest.raises(ValueError, match=match):
        load_task(d)
