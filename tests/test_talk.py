import json
from pathlib import Path

from g1 import talk
from g1.envs.moves import Moves, inventory
from g1.microphone import Microphone
from g1.talk import Talk, talk_menu, tools
from g1.vlm import ToolCall
from tests.envs.test_moves import Arm, Loco

MIMIC = {"fsm_id": 550, "api_id": 1, "motion": {
    "101": {"duration": 0.02, "name": "stance"},
    "401": {"duration": 3.08, "name": "vq_Y-qishen-faceup_001_get_up"},
    "516": {"duration": 4.92, "name": "bow"},
    "604": {"duration": 17.92, "name": "vq_YSKJ_JIXIEWU3_001_XX_b1f11bf1_G1_50hz"},
    "553": {"duration": 6.02, "name": "vq_shuangshoudazhaohu"}}}


class Model:
    """Answers each request with the next scripted (name, arguments); an Exception is raised."""
    def __init__(self, *replies):
        self.replies, self.asked = list(replies), []

    def complete(self, messages, *, tools):
        self.asked.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ToolCall("call_1", r[0], json.dumps(r[1]))


class Audio:
    def __init__(self):
        self.said = []

    def TtsMaker(self, text, speaker_id):
        self.said.append(text)
        return 0


def make(*replies, fsm=802):
    clock = {"t": 0.0}

    def sleep(dt):
        clock["t"] += dt
    now = lambda: clock["t"]
    loco = Loco(fsm)
    loco.GetMimicMotion = lambda: (0, MIMIC)
    moves = Moves(loco, Arm(), sleep=sleep, now=now, confirm=lambda q: True)
    mic = Microphone(now)
    menu = talk_menu(inventory(loco, Arm(), []))
    return Talk(mic, Audio(), moves, Model(*replies), menu, sleep=sleep, now=now), mic, clock


def heard(mic, text):
    mic.on_message(json.dumps({"text": text, "language": "<|en|>"}))


def test_only_gestures_and_dances_are_offered():
    t, _, _ = make()
    assert set(t.menu) == {516, 553, 604}                        # never the stance or a get-up
    assert t.menu[604]["name"] == "mechanical dance" and t.menu[553]["name"] == "greet with both hands"
    move = next(x for x in tools(t.menu) if x["function"]["name"] == "move")["function"]
    assert move["parameters"]["properties"]["move_id"]["enum"] == [516, 553, 604]
    assert {x["function"]["name"] for x in t.tools} == {"say", "move", "walk", "ignore"}


def test_listen_joins_what_is_said_until_a_pause():
    t, mic, clock = make()
    heard(mic, "hello robot")
    clock["t"] += 0.3
    heard(mic, "how tall are you")
    text, seq = t.listen(0)
    assert text == "hello robot how tall are you" and seq == 2


def test_say_move_ignore_and_a_model_error():
    t, mic, _ = make(("say", {"text": "I am about one point three metres tall."}),
                     ("move", {"move_id": 604, "say": "Here is my robot dance!"}),
                     ("ignore", {"reason": "side chatter"}),
                     ("move", {"move_id": 401}),                      # not allowed: nothing moves
                     RuntimeError("network down"))
    t.answer("how tall are you")
    t.answer("can you dance")
    assert t.moves.loco.calls == [550604] and t.moves.loco.fsm == 802
    t.answer("no I was talking to her")
    t.answer("lie down")
    t.answer("what is your name")
    assert t.moves.loco.calls == [550604]
    assert t.audio.said == ["I am about one point three metres tall.", "Here is my robot dance!",
                            "Sorry, I could not think of an answer just now."]
    assert t.counts == {"heard": 0, "said": 3, "moves": 1, "walks": 0, "ignored": 2, "errors": 1}
    last = t.model.asked[-1]
    assert last[0]["role"] == "system" and last[-1] == {"role": "user", "content": "what is your name"}
    assert {"role": "assistant", "content": "[played move 604: mechanical dance]"} in last   # it remembers


def test_talk_never_reaches_tasks_tools_or_the_arms():
    import ast
    tree = ast.parse(Path(talk.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if m.startswith(("g1.agent", "g1.tools", "g1.envs.robot", "g1.cli"))}, imported
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for forbidden in ("arm", "arm_sdk", "run_task", "switch_to", "Move", "SetFsmId", "ExecuteAction", "base"):
        assert forbidden not in used, forbidden
    # the only robot calls it makes itself: speech; moves go through Moves.play, the FSM through Moves.switch
    calls = {n.args[1].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "call"
             and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)}
    assert calls == {"TtsMaker"}, calls


def test_walk_is_capped_and_said_first():
    t, _, _ = make(("walk", {"turn_deg": 0, "forward_m": 5.0, "side_m": 0, "say": "Stepping forward!"}),
                   ("walk", {"turn_deg": 0, "forward_m": 0, "side_m": 0}))
    t.answer("come forward")
    vel = [c for c in t.moves.loco.calls if isinstance(c, tuple) and c[1] > 0]
    speed, poll = talk.limits.get("walk_speed"), talk.limits.get("base_send_period_s")
    assert len(vel) <= talk.limits.get("talk_walk_max_m") / speed / poll + 2             # 5 m asked, the cap walked
    assert t.audio.said == ["Stepping forward!"] and t.counts["walks"] == 1
    t.answer("walk")                                                                     # nothing to walk: ignored
    assert t.counts["walks"] == 1 and t.counts["ignored"] == 1
    assert "1 m" in talk.system_prompt() and "180 degrees" in talk.system_prompt()
