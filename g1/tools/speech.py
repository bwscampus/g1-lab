"""The speaker: ``say`` speaks through the robot's onboard text-to-speech
(``AudioClient.TtsMaker``), a one-shot call like the gestures, holding the
current pose for as long as the speech takes. Sim prints the text. The one
tool GPT-Policy's menu does not have."""
from __future__ import annotations

from g1.core import limits
from g1.core.action import Segment
from g1.tools.base import Tool, limit, num


class Say(Tool):
    name = "say"
    order = 45
    command = "TtsMaker"
    prompt = ("Say text aloud through the robot's speaker (up to {limit:say_max_chars} characters). The host keeps "
              "the pose and waits about {limit:speech_chars_per_s} characters per second for the speech to finish, "
              "plus pause_s. Use it to tell a person what you found, to ask someone to step aside, or to report "
              "before done. It moves nothing; not a search move.")
    params = {"text": {"type": "string", "minLength": 1, "maxLength": limit("say_max_chars"),
                       "description": "what to say, one or two short sentences"},
              "pause_s": num(0.0, limit("say_pause_max_s"), 0.0, "extra seconds to wait after speaking")}

    def __init__(self, **args) -> None:
        super().__init__(**args)
        if not self.text.strip():
            raise ValueError("say: text must not be empty")
        cap = int(limits.get("say_max_chars"))
        if len(self.text) > cap:                        # a clamp, like a number out of range
            self.notes.append(f"text cut to {cap} characters")
            self.text = self.args["text"] = self.text[:cap]

    @property
    def seconds(self) -> float:
        return max(limits.get("say_min_s"), len(self.text) / limits.get("speech_chars_per_s")) + self.pause_s

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({}, self.seconds, label=f"saying {self.text[:40]!r}",
                        command=("TtsMaker", {"text": self.text, "speaker_id": int(limits.get("tts_speaker_id"))})),)
