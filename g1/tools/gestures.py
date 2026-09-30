"""Onboard gestures: the LocoClient does the motion, arm_sdk lets go meanwhile
(GPT-Policy's ``set_gripper``: a bounded SDK call with a measured result).
Robot only; sim has no onboard gestures and aborts on the call."""
from __future__ import annotations

from g1.core import limits
from g1.core.action import Segment
from g1.core.poses import STAND
from g1.tools.base import Tool, flag, integer, limit, num


class Gesture(Tool):
    """Hand the arms to the onboard controller (weight 1->0), call the
    LocoClient method once, hold at weight 0 for the gesture's length, then
    take the arms back (0->1)."""

    needs_loco = True
    ramp_s = limits.get("gesture_ramp_s")

    def call_args(self) -> dict:
        return {}

    def segments(self) -> tuple[Segment, ...]:
        assert self.command is not None
        return (Segment(STAND, self.ramp_s, weight=lambda a: 1.0 - a, label="handing the arms to the onboard controller"),
                Segment(STAND, self.seconds, weight=lambda a: 0.0, command=(self.command, self.call_args()),
                        label=f"{self.name} (onboard)"),
                Segment(STAND, self.ramp_s, weight=lambda a: a, label="taking the arms back"))


class WaveHand(Gesture):
    name = "wave_hand"
    order = 40
    command = "WaveHand"
    prompt = ("Wave with the onboard controller's built-in gesture; turn_flag true adds a turn of the hand. The host "
              "hands the arms to the onboard controller for seconds (default {limit:wave_hand_seconds}), then takes "
              "them back. A greeting toward a person; not a search move.")
    params = {"turn_flag": flag(False, "wave with a turn of the hand"),
              "seconds": num(1.0, limit("gesture_seconds_max"), limit("wave_hand_seconds"), "how long the onboard gesture takes")}

    def call_args(self) -> dict:
        return {"turn_flag": bool(self.turn_flag)}


class ShakeHand(Gesture):
    name = "shake_hand"
    order = 41
    command = "ShakeHand"
    prompt = ("Offer a handshake with the onboard controller's built-in gesture (stage -1 runs the whole gesture). "
              "The host hands the arms to the onboard controller for seconds (default {limit:shake_hand_seconds}), "
              "then takes them back. Only when a person stands within reach in front of the robot.")
    params = {"stage": integer(-1, 1, -1, "-1: the whole gesture; 0 and 1 are the SDK's two halves"),
              "seconds": num(1.0, limit("gesture_seconds_max"), limit("shake_hand_seconds"), "how long the onboard gesture takes")}

    def call_args(self) -> dict:
        return {"stage": int(self.stage)}
