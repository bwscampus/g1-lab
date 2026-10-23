"""The twin's SDK services: ``(service, api_id, parameter JSON) -> (code, data JSON)``,
the wire format of unitree_sdk2py's RPC, so the same handlers answer the in-process
loopback and the DDS servers of ``g1 twin``.

  sport            7001 GetFsmId, 7101 SetFsmId, 7105 SetVelocity (Move / StopMove),
                   7106 SetArmTask (WaveHand / ShakeHand)
  voice            1001 TtsMaker (printed and logged), 1005 GetVolume, 1006 SetVolume,
                   1010 LED (logged)
  motion_switcher  1001 CheckMode ("ai"), 1003 ReleaseMode
  twin             9001 Summary: the twin's own verdict, which no robot can give (did it
                   fall, where is the base really); ``g1 run --sim`` reads it at the end

Anything else answers 3203 (API not implemented), which is what an unknown id gets
from the robot too.
"""
from __future__ import annotations

import json

from g1.core import limits

NOT_IMPLEMENTED = 3203
BAD_PARAMETER = 3204

# name, version: as the SDK's clients set them (g1/loco/g1_loco_api.py, g1/audio/g1_audio_api.py,
# comm/motion_switcher/motion_switcher_api.py)
SERVICES = {"sport": "1.0.0.0", "voice": "1.0.0.0", "motion_switcher": "1.0.0.1", "twin": "1.0.0.0"}
APIS = {"sport": (7001, 7101, 7105, 7106), "voice": (1001, 1005, 1006, 1010), "motion_switcher": (1001, 1003),
        "twin": (9001,)}
TWIN_SUMMARY = 9001


class Services:
    def __init__(self, onboard, clock, voice: str = "sim", summary=None) -> None:
        self.onboard = onboard
        self.summary = summary                 # () -> dict, the twin's own report
        self.clock = clock                     # sim seconds
        self.voice = voice                     # the prefix spoken text is printed with
        self.volume = int(limits.get("twin_volume"))
        self.log: list[dict] = []              # every call: t, service, api, parameter, code
        self.spoken: list[str] = []

    def call(self, service: str, api: int, parameter: str) -> tuple[int, str]:
        try:
            p = json.loads(parameter) if parameter else {}
        except json.JSONDecodeError:
            code, data = BAD_PARAMETER, ""
        else:
            try:
                code, data = getattr(self, f"_{service}")(int(api), p)
            except (KeyError, TypeError, ValueError):
                code, data = BAD_PARAMETER, ""
        self.log.append({"t": round(self.clock(), 4), "service": service, "api": int(api),
                         "parameter": parameter, "code": code})
        return code, data

    def _sport(self, api: int, p: dict) -> tuple[int, str]:
        o, t = self.onboard, self.clock()
        if api == 7001:
            return 0, json.dumps({"data": o.fsm})
        if api == 7101:
            return o.set_fsm(int(p["data"]), t), ""
        if api == 7105:
            v = [float(x) for x in p["velocity"]]
            if len(v) != 3:
                return BAD_PARAMETER, ""
            o.set_velocity(v, float(p.get("duration", 1.0)), t)
            return 0, ""
        if api == 7106:
            o.arm_task(int(p["data"]), t)
            return 0, ""
        return NOT_IMPLEMENTED, ""

    def _voice(self, api: int, p: dict) -> tuple[int, str]:
        if api == 1001:
            text = str(p["text"])
            self.spoken.append(text)
            print(f"{self.voice}: says {text!r}")    # no speaker here: the text is the record
            return 0, ""
        if api == 1005:
            return 0, json.dumps({"volume": self.volume})
        if api == 1006:
            self.volume = max(0, min(100, int(p["volume"])))
            return 0, ""
        if api == 1010:
            return 0, ""
        return NOT_IMPLEMENTED, ""

    def _motion_switcher(self, api: int, p: dict) -> tuple[int, str]:
        if api == 1001:
            return 0, json.dumps({"form": "0", "name": "ai"})
        if api == 1003:
            return 0, ""
        return NOT_IMPLEMENTED, ""

    def _twin(self, api: int, p: dict) -> tuple[int, str]:
        if api == TWIN_SUMMARY and self.summary is not None:
            return 0, json.dumps(self.summary())
        return NOT_IMPLEMENTED, ""
