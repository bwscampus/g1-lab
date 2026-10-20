"""The SDK seam: everything ``RobotEnv`` takes from unitree_sdk2py, behind one
object, so the same robot code drives the G1 over DDS (``DdsLink``) or the twin
in-process on sim time (``g1/twin/loopback.py:LoopbackLink``).

    init(domain, iface)       ChannelFactoryInitialize
    loco() / audio()          LocoClient / AudioClient
    twin_client()             the twin's own service (Summary), only when talking to a twin
    lowstate(callback)        subscribe rt/lowstate
    arm_sdk() / lowcmd()      the rt/arm_sdk publisher and a LowCmd_ buffer to fill
    crc()                     the LowCmd_ CRC
    now() / sleep(dt)         the clock robot code paces itself on
    start_commander(base)     run BaseCommander (its own thread over DDS)

The SDK is imported inside each method, so nothing here needs it until a robot
run starts.

The DDS domain: ``--domain``, else ``$UNITREE_DOMAIN``, else the twin's domain
(``limits twin_domain``) on a loopback interface and the robot's (0) on any other.
"""
from __future__ import annotations

import json
import os
import time

from g1.core import limits

LOOPBACK = frozenset({"lo", "lo0", "127.0.0.1", "localhost"})
ROBOT_DOMAIN = 0


def is_loopback(iface: str | None) -> bool:
    return (iface or "").strip() in LOOPBACK


def dds_domain(iface: str | None, domain: int | None = None) -> int:
    if domain is not None:
        return int(domain)
    if os.environ.get("UNITREE_DOMAIN"):
        return int(os.environ["UNITREE_DOMAIN"])
    return int(limits.get("twin_domain")) if is_loopback(iface) else ROBOT_DOMAIN


class DdsLink:
    name = "dds"

    def init(self, domain: int, iface: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(domain, iface)

    def loco(self):
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        return LocoClient()

    def audio(self):
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        return AudioClient()

    def twin_client(self):
        from unitree_sdk2py.rpc.client import Client

        class TwinClient(Client):
            """``g1 twin``'s service: one call, its verdict (see g1/twin/services.py)."""
            def __init__(self):
                super().__init__("twin", False)

            def Init(self):
                self._SetApiVerson("1.0.0.0")
                self._RegistApi(9001, 0)

            def Summary(self):
                code, data = self._Call(9001, "{}")
                return (code, json.loads(data)) if code == 0 else (code, None)
        return TwinClient()

    def lowstate(self, callback):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(callback, 10)
        return sub

    def arm_sdk(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        pub.Init()
        return pub

    def lowcmd(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        return unitree_hg_msg_dds__LowCmd_()

    def crc(self):
        from unitree_sdk2py.utils.crc import CRC
        return CRC()

    def now(self) -> float:
        return time.time()

    def sleep(self, dt: float) -> None:
        time.sleep(dt)

    def start_commander(self, base) -> None:
        base.start()
