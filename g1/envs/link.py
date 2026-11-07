"""The SDK seam: everything ``RobotEnv`` takes from unitree_sdk2py, behind one
object, so the same robot code drives the G1 over DDS (``DdsLink``) or the twin
in-process on sim time (``g1/twin/loopback.py:LoopbackLink``).

    init(domain, iface)       ChannelFactoryInitialize
    loco() / audio()          LocoClient / AudioClient
    mimic(loco)               LocoClient + GetMimicMotion (the robot's stored motions; g1 moves)
    arm_action()              the arm service: preset and taught upper-body actions (g1 moves)
    twin_client()             the twin's own service (Summary), only when talking to a twin
    lowstate(callback)        subscribe rt/lowstate (short probes only: g1 status, g1 moves)
    lowstate_reader()         rt/lowstate read on demand, the newest sample (a run's ArmSdk)
    audio_msg(callback)       subscribe rt/audio_msg: the robot's speech recognition (g1/microphone.py)
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

    domain: int | None = None

    def init(self, domain: int, iface: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(domain, iface)
        self.domain = domain

    def loco(self):
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        return LocoClient()

    def mimic(self, loco):
        """``loco`` + GetMimicMotion, the C++ SDK's (unitree_sdk2 g1_loco_client.hpp) that the Python
        SDK lacks: the FSM API (7100) asked for FSM 550's motions. Play one with SetFsmId(550<id>).
        Call after ``loco.Init()``."""
        loco._RegistApi(7100, 0)

        class Mimic:
            def __getattr__(self, name):
                return getattr(loco, name)

            def GetMimicMotion(self):
                code, data = loco._Call(7100, json.dumps({"fsm_id": 550, "api_id": 1}))
                return (code, json.loads(data)) if code == 0 and data else (code, None)
        return Mimic()

    def arm_action(self):
        from unitree_sdk2py.rpc.client import Client

        class ArmActionClient(Client):
            """The ``arm`` service: Unitree's preset upper-body actions (by id; 99 releases one that
            holds) and the app's taught ones (by name). The C++ SDK's parameters (g1_arm_action_client.hpp)."""
            def __init__(self):
                super().__init__("arm", False)

            def Init(self):
                self._SetApiVerson("1.0.0.14")
                for api in (7106, 7107, 7108, 7113):
                    self._RegistApi(api, 0)

            def ExecuteAction(self, action_id: int):
                return self._Call(7106, json.dumps({"action_id": int(action_id)}))

            def ExecuteCustomAction(self, name: str):
                return self._Call(7108, json.dumps({"action_name": name}))

            def StopCustomAction(self):
                return self._Call(7113, "{}")

            def GetActionList(self):
                code, data = self._Call(7107, "{}")
                return (code, json.loads(data)) if code == 0 and data else (code, None)
        return ArmActionClient()

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

    def robot_state(self):
        """The robot's service list (read-only here: g1 lidar never calls ServiceSwitch)."""
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
        return RobotStateClient()

    def slam(self):
        from unitree_sdk2py.rpc.client import Client

        class SlamClient(Client):
            """Unitree's SLAM service (``slam_operate``): only the calls in config.SLAM_METHODS.
            JSON in, ``(code, data)`` out. Navigation (1102, 1201/1202) and relocation (1804) are
            not registered."""
            API = {"StartMapping": 1801, "EndMapping": 1802, "CloseSlam": 1901}

            def __init__(self):
                super().__init__("slam_operate", False)

            def Init(self):
                self._SetApiVerson("1.0.0.1")
                for api in self.API.values():
                    self._RegistApi(api, 0)

            def _json(self, name, data):
                return self._Call(self.API[name], json.dumps({"data": data}))

            def StartMapping(self, slam_type):
                return self._json("StartMapping", {"slam_type": slam_type})

            def EndMapping(self, address):
                return self._json("EndMapping", {"address": address})

            def CloseSlam(self):
                return self._json("CloseSlam", {})
        return SlamClient()

    def lowstate(self, callback):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(callback, 10)
        return sub

    def lowstate_reader(self) -> "LatestReader":
        """rt/lowstate without a callback: cyclonedds keeps the newest sample (in C) and the
        caller takes it, once a tick. A callback (``lowstate``) runs Python on DDS's receive
        thread for each of the robot's ~1000 samples a second; once another thread is busy
        it falls behind for good, and every RPC reply in the process waits behind it (3104
        after the full timeout, on the Jetson, while the robot carries the command out)."""
        if self.domain is None:
            raise RuntimeError("lowstate_reader before init(domain, iface)")
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        participant = DomainParticipant(self.domain)          # the domain (and interface) init() set up
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        return LatestReader(DataReader(participant, Topic(participant, "rt/lowstate", LowState_), qos=qos),
                            participant)

    def audio_msg(self, callback):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        sub = ChannelSubscriber("rt/audio_msg", String_)
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


class LatestReader:
    """A DDS reader that holds only the newest sample (KeepLast 1). ``take()`` returns it once,
    or None when nothing new arrived since the last take; ``Close()`` deletes the reader."""

    def __init__(self, reader, participant=None) -> None:
        self.reader = reader
        self.participant = participant

    def take(self):
        if self.reader is None:
            return None
        from cyclonedds.internal import InvalidSample
        samples = [s for s in self.reader.take(N=1) if not isinstance(s, InvalidSample)]
        return samples[-1] if samples else None

    def Close(self) -> None:
        reader, participant, self.reader, self.participant = self.reader, self.participant, None, None
        for entity in (reader, participant):        # deleting the participant deletes its topic too
            if entity is not None:
                entity.__del__()                     # cyclonedds: deletes the DDS entity now, once

