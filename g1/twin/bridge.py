"""The DDS transport of ``g1 twin``: the twin on the loopback interface, speaking
the robot's protocol, so unmodified robot code (``g1 run TASK --sim``, the
SDK's examples, ``g1 status``) talks to it as it would to the G1.

  services     an ``unitree_sdk2py.rpc.server.Server`` per service (sport, voice,
               motion_switcher) answering on rt/api/<name>/request, every API id
               routed to ``Twin.call``
  rt/lowstate  unitree_hg LowState_ from its own thread at ``twin_lowstate_hz``
               (the CRC is left 0: the SDK's Python CRC is too slow for 500 Hz, and
               nothing on the receiving side checks it)
  rt/arm_sdk   unitree_hg LowCmd_, applied by the onboard controller's blend
  rt/lowcmd    ignored, with a warning: g1-lab never drives the legs, and the twin's
               legs belong to its walker

The bridge refuses any interface but loopback: a fake LowState on the robot's
network would be read as the robot's.
"""
from __future__ import annotations

import threading
import time

from g1.core import limits
from g1.core.config import NUM_JOINTS
from g1.envs.link import is_loopback
from g1.twin.onboard import ArmSdkCommand
from g1.twin.services import APIS, SERVICES


class DdsBridge:
    def __init__(self, twin, domain: int, iface: str) -> None:
        if not is_loopback(iface):
            raise SystemExit(f"g1 twin only serves on the loopback interface (lo0 on macOS, lo on Linux), not "
                             f"{iface!r}: its LowState must never reach a network the robot is on")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.rpc.server import Server

        self.twin = twin
        self.domain = domain
        ChannelFactoryInitialize(domain, iface)
        self.servers = []
        for name, version in SERVICES.items():
            server = Server(name)
            server._SetApiVersion(version)
            for api in APIS[name]:
                server._RegistHandler(api, self._handler(name, api), False)
            server.Start(False)
            self.servers.append(server)
        self.pub = ChannelPublisher("rt/lowstate", LowState_)
        self.pub.Init()
        self.msg = unitree_hg_msg_dds__LowState_()
        self.arm_sub = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        self.arm_sub.Init(self._on_arm_sdk, 10)
        self._lowcmd_warned = False
        self.lowcmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_sub.Init(self._on_lowcmd, 10)
        self.published = 0
        self.arm_messages = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _handler(self, service: str, api: int):
        def handle(parameter: str):
            return self.twin.call(service, api, parameter)
        return handle

    def _on_arm_sdk(self, msg) -> None:
        self.arm_messages += 1
        self.twin.on_arm_sdk(ArmSdkCommand.of(msg))

    def _on_lowcmd(self, msg) -> None:
        if not self._lowcmd_warned:
            self._lowcmd_warned = True
            print("twin: rt/lowcmd ignored (the twin's legs belong to its walker; arm_sdk drives the upper body)")

    def publish(self) -> None:
        s = self.twin.lowstate()
        m = self.msg
        m.tick = s.tick
        for i in range(NUM_JOINTS):
            ms = m.motor_state[i]
            ms.q, ms.dq, ms.tau_est = float(s.q[i]), float(s.dq[i]), float(s.tau[i])
        m.imu_state.quaternion = [float(v) for v in s.quat]
        m.imu_state.gyroscope = [float(v) for v in s.gyro]
        m.imu_state.accelerometer = [float(v) for v in s.acc]
        self.pub.Write(m)
        self.published += 1

    def start(self, hz: float = limits.get("twin_lowstate_hz")) -> None:
        def run():
            period = 1.0 / hz
            next_t = time.perf_counter()
            while not self._stop.is_set():
                self.publish()
                next_t += period
                lag = next_t - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                else:
                    next_t = time.perf_counter()          # behind: publish as fast as it can, never in a burst
        self._thread = threading.Thread(target=run, name="twin-lowstate", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
