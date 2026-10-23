"""The in-process transport: ``RobotEnv``'s link (``envs/link.py``) wired straight
to a ``Twin``, in lockstep on sim time. The tests run the robot
code through it.

``sleep(dt)`` is where time passes: it advances the twin one control tick at a
time and, after each, delivers rt/lowstate to the subscribers, lets a
BaseCommander send what it would have sent, renders the head camera when one is
due and syncs the viewer. ``now()`` is sim time, so the robot code's pacing, its
bring-up waits and the 1 s Move dead-man all run on the twin's clock, faster
than real time headless and paced with ``realtime``.

The clients encode every call exactly as unitree_sdk2py's do (the same API id
and the same JSON, the same return shape, ``Damp`` returning None included);
``tests/twin/test_loopback_contract.py`` holds them to that.
"""
from __future__ import annotations

import json
import time

from g1.core import limits
from g1.core.config import CONTROL_DT, NUM_JOINTS
from g1.twin.onboard import ArmSdkCommand

MOTORS = 35                       # unitree_hg LowCmd_ / LowState_ carry 35 motor slots


class Motor:
    __slots__ = ("mode", "q", "dq", "tau", "kp", "kd", "tau_est")

    def __init__(self) -> None:
        self.mode = 0
        self.q = self.dq = self.tau = self.kp = self.kd = self.tau_est = 0.0


class LowCmd:
    def __init__(self) -> None:
        self.motor_cmd = [Motor() for _ in range(MOTORS)]
        self.crc = 0


class Imu:
    __slots__ = ("quaternion", "gyroscope", "accelerometer", "rpy")


class LowStateMsg:
    """What a LowState_ subscriber reads: ``motor_state[i].q/dq/tau_est``, ``imu_state``, ``tick``."""

    def __init__(self, s) -> None:
        self.tick = s.tick
        self.mode_machine = 0
        self.motor_state = [Motor() for _ in range(MOTORS)]
        for i in range(NUM_JOINTS):
            m = self.motor_state[i]
            m.q, m.dq, m.tau_est = float(s.q[i]), float(s.dq[i]), float(s.tau[i])
        self.imu_state = Imu()
        self.imu_state.quaternion = list(map(float, s.quat))
        self.imu_state.gyroscope = list(map(float, s.gyro))
        self.imu_state.accelerometer = list(map(float, s.acc))
        self.imu_state.rpy = [0.0, 0.0, 0.0]


class _Crc:
    def Crc(self, msg) -> int:
        return 0                  # the twin does not check it


class _Publisher:
    def __init__(self, twin) -> None:
        self.twin = twin

    def Write(self, msg) -> bool:
        self.twin.on_arm_sdk(ArmSdkCommand.of(msg))
        return True


class _Client:
    service = ""

    def __init__(self, twin) -> None:
        self.twin = twin

    def SetTimeout(self, timeout: float) -> None:
        pass

    def Init(self) -> None:
        pass

    def _Call(self, api: int, parameter: str) -> tuple[int, str]:
        return self.twin.call(self.service, api, parameter)


class LoopbackLoco(_Client):
    """``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient``, the calls g1-lab makes."""
    service = "sport"

    def __init__(self, twin) -> None:
        super().__init__(twin)
        self.first_shake_hand_stage_ = -1

    def GetFsmId(self):
        code, data = self._Call(7001, json.dumps({}))
        if code != 0:
            return code, None
        return code, json.loads(data).get("data")

    def SetFsmId(self, fsm_id: int):
        return self._Call(7101, json.dumps({"data": fsm_id}))[0]

    def SetVelocity(self, vx: float, vy: float, omega: float, duration: float = 1.0):
        return self._Call(7105, json.dumps({"velocity": [vx, vy, omega], "duration": duration}))[0]

    def SetTaskId(self, task_id: float):
        return self._Call(7106, json.dumps({"data": task_id}))[0]

    def Damp(self):
        self.SetFsmId(1)

    def ZeroTorque(self):
        self.SetFsmId(0)

    def Start(self):
        self.SetFsmId(500)

    def StopMove(self):
        self.SetVelocity(0., 0., 0.)

    def Move(self, vx: float, vy: float, vyaw: float, continous_move: bool = False):
        self.SetVelocity(vx, vy, vyaw, 864000.0 if continous_move else 1)

    def WaveHand(self, turn_flag: bool = False):
        self.SetTaskId(1 if turn_flag else 0)

    def ShakeHand(self, stage: int = -1):
        if stage == 0:
            self.first_shake_hand_stage_ = False
            self.SetTaskId(2)
        elif stage == 1:
            self.first_shake_hand_stage_ = True
            self.SetTaskId(3)
        else:
            self.first_shake_hand_stage_ = not self.first_shake_hand_stage_
            return self.SetTaskId(3 if self.first_shake_hand_stage_ else 2)


class LoopbackAudio(_Client):
    """``unitree_sdk2py.g1.audio.g1_audio_client.AudioClient``, the calls g1-lab makes."""
    service = "voice"

    def __init__(self, twin) -> None:
        super().__init__(twin)
        self.tts_index = 0

    def TtsMaker(self, text: str, speaker_id: int):
        self.tts_index += self.tts_index
        return self._Call(1001, json.dumps({"index": self.tts_index, "text": text, "speaker_id": speaker_id}))[0]

    def GetVolume(self):
        code, data = self._Call(1005, json.dumps({}))
        return (code, json.loads(data)) if code == 0 else (code, None)

    def SetVolume(self, volume: int):
        return self._Call(1006, json.dumps({"volume": volume}))[0]

    def LedControl(self, R: int, G: int, B: int):
        return self._Call(1010, json.dumps({"R": R, "G": G, "B": B}))[0]


class LoopbackTwin(_Client):
    """The twin's own service, ``Summary``, as ``envs/link.py:DdsLink.twin`` calls it."""
    service = "twin"

    def Summary(self):
        code, data = self._Call(9001, "{}")
        return (code, json.loads(data)) if code == 0 else (code, None)


class LoopbackLink:
    name = "loopback"

    def __init__(self, twin, *, realtime: float | None = None, viewer=None, renderer=None, camera=None,
                 camera_every: int = max(1, round(1.0 / (limits.get("twin_camera_fps") * CONTROL_DT))),
                 stamp=None) -> None:
        self.twin = twin
        self.stamp = stamp or (lambda: twin.time)     # the clock camera frames are stamped on
        self.realtime = realtime
        self.viewer = viewer
        self.renderer = renderer          # with ``camera``: render the head camera every ``camera_every`` ticks
        self.camera = camera
        self.camera_every = max(1, int(camera_every))
        self._subscribers: list = []
        self._commanders: list = []
        self._ticks = 0
        self._wall0 = None

    # -- the link ---------------------------------------------------------------------------------
    def init(self, domain: int, iface: str) -> None:
        pass

    def loco(self) -> LoopbackLoco:
        return LoopbackLoco(self.twin)

    def audio(self) -> LoopbackAudio:
        return LoopbackAudio(self.twin)

    def twin_client(self) -> LoopbackTwin:
        return LoopbackTwin(self.twin)

    def lowstate(self, callback):
        self._subscribers.append(callback)
        return callback

    def arm_sdk(self) -> _Publisher:
        return _Publisher(self.twin)

    def lowcmd(self) -> LowCmd:
        return LowCmd()

    def crc(self) -> _Crc:
        return _Crc()

    def now(self) -> float:
        return self.twin.time

    def start_commander(self, base) -> None:
        self._commanders.append(base)     # polled every tick on sim time instead of its own thread

    def sleep(self, dt: float) -> None:
        end = self.twin.time + dt - 1e-9
        while self.twin.time < end:
            self.tick()

    # -- one control tick -------------------------------------------------------------------------
    def tick(self) -> None:
        for base in self._commanders:
            base.poll(self.twin.time)
        self.twin.step()
        self._ticks += 1
        if self._subscribers:
            msg = LowStateMsg(self.twin.lowstate())
            for fn in self._subscribers:
                fn(msg)
        if self.renderer is not None and self._ticks % self.camera_every == 0:
            self.render()
        if self.viewer is not None:
            if not self.viewer.is_running():
                self.viewer = None            # once: the run's safe return carries on headless
                raise KeyboardInterrupt("viewer closed")
            self.viewer.sync()
        if self.realtime is not None:
            now = time.perf_counter()
            if self._wall0 is None:
                self._wall0 = (now, self.twin.time)
            lag = self._wall0[0] + (self.twin.time - self._wall0[1]) / max(self.realtime, 1e-6) - now
            if lag > 0:
                time.sleep(lag)

    def render(self) -> None:
        from g1.twin.model import HEAD_CAMERA
        with self.twin.lock:
            self.renderer.update_scene(self.twin.data, camera=HEAD_CAMERA)
            image = self.renderer.render()
        self.camera.publish(image, self.stamp())
