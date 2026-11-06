"""The robot env's onboard-call dispatch, without the SDK: stub clients stand in."""
import numpy as np
import pytest

from g1.cli import run_settings
from g1.core.action import Action
from g1.core.config import NUM_JOINTS, STAND_Q, UPPER_BODY
from g1.envs import RobotEnv


class Client:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls.append((name, kw) if not a else (name, a, kw))
            if self.fail:
                raise RuntimeError("rpc failed")
            return (0, 50) if name == "GetVolume" else 0
        return call


class Arm:
    def __init__(self):
        self.sent = []

        class M:
            q = 0.0
        self.state = type("S", (), {"motor_state": [M() for _ in range(NUM_JOINTS)]})()

    def send(self, action):
        self.sent.append(action)

    def release(self, duration=1.0):
        self.sent.append("release")

    def close(self):
        self.sent.append("close")


def robot(**settings):
    env = RobotEnv(run_settings(iface="lo", **{"mode": "standing", **settings}))
    env.arm, env.loco, env.audio = Arm(), Client(), Client()
    env.monitor, env.base = None, None
    env.audio_calls, env._audio_warned, env.overruns, env.ticks = 0, False, 0, 0
    import time
    env._wall = time.time()
    env._health_at = time.monotonic()
    return env


def act(command=None):
    return Action(STAND_Q.copy(), list(UPPER_BODY), command=command)


def test_commands_are_routed_by_their_allow_list(capsys):
    env = robot()
    env.step(act(("TtsMaker", {"text": "hello", "speaker_id": 0})))
    assert env.audio.calls == [("TtsMaker", {"text": "hello", "speaker_id": 0})] and env.loco.calls == []
    env.step(act(("WaveHand", {"turn_flag": True})))
    assert env.loco.calls == [("WaveHand", {"turn_flag": True})] and len(env.audio.calls) == 1
    assert env.audio_calls == 1 and env.has_audio
    with pytest.raises(RuntimeError, match="not allowed"):
        env.step(act(("Damp", {})))
    with pytest.raises(RuntimeError, match="not allowed"):
        env.step(act(("SetVolume", {"volume": 100})))
    assert len(env.arm.sent) == 2                                 # the refused calls sent nothing
    env.report()
    assert "audio: 1 call(s)" in capsys.readouterr().out


def test_with_the_speaker_off_speech_is_printed_never_said(capsys):
    env = robot(speaker=False)                                    # g1 start's /speaker off
    env.step(act(("TtsMaker", {"text": "I found it", "speaker_id": 0})))
    env.step(act(("WaveHand", {"turn_flag": True})))
    assert env.audio.calls == [] and env.audio_calls == 0 and env.loco.calls == [("WaveHand", {"turn_flag": True})]
    assert "robot (speaker off): I found it" in capsys.readouterr().out and len(env.arm.sent) == 2


def test_a_missing_or_failing_speaker_never_stops_the_run(capsys):
    env = robot()
    env.audio = None
    env.step(act(("TtsMaker", {"text": "hi", "speaker_id": 0})))
    env.step(act(("TtsMaker", {"text": "hi", "speaker_id": 0})))
    out = capsys.readouterr().out
    assert out.count("audio unavailable") == 1 and len(env.arm.sent) == 2 and not env.has_audio
    env = robot()
    env.audio = Client(fail=True)
    env.step(act(("TtsMaker", {"text": "hi", "speaker_id": 0})))
    assert "TtsMaker returned RuntimeError" in capsys.readouterr().out and len(env.arm.sent) == 1
    env.loco = Client(fail=True)
    with pytest.raises(RuntimeError, match="rpc failed"):          # a failed gesture still raises
        env.step(act(("WaveHand", {"turn_flag": False})))




class Loco:
    def __init__(self, fsm, set_code=0, get_code=0):
        self.fsm = fsm
        self.calls = []
        self.set_code = set_code
        self.get_code = get_code

    def SetTimeout(self, t):
        pass

    def Init(self):
        pass

    def GetFsmId(self):
        return self.get_code, self.fsm

    def SetFsmId(self, fsm):
        self.calls.append(("SetFsmId", fsm))
        if self.set_code == 0:
            self.fsm = fsm
        return self.set_code

    def Damp(self):
        self.calls.append(("Damp",))


def bring_up(monkeypatch, fsm, capsys, set_code=0, get_code=0, arm_cls=None, **settings):
    """setup() with the SDK stubbed out: the FSM handling and the messages."""
    import sys, types
    monkeypatch.setattr("time.sleep", lambda s: None)
    loco = Loco(fsm, set_code, get_code)
    core = types.ModuleType("unitree_sdk2py.core.channel"); core.ChannelFactoryInitialize = lambda *a: None
    lc = types.ModuleType("unitree_sdk2py.g1.loco.g1_loco_client"); lc.LocoClient = lambda: loco
    au = types.ModuleType("unitree_sdk2py.g1.audio.g1_audio_client"); au.AudioClient = lambda: Client()
    for name, mod in {"unitree_sdk2py": types.ModuleType("unitree_sdk2py"), "unitree_sdk2py.core": types.ModuleType("c"),
                      "unitree_sdk2py.core.channel": core, "unitree_sdk2py.g1": types.ModuleType("g"),
                      "unitree_sdk2py.g1.loco": types.ModuleType("l"), "unitree_sdk2py.g1.loco.g1_loco_client": lc,
                      "unitree_sdk2py.g1.audio": types.ModuleType("a"), "unitree_sdk2py.g1.audio.g1_audio_client": au}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    built = []

    def arm_sdk(health=None, link=None):
        built.append(1)
        return Arm()
    monkeypatch.setattr("g1.envs.robot.ArmSdk", arm_cls or arm_sdk)
    env = RobotEnv(run_settings(iface="lo", **{"mode": "standing", **settings}))
    env.setup()
    out = capsys.readouterr().out
    env.arms_built = len(built)
    return env, loco, out


def test_standing_mode_skips_the_transition_when_already_in_control(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 200, capsys)
    assert loco.calls == [] and "control mode in FSM 200 (main operation): no FSM change" in out and "warning" not in out
    assert "Taking over" not in out and "countdown" not in out
    env.arm.state = None
    env._teardown_robot()
    assert loco.calls == [] and "leaving control mode: releasing the arms" in capsys.readouterr().out


def test_bring_up_stops_before_the_arms_on_a_failed_transition(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="3104.*g1 status"):
        bring_up(monkeypatch, 1, capsys, set_code=3104, mode="gantry")  # no reply: the link
    with pytest.raises(SystemExit, match="refused"):
        bring_up(monkeypatch, 1, capsys, set_code=7001, mode="gantry")  # a service code: the robot refused
    with pytest.raises(SystemExit, match="GetFsmId failed with 3102"):
        bring_up(monkeypatch, 500, capsys, get_code=3102)
    with pytest.raises(SystemExit, match="did not enter FSM 200"):
        loco_stuck = Loco(1)
        loco_stuck.SetFsmId = lambda fsm: 0                             # accepted but never happens
        monkeypatch.setattr("tests.envs.test_robot.Loco", lambda *a: loco_stuck)
        bring_up(monkeypatch, 1, capsys, mode="gantry")


def test_a_run_stays_in_the_fsm_it_finds_and_warns_outside_200(monkeypatch, capsys):
    for fsm in (4, 500, 501, 801, 802):
        env, loco, out = bring_up(monkeypatch, fsm, capsys)
        assert loco.calls == [] and f"control mode in FSM {fsm}" in out and "verified only in FSM 200" in out
        env.arm.state = None
        env._teardown_robot()
        assert loco.calls == [] and f"fsm: {fsm} (" in capsys.readouterr().out
def test_standing_mode_refuses_a_robot_that_is_not_standing(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="1 \\(damp\\)"):
        bring_up(monkeypatch, 1, capsys)


def test_gantry_from_limp_gets_the_bring_up_and_stays_in_200(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 1, capsys, mode="gantry")
    assert loco.calls == [("Damp",), ("SetFsmId", 4), ("SetFsmId", 200)] and "fsm: 4 (locked stand)" in out
    env._teardown_robot()
    assert loco.calls[-1] == ("SetFsmId", 200) and "Damp" not in capsys.readouterr().out
    env, loco, out = bring_up(monkeypatch, 200, capsys, mode="gantry")   # g1 start brings it up to 200
    env.arm.state = None
    env._teardown_robot()
    assert loco.calls == []


def test_teardown_never_raises(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 4, capsys)
    loco.get_code = 3104
    env._teardown_robot()                                                # no answer: said, not raised
    assert "3104" in capsys.readouterr().out


def test_verbose_prints_every_call_and_the_health_line(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 1, capsys, verbose=True, mode="gantry")
    assert "sdk " in out and "LocoClient.GetFsmId() -> 0 (ok)" in out and "fsm at start: 1 (damp)" in out
    assert "SetFsmId(200) -> 0 (ok)" in out and "dds: ChannelFactoryInitialize" in out
    env.arm, env.monitor, env.base = Arm(), None, None
    env.overruns, env.ticks = 0, 0
    import time
    env._wall = time.time()
    env._health_at = time.monotonic() - 2.0
    env.step(act())
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("t=") and "lowstate" in line and "ticks 1 overruns 0" in line
    assert env.health_line().startswith("t=")
    env2, _, out2 = bring_up(monkeypatch, 500, capsys)
    assert "sdk " not in out2 and "dds:" not in out2                     # quiet by default
    assert len(env.sdk_log) >= 3 and all("code" in e for e in env.sdk_log)
    env.report()
    assert "sdk: " in capsys.readouterr().out


def test_probe_verdicts(monkeypatch, capsys):
    from g1.envs.robot import probe, status_main
    monkeypatch.setattr("time.sleep", lambda s: None)

    closed = []

    class Sub:
        def Close(self):
            closed.append(True)

    def run(fsm, code, lowstate):
        def subscribe(on_state):
            if lowstate:
                class Msg:
                    motor_state = [type("M", (), {"q": 0.1})()]
                    bms_state = type("B", (), {"soc": 77})()
                for _ in range(10):
                    on_state(Msg())
            return Sub()
        return probe("lo", 0.01, factory=lambda: None, loco_cls=lambda: Loco(fsm, get_code=code), subscribe=subscribe)

    out = run(200, 0, True)
    assert closed == [True]                           # /status in g1 start: its LowState reader ends with the probe
    assert out["verdict"].startswith("link ok, FSM 200") and out["lowstate"]["battery_soc"] == 77
    assert "not a standing state" in run(1, 0, True)["verdict"]
    assert run(0, 3104, False)["verdict"].startswith("no link")
    assert "refuses" in run(0, 7001, True)["verdict"]
    assert "no LowState" in run(200, 0, False)["verdict"]
    monkeypatch.setattr("g1.envs.robot.probe", lambda iface, seconds, verbose, **kw: run(200, 0, True))
    monkeypatch.setenv("UNITREE_IFACE", "lo")
    assert status_main([]) == 0
    text = capsys.readouterr().out
    assert "verdict:  link ok" in text and "lowstate: 10 messages" in text and "on lo" in text and "battery 77%" in text
    monkeypatch.delenv("UNITREE_IFACE")
    with pytest.raises(SystemExit):
        status_main([])                                                   # the interface comes from .env only
    assert "UNITREE_IFACE" in capsys.readouterr().err



def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("UNITREE_IFACE", "en7")
    monkeypatch.setenv("UNITREE_ROBOT_IP", "192.168.123.164")
    monkeypatch.setenv("G1_VERBOSE", "1")
    s = run_settings(mode="gantry")
    assert (s.iface, s.camera_ip, s.verbose, s.mode) == ("en7", "192.168.123.164", True, "gantry")
    assert RobotEnv(s).can_walk is False and RobotEnv(run_settings(mode="standing")).can_walk is True
    monkeypatch.delenv("UNITREE_IFACE")
    with pytest.raises(SystemExit, match="UNITREE_IFACE"):
        RobotEnv(run_settings(mode="standing"))._check_args()
    with pytest.raises(TypeError, match="no setting"):
        run_settings(walk=True)


def test_the_run_closes_its_lowstate_subscriber_after_the_release(monkeypatch, capsys):
    from g1.envs.robot import ArmSdk

    class Sub:
        closed = False

        def Close(self):
            Sub.closed = True

    class Link:
        def crc(self): return None
        def lowstate(self, cb): return Sub()
        def arm_sdk(self): return None
        def lowcmd(self):
            class M:
                q = dq = tau = kp = kd = 0.0
            return type("C", (), {"motor_cmd": [M() for _ in range(NUM_JOINTS + 6)]})()
    arm = ArmSdk(link=Link())
    arm.close()
    assert Sub.closed and arm.sub is None              # in g1 start the process outlives the run
    env, loco, out = bring_up(monkeypatch, 200, capsys, arm_cls=lambda health=None, link=None: Arm())
    env._teardown_robot()
    assert env.arm.sent[-2:] == ["release", "close"]


def test_a_session_run_uses_the_sessions_clients(monkeypatch, capsys):
    shared_loco, shared_audio = Loco(200), Client()
    env, loco, out = bring_up(monkeypatch, 200, capsys, loco_shared=shared_loco, audio_shared=shared_audio)
    assert env.loco is shared_loco and env.audio is shared_audio and "audio: volume" not in out


def test_the_lowstate_reader_is_closed_when_setup_fails_after_it_or_the_release_raises(monkeypatch, capsys):
    made = []

    class Closing(Arm):
        def __init__(self, *a, **k):
            super().__init__()
            made.append(self)

        def release(self, duration=1.0):
            raise RuntimeError("write failed")
    monkeypatch.setattr("g1.envs.robot.RobotEnv._make_monitor", lambda self: (_ for _ in ()).throw(RuntimeError("monitor")))
    with pytest.raises(RuntimeError, match="monitor"):
        bring_up(monkeypatch, 200, capsys, arm_cls=Closing)
    assert made[-1].sent == ["close"]                                    # no teardown follows a failed setup
    monkeypatch.undo()
    env, loco, out = bring_up(monkeypatch, 200, capsys, arm_cls=Closing)
    with pytest.raises(RuntimeError, match="write failed"):
        env._teardown_robot()
    assert made[-1].sent == ["close"]                                    # closed even though the release raised
