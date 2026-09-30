"""The robot env's onboard-call dispatch, without the SDK: stub clients stand in."""
import numpy as np
import pytest

from g1.cli import run_parser
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


def robot(*extra):
    env = RobotEnv(run_parser().parse_args(["--env", "robot", "--tools", "x", "--iface", "lo", "--mode", "standing", *extra]))
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


def test_volume_flag_parses():
    args = run_parser().parse_args(["--env", "robot", "--tools", "x", "--iface", "lo", "--mode", "standing", "--volume", "40"])
    assert args.volume == 40
    assert run_parser().parse_args(["--env", "robot", "--tools", "x"]).volume is None


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


def bring_up(monkeypatch, fsm, capsys, set_code=0, get_code=0, extra=(), arm_cls=None):
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

    def arm_sdk(health=None):
        built.append(1)
        return Arm()
    monkeypatch.setattr("g1.envs.robot.ArmSdk", arm_cls or arm_sdk)
    env = RobotEnv(run_parser().parse_args(["--env", "robot", "--tools", "x", "--iface", "lo", "--mode", "standing", *extra]))
    env.setup()
    out = capsys.readouterr().out
    env.arms_built = len(built)
    return env, loco, out


def test_standing_mode_skips_the_transition_when_already_in_control(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 200, capsys)
    assert loco.calls == [] and "already in control mode (FSM 200" in out
    assert "Taking over" not in out and "countdown" not in out
    env.arm.state = None
    env._teardown_robot()
    assert loco.calls == [] and "leaving control mode: releasing the arms" in capsys.readouterr().out


def test_standing_mode_enters_and_restores_another_fsm(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 500, capsys)
    assert loco.calls == [("SetFsmId", 200)] and "entering control mode (FSM 200" in out and "FSM 500" in out
    env._teardown_robot()
    assert loco.calls[-1] == ("SetFsmId", 500) and "FSM 500 (start) restored" in capsys.readouterr().out


def test_countdown_is_gone():
    with pytest.raises(SystemExit):
        run_parser().parse_args(["--env", "robot", "--tools", "x", "--countdown", "3"])


def test_bring_up_stops_before_the_arms_on_a_failed_transition(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="3104.*g1 status"):
        bring_up(monkeypatch, 500, capsys, set_code=3104)               # no reply: the link
    with pytest.raises(SystemExit, match="refused"):
        bring_up(monkeypatch, 500, capsys, set_code=7001)               # a service code: the robot refused
    with pytest.raises(SystemExit, match="GetFsmId failed with 3102"):
        bring_up(monkeypatch, 500, capsys, get_code=3102)
    with pytest.raises(SystemExit, match="did not enter FSM 200"):
        loco_stuck = Loco(500)
        loco_stuck.SetFsmId = lambda fsm: 0                             # accepted but never happens
        monkeypatch.setattr("tests.envs.test_robot.Loco", lambda *a: loco_stuck)
        bring_up(monkeypatch, 500, capsys)


def test_standing_mode_refuses_a_robot_that_is_not_standing(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="1 \\(damp\\)"):
        bring_up(monkeypatch, 1, capsys)
    env, loco, out = bring_up(monkeypatch, 4, capsys)                    # locked stand: fine, restored at the end
    assert loco.calls == [("SetFsmId", 200)]
    env._teardown_robot()
    assert loco.calls[-1] == ("SetFsmId", 4) and "4 (locked stand) restored" in capsys.readouterr().out


def test_teardown_never_raises(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 4, capsys)
    loco.set_code = 3104
    env._teardown_robot()                                                # a failed restore is a warning
    text = capsys.readouterr().out
    assert "could not restore FSM 4" in text and "3104" in text and "done" in text


def test_verbose_prints_every_call_and_the_health_line(monkeypatch, capsys):
    env, loco, out = bring_up(monkeypatch, 500, capsys, extra=["--verbose"])
    assert "sdk " in out and "LocoClient.GetFsmId() -> 0 (ok)" in out and "fsm at start: 500 (start)" in out
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

    def run(fsm, code, lowstate):
        def subscribe(on_state):
            if lowstate:
                class Msg:
                    motor_state = [type("M", (), {"q": 0.1})()]
                    bms_state = type("B", (), {"soc": 77})()
                for _ in range(10):
                    on_state(Msg())
        return probe("lo", 0.01, factory=lambda: None, loco_cls=lambda: Loco(fsm, get_code=code), subscribe=subscribe)

    out = run(200, 0, True)
    assert out["verdict"].startswith("link ok, FSM 200") and out["lowstate"]["battery_soc"] == 77
    assert "not a standing state" in run(1, 0, True)["verdict"]
    assert run(0, 3104, False)["verdict"].startswith("no link")
    assert "refuses" in run(0, 7001, True)["verdict"]
    assert "no LowState" in run(200, 0, False)["verdict"]
    monkeypatch.setattr("g1.envs.robot.probe", lambda iface, seconds, verbose: run(200, 0, True))
    assert status_main(["--iface", "lo"]) == 0
    text = capsys.readouterr().out
    assert "verdict:  link ok" in text and "lowstate: 10 messages" in text and "battery 77%" in text
    assert status_main(["--iface", "lo", "--json"]) == 0
    import json
    assert json.loads(capsys.readouterr().out)["rpc"]["fsm"] == 200
    with pytest.raises(SystemExit):
        status_main([])


def test_iface_defaults_to_the_environment(monkeypatch):
    monkeypatch.setenv("UNITREE_IFACE", "en7")
    assert run_parser().parse_args(["--env", "robot", "--tools", "x"]).iface == "en7"
    assert run_parser().parse_args(["--env", "robot", "--tools", "x", "--iface", "lo"]).iface == "lo"
    monkeypatch.delenv("UNITREE_IFACE")
    assert run_parser().parse_args(["--env", "robot", "--tools", "x"]).iface is None
