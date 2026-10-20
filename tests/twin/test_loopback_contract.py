"""The loopback's clients send exactly what unitree_sdk2py's do: the same API id, the
same JSON, the same return shape. The SDK clients are built without DDS (their
constructors open channels) and their ``_Call`` captured."""
import json

import pytest

from g1.twin.loopback import LoopbackAudio, LoopbackLoco

CALLS = [("GetFsmId", ()), ("SetFsmId", (200,)), ("SetFsmId", (4,)), ("Damp", ()), ("ZeroTorque", ()), ("Start", ()),
         ("Move", (0.2, -0.1, 0.4)), ("Move", (0.2, 0.0, 0.0, True)), ("StopMove", ()), ("SetVelocity", (0.1, 0.0, 0.0, 2.0)),
         ("WaveHand", ()), ("WaveHand", (True,)), ("ShakeHand", ()), ("ShakeHand", ()), ("ShakeHand", (0,)), ("ShakeHand", (1,))]
AUDIO = [("TtsMaker", ("hello", 0)), ("TtsMaker", ("again", 1)), ("GetVolume", ()), ("SetVolume", (60,)), ("LedControl", (1, 2, 3))]


class Recorder:
    """A twin that records what reaches it and answers like the robot."""

    def __init__(self):
        self.calls = []

    def call(self, service, api, parameter):
        self.calls.append((api, json.loads(parameter)))
        return 0, json.dumps({"data": 200} if api == 7001 else {"volume": 50} if api == 1005 else {})


def sdk_client(cls, **attrs):
    c = cls.__new__(cls)
    for k, v in attrs.items():
        setattr(c, k, v)
    c.sent = []

    def _call(api, parameter):
        c.sent.append((api, json.loads(parameter)))
        return 0, json.dumps({"data": 200} if api == 7001 else {"volume": 50} if api == 1005 else {})
    c._Call = _call
    return c


def test_loco_calls_match_the_sdk():
    loco = pytest.importorskip("unitree_sdk2py.g1.loco.g1_loco_client")
    sdk = sdk_client(loco.LocoClient, first_shake_hand_stage_=-1)
    twin = Recorder()
    ours = LoopbackLoco(twin)
    for name, args in CALLS:
        assert getattr(ours, name)(*args) == getattr(sdk, name)(*args), name     # Damp -> None, GetFsmId -> (0, 200)
    assert twin.calls == sdk.sent


def test_audio_calls_match_the_sdk():
    audio = pytest.importorskip("unitree_sdk2py.g1.audio.g1_audio_client")
    sdk = sdk_client(audio.AudioClient, tts_index=0)
    twin = Recorder()
    twin.call = lambda service, api, p: (twin.calls.append((api, json.loads(p))) or
                                         (0, json.dumps({"volume": 50} if api == 1005 else {})))
    ours = LoopbackAudio(twin)
    for name, args in AUDIO:
        assert getattr(ours, name)(*args) == getattr(sdk, name)(*args), name
    assert twin.calls == sdk.sent


def test_services_answer_the_api_ids_the_sdk_registers():
    loco_api = pytest.importorskip("unitree_sdk2py.g1.loco.g1_loco_api")
    audio_api = pytest.importorskip("unitree_sdk2py.g1.audio.g1_audio_api")
    from g1.twin.services import APIS, SERVICES
    assert SERVICES["sport"] == loco_api.LOCO_API_VERSION and SERVICES["voice"] == audio_api.AUDIO_API_VERSION
    assert {loco_api.ROBOT_API_ID_LOCO_GET_FSM_ID, loco_api.ROBOT_API_ID_LOCO_SET_FSM_ID,
            loco_api.ROBOT_API_ID_LOCO_SET_VELOCITY, loco_api.ROBOT_API_ID_LOCO_SET_ARM_TASK} <= set(APIS["sport"])
    assert {audio_api.ROBOT_API_ID_AUDIO_TTS, audio_api.ROBOT_API_ID_AUDIO_GET_VOLUME,
            audio_api.ROBOT_API_ID_AUDIO_SET_VOLUME} <= set(APIS["voice"])
