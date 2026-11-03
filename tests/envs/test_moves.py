import pytest

from g1.envs import moves
from g1.envs.moves import Moves, fsm_path, inventory, resolve

# The shapes the robot answered with (G1, 2026-11-02)
MIMIC = {"fsm_id": 550, "api_id": 1, "motion": {"516": {"duration": 4.92, "name": "bow"},
                                                "504": {"duration": 30.02, "name": "vq_POPPING9_001_v1_G1_50hz"},
                                                "535": {"duration": 3.6, "name": "vq_clap"}}}
ARM = [[{"id": 99, "name": "release_arm"}, {"fsm": [500, 501], "id": 1, "name": "turn_back_wave"},
        {"id": 18, "name": "high_five"}, {"id": 20, "mode_machine": [5, 6], "name": "make_heart_with_both_hands"}],
       [{"name": "Waist_Drum_Dance", "time": 9.5}]]


class Loco:
    """SetFsmId moves the reported FSM; a mimic id (550xxx) reports 550 for ``play_polls``
    GetFsmId calls, then 802, as the robot did (2026-11-03)."""
    def __init__(self, fsm, play_polls=5):
        self.fsm, self.calls, self.play_polls, self.left = fsm, [], play_polls, 0

    def GetFsmId(self):
        if self.fsm == 550:
            self.left -= 1
            if self.left < 0:
                self.fsm = 802
        return 0, self.fsm

    def SetFsmId(self, fsm_id):
        self.calls.append(fsm_id)
        if fsm_id >= 550000:
            self.fsm, self.left = 550, self.play_polls
        else:
            self.fsm = fsm_id
        return 0

    def GetMimicMotion(self):
        return 0, MIMIC

    def SetVelocity(self, vx, vy, vyaw, duration):
        self.calls.append(("vel", round(vx, 3), round(vy, 3), round(vyaw, 3)))
        return 0


class Arm:
    def __init__(self):
        self.calls = []

    def GetActionList(self):
        return 0, ARM

    def ExecuteAction(self, action_id):
        self.calls.append(action_id)
        return 0, None

    def ExecuteCustomAction(self, name):
        self.calls.append(name)
        return 0, None

    def StopCustomAction(self):
        self.calls.append("stop")
        return 0, None


def make(fsm, confirm=True, interrupt_after=None, play_polls=5):
    clock = {"t": 0.0}

    def sleep(dt):
        clock["t"] += dt
        if interrupt_after is not None and clock["t"] > interrupt_after:
            raise KeyboardInterrupt
    m = Moves(Loco(fsm, play_polls), Arm(), sleep=sleep, now=lambda: clock["t"], confirm=lambda q: confirm)
    return m


def test_fsm_order_never_goes_through_damp():
    assert fsm_path(0, 501) == [1, 4, 501]                 # zero torque: damp, locked stand, walk (29-DoF)
    assert fsm_path(4, 200) == [200]
    assert fsm_path(3, 501) == [801, 501]                   # sitting: stand up first
    assert fsm_path(500, 1) == [1]                          # damp is reachable as the end
    assert fsm_path(200, 4) is None                         # but never a way through (it would drop it)
    assert fsm_path(500, 500) == []


def test_inventory_and_resolve_match_ids_and_names():
    inv = inventory(Loco(500), Arm(), [])
    assert set(inv["mimic"]) == {516, 504, 535} and [a["id"] for a in inv["arm"]] == [99, 1, 18, 20]
    assert inv["custom"] == [{"name": "Waist_Drum_Dance", "time": 9.5}]
    assert resolve("516", inv)["label"] == "bow" and resolve("Bow", inv)["id"] == 516
    assert resolve("clap", inv)["id"] == 535 and resolve("vq_clap", inv)["id"] == 535   # the vq_ prefix is optional
    assert resolve("arm:high_five", inv)["id"] == 18 and resolve("arm:20", inv)["mode_machine"] == [5, 6]
    assert resolve("custom:waist_drum_dance", inv)["duration"] == 9.5
    assert resolve("moonwalk", inv) is None and resolve("arm:999", inv) is None
    text = "\n".join(moves.list_lines(inv, 5))
    assert "SetFsmId(550516)" in text and "popping dance" in text and "release_arm" not in text


def test_switch_walks_the_order_and_asks_first():
    m = make(0)
    m.switch(501)
    assert m.loco.calls == [1, 4, 501] and m.loco.fsm == 501
    m = make(0, confirm=False)
    with pytest.raises(SystemExit, match="not switched"):
        m.switch(500)
    assert m.loco.calls == []
    with pytest.raises(SystemExit, match="no allowed way"):
        make(200).switch(4)


def test_play_from_standing_only_then_mimic_arm_custom_in_order():
    inv = inventory(Loco(500), Arm(), [])
    with pytest.raises(SystemExit, match="play from standing"):
        make(0).play([resolve("bow", inv)])
    with pytest.raises(SystemExit, match="only in FSM"):
        make(200).play([resolve("arm:turn_back_wave", inv)])
    m = make(500)
    assert m.play([resolve("bow", inv), resolve("arm:18", inv), resolve("custom:Waist_Drum_Dance", inv)]) == 0
    assert m.loco.calls == [550516]
    assert m.arm.calls == [18, moves.ARM_RELEASE, "Waist_Drum_Dance"]       # an arm action is released after
    m = make(500, confirm=False)
    with pytest.raises(SystemExit, match="not played"):
        m.play([resolve("bow", inv)])
    assert m.loco.calls == []


def test_ctrl_c_stops_the_move_back_to_the_start_fsm():
    inv = inventory(Loco(500), Arm(), [])
    m = make(500, interrupt_after=1.0)
    assert m.play([resolve("vq_POPPING9_001_v1_G1_50hz", inv), resolve("bow", inv)]) == 130
    assert m.loco.calls == [550504, 500]                    # the second move never starts


def test_main_refuses_the_twin(monkeypatch, capsys):
    monkeypatch.setenv("UNITREE_IFACE", "lo0")
    with pytest.raises(SystemExit):
        moves.main(["list"])
    assert "twin has none" in capsys.readouterr().err


def test_a_move_that_runs_past_its_duration_is_waited_for():
    inv = inventory(Loco(500), Arm(), [])
    poll = moves.limits.get("fsm_poll_s")
    bow = resolve("bow", inv)                                               # 4.92 s listed
    m = make(501, play_polls=int((bow["duration"] + 6.0) / poll))           # it reports 550 for about 11 s
    assert m.play([bow, resolve("clap", inv)]) == 0
    assert m.loco.calls == [550516, 550535] and m.loco.fsm == 802          # the second played from 802
    m = make(501, play_polls=10 ** 6)                                       # it never leaves 550
    m.play([bow])
    assert m.now() < bow["duration"] + moves.limits.get("moves_settle_s") + moves.limits.get("moves_overrun_s") + 1


def test_walk_turns_walks_sidesteps_and_always_stops():
    m = make(802)                                                     # after a stored move: run mode first
    assert m.walk(forward_m=0.4, side_m=-0.15, turn_deg=90) == 0
    assert m.loco.calls[0] == moves.limits.get("walk_fsm") == 801
    vel = [c for c in m.loco.calls if isinstance(c, tuple)]
    speed, side, rate = (moves.limits.get(k) for k in ("walk_speed", "side_speed", "turn_rate"))
    assert ("vel", 0.0, 0.0, round(rate, 3)) in vel and ("vel", round(speed, 3), 0.0, 0.0) in vel
    assert ("vel", 0.0, round(-side, 3), 0.0) in vel
    assert vel[-1] == ("vel", 0.0, 0.0, 0.0)                          # it ends stopped
    poll = moves.limits.get("base_send_period_s")
    assert len([v for v in vel if v[1] > 0]) == pytest.approx(0.4 / speed / poll, abs=2)   # re-sent: the 1 s dead-man
    m = make(801, interrupt_after=0.5)
    assert m.walk(forward_m=1.0) == 130
    assert m.loco.calls[-1] == ("vel", 0.0, 0.0, 0.0)                 # Ctrl-C: stopped at once
    assert make(501).walk() == 0
    m = make(501)
    m.walk(turn_deg=10)
    assert m.loco.calls[0] == 801                                     # walk mode -> run mode before walking
    assert fsm_path(0, 801) == [1, 4, 801] and fsm_path(802, 801) == [801]
