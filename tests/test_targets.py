import math
import time

import numpy as np
import pytest

from behaviors import Face, GoTo, Look
from camera import ClockedCamera, Frame
from config import BASE_VEL_MAX, CONTROL_DT, STAND_Q, UPPER_BODY, joint_index
from envs import RobotEnv
from envs.robot import BaseCommander
from perception import Detected, Percept
from policy import Action, Obs, ReactivePolicy
from run import build_parser, run
from targets import Doorway, Labeled, RedDot, Salient, Sighting, Target, seen
from vision import bearing, elevation

WAIST_YAW = joint_index("waist_yaw")
SHAPE = (48, 64, 3)


def solid(color=(0, 0, 0), h=48, w=64):
    return np.full((h, w, 3), color, dtype=np.uint8)


def frame(seq=1, image=None, stamp=0.0):
    return Frame(solid() if image is None else image, stamp, seq)


def red_square(cx, cy, side, h=48, w=64):
    img = solid(h=h, w=w)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    img[max(0, y0):y0 + side, max(0, x0):x0 + side] = (230, 20, 20)
    return img


from tests.doubles import scripted_sim, sim_env as check_env


# -- Sighting / Target --------------------------------------------------------------

def test_sighting_geometry():
    s = Sighting.from_box(1.0, 1.0, 0.5, 0.2, SHAPE, 3, 1.5, "x")
    assert s.bearing == pytest.approx(bearing(1.0, SHAPE)) and s.bearing > 0
    assert s.elevation == pytest.approx(elevation(1.0, SHAPE)) and s.elevation < 0
    assert s.area == pytest.approx(0.1) and s.frame_seq == 3 and s.stamp == 1.5
    b = Sighting.from_blob(0.0, 0.0, 0.04, frame(7, stamp=2.0))
    assert b.bearing == 0 and b.elevation == 0 and b.width == pytest.approx(0.2) and b.area == pytest.approx(0.04)
    d = Detected("chair", 0.25, 0.5, 0.1, 0.2, 1.5, bearing(-0.5, SHAPE), 0.0)
    p = Percept("s", [d], True, 9, 4.0)
    c = Sighting.from_detected(d, p)
    assert c.label == "chair" and c.distance_m == 1.5 and c.frame_seq == 9 and c.bearing < 0


class Counting(Target):
    name = "counting"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def locate(self, obs):
        self.calls += 1
        return None if obs.frame.image[0, 0, 0] == 0 else Sighting.from_box(0.5, 0.5, 0.1, 0.1, SHAPE, obs.frame.seq, obs.frame.stamp)


def test_target_update_only_on_new_input():
    tgt = Counting()
    assert tgt.update(Obs(STAND_Q), 0.0) is None and tgt.calls == 0        # no frame
    miss = frame(1)
    assert tgt.update(Obs(STAND_Q, miss, 0.0), 0.0) is None and tgt.calls == 1
    assert tgt.update(Obs(STAND_Q, miss, 0.02), 0.02) is None and tgt.calls == 1   # same seq
    assert tgt.age(1.0) == math.inf and not tgt.seen(1.0, 5.0)
    hit = frame(2, solid((255, 0, 0)))
    s = tgt.update(Obs(STAND_Q, hit, 0.1), 1.0)
    assert s is not None and tgt.calls == 2 and tgt.last is s and tgt.current is s
    assert tgt.age(1.0) == pytest.approx(0.1) and tgt.age(2.0) == pytest.approx(1.1)
    assert tgt.seen(1.5, 1.0) and not tgt.seen(3.0, 1.0)
    assert tgt.update(Obs(STAND_Q, frame(3)), 3.0) is None and tgt.current is None and tgt.last is s
    tgt.reset()
    assert tgt.last is None and tgt.age(0.0) == math.inf


def test_reddot_locate_and_reached():
    dot = RedDot()
    obs = Obs(STAND_Q, frame(1, red_square(48, 24, 8)), 0.0)
    s = dot.locate(obs)
    assert s is not None and s.bearing > 0 and abs(s.elevation) < 1e-6 and s.label == "red_dot"
    assert not dot.reached(s)
    assert dot.locate(Obs(STAND_Q, frame(2), 0.0)) is None
    big = dot.locate(Obs(STAND_Q, frame(3, red_square(32, 24, 12)), 0.0))    # 144/3072 = 0.047
    assert dot.reached(big)
    low = dot.locate(Obs(STAND_Q, frame(4, red_square(32, 45, 4)), 0.0))     # bottom of the frame
    assert low.elevation < -0.2 and dot.reached(low)
    assert seen(dot)(obs) and not seen(dot)(Obs(STAND_Q)) and not seen(dot)(Obs(STAND_Q, frame(5)))


def test_labeled_salient_doorway():
    chair = Detected("wooden chair", 0.3, 0.5, 0.2, 0.3, 1.2, bearing(-0.4, SHAPE), 0.0)
    door = Detected("open door", 0.7, 0.4, 0.3, 0.8, 3.0, bearing(0.4, SHAPE), 0.05)
    p = Percept("a room", [chair, door], True, 5, 1.0)
    p.seq = 1
    obs = Obs(STAND_Q, None, math.inf, p, 0.5)
    assert Labeled(("chair",)).locate(obs).label == "wooden chair"
    assert Labeled(("sofa", "door"), name="furniture").locate(obs).label == "open door"
    assert Labeled(("cat",)).locate(obs) is None
    assert Salient().locate(obs).label == "open door"        # largest area
    dw = Doorway()
    s = dw.locate(obs)
    assert s.distance_m == 3.0 and s.frame_seq == 5 and dw.uses_vision and not dw.can_reach
    assert dw.reached(s) is False
    assert dw.key(obs) == 1 and dw.input_age(obs) == 0.5 and dw.key(Obs(STAND_Q, frame(1))) is None
    with pytest.raises(ValueError):
        GoTo(Doorway())


# -- Face --------------------------------------------------------------------------

def test_face_reddot_matches_look():
    def yaw_after(policy, img):
        policy.reset(Obs(STAND_Q))
        a = None
        for n in range(40):
            a = policy.step(n * CONTROL_DT, Obs(STAND_Q, Frame(img, 0.0, 1), 0.0))
        return a.q[WAIST_YAW]

    img = red_square(56, 24, 8)                     # far right
    y_face = yaw_after(Face(RedDot(), 5.0, ramp=0.0, to_stand=0.0), img)
    y_look = yaw_after(Look(5.0, ramp=0.0, to_stand=0.0), img)
    u = 2 * 55.5 / 63 - 1                          # the 8 px square spans columns 52..59
    assert y_face == y_look == pytest.approx(-0.6 * bearing(u, SHAPE), abs=1e-6)
    assert Face(Doorway()).uses_vision and Face(Doorway()).name == "face_doorway"


# -- GoTo through check --------------------------------------------------------------

class Scripted(ClockedCamera):
    def __init__(self, images, fps=10.0):
        super().__init__(fps)
        self.images = images

    def image(self, i):
        return self.images[min(i, len(self.images) - 1)]


def scripted_env(images, fps=10.0):
    return scripted_sim(Scripted(images, fps))


def bases(env):
    return [a.base for a in env.actions]


def test_goto_reaches_and_stops():
    n = 40                                              # 4 s of frames: dot drifts to centre and grows
    images = [red_square(32 + 20 * (1 - i / n), 24, 4 + int(10 * i / n)) for i in range(n)]
    env = scripted_env(images)
    p = GoTo(RedDot(), timeout=20.0, ramp=0.2, to_stand=0.2)
    assert run(p, env) is True
    assert env.violations == [] and p.reached
    cmds = [b for b in bases(env) if b is not None]
    assert cmds and all(abs(b[i]) <= BASE_VEL_MAX[i] + 1e-9 for b in cmds for i in range(3))
    assert any(b[0] == 0 and b[2] < 0 for b in cmds)   # turned right first, without walking
    assert any(b[0] > 0.15 for b in cmds)              # then walked
    assert env.base_path > 0 and env.base_pose()[2] < 0
    last = max(i for i, b in enumerate(bases(env)) if b is not None)
    assert all(b is None for b in bases(env)[last + 1:])          # base stopped before handback
    assert len(bases(env)) - last > 0.4 / CONTROL_DT
    assert env.ticks < (20.0 + 0.8) / CONTROL_DT          # finished early


def test_goto_holds_when_lost_then_times_out():
    images = [red_square(32, 24, 6)] * 10 + [solid()] * 30
    env = scripted_env(images)
    p = GoTo(RedDot(), timeout=4.0, ramp=0.2, to_stand=0.2, lost_after=0.5)
    assert run(p, env) is True
    assert env.violations == [] and not p.reached and env.base_path > 0
    drive = [b for b in bases(env) if b is not None]
    assert drive[-1] == (0.0, 0.0, 0.0)                    # decayed to a stop before timeout
    assert env.ticks == round((4.0 + 0.8) / CONTROL_DT)


def test_goto_passes_check_under_noise():
    env = check_env("--camera-noise")
    p = GoTo(RedDot(), timeout=3.0)
    assert run(p, env) is True and env.violations == []


def test_check_flags_base_over_limit(capsys):
    class Fast(ReactivePolicy):
        name = "fast"

        def track(self, t, obs):
            return {}

        def drive(self, t, obs):
            return (1.0, 0.0, 0.0)

    env = check_env()
    assert run(Fast(0.5, ramp=0.0, to_stand=0.0), env) is False
    kinds = {v.kind for v in env.violations}
    assert kinds == {"base_vx"}
    out = capsys.readouterr().out
    assert "base:" in out
    lines = [l for l in out.splitlines() if "base_vx" in l and "t=" in l]
    assert lines and all(" -  " in l for l in lines)


def test_finish_shortens_reactive_policy():
    class Quit(ReactivePolicy):
        name = "quit"

        def track(self, t, obs):
            return {}

        def drive(self, t, obs):
            if t > 0.5:
                self.finish()
            return (0.1, 0.0, 0.0)

    env = check_env()
    p = Quit(10.0, ramp=0.2, to_stand=0.2)
    assert run(p, env) is True
    assert env.ticks == pytest.approx((0.2 + 0.2 + 0.52 + 0.2 + 0.2) / CONTROL_DT, abs=2)
    assert p.phase == "handback"


def test_action_base_validation():
    Action(STAND_Q, UPPER_BODY, base=(0.1, 0, 0))
    with pytest.raises(ValueError):
        Action(STAND_Q, UPPER_BODY, base=(0.1, 0))
    with pytest.raises(ValueError):
        Action(STAND_Q, UPPER_BODY, base=(math.nan, 0, 0))


# -- robot env (no SDK) ---------------------------------------------------------------

class StubMotor:
    q = 0.0


class StubArm:
    def __init__(self):
        self.sent = []
        self.state = type("S", (), {"motor_state": [StubMotor()] * 35})()

    def send(self, action):
        self.sent.append(action)


def test_robot_base_requires_walk():
    args = build_parser().parse_args(["--env", "robot", "--policy", "x", "--iface", "lo", "--mode", "standing"])
    env = RobotEnv(args)
    env.arm = StubArm()
    env.camera = None
    env.overruns = 0
    env._wall = time.time()
    env.step(Action(STAND_Q, UPPER_BODY))
    with pytest.raises(RuntimeError, match="--walk"):
        env.step(Action(STAND_Q, UPPER_BODY, base=(0.1, 0, 0)))
    assert len(env.arm.sent) == 1


class StubLoco:
    def __init__(self):
        self.moves = []
        self.stops = 0

    def Move(self, vx, vy, vyaw):
        self.moves.append((vx, vy, vyaw))
        return 0

    def StopMove(self):
        self.stops += 1


def test_base_commander_cadence_and_stop():
    loco = StubLoco()
    cmd = BaseCommander(loco, period=0.02)
    cmd.start()
    cmd.command((1.0, 0.0, 0.0))                          # clamped to BASE_VEL_MAX
    time.sleep(0.15)
    assert len(loco.moves) >= 3 and loco.moves[0] == (BASE_VEL_MAX[0], 0.0, 0.0)
    cmd.command(None)
    time.sleep(0.06)
    assert loco.stops == 1
    n = len(loco.moves)
    time.sleep(0.06)
    assert len(loco.moves) == n                           # nothing re-sent while cleared
    cmd.command((0.1, 0.0, 0.1))
    time.sleep(0.05)
    cmd.stop()
    assert loco.stops == 2 and "command(s)" in cmd.summary()
    assert not cmd._thread.is_alive()
