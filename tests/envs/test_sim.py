import math

import numpy as np
import pytest

from g1.cli import run, run_settings
from g1.envs import SimEnv
from g1.tools import Chain
from g1.tools.arms import TPose
from g1.tools.control import Hold
from g1.tools.move import Move


def sim(**settings):
    pytest.importorskip("mujoco")
    return SimEnv(run_settings(**settings))


def test_tpose_on_the_twin_keeps_it_standing_and_sags_like_the_robot():
    env = sim()
    assert run(Chain(TPose()), env) is True
    assert env.twin.pelvis_z == pytest.approx(0.75, abs=0.02) and env.twin.onboard.fallen_at is None   # GR00T's height
    x, y, yaw = env.base_pose()
    assert math.hypot(x, y) < 0.02 and abs(yaw) < math.radians(2)       # an arm tool does not move the base
    # the arms are on arm_sdk's kp 60 now, not stiff servos: the measured pose sags below the command
    assert env.cmd_max[16] == pytest.approx(1.57) and 1.40 < env.q_max[16] < 1.57
    assert np.all(np.isfinite(env.twin.lowstate().q))


def test_the_robot_code_path_runs_against_the_twin():
    env = sim()
    assert run(Chain(Hold(seconds=0.5), pause=0.0), env) is True
    calls = [(e["client"], e["name"]) for e in env.sdk_log]
    assert ("LocoClient", "GetFsmId") in calls and ("AudioClient", "GetVolume") in calls      # the robot's bring-up
    apis = [(c["service"], c["api"]) for c in env.twin.services.log]
    assert ("sport", 7001) in apis and ("voice", 1005) in apis
    assert env.twin.onboard.arm is not None and env.twin.onboard.arm.weight == 0.0           # released at the end


def test_move_steps_the_robot_to_the_target():
    env = sim()
    assert run(Chain(Move(dx_m=0.6, dyaw_deg=30)), env) is True
    assert env.monitor.base_pose()[0] == pytest.approx(0.6, abs=1e-6)       # what was commanded, exactly
    x, y, yaw = env.base_pose()                                             # where the twin actually went
    assert x == pytest.approx(0.6, abs=0.12) and abs(y) < 0.12
    assert math.degrees(yaw) == pytest.approx(30, abs=10)
    assert env.twin.onboard.walk_ticks > 0 and not env.twin.onboard.walking   # it stepped (Walk), then stood (Balance)


def test_damp_drops_it_unless_it_hangs_in_the_gantry():
    from g1.twin.core import Twin
    twin = Twin()
    twin.call("sport", 7101, '{"data": 1}')
    twin.advance(3.0)
    assert twin.onboard.fallen_at is not None and twin.pelvis_z < 0.45
    env = sim(mode="gantry")                                          # Damp -> FSM 4 -> FSM 200 -> run -> Damp
    assert run(Chain(Hold(seconds=0.5), pause=0.0), env) is True
    assert env.twin.onboard.fallen_at is None and env.twin.onboard.fsm == 1 and env.twin.pelvis_z > 0.5


def test_the_dead_man_stops_the_base_a_second_after_the_last_move():
    from g1.twin.core import Twin
    twin = Twin()
    twin.call("sport", 7105, '{"velocity": [0.2, 0, 0], "duration": 1}')
    twin.advance(0.5)
    assert twin.onboard.walking
    twin.advance(1.0)
    assert not twin.onboard.walking                     # the 1 s dead-man ran out: Balance, at once
    twin.advance(3.0)
    x = twin.base_pose()[0]
    assert 0.1 < x < 0.4 and not twin.onboard.walking
    twin.advance(2.0)
    assert twin.base_pose()[0] == pytest.approx(x, abs=0.01)


def test_torques_stay_inside_the_motor_limits_and_runs_repeat():
    from g1.twin.core import Twin

    def trial():
        twin = Twin()
        for _ in range(20):
            twin.call("sport", 7105, '{"velocity": [0.3, 0, 0.3], "duration": 1}')
            twin.advance(0.1)
            lo, hi = twin.model.actuator_ctrlrange.T
            assert np.all(twin.data.actuator_force >= lo - 1e-9) and np.all(twin.data.actuator_force <= hi + 1e-9)
        return twin.data.qpos.copy()
    assert np.array_equal(trial(), trial())                                # bit-identical


def test_head_camera_renders_on_sim_time_only_when_asked():
    env = sim(camera="on", sim_obstacle=(1.2, 0.0, 0.225))
    env2 = sim()
    assert run(Chain(Hold(seconds=1.0), pause=0.0), env) is True
    f = env.frame()
    assert f is not None and f.image.shape == (480, 640, 3) and f.image.std() > 0
    assert env.camera.count > 50                     # every --camera-every ticks, stamped on sim time
    assert f.stamp <= env.clock()
    assert run(Chain(Hold(seconds=0.5), pause=0.0), env2) is True   # a chain does not use the camera: no render
    assert env2.frame() is None and env2.renderer is None


def test_a_fall_fails_the_run_through_the_twins_summary(capsys):
    env = sim()
    original = env.report

    def report():                       # the run went fine, then the twin reports it fell
        env.twin.onboard.fallen_at = 1.23
        return original()
    env.report = report
    assert run(Chain(Hold(seconds=0.3), pause=0.0), env) is False
    out = capsys.readouterr().out
    assert "PASS" in out and "FAIL: the robot fell at t = 1.23 s" in out and "twin: pelvis z =" in out


def test_it_stands_still_while_the_arms_swing_and_takes_a_push():
    """GR00T's Balance network: no marching in place, no drift, and a shove does not floor it."""
    from g1.tools.arms import SixSeven
    env = sim()
    assert run(Chain(SixSeven(reps=2, settle_s=1.0, hold_s=0.1)), env) is True
    x, y, yaw = env.base_pose()
    assert math.hypot(x, y) < 0.05 and abs(math.degrees(yaw)) < 3 and env.twin.onboard.walk_ticks == 0
    from g1.twin.core import Twin
    for force in ((60.0, 0, 0), (0, 120.0, 0)):           # the old walker fell to the 120 N side push
        twin = Twin()
        twin.advance(1.0)
        twin.data.xfrc_applied[twin.torso, :3] = force
        twin.advance(0.1)
        twin.data.xfrc_applied[twin.torso, :3] = 0
        twin.advance(4.0)
        assert twin.onboard.fallen_at is None and twin.pelvis_z > 0.7, force
