"""point_to: the model names a pixel and a distance, the host aims the arm's line through that point.
Given the laptop's true pixel and distance (test-only, from the twin's truth), one point_to lands on
it, and the next observation's aim reports the small miss the twin measures too."""
import json
import math

import numpy as np
import pytest

from g1.agent.agent import Agent
from g1.agent.decider import Decider
from g1.cli import run
from g1.core import aim
from g1.core.config import WAIST
from g1.envs import scene
from g1.tools import menu
from tests.doubles import select, sim_env

pytestmark = pytest.mark.skipif(bool(scene.missing()), reason="room assets not fetched (g1 twin fetch)")

LAPTOP = {"scene": "room", "objects": ["laptop@1.75,-1.5,0.77"], "start": [1.2, -0.95, 0]}


class ClickTheTruth(Decider):
    """Turns 1..shots: point_to the target's true pixel and distance in that turn's image (or a pixel
    ``off`` to the side); then done."""

    model = "truth"

    def __init__(self, env, off_px=0.0, shots=1):
        super().__init__(threaded=False)
        self.env, self.off_px, self.shots, self.turns = env, off_px, shots, []

    def decide(self, turn):
        self.turns.append(turn)
        if len(self.turns) > self.shots:
            return select(self, turn, "done", {"summary": "s", "hindsight": ""})
        twin = self.env.twin
        d = twin.data
        target = twin.evaluator.target() - d.xpos[1]        # in the IMU's frame: world axes at the pelvis
        image = json.loads(turn.observation)["images"][0]
        q = json.loads(turn.observation)["state"]["joint_pos"]
        waist = tuple(q[j] for j in WAIST)
        imu = d.qpos[3:7].copy()
        px = aim.project(target, image["width"], image["height"], waist, imu)
        dist = float(np.linalg.norm(target - aim.camera(waist, imu)[1]))
        return select(self, turn, "point_to", {"x_px": px[0] + self.off_px, "y_px": px[1], "distance_m": dist,
                                               "hold_s": 1.5, "note": "the laptop"})


def point_to_the_laptop(off_px=0.0, shots=1):
    env = sim_env(scene="room", sim_objects=[scene.parse_object(o) for o in LAPTOP["objects"]],
                  sim_start=LAPTOP["start"], success={"point_at": "laptop", "within_deg": 10, "for_s": 1.0},
                  camera="on", mode="gantry")
    dec = ClickTheTruth(env, off_px, shots)
    agent = Agent("point at the laptop", dec, menu(False, True), max_decisions=shots + 2)
    assert run(agent, env, max_time=60) is True
    agent.close()
    return env, [json.loads(t.observation) for t in dec.turns]


def test_the_first_point_to_reports_the_twist_and_the_second_lands():
    # raising the arm with the waist turned twists the pelvis over the planted feet (the twin: about 10
    # degrees), after the image was taken: the first aim misses by it and says so; the second, solved
    # from the twisted pose and the arm's measured sag, lands
    env, obs = point_to_the_laptop(shots=2)
    first, second = (o["previous_result"]["result"]["aim"] for o in obs[1:3])
    assert first["target_left_of_arm_deg"] > 5, first                          # the twist, measured
    assert abs(second["target_left_of_arm_deg"]) < 3 and abs(second["target_above_arm_deg"]) < 3, second
    assert env.twin.evaluator.result()["measured"]["pointing_error_min_deg"] < 3.0
    assert second["arm_line_px"] is not None


def test_a_pixel_off_the_target_is_hit_and_the_target_missed():
    # clicked 120 px right of the laptop: a bigger waist turn, a bigger twist, three shots to converge;
    # the arm's line lands on what was clicked, not the laptop
    env, obs = point_to_the_laptop(off_px=120.0, shots=3)
    third = obs[3]["previous_result"]["result"]["aim"]
    assert abs(third["target_left_of_arm_deg"]) < 3 and abs(third["target_above_arm_deg"]) < 3, third
    assert env.twin.evaluator.result()["measured"]["pointing_error_min_deg"] > 8


def test_point_to_needs_an_image_and_a_reachable_target():
    from g1.tools import TOOLS
    with pytest.raises(ValueError, match="no camera image"):
        TOOLS["point_to"](x_px=10, y_px=10, distance_m=1.0).bind({"frame_size": None})
    with pytest.raises(ValueError, match="outside"):
        TOOLS["point_to"](x_px=10, y_px=600, distance_m=1.0).bind({"frame_size": (640, 480), "q": None})
    feet = TOOLS["point_to"](x_px=320, y_px=479, distance_m=0.6)    # near the bottom, close by: at the feet
    feet.bind({"frame_size": (640, 480), "q": None})
    assert feet._solve()[3] == 60.0 and "below horizontal" in feet.notes[-1]       # clamped, with a note
    with pytest.raises(ValueError, match="turn the feet"):           # far round: beyond the waist's reach
        TOOLS["point_to"](x_px=0, y_px=200, distance_m=1.0).bind(
            {"frame_size": (640, 480), "q": [0.0] * 12 + [1.4, 0.0, 0.0] + [0.0] * 14})
