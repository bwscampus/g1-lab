"""The model knows only what the robot knows: its tools, the camera, the task's objective
and rules, its own joints and commands, and its own earlier turns. Never the room's layout
or the twin's truth, which would be false on the robot (and GPT-Policy gives none either)."""
import json
import re

import pytest

from g1.agent.agent import Agent
from g1.agent.decider import build_context
from g1.agent.task import find, load_task
from g1.cli import run
from g1.core import limits
from g1.envs import scene
from g1.tools import menu
from tests.agent.test_agent import Sequence
from tests.doubles import sim_env

ROOT = limits.ROOT
WORDS = ("wall", "doorway", r"(?<!joint )table", "room position", "room's origin", "hidden obstacle", "simulator",
         "base_pose_env", "base_measured", "base_error")
# the room's own numbers (the ones no tool range or joint limit shares): where the table and door are
ROOM_NUMBERS = sorted({f"{v:g}" for v in (scene.TABLE[0], scene.TABLE_TOP_Z, scene.DOOR_Y[1], scene.ROOM_H)})
STATE = {"joint_pos", "joint_vel", "joint_torque", "waist_yaw_deg", "base_pose_cmd"}


def instruction_tasks():
    return [d.name for d in sorted((ROOT / "tasks").iterdir())
            if (d / "task.json").is_file() and load_task(d)["instruction"]]


def leaks(text: str, numbers=()) -> list[str]:
    low = text.lower()
    found = [w for w in WORDS if re.search(w, low)]
    found += [n for n in numbers if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", text)]
    return found


@pytest.mark.parametrize("name", instruction_tasks())
def test_the_prompt_tells_no_world_facts(name):
    task = load_task(find(name))
    ctx = build_context(menu(True, True), can_walk=True, max_decisions=task["max_decisions"],
                        safety_notes=task["safety_notes"])
    assert not leaks(ctx.instructions + task["instruction"], ROOM_NUMBERS), name


@pytest.mark.skipif(bool(scene.missing()), reason="room assets not fetched (g1 twin fetch)")
def test_the_observations_carry_only_what_the_robot_measures():
    task = load_task(find("point_at_the_laptop"))          # the twin starts beside the table, a laptop in the room
    env = sim_env(scene=task["scene"], sim_objects=[scene.parse_object(o) for o in task["objects"]],
                  sim_start=task["start"], camera="on")
    dec = Sequence([("look", {"yaw_deg": -30.0}), ("locate_point", {"x_px": 10, "y_px": 10}),
                    ("done", {"summary": "s", "hindsight": "h"})])
    agent = Agent(task["instruction"], dec, menu(False), safety_notes=task["safety_notes"], max_decisions=5)
    assert run(agent, env, max_time=120) is True
    sent = [json.loads(t.observation) for t in dec.turns]
    assert len(sent) == 3
    for o in sent:
        assert set(o["state"]) == STATE
        assert not leaks(json.dumps(o)), o
    located = sent[2]["previous_result"]["result"]
    assert located["tool"] == "locate_point" and located["image_size"] == [o["images"][0]["width"], o["images"][0]["height"]]
    assert located["look_yaw_deg"] > -30                    # top-left of the image: left of where it looks
