"""The twin scores a task's ``success`` from its truth, and a --sim run is labelled from it:
the model's "done" is a claim, the twin's verdict is the outcome."""
import json

import numpy as np
import pytest

from g1.agent.task import load_task
from g1.cli import run_once
from g1.envs import scene
from tests.doubles import sim_env

pytestmark = pytest.mark.skipif(bool(scene.missing()), reason="room assets not fetched (g1 twin fetch)")

LAPTOP = {"scene": "room", "objects": ["laptop@1.75,-1.5,0.77"], "start": [1.2, -0.95, 0]}
SUCCESS = {"point_at": "laptop", "within_deg": 10, "for_s": 1.0, "feet_drift_max_m": 0.05,
           "feet_yaw_drift_max_deg": 10, "no_contact": ["table_top", "table_leg", "laptop"]}


def run_chain(tmp_path, chain, success=SUCCESS, **spec):
    d = tmp_path / "tasks" / chain.replace(":", "_").replace(",", "-")
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({**LAPTOP, "tools": chain, "success": success, **spec}))
    task = load_task(d)
    env = sim_env(scene=task["scene"], sim_objects=[scene.parse_object(o) for o in task["objects"]],
                  sim_start=task["start"], success=task["success"])
    code, folder = run_once(task, sim=True, mode=None, env=env)
    return code, folder, json.loads((folder / "summary.json").read_text())


def evaluation(summary):
    return summary["health"]["twin"]["evaluation"]


def test_pointing_at_the_laptop_passes_and_labels_the_run(tmp_path):
    # the twin's pelvis twists about 24 deg over planted feet when the arm rises with the waist turned,
    # so the waist turn that points at the laptop (45 deg right of the start) is about 20 deg
    code, folder, s = run_chain(tmp_path, "look:-20,point:18")
    e = evaluation(s)
    assert e["passed"], e
    assert e["measured"]["pointing_within_s"] >= 1.0 and e["measured"]["pointing_error_min_deg"] < 10
    assert folder.name.endswith("_success") and s["outcome"] == "success" and s["outcome_source"] == "twin"
    row = json.loads((folder.parent.parent / "results.jsonl").read_text())
    assert row["twin_passed"] is True and row["outcome_source"] == "twin" and row["timing"] == "ok"


def test_pointing_away_fails_on_point_at(tmp_path):
    code, folder, s = run_chain(tmp_path, "look:20,point:18")
    e = evaluation(s)
    assert e["failed"] == ["point_at"] and e["measured"]["pointing_error_min_deg"] > 30, e
    assert folder.name.endswith("_failed") and s["outcome_source"] == "twin"


def test_walking_fails_on_the_feet_not_the_pelvis(tmp_path):
    code, folder, s = run_chain(tmp_path, "walk_forward:0.3",
                                success={"feet_drift_max_m": 0.05, "feet_yaw_drift_max_deg": 10},
                                start=[0, 0, 0], objects=[])
    assert "feet_drift_max_m" in evaluation(s)["failed"], evaluation(s)


def test_touching_a_listed_geom_fails(tmp_path):
    # the feet always touch the floor: listing it must fail the run, naming the contact
    code, folder, s = run_chain(tmp_path, "hold:1", success={"no_contact": ["room_floor", "floor"]},
                                start=[0, 0, 0], objects=[])
    e = evaluation(s)
    assert e["failed"] == ["no_contact"] and any("floor" in k for k in e["measured"]["contacts"]), e


def test_the_twin_fills_base_true_in_the_trace(tmp_path):
    code, folder, s = run_chain(tmp_path, "hold:0.5", success={"feet_drift_max_m": 0.05})
    b = np.load(folder / "trace.npz")["base_true"]
    assert np.isfinite(b).all()


def test_criteria_are_checked():
    from g1.twin.core import Evaluator

    class Twin:            # only what the constructor checks before it touches a model
        pass
    with pytest.raises(ValueError, match="unknown"):
        Evaluator(Twin(), {"point_at": "laptop", "within_deg": 5, "for_s": 1, "acuracy": 1})
    with pytest.raises(ValueError, match="needs within_deg and for_s"):
        Evaluator(Twin(), {"point_at": "laptop"})


@pytest.mark.parametrize("error_deg, passed", [(10.5, True), (11.0, True), (11.5, False)])
def test_angles_pass_within_the_margin(error_deg, passed):
    # a task's 10 deg accepts up to 10 + success_margin_deg (1 deg) for measurement error
    from g1.core import limits
    from g1.twin.core import Twin
    assert limits.get("success_margin_deg") == 1.0
    twin = Twin(scene="room", objects=[scene.parse_object(o) for o in LAPTOP["objects"]], start=LAPTOP["start"],
                success={"point_at": "laptop", "within_deg": 10, "for_s": 1.0})
    twin.evaluator.pointing_error_deg = lambda: error_deg
    for _ in range(60):                                     # 1.2 s of control ticks
        twin.evaluator.tick(0.02)
    r = twin.evaluator.result()
    assert r["passed"] is passed and r["margin_deg"] == 1.0
