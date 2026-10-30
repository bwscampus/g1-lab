"""configs/limits.json is the one place a limit lives."""
import json
import re
from pathlib import Path

import pytest

from g1.agent.decider import build_context
from g1.core import limits
from g1.core.config import BASE_VEL_MAX, CONTROL_DT
from g1.tools import TOOLS, menu, validate_args

ROOT = Path(__file__).resolve().parents[2]


def test_the_file_defines_exactly_the_names_the_code_reads():
    data = json.loads(limits.PATH.read_text())
    assert set(data["limits"]) == set(limits.NAMES) and len(limits.NAMES) == len(set(limits.NAMES))
    for name, e in data["limits"].items():
        assert e["source"] in limits.SOURCES and e["unit"] and e["note"], name
    assert limits.get("max_time_s") == data["limits"]["max_time_s"]["value"]


def test_code_reads_its_numbers_from_the_file():
    from g1 import cli, vlm
    from g1.agent import agent, episode
    from g1.envs.monitor import JointMonitor
    from g1.tools import move
    assert CONTROL_DT == limits.get("control_dt_s")
    assert list(BASE_VEL_MAX) == [limits.get("base_vx_max"), limits.get("base_vy_max"), limits.get("base_vyaw_max")]
    assert (move.WALK_SPEED, move.SIDE_SPEED, move.TURN_RATE) == tuple(
        limits.get(n) for n in ("walk_speed", "side_speed", "turn_rate"))
    assert (cli.TO_STAND, cli.RAMP, cli.RELEASED) == tuple(
        limits.get(n) for n in ("return_to_stand_s", "return_ramp_s", "return_released_s"))
    a = vlm.AGENT_SETTINGS                          # the agent loop's settings: configs/model.json
    assert vlm.IMAGE_WIDTH == a["image_width_px"]
    assert len(agent.RETRY_DELAYS) == a["retry_max"] and max(agent.RETRY_DELAYS) == a["retry_delay_cap_s"]
    assert agent.RECOVERY_TIMEOUT == a["recovery_timeout_s"]
    m = JointMonitor()
    assert (m.margin, m.max_vel, m.max_violations) == tuple(
        limits.get(n) for n in ("joint_margin_rad", "command_vel_max", "max_violations"))
    a = agent.Agent("g", None.__class__ and __import__("tests.doubles", fromlist=["RedBallDecider"]).RedBallDecider(), menu(True))
    assert a.max_decisions == vlm.AGENT_SETTINGS["max_decisions"]
    assert (a.settle_tol, a.settle_vel_tol, a.settle_samples, a.frame_timeout) == tuple(
        limits.get(n) for n in ("settle_pos_tol_rad", "settle_vel_tol", "settle_samples", "frame_timeout_s"))


def test_the_tools_and_the_prompt_quote_the_limits():
    # no range in a tool is a literal for anything limits.json governs
    move = TOOLS["move"].schema()["properties"]
    assert move["dx_m"]["maximum"] == limits.get("move_dx_max_m") and move["dx_m"]["minimum"] == -limits.get("move_dx_max_m")
    assert move["dy_m"]["maximum"] == limits.get("move_dy_max_m") and move["dyaw_deg"]["minimum"] == -limits.get("move_dyaw_max_deg")
    wp = TOOLS["arm_path"].schema()["properties"]["waypoints"]
    assert wp["maxItems"] == limits.get("arm_path_max_waypoints")
    assert wp["items"]["properties"]["seconds"]["default"] == limits.get("arm_path_seconds_default")
    assert TOOLS["wave_hand"].schema()["properties"]["seconds"]["default"] == limits.get("wave_hand_seconds")
    assert TOOLS["takeover"].schema()["properties"]["ramp_s"]["default"] == limits.get("bookend_ramp_s")
    assert "{limit:" not in TOOLS["move"].describe()
    assert f"up to {limits.get('move_dx_max_m'):g} m either way" in TOOLS["move"].describe()
    assert f"{limits.get('walk_speed'):g} m/s" in TOOLS["move"].describe()
    text = build_context(menu(True), can_walk=True, max_decisions=5).instructions
    assert "{limit:" not in text and "$limit" not in text
    for path in (ROOT / "g1" / "tools").glob("*.py"):
        if path.name == "base.py":
            continue                                   # the contract's docstring, not a tool
        raw = path.read_text()
        for name in re.findall(r"\{limit:([a-z0-9_]+)\}", raw) + re.findall(r'limit\("([a-z0-9_]+)"\)', raw):
            assert name in limits.NAMES, (path.name, name)


def test_a_changed_limit_changes_what_the_model_is_offered(tmp_path):
    data = json.loads(limits.PATH.read_text())
    data["limits"]["move_dx_max_m"]["value"] = 1.0
    p = tmp_path / "limits.json"
    p.write_text(json.dumps(data))
    try:
        limits.use(p)
        assert TOOLS["move"].schema()["properties"]["dx_m"]["maximum"] == 1.0
        assert "up to 1 m either way" in TOOLS["move"].describe()
        assert validate_args(TOOLS["move"], {"dx_m": 2.0})[0]["dx_m"] == 1.0       # clamped
        tools = build_context(menu(True), can_walk=True, max_decisions=5).tools     # what each request offers
        move = next(t["function"] for t in tools if t["function"]["name"] == "move")
        assert "up to 1 m either way" in move["description"] and move["parameters"]["properties"]["dx_m"]["maximum"] == 1.0
    finally:
        limits.use(None)
    assert TOOLS["move"].schema()["properties"]["dx_m"]["maximum"] == limits.get("move_dx_max_m")


def test_a_bad_limits_file_is_refused_with_the_name(tmp_path):
    base = json.loads(limits.PATH.read_text())

    def broken(mutate, match):
        d = json.loads(json.dumps(base))
        mutate(d)
        p = tmp_path / "l.json"
        p.write_text(json.dumps(d))
        with pytest.raises(ValueError, match=match):
            limits.load(p)

    broken(lambda d: d["limits"].pop("max_time_s"), "missing .*max_time_s")
    broken(lambda d: d["limits"].update(max_tme_s=d["limits"]["max_time_s"]), "unknown .*max_tme_s")
    broken(lambda d: d["limits"]["walk_speed"].update(value="fast"), "walk_speed.value must be a number")
    broken(lambda d: d["limits"]["walk_speed"].update(value=-1), "must not be negative")
    broken(lambda d: d["limits"]["walk_speed"].update(value=0.5), "walk_speed .* must not exceed base_vx_max")
    broken(lambda d: d["limits"]["side_speed"].update(value=0.3), "side_speed .* must not exceed base_vy_max")
    broken(lambda d: d["limits"]["walk_speed"].update(source="vibes"), "source must be one of")
    broken(lambda d: d["limits"]["walk_speed"].update(note=""), "needs a note")
    broken(lambda d: d.update(version=2), "version 1")
    with pytest.raises(ValueError, match="not found"):
        limits.load(tmp_path / "none.json")


def test_robot_and_twin_timing_is_never_a_literal():
    # bring-up, teardown and RPC timeouts move the real robot: they come from limits.json
    pattern = re.compile(r"(?:sleep|SetTimeout|release)\(\s*[0-9.]+\s*\)")
    for rel in ("g1/envs/robot.py", "g1/envs/sim.py", "g1/twin/core.py", "g1/twin/onboard.py", "g1/twin/main.py",
                "g1/twin/services.py", "g1/cli.py"):
        hits = pattern.findall((ROOT / rel).read_text())
        assert not hits, (rel, hits)


def test_env_holds_nothing_editable():
    # .env is private to one machine; every setting anyone may edit is in limits.json
    from g1 import vlm
    assert vlm.PROVIDER == vlm.MODEL["provider"] and vlm.REASONING_EFFORT == vlm.MODEL["reasoning_effort"]
    keys = [line.split("=")[0].lstrip("# ").strip() for line in (ROOT / ".env.example").read_text().splitlines()
            if "=" in line and line.lstrip("# ").split("=")[0].strip().isupper()]
    assert set(keys) <= {"HF_TOKEN", "VLM_BASE_URL", "OPENROUTER_API_KEY", "UNITREE_IFACE", "UNITREE_ROBOT_IP", "UNITREE_AES_128_KEY",
                         "UNITREE_DOMAIN", "G1_VERBOSE"}, keys
