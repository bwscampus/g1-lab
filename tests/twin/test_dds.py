"""``g1 run TASK --sim`` end to end: it starts the twin in its own process, runs the robot
env's code against it over DDS on the loopback interface, and stops it. Real time and
needs unitree_sdk2py: run with ``pytest -m dds``."""
import json
import os
import subprocess
import sys

import pytest

from g1.core import limits

pytestmark = pytest.mark.dds


def g1(*args, timeout=180):
    pytest.importorskip("unitree_sdk2py")
    return subprocess.run([sys.executable, "-m", "g1", *args], cwd=limits.ROOT, capture_output=True, text=True,
                          timeout=timeout, env={**{k: v for k, v in os.environ.items() if k != "G1_VERBOSE"},
                                                "G1_TWIN_HEADLESS": "1"})                   # tests only: no window


def task(tmp_path, name, **spec):
    d = tmp_path / name
    d.mkdir()
    (d / "task.json").write_text(json.dumps(spec))
    return d


def twins_left() -> bool:
    out = subprocess.run(["pgrep", "-f", "g1.twin.main serve"], capture_output=True, text=True).stdout
    return bool(out.strip())


def test_sim_runs_a_task_against_the_twin_and_stops_it():
    r = g1("run", "tpose", "--sim")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "twin: serving" in r.stdout and "PASS" in r.stdout and "twin: stopped" in r.stdout
    assert "twin: pelvis z =" in r.stdout                                   # its Summary, read over DDS
    assert "result: success" in r.stdout and not twins_left()


def test_sim_gantry_hangs_it_and_never_walks(tmp_path):
    r = g1("run", "tpose", "--sim", "--mode", "gantry")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "fsm: 4 (locked stand)" in r.stdout and "fsm: 1 (damp); done" in r.stdout and "FELL" not in r.stdout
    r = g1("run", str(task(tmp_path, "walk", tools="walk_forward:0.3")), "--sim", "--mode", "gantry")
    assert r.returncode != 0 and "never walks" in r.stderr and "twin: serving" not in r.stdout

