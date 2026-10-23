import numpy as np

from g1.cli import main, run
from g1.envs.trace import compare, load
from g1.tools import parse_chain
from tests.doubles import sim_env


def test_trace_records_and_compares_two_runs_of_a_chain(tmp_path, capsys):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        assert run(parse_chain("tpose:0.5:1"), sim_env(trace=str(tmp_path / name / "trace.npz"))) is True
    a, b = load(tmp_path / "a"), load(tmp_path / "b" / "trace.npz")              # a run directory or the file
    assert str(a["env"]) == "sim" and len(a["t"]) == len(b["t"]) and a["q"].shape == (len(a["t"]), 29)
    assert np.isfinite(a["base_true"]).all() and np.isnan(a["q_cmd"][:, 0]).all()     # legs: never commanded
    text = compare(a, b)
    assert "left_shoulder_roll" in text and "left_knee" not in text and "warning" not in text
    line = next(l for l in text.splitlines() if l.startswith("left_shoulder_roll"))
    assert float(line.split()[-1]) == 0.0                                            # the twin repeats exactly
    capsys.readouterr()
    assert main(["trace", "show", str(tmp_path / "a")]) == 0
    assert "tracking RMS" in capsys.readouterr().out
