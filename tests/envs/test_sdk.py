"""The SDK's return codes, named; one checked call."""
import pytest

from g1.envs import sdk


class Client:
    def __init__(self, **returns):
        self.returns = returns
        self.calls = []

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls.append((name, a, kw))
            r = self.returns.get(name, 0)
            if isinstance(r, Exception):
                raise r
            return r
        return call


def test_codes_are_named():
    assert sdk.explain(0) == "0 (ok)"
    assert "timeout" in sdk.explain(3104) and "DDS" in sdk.explain(3104)
    assert "refused" in sdk.explain(7001) and sdk.explain("x").endswith("(not a code)")
    assert sdk.is_transport(3104) and sdk.is_transport(3102) and not sdk.is_transport(7001)
    assert sdk.fsm_name(200) == "200 (main operation)" and sdk.fsm_name(999) == "999 (unknown)"
    assert {4, 200} <= sdk.STANDING_FSMS and 1 not in sdk.STANDING_FSMS


def test_call_normalises_times_logs_and_prints(capsys):
    log = []
    c = Client(GetFsmId=(0, 200), SetFsmId=3104, Damp=None, Boom=RuntimeError("no"))
    assert sdk.call(c, "GetFsmId", log=log) == (0, 200)
    assert sdk.call(c, "Damp", log=log) == (0, None)                      # None means ok
    assert sdk.call(c, "SetFsmId", 200, log=log)[0] == 3104
    code, data = sdk.call(c, "Boom", log=log, x=1)
    assert code == -1 and "RuntimeError: no" in data
    assert [e["name"] for e in log] == ["GetFsmId", "Damp", "SetFsmId", "Boom"]
    assert log[2]["args"] == [200] and log[3]["kwargs"] == {"x": 1} and log[0]["elapsed_s"] >= 0
    out = capsys.readouterr().out
    assert "SetFsmId(200) -> 3104" in out and "Boom(x=1) -> RuntimeError" in out and "GetFsmId" not in out
    sdk.call(c, "GetFsmId", verbose=True, label="Loco")
    assert "Loco.GetFsmId() -> 0 (ok)" in capsys.readouterr().out
