"""The Unitree SDK's return codes, named, and one way to call it.

Every RPC on the robot (LocoClient, AudioClient) returns a code; the SDK's own
codes (``unitree_sdk2py/rpc/internal.py``) say whether the *request* failed
(never sent, no reply: the DDS link) and the service's codes say the robot
*refused*. Telling those apart is the whole diagnosis of "the robot is stuck":

    no reply (3102/3104) + no LowState   -> the link: interface, subnet, the robot's services
    a service code + LowState streaming  -> the robot refuses: protection state, the app or
                                            the remote owns the FSM

``call`` makes one SDK call, times it, normalises the code, records it, and
prints it when it fails (or always, in verbose mode). Nothing in the repo may
call the SDK and ignore the code.
"""
from __future__ import annotations

import time
from typing import Any, Optional

# unitree_sdk2py/rpc/internal.py
RPC_CODES = {
    0: "ok",
    3001: "unknown RPC error",
    3102: "client send failed: the request never left (DDS link?)",
    3103: "API not registered on the client",
    3104: "client timeout: no reply from the robot (DDS link?)",
    3105: "API version mismatch",
    3106: "bad reply data",
    3107: "lease invalid",
    3201: "server send failed",
    3202: "server internal error",
    3203: "API not implemented on this robot",
    3204: "bad parameter",
}
TRANSPORT = frozenset({3102, 3104})              # the request did not make the round trip

# From Unitree's G1 documentation; verify against your firmware (`g1 status` prints the raw id).
FSM_NAMES = {0: "zero torque", 1: "damp", 2: "squat", 3: "sit", 4: "locked stand",
             200: "main operation", 500: "start", 501: "squat to stand", 702: "lie to stand",
             706: "stand to squat", 801: "sit to stand"}
STANDING_FSMS = frozenset({4, 200, 500})        # states a standing run may start in and return to


def fsm_name(fsm: Any) -> str:
    try:
        return f"{int(fsm)} ({FSM_NAMES.get(int(fsm), 'unknown')})"
    except (TypeError, ValueError):
        return str(fsm)


def explain(code: Any) -> str:
    """``"3104 (client timeout: no reply from the robot (DDS link?))"``."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return f"{code!r} (not a code)"
    if c in RPC_CODES:
        return f"{c} ({RPC_CODES[c]})"
    return f"{c} (service error: the robot refused)"


def is_transport(code: Any) -> bool:
    try:
        return int(code) in TRANSPORT
    except (TypeError, ValueError):
        return False


def call(client: Any, name: str, *args, log: Optional[list] = None, verbose: bool = False,
         label: Optional[str] = None, **kw) -> tuple[int, Any]:
    """One SDK call. Returns ``(code, data)``: the SDK's ``int`` or ``(code, data)``
    normalised, an exception as code -1 with the message as data. Appends an
    entry to ``log`` and prints a line on failure (always with ``verbose``)."""
    label = label or type(client).__name__
    t0 = time.monotonic()
    try:
        result = getattr(client, name)(*args, **kw)
        code, data = (result if isinstance(result, tuple) and len(result) == 2 else (result, None))
        code = 0 if code is None else int(code)
    except Exception as e:                       # the SDK raised: the call did not happen
        code, data = -1, f"{type(e).__name__}: {e}"
    elapsed = time.monotonic() - t0
    shown = ", ".join([*(repr(a) for a in args), *(f"{k}={v!r}" for k, v in kw.items())])
    entry = {"at_s": time.time(), "client": label, "name": name, "args": list(args), "kwargs": dict(kw),
             "code": code, "elapsed_s": round(elapsed, 6)}
    if log is not None:
        log.append(entry)
    if verbose or code != 0:
        print(f"sdk {elapsed:7.3f}s {label}.{name}({shown}) -> {explain(code) if code != -1 else data}")
    return code, data
