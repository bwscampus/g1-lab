"""Routines as tools: ``play_routine`` lets a task's model use a routine as one step, and
``RoutineProgram`` plays one on its own (``g1 run <routine>``, talk). Both compile the routine
(``g1/routines.py``) to the segments every tool plays.

Inside a task only upper-body routines are offered: the base and onboard tracks need the env to
walk and the arms to be handed over, which the agent's own tools (move, wave_hand) already do.
"""
from __future__ import annotations

from typing import Optional

from g1.core.action import Obs, Segment
from g1.core.config import UPPER_BODY
from g1.routines import RoutineError, compile_segments, load_routine, routine_names
from g1.tools.base import Tool


def _offered() -> list[str]:
    """The routines a task may play: upper body only, loadable."""
    names = []
    for n in routine_names():
        try:
            if load_routine(n).upper_body_only:
                names.append(n)
        except RoutineError as e:
            print(f"warning: routine {n} skipped: {e}")
    return names


OFFERED = _offered()


def _menu_text(names: list[str]) -> str:
    lines = []
    for n in names:
        notes = load_routine(n).notes.strip().splitlines()
        lines.append(f"{n} ({notes[0]})" if notes else n)
    return "; ".join(lines)


class RoutineTool(Tool):
    """Any routine, by name: its whole timeline as segments (the base and onboard tracks too).
    Hidden; ``RoutineProgram`` plays it with the takeover and handback."""

    name = "routine"
    order = 90
    visible = False
    params = {"routine": {"type": "string", "description": "a folder under routines/"}}

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.plan = load_routine(self.routine)
        if self.plan.full_body:
            raise RoutineError(f"routine {self.plan.name}: full-body routines need a tracking controller "
                               "(rt/lowcmd), which g1 does not have yet")
        self.needs_base = self.plan.uses_base
        self.needs_loco = self.plan.uses_onboard

    def segments(self) -> tuple[Segment, ...]:
        return tuple(compile_segments(self.plan))

    def __repr__(self) -> str:
        return f"{self.name}({self.routine})"


class PlayRoutine(RoutineTool):
    name = "play_routine"
    order = 62
    visible = bool(OFFERED)
    prompt = ("Play one of the routines start to finish: a fixed upper-body motion written in advance, "
              "the same every time. The routines: " + (_menu_text(OFFERED) or "none") + ".")
    params = {"routine": {"type": "string", "enum": OFFERED or ["none"], "description": "the routine to play"}}

    def __init__(self, **args) -> None:
        name = args.get("routine")
        if name not in OFFERED:
            raise ValueError(f"play_routine: {name!r} is not a routine a task may play "
                             f"(upper-body routines: {', '.join(OFFERED) or 'none'})")
        super().__init__(**args)


def routine_program(name: str):
    """A routine (a name under routines/, or its folder) played on its own: dry-run from the commanded
    start pose, then the takeover, the routine, a raised arm lowered, the handback (``tools.Chain``)."""
    from g1.tools import Chain

    class RoutineProgram(Chain):
        def __init__(self) -> None:
            self.tool = RoutineTool(routine=str(name))
            super().__init__(self.tool, name=f"routine:{self.tool.plan.name}")
            self.dry_run: Optional[dict] = None

        def reset(self, obs: Obs) -> None:
            from g1.tools.control import dry_run
            report = dry_run(self.tool, obs.q, UPPER_BODY)
            self.dry_run = report
            if report["status"] != "ok":
                raise SystemExit(f"routine {self.tool.plan.name} refused before it moved (dry run): "
                                 + "; ".join(report["violations"]))
            super().reset(obs)

    return RoutineProgram()
