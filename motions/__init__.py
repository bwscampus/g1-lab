"""Motion registry: reusable building blocks a Routine strings together.

``--policy a,b,c`` looks each name up here. To add a motion: subclass
``policy.Motion`` in this package and register it in ``MOTIONS``.
"""
from __future__ import annotations

from typing import Callable

from motions.bookends import Handback, Hold, Takeover
from motions.sixseven import SixSeven
from motions.tpose import TPose
from policy import Motion

MOTIONS: dict[str, Callable[[], Motion]] = {
    TPose.name: TPose,
    SixSeven.name: SixSeven,
}

__all__ = ["MOTIONS", "Motion", "TPose", "SixSeven", "Takeover", "Handback", "Hold"]
