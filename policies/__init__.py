"""Policy registry. ``--policy <name>`` picks one of these.

To add a policy: write a Policy subclass in this package and register it here.
"""
from __future__ import annotations

from policy import Policy
from policies.sixseven import SixSeven
from policies.tpose import TPose

POLICIES: dict[str, type[Policy]] = {
    TPose.name: TPose,
    SixSeven.name: SixSeven,
}

__all__ = ["POLICIES", "Policy", "TPose", "SixSeven"]
