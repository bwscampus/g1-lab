"""Targets: what a policy is looking for.

A policy names its Target (a red dot, a doorway, whatever the vision model
calls "person"); the target knows how to find itself in an observation and
when it counts as reached. What every target carries:

  location    where it is relative to the camera: ``bearing`` (+right) and
              ``elevation`` (+up) in rad, plus the normalised image box
  dimensions  apparent size only (image fractions); ``size_m`` is a nominal
              real size kept as metadata, never used to estimate distance
  distance    ``distance_m`` only when a detector *reports* one (the vision
              model does; a pixel detector does not). Targets never estimate it.
  reached     ``reached(sighting)``: the target's own arrival criterion
  freshness   ``update`` tracks the last sighting and when it was seen

  RedDot      pixels only (vision.red_blob); reached when it looms or drops
              to the bottom of the frame
  Labeled     the vision model's object with a matching label
  Salient     whatever the vision model finds largest
  Doorway     Labeled("door", "doorway"); a STUB with no arrival criterion, so
              GoTo refuses it (can_reach = False) and Face accepts it
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from vision import bearing, elevation, red_blob

if TYPE_CHECKING:
    from camera import Frame
    from perception import Detected, Percept
    from policy import Obs


@dataclass(frozen=True)
class Sighting:
    bearing: float              # rad, camera frame, positive right
    elevation: float            # rad, camera frame, positive up
    x: float                    # normalised box centre, 0 = left .. 1 = right
    y: float                    # 0 = top .. 1 = bottom
    width: float                # fractions of the image
    height: float
    frame_seq: int              # the frame it was seen in
    stamp: float                # ... and its stamp on the env clock
    label: str = ""
    distance_m: Optional[float] = None    # only as reported by the detector

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_box(cls, x: float, y: float, width: float, height: float, image_shape,
                 frame_seq: int, stamp: float, label: str = "",
                 distance_m: Optional[float] = None) -> "Sighting":
        return cls(bearing(2 * x - 1, image_shape), elevation(2 * y - 1, image_shape),
                   x, y, width, height, frame_seq, stamp, label, distance_m)

    @classmethod
    def from_blob(cls, u: float, v: float, fraction: float, frame: "Frame",
                  label: str = "red") -> "Sighting":
        size = math.sqrt(fraction)
        return cls.from_box((u + 1) / 2, (v + 1) / 2, size, size, frame.image.shape,
                            frame.seq, frame.stamp, label)

    @classmethod
    def from_detected(cls, d: "Detected", percept: "Percept") -> "Sighting":
        return cls(d.bearing, d.elevation, d.x, d.y, d.width, d.height,
                   percept.frame_seq, percept.frame_stamp, d.label, d.distance_m)


class Target:
    """Subclass and implement ``locate``; override ``reached`` for targets a
    policy may walk to. ``locate`` is pure; ``update`` is the stateful tracker
    policies call once per tick."""

    name: str = "target"
    uses_vision: bool = False           # locating needs the vision model's percept
    can_reach: bool = True              # False for stubs with no arrival criterion
    size_m: Optional[tuple[float, float]] = None   # nominal (width, height), metadata only

    def __init__(self) -> None:
        self.reset()

    # -- pure --------------------------------------------------------------
    def key(self, obs: "Obs") -> Optional[int]:
        """Identity of the input ``locate`` would look at (None when absent)."""
        if self.uses_vision:
            return None if obs.percept is None else obs.percept.seq
        return None if obs.frame is None else obs.frame.seq

    def input_age(self, obs: "Obs") -> float:
        return obs.percept_age if self.uses_vision else obs.frame_age

    def locate(self, obs: "Obs") -> Optional[Sighting]:
        """Find the target in ``obs``; may assume ``key(obs)`` is not None."""
        raise NotImplementedError

    def reached(self, s: Sighting) -> bool:
        return False

    # -- stateful ----------------------------------------------------------
    def reset(self) -> None:
        self.current: Optional[Sighting] = None    # result for the newest input (None = not in it)
        self.last: Optional[Sighting] = None       # most recent hit
        self._key: Optional[int] = None
        self._seen_t = -math.inf
        self._seen_age = math.inf

    def update(self, obs: "Obs", t: float) -> Optional[Sighting]:
        """Locate once per new input. Returns the new sighting, or None when
        there was no new input or the target is not in it."""
        k = self.key(obs)
        if k is None or k == self._key:
            return None
        self._key = k
        s = self.locate(obs)
        self.current = s
        if s is not None:
            self.last = s
            self._seen_t = t
            self._seen_age = self.input_age(obs)
        return s

    def age(self, t: float) -> float:
        """Seconds since the target was last seen, counting the input's own age."""
        if self.last is None:
            return math.inf
        return (t - self._seen_t) + self._seen_age

    def seen(self, t: float, within: float) -> bool:
        return self.age(t) <= within


class RedDot(Target):
    """A red blob in the raw frame. Reached when it looms past ``reach_fraction``
    of the image, or sinks below ``reach_elevation`` (the head camera looks
    47 deg down, so a floor-level dot leaves the bottom of the frame while it is
    still small: about 0.75 m ahead for the sim sphere)."""

    name = "red_dot"
    size_m = (0.2, 0.2)

    def __init__(self, min_fraction: float = 0.002, reach_fraction: float = 0.03,
                 reach_elevation: float = -0.2) -> None:
        super().__init__()
        self.min_fraction = min_fraction
        self.reach_fraction = reach_fraction
        self.reach_elevation = reach_elevation

    def locate(self, obs: "Obs") -> Optional[Sighting]:
        image = obs.frame.image
        if image.shape[1] > 800:
            image = image[::2, ::2]     # the robot streams 1280x720: 5 ms -> 1 ms, same geometry
        blob = red_blob(image, self.min_fraction)
        if blob is None:
            return None
        return Sighting.from_blob(*blob, obs.frame, label=self.name)

    def reached(self, s: Sighting) -> bool:
        return s.area >= self.reach_fraction or s.elevation <= self.reach_elevation


class Labeled(Target):
    """The vision model's object whose label contains one of ``labels``."""

    uses_vision = True

    def __init__(self, labels: Sequence[str], name: Optional[str] = None) -> None:
        super().__init__()
        self.labels = tuple(labels)
        self.name = name or self.labels[0]

    def locate(self, obs: "Obs") -> Optional[Sighting]:
        for label in self.labels:
            d = obs.percept.find(label)
            if d is not None:
                return Sighting.from_detected(d, obs.percept)
        return None


class Salient(Target):
    """Whatever the vision model lists as the largest object."""

    name = "salient"
    uses_vision = True

    def locate(self, obs: "Obs") -> Optional[Sighting]:
        d = obs.percept.salient()
        return None if d is None else Sighting.from_detected(d, obs.percept)


class Doorway(Labeled):
    """STUB. Finds a door/doorway in the percept; ``reached`` is not defined
    yet (what counts as "at the doorway" is future work), so ``can_reach`` is
    False: Face works, GoTo refuses it."""

    can_reach = False
    size_m = (0.9, 2.0)

    def __init__(self) -> None:
        super().__init__(("doorway", "door"), name="doorway")


def seen(target: Target) -> Callable[["Obs"], bool]:
    """A Selector predicate: the target is in the current input."""
    return lambda obs: target.key(obs) is not None and target.locate(obs) is not None
