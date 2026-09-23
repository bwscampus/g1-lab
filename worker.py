"""Background workers with a latest-only result slot.

A Worker turns one input into one output on a daemon thread, **only when
asked**: ``request(item)`` never blocks and is ignored while a request is
pending or within ``min_interval`` of the last accepted one; ``latest()`` is
the newest result; ``pending`` says whether one is in flight. Results get
``seq`` and ``latency`` set. Errors are counted and printed sparingly, never
raised into the caller. ``threaded=False`` serves requests inline, which keeps
check (which runs far faster than realtime) deterministic in tests.
"""
from __future__ import annotations

import math
import statistics
import threading
import time
from typing import Generic, Optional, TypeVar

I = TypeVar("I")
O = TypeVar("O")


class Worker(Generic[I, O]):
    def __init__(self, *, threaded: bool = True, min_interval: float = 0.0,
                 name: str = "worker") -> None:
        self.threaded = threaded
        self.min_interval = min_interval
        self.name = name
        self._cond = threading.Condition()
        self._queued: Optional[I] = None
        self._inflight = False
        self._lock = threading.Lock()
        self._latest: Optional[O] = None
        self._seq = 0
        self._last_request = -math.inf
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.requests = 0
        self.errors = 0
        self.last_error: Optional[BaseException] = None
        self._latencies: list[float] = []

    def process(self, item: I) -> O:
        """Blocking: turn one input into one result. May raise."""
        raise NotImplementedError

    # -- caller side ---------------------------------------------------------
    def request(self, item: I) -> bool:
        """Ask for ``item`` to be processed. Never blocks. False = ignored."""
        with self._cond:
            if self._queued is not None or self._inflight:
                return False
            now = time.monotonic()
            if now - self._last_request < self.min_interval:
                return False
            self._last_request = now
            self._queued = item
            self._cond.notify()
        if not self.threaded:
            self._drain()
        return True

    @property
    def pending(self) -> bool:
        with self._cond:
            return self._queued is not None or self._inflight

    def latest(self) -> Optional[O]:
        with self._lock:
            return self._latest

    @property
    def count(self) -> int:
        return self._seq

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        if not self.threaded or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, join: float = 1.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        t = self._thread
        if t is not None:
            t.join(timeout=join)     # a hung request must never block teardown
            self._thread = None

    # -- worker side -------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while self._queued is None and not self._stop.is_set():
                    self._cond.wait(0.5)
            if self._stop.is_set():
                break
            self._drain()

    def _drain(self) -> None:
        with self._cond:
            item, self._queued = self._queued, None
            if item is None:
                return
            self._inflight = True
        try:
            self._process(item)
        finally:
            with self._cond:
                self._inflight = False

    def _process(self, item: I) -> None:
        self.requests += 1
        t0 = time.monotonic()
        try:
            out = self.process(item)
        except Exception as e:
            self.errors += 1
            self.last_error = e
            if self.errors <= 3 or self.errors % 50 == 0:
                print(f"{self.name}: error {self.errors}: {e}")
            return
        latency = time.monotonic() - t0
        self._latencies.append(latency)
        with self._lock:
            self._seq += 1
            try:
                out.seq = self._seq
                out.latency = latency
            except AttributeError:
                pass
            self._latest = out

    def summary(self) -> str:
        if self._latencies:
            lat = (f"latency mean {statistics.mean(self._latencies):.2f}s, "
                   f"max {max(self._latencies):.2f}s")
        else:
            lat = "no results"
        s = (f"{self.name}: {self.requests} request(s), {self._seq} result(s), "
             f"{self.errors} error(s); {lat}")
        if self.last_error is not None:
            s += f"; last error: {self.last_error}"
        return s
