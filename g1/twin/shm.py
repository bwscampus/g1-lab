"""The head camera from ``g1 twin`` to a robot env on the same machine: one
shared-memory segment per DDS domain, a header and one RGB frame behind a
seqlock (the writer makes the counter odd, copies, makes it even; a reader keeps
a copy only if the counter was even and unchanged across it).

Frames are stamped with ``time.monotonic()``, which every process on the machine
shares, so a robot env's frame age is right. The robot sends video over WebRTC
instead; the twin does not emulate that transport.
"""
from __future__ import annotations

import struct
import time
from multiprocessing import shared_memory

import numpy as np

HEADER = struct.Struct("<QdII")        # seq, stamp, height, width
HEADER_BYTES = 64


def segment_name(domain: int) -> str:
    return f"g1twin_{int(domain)}_head"          # macOS caps the name at 31 characters


class FrameWriter:
    def __init__(self, domain: int, height: int, width: int) -> None:
        self.name = segment_name(domain)
        try:                                      # a stale segment from a twin that died
            old = shared_memory.SharedMemory(name=self.name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        self.shape = (height, width, 3)
        self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=HEADER_BYTES + height * width * 3)
        self.seq = 0
        HEADER.pack_into(self.shm.buf, 0, 0, 0.0, height, width)

    def write(self, image: np.ndarray, stamp: float | None = None) -> None:
        if image.shape != self.shape:
            raise ValueError(f"frame {image.shape} does not fit the segment {self.shape}")
        stamp = time.monotonic() if stamp is None else stamp
        buf = self.shm.buf
        struct.pack_into("<Q", buf, 0, self.seq + 1)                 # odd: writing
        np.frombuffer(buf, dtype=np.uint8, count=image.size, offset=HEADER_BYTES)[:] = image.reshape(-1)
        self.seq += 2
        HEADER.pack_into(buf, 0, self.seq, stamp, *self.shape[:2])   # even: done

    def close(self) -> None:
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


class FrameReader:
    def __init__(self, domain: int) -> None:
        self.shm = shared_memory.SharedMemory(name=segment_name(domain))
        try:                                      # Python 3.10 would unlink the twin's segment when we exit
            from multiprocessing import resource_tracker
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception:
            pass
        self.seq = 0

    def read(self) -> tuple[np.ndarray, float] | None:
        """The newest frame as ``(image, stamp)`` if it is new since the last read, else None."""
        buf = self.shm.buf
        for _ in range(3):
            seq, stamp, h, w = HEADER.unpack_from(buf, 0)
            if seq == 0 or seq % 2 or seq == self.seq:
                if seq % 2:
                    time.sleep(0.001)
                    continue
                return None
            image = np.frombuffer(buf, dtype=np.uint8, count=h * w * 3, offset=HEADER_BYTES).copy()
            if HEADER.unpack_from(buf, 0)[0] == seq:
                self.seq = seq
                return image.reshape(h, w, 3), stamp
        return None

    def close(self) -> None:
        self.shm.close()
