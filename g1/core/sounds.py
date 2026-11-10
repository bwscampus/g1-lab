"""Sound files for the robot's speaker, through PyAV.

The only place a sound is read or converted (files here, and the live voice's streams with ``Resampler``). The G1 speaker (``AudioClient.PlayStream``)
plays raw 16 kHz mono 16-bit little-endian PCM with no header and no rate field, so a file
in any other shape would play at the wrong speed. ``load`` decodes whatever FFmpeg reads
(WAV, AIFF, MP3, M4A/AAC/ALAC, FLAC, OGG/Vorbis/Opus, WebM) and resamples it to exactly that.
PyAV is the same library that decodes the robot's video (see ``g1/core/images.py`` for why
OpenCV is not used).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from g1.core import limits
from g1.core.config import SPEAKER_RATE_HZ, SPEAKER_SAMPLE_BYTES


@dataclass(frozen=True)
class Sound:
    pcm: bytes          # 16 kHz mono s16le, what PlayStream plays
    source: str         # the file it came from

    @property
    def seconds(self) -> float:
        return len(self.pcm) / (SPEAKER_RATE_HZ * SPEAKER_SAMPLE_BYTES)


def load(path: Path | str, max_s: float = limits.get("sound_max_s")) -> Sound:
    """A sound file as the speaker's PCM. Raises IOError when it cannot be decoded, has no
    audio, or is longer than ``max_s``."""
    import av
    path = Path(path)
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise IOError(f"{path}: no audio stream")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SPEAKER_RATE_HZ)
            chunks = []
            for frame in container.decode(audio=0):
                chunks += [f.to_ndarray().tobytes() for f in resampler.resample(frame)]
            chunks += [f.to_ndarray().tobytes() for f in resampler.resample(None)]      # flush
    except (av.FFmpegError, ValueError) as e:      # a missing file is an FFmpegError too
        raise IOError(f"could not read sound {path}: {e}") from None
    sound = Sound(b"".join(chunks), str(path))
    if not sound.pcm:
        raise IOError(f"{path}: no audio in it")
    if sound.seconds > max_s:
        raise IOError(f"{path}: {sound.seconds:.1f} s, longer than sound_max_s ({max_s:g} s)")
    return sound


class Resampler:
    """A stream of mono s16le PCM from ``src_hz`` to ``dst_hz``, piece by piece (the filter keeps its
    state between pieces, so the joins are clean). ``flush`` ends the stream: a new one needs a new
    Resampler. The live voice uses it both ways (the mic's 16 kHz up to the model's 24 kHz, the model's
    voice down to the speaker's 16 kHz)."""

    def __init__(self, src_hz: int, dst_hz: int) -> None:
        import av
        self.src_hz, self.dst_hz = int(src_hz), int(dst_hz)
        self._av = av
        self._r = av.AudioResampler(format="s16", layout="mono", rate=self.dst_hz)
        self._odd = b""                              # half a sample left from the last piece

    def push(self, pcm: bytes) -> bytes:
        import numpy as np
        pcm, self._odd = self._odd + pcm, b""
        if len(pcm) % SPEAKER_SAMPLE_BYTES:
            pcm, self._odd = pcm[:-1], pcm[-1:]
        if not pcm:
            return b""
        if self.src_hz == self.dst_hz:
            return pcm
        frame = self._av.AudioFrame.from_ndarray(np.frombuffer(pcm, "<i2").reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.src_hz
        return b"".join(f.to_ndarray().tobytes() for f in self._r.resample(frame))

    def flush(self) -> bytes:
        if self.src_hz == self.dst_hz:
            return b""
        return b"".join(f.to_ndarray().tobytes() for f in self._r.resample(None))


def dbfs(pcm: bytes) -> float:
    """How loud a piece of s16le PCM is: its RMS in dB below full scale (-120 for silence)."""
    import math

    import numpy as np
    x = np.frombuffer(pcm[: len(pcm) - len(pcm) % SPEAKER_SAMPLE_BYTES], "<i2").astype(np.float64)
    if not x.size:
        return -120.0
    rms = math.sqrt(float(np.mean(x * x))) / 32768.0
    return max(-120.0, 20.0 * math.log10(rms)) if rms > 0 else -120.0
