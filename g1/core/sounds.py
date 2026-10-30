"""Sound files for the robot's speaker, through PyAV.

The only place a sound is read or converted. The G1 speaker (``AudioClient.PlayStream``)
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
