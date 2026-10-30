import wave

import numpy as np
import pytest

from g1.core import sounds
from g1.core.config import SPEAKER_RATE_HZ

av = pytest.importorskip("av")


def tone(hz: float, seconds: float, rate: int, channels: int) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    x = (0.3 * 32767 * np.sin(2 * np.pi * hz * t)).astype(np.int16)
    return np.repeat(x[:, None], channels, axis=1)


def write_wav(path, x: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(x.astype("<i2").tobytes())


def write_mp3(path, x: np.ndarray, rate: int) -> None:
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mp3", rate=rate, layout="stereo" if x.shape[1] == 2 else "mono")
        frame = av.AudioFrame.from_ndarray(x.reshape(1, -1),            # packed: L R L R ...
                                           format="s16", layout=stream.layout.name)
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def peak_hz(pcm: bytes) -> float:
    x = np.frombuffer(pcm, "<i2").astype(float)
    return float(np.fft.rfftfreq(len(x), 1 / SPEAKER_RATE_HZ)[np.argmax(np.abs(np.fft.rfft(x)))])


def test_a_44k_stereo_wav_becomes_16k_mono_at_the_same_pitch_and_length(tmp_path):
    write_wav(tmp_path / "a.wav", tone(440, 2.0, 44100, 2), 44100)
    s = sounds.load(tmp_path / "a.wav")
    assert len(s.pcm) % 2 == 0 and s.seconds == pytest.approx(2.0, abs=0.02)
    assert peak_hz(s.pcm) == pytest.approx(440, abs=2)       # resampled, not replayed at 16 kHz


def test_an_mp3_decodes_to_the_speaker_format(tmp_path):
    write_mp3(tmp_path / "a.mp3", tone(880, 1.5, 48000, 2), 48000)
    s = sounds.load(tmp_path / "a.mp3")
    assert s.seconds == pytest.approx(1.5, abs=0.1)           # mp3 pads a frame or two
    assert peak_hz(s.pcm) == pytest.approx(880, abs=2)


def test_a_16k_mono_wav_passes_through_unchanged(tmp_path):
    x = tone(300, 1.0, SPEAKER_RATE_HZ, 1)
    write_wav(tmp_path / "a.wav", x, SPEAKER_RATE_HZ)
    assert sounds.load(tmp_path / "a.wav").pcm == x.astype("<i2").tobytes()


def test_unreadable_missing_and_too_long_files_are_refused_by_name(tmp_path):
    (tmp_path / "junk.mp3").write_bytes(b"not audio at all")
    with pytest.raises(IOError, match="junk.mp3"):
        sounds.load(tmp_path / "junk.mp3")
    with pytest.raises(IOError, match="missing.wav"):
        sounds.load(tmp_path / "missing.wav")
    write_wav(tmp_path / "long.wav", tone(440, 3.0, SPEAKER_RATE_HZ, 1), SPEAKER_RATE_HZ)
    with pytest.raises(IOError, match="longer than sound_max_s"):
        sounds.load(tmp_path / "long.wav", max_s=2.0)
