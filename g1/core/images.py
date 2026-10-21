"""Image files and encoding, in RGB, through Pillow.

The only place an image is read, written, resized or encoded. Everything is
``uint8`` RGB, ``image[row, col, channel]`` with row 0 at the top — the same
layout as camera frames, so there is no channel swap anywhere.

OpenCV is deliberately not used. Its wheel bundles its own copy of ffmpeg, and
so does PyAV (which decodes the robot's video); loading both in one process on
macOS makes the Objective-C runtime warn that ``AVFFrameReceiver`` is
implemented twice, "which may cause spurious casting failures and mysterious
crashes". Pillow bundles no ffmpeg.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from g1.core import limits


def read_rgb(path: Path | str) -> np.ndarray:
    """An image file as a uint8 RGB array. Raises IOError when it cannot be read."""
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, ValueError) as e:
        raise IOError(f"could not read {path}: {e}") from None


def write_png(path: Path | str, image_rgb: np.ndarray, compress: int = limits.get("png_compress_level")) -> None:
    """Lossless: the exact RGB array reads back."""
    try:
        Image.fromarray(_rgb(image_rgb)).save(path, format="PNG", compress_level=compress)
    except (OSError, ValueError) as e:
        raise IOError(f"could not write {path}: {e}") from None


def sent_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """(width, height) of a frame downscaled to ``max_width`` (never enlarged)."""
    if width <= max_width:
        return width, height
    return max_width, int(round(height * max_width / width))


def encode_jpeg(image_rgb: np.ndarray, max_width: int, quality: int) -> bytes:
    """JPEG bytes of an RGB frame, downscaled to ``max_width`` (never enlarged)."""
    im = Image.fromarray(_rgb(image_rgb))
    if im.size[0] > max_width:
        im = im.resize(sent_size(*im.size, max_width), Image.Resampling.BOX)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    """Encoded image bytes (JPEG, PNG, ...) as a uint8 RGB array."""
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _rgb(image: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected a uint8 RGB image (H, W, 3), got {a.dtype} {a.shape}")
    return a
