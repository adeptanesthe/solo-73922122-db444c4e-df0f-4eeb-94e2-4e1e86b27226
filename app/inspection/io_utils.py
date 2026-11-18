"""Image decoding / encoding helpers for the re-inspection API.

The detector only ever receives decoded numpy arrays.  Nothing in this
package reads generation parameters or ground-truth data.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

IMAGE_SIZE = 512


class ImageValidationError(ValueError):
    """Raised when an uploaded image is not a 512x512 grayscale PNG."""


def decode_gray_png(data: bytes) -> np.ndarray:
    """Decode an uploaded PNG as an 8-bit grayscale 512x512 image."""
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise ImageValidationError("empty upload")
    buffer = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageValidationError("file is not a decodable image")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ImageValidationError("unsupported channel count")
    if img.ndim != 2:
        raise ImageValidationError("image must be single-channel grayscale")
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ImageValidationError(
            f"image must be {IMAGE_SIZE}x{IMAGE_SIZE} grayscale, got {img.shape}"
        )
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def encode_png_base64(img: np.ndarray) -> str:
    return base64.b64encode(encode_png(img)).decode("ascii")


def decode_png_base64(text: str) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(text), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
