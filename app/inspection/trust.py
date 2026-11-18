"""Trust gates: decide whether a correction / defect verdict is defensible.

The system must say "undeterminable" instead of "pass" when:

* an image is solid color or near-flat (no information to align with);
* a periodic texture admits multiple equally-good alignments;
* the ECC fit did not correlate well or implies an implausible warp;
* the overlap region is too small or too few inliers remain.

Periodicity is measured from the normalized autocorrelation of the
high-pass filtered image: a strong secondary peak at a non-trivial lag
means shifting by that lag leaves the content nearly unchanged, i.e. the
alignment problem has multiple solutions.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE = 512

# Texture gate (mean Sobel magnitude on 0..255 image).
MIN_GRADIENT_MAGNITUDE = 2.0
MIN_STD = 4.0

# Autocorrelation gate.
LAG_MIN = 4            # ignore lags smaller than this (local smoothness)
LAG_MAX = 96           # half-image ambiguity is a structural ambiguity too
PERIOD_PEAK_MIN = 0.55  # normalized ACF height that proves repetition
AMBIGUITY_MARGIN = 0.10  # alt alignment must be nearly as good as zero lag
MIN_CC = 0.75          # minimum final ECC correlation coefficient
MIN_OVERLAP_RATIO = 0.85
MIN_INLIER_RATIO = 0.55


@dataclass
class ContentVerdict:
    decidable: bool
    reason_code: str | None
    reason: str | None
    metrics: dict


def _gradient_strength(img: np.ndarray) -> float:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(cv2.magnitude(gx, gy)))


def _highpass(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3.0)
    hp = img.astype(np.float32) - blurred
    return hp - hp.mean()


def autocorrelation(img: np.ndarray, max_lag: int = LAG_MAX) -> np.ndarray:
    """Normalized autocorrelation of the high-pass image, shape (2L+1, 2L+1)."""
    hp = _highpass(img)
    energy = float(np.sum(hp * hp))
    if energy < 1e-6:
        return np.zeros((2 * max_lag + 1, 2 * max_lag + 1), np.float32)
    # Full normalized ACF via FFT.
    n = IMAGE_SIZE
    f = np.fft.rfft2(hp, s=(2 * n, 2 * n))
    acf_full = np.fft.irfft2(f * np.conj(f), s=(2 * n, 2 * n))
    acf_full = np.fft.fftshift(acf_full)
    center = np.array(acf_full.shape) // 2
    tile = acf_full[
        center[0] - max_lag:center[0] + max_lag + 1,
        center[1] - max_lag:center[1] + max_lag + 1,
    ].copy()
    tile /= max(energy, 1e-9)
    return tile.astype(np.float32)


def periodic_ambiguity(img: np.ndarray) -> dict:
    """Find strong non-zero-lag autocorrelation peaks (periodic repeats)."""
    acf = autocorrelation(img)
    l = LAG_MAX
    yy, xx = np.mgrid[-l:l + 1, -l:l + 1]
    dist = np.sqrt(xx * xx + yy * yy)
    acf[dist < LAG_MIN] = -np.inf  # suppress zero-lag neighborhood

    # Local maxima via grayscale dilation.
    dilated = cv2.dilate(acf, np.ones((3, 3), np.uint8))
    peaks = (acf == dilated) & np.isfinite(acf)
    peak_values = acf[peaks]
    peak_coords = np.argwhere(peaks)  # (row, col) in tile coords

    candidates = []
    order = np.argsort(peak_values)[::-1][:6]
    for idx in order:
        value = float(peak_values[idx])
        r, c = peak_coords[idx]
        lag_y, lag_x = int(r - l), int(c - l)
        lag_mag = float(np.hypot(lag_x, lag_y))
        if value >= PERIOD_PEAK_MIN and value >= 1.0 - AMBIGUITY_MARGIN \
                and LAG_MIN <= lag_mag <= LAG_MAX:
            candidates.append(
                {"lag_x_px": lag_x, "lag_y_px": lag_y,
                 "lag_mag_px": lag_mag, "correlation": value}
            )
    return {
        "periodic": len(candidates) > 0,
        "secondary_peaks": candidates,
        "acf_secondary_max": float(np.nanmax(acf)) if np.isfinite(acf).any() else 0.0,
    }


def assess_content(img: np.ndarray, role: str) -> ContentVerdict:
    """Pre-registration gate for one image (``role`` is 'golden'/'inspected')."""
    std = float(img.std())
    grad = _gradient_strength(img)
    metrics = {f"{role}_std": std, f"{role}_gradient_mean": grad}
    if std < MIN_STD:
        return ContentVerdict(
            False, "SOLID_COLOR",
            f"{role} image is effectively solid color (std={std:.2f}); "
            "registration has no information",
            metrics,
        )
    if grad < MIN_GRADIENT_MAGNITUDE:
        return ContentVerdict(
            False, "LOW_TEXTURE",
            f"{role} image has too little texture "
            f"(mean gradient={grad:.2f}); alignment is under-constrained",
            metrics,
        )
    period = periodic_ambiguity(img)
    metrics[f"{role}_acf_secondary_max"] = period["acf_secondary_max"]
    if period["periodic"]:
        top = period["secondary_peaks"][0]
        return ContentVerdict(
            False, "PERIODIC_TEXTURE",
            f"{role} image contains a periodic texture with an alternative "
            f"alignment at lag ({top['lag_x_px']},{top['lag_y_px']}) px, "
            f"correlation={top['correlation']:.3f}; the correction is not "
            "unique, so defects cannot be judged",
            metrics,
        )
    return ContentVerdict(True, None, None, metrics)
