"""Linear exposure compensation and residual defect detection.

Observed-image model (in golden-frame geometry, overlap region only)::

    I = a * G + b          (a = generation gain, b = generation bias)

The Huber-IRLS fit estimates ``a`` and ``b`` on inlier pixels so a real
defect cannot be absorbed into the exposure correction. The inspected
image is then brought back to the golden intensity scale by the inverse
relation::

    I_corrected = (I - b) / a

and defects are the statistically significant residuals
``I_corrected - G`` that survive morphology and area filtering.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE = 512

# Plausibility of the *generation* gain/bias that maps G onto I.
GAIN_BOUND = (0.5, 2.0)
BIAS_BOUND = (-60.0, 60.0)
HUBER_K = 1.345
IRLS_ITERATIONS = 6
MIN_DEFECT_AREA = 36           # px; suppresses isolated sensor noise
# A genuine added/removed block is a sustained, same-sign intensity change.
# Edge-misregistration artifacts are small and bipolar (residuals cancel),
# so a component must either be large or have strongly coherent sign.
MIN_DEFECT_AREA_STRICT = 120  # px: above this, accept on area alone
MIN_SIGN_COHERENCE = 0.60      # |mean residual| / mean |residual|
MORPH_KERNEL = 3
EROSION_BORDER_PX = 3          # never judge within 3px of the overlap edge
MIN_RESIDUAL_ABS = 15.0        # gray-level floor for a defect pixel
EDGE_GRADIENT_GAIN = 0.06      # how strongly local edges raise the threshold
EDGE_FACTOR_MAX = 4.0          # cap on the edge-adaptive threshold multiplier


@dataclass
class ExposureFit:
    gain: float                 # a, such that I = a * G + b
    bias: float                 # b
    robust_sigma: float         # residual scale of the I = aG+b model
    inlier_ratio: float
    iterations: int


def fit_exposure(
    observed: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
) -> ExposureFit:
    """Robust linear fit ``observed (y) = gain * reference (x) + bias``.

    The regression runs on Gaussian-smoothed images. After image
    registration a residual sub-pixel misalignment always remains; on the
    raw images that decorrelation biases the slope toward 1. Smoothing over
    a kernel larger than the residual misalignment removes this artifact.
    The robust sigma is still measured on full-resolution residuals so
    defect sensitivity is unaffected.
    """
    sel = valid_mask > 0
    # Low-pass both images identically before the regression.
    ref_s = cv2.GaussianBlur(reference.astype(np.float32), (15, 15), 0)
    obs_s = cv2.GaussianBlur(observed.astype(np.float32), (15, 15), 0)
    x = ref_s[sel].astype(np.float64)
    y = obs_s[sel].astype(np.float64)
    n = x.size
    if n < 100:
        raise ValueError("overlap region too small for exposure fit")

    # OLS initialization.
    x_mean, y_mean = x.mean(), y.mean()
    var = np.sum((x - x_mean) ** 2)
    if var < 1e-6:
        gain = 1.0
        bias = float(y_mean - x_mean)
    else:
        gain = float(np.sum((x - x_mean) * (y - y_mean)) / var)
        bias = float(y_mean - gain * x_mean)

    weights = np.ones(n, np.float64)
    sigma = 1.0
    for _ in range(IRLS_ITERATIONS):
        w_sum = weights.sum()
        wx = np.sum(weights * x)
        wy = np.sum(weights * y)
        wxx = np.sum(weights * x * x)
        wxy = np.sum(weights * x * y)
        denom = w_sum * wxx - wx * wx
        if abs(denom) < 1e-9:
            break
        gain = (w_sum * wxy - wx * wy) / denom
        bias = (wy - gain * wx) / w_sum
        resid = y - (gain * x + bias)
        mad = np.median(np.abs(resid - np.median(resid)))
        sigma = max(1.4826 * mad, 1.0)
        scaled = np.abs(resid) / (HUBER_K * sigma)
        # Huber weights; guard the reciprocal against an exact-zero residual.
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(
                scaled <= 1.0, 1.0, np.where(scaled > 0, 1.0 / scaled, 1.0)
            )

    # Robust residual scale on the *full-resolution* images so the defect
    # threshold reflects sensor-level noise, not smoothing residuals.
    full_resid = observed.astype(np.float64) - (
        gain * reference.astype(np.float64) + bias
    )
    full_in = full_resid[sel]
    mad_full = np.median(np.abs(full_in - np.median(full_in)))
    sigma_full = max(1.4826 * mad_full, 1.0)
    inlier = np.abs(full_in) <= max(2.5 * sigma_full, MIN_RESIDUAL_ABS * 0.8)
    return ExposureFit(
        gain=float(gain),
        bias=float(bias),
        robust_sigma=float(sigma_full),
        inlier_ratio=float(inlier.mean()),
        iterations=IRLS_ITERATIONS,
    )


def exposure_plausible(fit: ExposureFit) -> tuple[bool, str | None]:
    if not (GAIN_BOUND[0] <= fit.gain <= GAIN_BOUND[1]):
        return False, f"gain {fit.gain:.3f} outside plausible {GAIN_BOUND}"
    if not (BIAS_BOUND[0] <= fit.bias <= BIAS_BOUND[1]):
        return False, f"bias {fit.bias:.2f} outside plausible {BIAS_BOUND}"
    return True, None


def correct_to_reference(
    observed: np.ndarray, fit: ExposureFit
) -> np.ndarray:
    """Invert the exposure model: map observed intensities back to reference."""
    out = (observed.astype(np.float64) - fit.bias) / fit.gain
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class DefectResult:
    mask: np.ndarray                 # uint8 0/255, golden-frame coordinates
    boxes: list[list[int]]           # [x, y, w, h]
    residual: np.ndarray             # signed float residual in overlap
    threshold_gray: float
    eroded_valid_mask: np.ndarray


def detect_defects(
    observed: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    fit: ExposureFit,
) -> DefectResult:
    """Threshold robust residuals inside the eroded overlap region.

    ``observed`` is the inspected image rendered into golden-frame geometry
    and ``reference`` is the golden image. The residual is computed after
    inverting the fitted exposure model, so perfectly aligned normal pixels
    have residual ~0 regardless of gain/bias.
    """
    corrected = (observed.astype(np.float64) - fit.bias) / fit.gain
    residual = corrected - reference.astype(np.float64)

    # sigma was measured in observed (I=aG+b) units; rescale to reference.
    sigma_ref = max(fit.robust_sigma / abs(fit.gain), 1.0)

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    eroded = cv2.erode(valid_mask, kernel, iterations=EROSION_BORDER_PX)

    # Gradient-adaptive threshold: a global affine cannot perfectly align
    # sharp local edges, so residuals there are expected to be larger even
    # for a defect-free board. Raise the bar with the local reference
    # gradient while keeping flat regions at the sensitive floor.
    gx = cv2.Sobel(reference, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(reference, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edge_factor = 1.0 + EDGE_GRADIENT_GAIN * grad / max(
        float(np.median(grad[eroded > 0])) + 1e-6, 1.0
    )
    edge_factor = np.clip(edge_factor, 1.0, EDGE_FACTOR_MAX)
    base_thr = max(3.0 * sigma_ref, MIN_RESIDUAL_ABS)
    thr_map = base_thr * edge_factor
    candidate = (np.abs(residual) >= thr_map) & (eroded > 0)
    mask = (candidate.astype(np.uint8) * 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[list[int]] = []
    keep = np.zeros_like(mask)
    for label in range(1, num):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_DEFECT_AREA:
            continue

        component = labels == label
        comp_residual = residual[component]
        mean_abs = float(np.abs(comp_residual).mean())
        coherence = abs(float(comp_residual.mean())) / max(mean_abs, 1e-9)

        # Reject bipolar edge-misregistration fragments: a real defect is a
        # coherent same-sign region. Large regions are accepted on area alone.
        if area < MIN_DEFECT_AREA_STRICT and coherence < MIN_SIGN_COHERENCE:
            continue

        boxes.append([x, y, w, h])
        keep[component] = 255

    return DefectResult(
        mask=keep,
        boxes=boxes,
        residual=residual.astype(np.float32),
        threshold_gray=float(base_thr),
        eroded_valid_mask=eroded,
    )
