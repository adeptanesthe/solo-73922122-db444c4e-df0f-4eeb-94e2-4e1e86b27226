"""Shared trustworthy correction: trust gates, registration, exposure.

Both the defect re-inspection endpoint and the polarity endpoint must
correct an inspected image with exactly the same machinery, otherwise a
verdict from one could be based on a different alignment/exposure model
than the other.

The function :func:`prepare_correction` returns either a rejection with a
machine-readable reason code (the caller must then answer
``undeterminable``) or the full correction state: the golden->inspected
warp, the exposure fit, the inspected image rendered into the golden
frame, and both coverage masks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry, trust
from .exposure import (
    ExposureFit,
    correct_to_reference,
    exposure_plausible,
    fit_exposure,
)
from .io_utils import IMAGE_SIZE

# Robust ECC: number of residual-mask refits. Defects are masked out so the
# final warp cannot absorb them.
ROBUST_REFITS = 2
RESIDUAL_REJECT_K = 3.0


@dataclass
class CorrectionResult:
    ok: bool
    reason_code: str | None
    reason: str | None
    metrics: dict = field(default_factory=dict)
    warp: np.ndarray | None = None              # golden -> inspected
    fit: ExposureFit | None = None
    inspected_in_golden: np.ndarray | None = None
    corrected_in_golden: np.ndarray | None = None
    coverage_g: np.ndarray | None = None
    coverage_i: np.ndarray | None = None
    overlap_ratio: float | None = None
    ecc_cc: float | None = None
    warp_params: dict | None = None

    @classmethod
    def reject(cls, code: str, reason: str, metrics: dict) -> "CorrectionResult":
        return cls(ok=False, reason_code=code, reason=reason, metrics=metrics)


def _build_inlier_mask_golden_frame(
    golden: np.ndarray,
    inspected_in_golden: np.ndarray,
    coverage_g: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Inlier mask in golden-frame geometry (ECC template frame).

    Uses a per-pixel *normalized* difference so a global gain/bias mismatch
    does not turn well-aligned pixels into outliers before exposure fitting.
    """
    g = golden.astype(np.float32)
    y = inspected_in_golden.astype(np.float32)
    valid = coverage_g > 0
    if valid.sum() < 100:
        return np.zeros_like(coverage_g), 0.0
    gz = np.zeros_like(g)
    yz = np.zeros_like(y)
    gz[valid] = (g[valid] - g[valid].mean()) / max(g[valid].std(), 1e-6)
    yz[valid] = (y[valid] - y[valid].mean()) / max(y[valid].std(), 1e-6)
    diff = np.abs(gz - yz)
    med = float(np.median(diff[valid]))
    mad = float(np.median(np.abs(diff[valid] - med)))
    sigma = max(1.4826 * mad, 0.1)
    thr = med + RESIDUAL_REJECT_K * sigma
    mask_g = ((diff < thr) & valid).astype(np.uint8) * 255
    return mask_g, float((mask_g[valid] > 0).mean())


def _fit_and_correct(
    golden: np.ndarray,
    inspected: np.ndarray,
    warp: np.ndarray,
    coverage_g: np.ndarray,
    metrics: dict,
) -> tuple[np.ndarray, np.ndarray, ExposureFit | None, str | None, str | None]:
    """Render inspected into the golden frame, then fit exposure.

    Returns (inspected_in_golden, corrected_in_golden, fit, code, reason).
    On failure ``fit`` is None and the corrected image is the identity of
    the rendered inspected image.
    """
    i_in_g = geometry.render_inspected_in_golden(inspected, warp)
    fit_kernel = np.ones((3, 3), np.uint8)
    coverage_fit = cv2.erode(coverage_g, fit_kernel, iterations=2)
    try:
        fit = fit_exposure(i_in_g, golden, coverage_fit)
    except ValueError as exc:
        return i_in_g, i_in_g, None, "EXPOSURE_FIT_FAILED", str(exc)
    metrics["exposure"] = {
        "gain": fit.gain,
        "bias": fit.bias,
        "robust_sigma": fit.robust_sigma,
        "inlier_ratio": fit.inlier_ratio,
        "irls_iterations": fit.iterations,
    }
    ok, why = exposure_plausible(fit)
    if not ok:
        return i_in_g, i_in_g, None, "IMPLAUSIBLE_EXPOSURE", why or "bad exposure"
    return i_in_g, correct_to_reference(i_in_g, fit), fit, None, None


def prepare_correction(golden: np.ndarray, inspected: np.ndarray) -> CorrectionResult:
    """Run the shared content gates, robust ECC and exposure correction."""
    if golden.shape != (IMAGE_SIZE, IMAGE_SIZE) or \
            inspected.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("both images must be 512x512")
    golden = golden.astype(np.uint8)
    inspected = inspected.astype(np.uint8)
    metrics: dict = {}

    # ---- 1. Content trust gates -------------------------------------------
    for img, role in ((golden, "golden"), (inspected, "inspected")):
        verdict = trust.assess_content(img, role)
        metrics.update(verdict.metrics)
        if not verdict.decidable:
            return CorrectionResult.reject(
                verdict.reason_code or "CONTENT_REJECTED",
                verdict.reason or "content not decidable",
                metrics,
            )

    # ---- 2. Robust multi-scale ECC ----------------------------------------
    try:
        reg = geometry.estimate_affine(golden, inspected)
        warp = reg.warp  # golden -> inspected (forward)
        inlier_ratios = []
        for _ in range(ROBUST_REFITS):
            i_in_g = geometry.render_inspected_in_golden(inspected, warp)
            coverage_g, _ = geometry.overlap_masks(warp)
            mask_g, inlier_ratio = _build_inlier_mask_golden_frame(
                golden, i_in_g, coverage_g
            )
            inlier_ratios.append(inlier_ratio)
            if inlier_ratio < trust.MIN_INLIER_RATIO:
                metrics["ecc_inlier_ratio"] = inlier_ratios
                return CorrectionResult.reject(
                    "LOW_INLIER_RATIO",
                    f"only {inlier_ratio:.1%} of overlap pixels support the "
                    "alignment; the correction is not trustworthy",
                    metrics,
                )
            # ECC's inputImage mask must be expressed in inspected frame.
            mask_i = geometry.render_golden_in_inspected(
                mask_g, warp, nearest=True
            )
            # A refit can diverge on a degraded mask (e.g. heavily darkened
            # images); keep the previous valid warp rather than failing.
            try:
                refit = geometry.robust_refit_with_mask(
                    golden, inspected, mask_i, warp
                )
                warp = refit.warp
                reg.cc = refit.cc
                reg.levels = refit.levels
            except geometry.RegistrationError:
                break
        metrics["ecc_inlier_ratio"] = inlier_ratios
    except geometry.RegistrationError as exc:
        return CorrectionResult.reject("ECC_FAILED", str(exc), metrics)

    metrics["ecc_correlation"] = float(reg.cc)
    metrics["ecc_level_correlations"] = [float(x) for x in reg.levels]
    params = geometry.decompose_affine(warp)
    metrics["warp_params"] = params

    # ---- 3. Plausibility / correlation / overlap gates --------------------
    ok, why = geometry.warp_plausible(warp)
    if not ok:
        return CorrectionResult.reject(
            "IMPLAUSIBLE_WARP", why or "implausible warp", metrics
        )
    if reg.cc < trust.MIN_CC:
        return CorrectionResult.reject(
            "LOW_CORRELATION",
            f"alignment correlation {reg.cc:.3f} below trust threshold "
            f"{trust.MIN_CC}",
            metrics,
        )

    coverage_g, coverage_i = geometry.overlap_masks(warp)
    overlap_ratio = float((coverage_i > 0).mean())
    metrics["overlap_ratio"] = overlap_ratio
    if overlap_ratio < trust.MIN_OVERLAP_RATIO:
        return CorrectionResult.reject(
            "INSUFFICIENT_OVERLAP",
            f"overlap covers only {overlap_ratio:.1%} of the frame; "
            "non-overlap regions cannot be judged",
            metrics,
        )

    # ---- 4. Robust exposure fit (golden-frame geometry) -------------------
    i_in_g, corrected, fit, code, reason = _fit_and_correct(
        golden, inspected, warp, coverage_g, metrics
    )
    if fit is None:
        return CorrectionResult.reject(code or "EXPOSURE_FIT_FAILED",
                                       reason or "exposure fit failed",
                                       metrics)

    return CorrectionResult(
        ok=True,
        reason_code=None,
        reason=None,
        metrics=metrics,
        warp=warp,
        fit=fit,
        inspected_in_golden=i_in_g,
        corrected_in_golden=corrected,
        coverage_g=coverage_g,
        coverage_i=coverage_i,
        overlap_ratio=overlap_ratio,
        ecc_cc=float(reg.cc),
        warp_params=params,
    )


def correction_payload(result: CorrectionResult) -> dict:
    """JSON representation shared by both endpoints."""
    assert result.warp is not None and result.fit is not None
    return {
        "warp_matrix": result.warp.tolist(),
        "warp_convention": "golden_coords -> inspected_coords",
        "warp_params": result.warp_params,
        "gain": result.fit.gain,
        "bias": result.fit.bias,
        "exposure_model": "observed I = gain * G + bias; "
                          "corrected = (I - bias) / gain",
        "correction_gain": 1.0 / result.fit.gain,
        "correction_bias": -result.fit.bias / result.fit.gain,
        "robust_sigma": result.fit.robust_sigma,
        "ecc_correlation": result.ecc_cc,
        "overlap_ratio": result.overlap_ratio,
    }
