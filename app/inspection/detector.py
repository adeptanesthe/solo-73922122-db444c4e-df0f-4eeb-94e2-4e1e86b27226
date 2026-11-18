"""End-to-end re-inspection pipeline.

This module is the *only* entry point used by both the API and the
independent evaluator.  Its sole inputs are two decoded grayscale numpy
arrays; it never reads files, generation parameters, or ground truth.

Pipeline
--------
1. Content trust gates (solid color / low texture / periodic ambiguity).
2. Multi-scale robust ECC affine registration. The returned warp ``W`` is
   golden->inspected (forward); inspected content is rendered into the
   golden frame with the inverse mapping for comparison.
3. Warp plausibility, correlation and overlap gates.
4. Robust (Huber-IRLS) linear gain/bias exposure fit on overlap inliers,
   evaluated in golden-frame geometry.
5. Residual defect detection inside the eroded overlap region, then the
   mask and boxes are forward-mapped into *inspected-image* coordinates.
6. Any failed trust gate yields status ``undeterminable`` -- never ``pass``.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry, trust
from .exposure import (
    correct_to_reference,
    detect_defects,
    exposure_plausible,
    fit_exposure,
)
from .io_utils import IMAGE_SIZE, encode_png_base64

# Robust ECC: number of residual-mask refits. Defects are masked out so the
# final warp cannot absorb them.
ROBUST_REFITS = 2
RESIDUAL_REJECT_K = 3.0
MIN_BOX_AREA = 36


def _undeterminable(code: str, reason: str, metrics: dict | None = None) -> dict:
    return {
        "status": "undeterminable",
        "reason_code": code,
        "reason": reason,
        "metrics": metrics or {},
        "correction": None,
        "defect_mask_png_base64": None,
        "defect_boxes": [],
        "debug_png_base64": None,
    }


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
    # Z-score each image inside the overlap; the difference is then exposure
    # invariant to first order.
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


def inspect_images(
    golden: np.ndarray, inspected: np.ndarray, include_debug: bool = True
) -> dict:
    """Run the full re-inspection. See module docstring for the contract."""
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
            return _undeterminable(
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
                return _undeterminable(
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
        return _undeterminable("ECC_FAILED", str(exc), metrics)

    metrics["ecc_correlation"] = float(reg.cc)
    metrics["ecc_level_correlations"] = [float(x) for x in reg.levels]
    params = geometry.decompose_affine(warp)
    metrics["warp_params"] = params

    # ---- 3. Plausibility / correlation / overlap gates --------------------
    ok, why = geometry.warp_plausible(warp)
    if not ok:
        return _undeterminable("IMPLAUSIBLE_WARP", why or "implausible warp",
                               metrics)
    if reg.cc < trust.MIN_CC:
        return _undeterminable(
            "LOW_CORRELATION",
            f"alignment correlation {reg.cc:.3f} below trust threshold "
            f"{trust.MIN_CC}",
            metrics,
        )

    coverage_g, coverage_i = geometry.overlap_masks(warp)
    overlap_ratio = float((coverage_i > 0).mean())
    metrics["overlap_ratio"] = overlap_ratio
    if overlap_ratio < trust.MIN_OVERLAP_RATIO:
        return _undeterminable(
            "INSUFFICIENT_OVERLAP",
            f"overlap covers only {overlap_ratio:.1%} of the frame; "
            "non-overlap regions cannot be judged",
            metrics,
        )

    # ---- 4. Robust exposure fit (golden-frame geometry) -------------------
    # fit_exposure signature: observed y = i_in_g, reference x = golden G,
    # and it fits y = gain * x + bias  <=>  I = gain * G + bias.
    # Erode the overlap first: warp-interpolation edge pixels would
    # otherwise bias the linear regression.
    i_in_g = geometry.render_inspected_in_golden(inspected, warp)
    fit_kernel = np.ones((3, 3), np.uint8)
    coverage_fit = cv2.erode(coverage_g, fit_kernel, iterations=2)
    try:
        fit = fit_exposure(i_in_g, golden, coverage_fit)
    except ValueError as exc:
        return _undeterminable("EXPOSURE_FIT_FAILED", str(exc), metrics)
    metrics["exposure"] = {
        "gain": fit.gain,
        "bias": fit.bias,
        "robust_sigma": fit.robust_sigma,
        "inlier_ratio": fit.inlier_ratio,
        "irls_iterations": fit.iterations,
    }
    ok, why = exposure_plausible(fit)
    if not ok:
        return _undeterminable("IMPLAUSIBLE_EXPOSURE", why or "bad exposure",
                               metrics)

    # ---- 5. Defect detection in golden frame, then map to inspected -------
    # detect_defects(inspected-frame image, golden reference, overlap mask).
    result = detect_defects(i_in_g, golden, coverage_g, fit)
    mask_g = result.mask
    mask_i = geometry.render_golden_in_inspected(mask_g, warp, nearest=True)
    # Restrict to the inspected-frame overlap and clean resampling artifacts.
    mask_i = cv2.bitwise_and(mask_i, coverage_i)
    kernel = np.ones((3, 3), np.uint8)
    mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_OPEN, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_i, connectivity=8
    )
    boxes: list[list[int]] = []
    keep = np.zeros_like(mask_i)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_BOX_AREA:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, w, h])
        keep[labels == label] = 255
    mask_i = keep

    defect_pixels = int((mask_i > 0).sum())
    metrics["defect_pixels"] = defect_pixels
    metrics["defect_area_ratio"] = defect_pixels / float(IMAGE_SIZE ** 2)
    metrics["residual_threshold_gray"] = result.threshold_gray

    corrected_g = correct_to_reference(i_in_g, fit)
    mask_b64 = encode_png_base64(mask_i)
    debug_b64 = None
    if include_debug:
        debug_b64 = encode_png_base64(
            _debug_panel(golden, corrected_g, mask_g, result.boxes,
                         result.residual, result.eroded_valid_mask)
        )

    status = "defect" if defect_pixels > 0 else "pass"
    return {
        "status": status,
        "reason_code": None,
        "reason": None,
        "metrics": metrics,
        "correction": {
            "warp_matrix": warp.tolist(),
            "warp_convention": "golden_coords -> inspected_coords",
            "warp_params": params,
            "gain": fit.gain,
            "bias": fit.bias,
            "exposure_model": "observed I = gain * G + bias; "
                              "corrected = (I - bias) / gain",
            "correction_gain": 1.0 / fit.gain,
            "correction_bias": -fit.bias / fit.gain,
            "robust_sigma": fit.robust_sigma,
            "ecc_correlation": float(reg.cc),
            "overlap_ratio": overlap_ratio,
        },
        "defect_mask_png_base64": mask_b64,
        "defect_boxes": boxes,
        "debug_png_base64": debug_b64,
    }


def _debug_panel(
    golden: np.ndarray,
    corrected_g: np.ndarray,
    mask_g: np.ndarray,
    boxes_g: list[list[int]],
    residual: np.ndarray,
    valid_g: np.ndarray,
) -> np.ndarray:
    """2x2 BGR panel in golden-frame geometry.

    golden | exposure-corrected inspected ; residual heat-map | overlay.
    """
    def to_bgr(gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    overlay = to_bgr(corrected_g)
    overlay[mask_g > 0] = (0, 0, 255)
    for x, y, w, h in boxes_g:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 1)

    r_show = np.zeros_like(residual)
    r_show[valid_g > 0] = residual[valid_g > 0]
    if (valid_g > 0).any():
        lim = max(1.0, float(np.percentile(np.abs(r_show[valid_g > 0]), 99)))
    else:
        lim = 1.0
    r_norm = np.clip(128 + 127 * r_show / lim, 0, 255).astype(np.uint8)
    r_norm[valid_g == 0] = 128
    heat = cv2.applyColorMap(r_norm, cv2.COLORMAP_JET)
    heat[valid_g == 0] = (60, 60, 60)

    top = np.hstack([to_bgr(golden), to_bgr(corrected_g)])
    bottom = np.hstack([heat, overlay])
    return np.vstack([top, bottom])
