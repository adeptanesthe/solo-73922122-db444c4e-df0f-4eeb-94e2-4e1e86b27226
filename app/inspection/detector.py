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

from . import geometry
from .correction import correction_payload, prepare_correction
from .exposure import detect_defects
from .io_utils import IMAGE_SIZE, encode_png_base64

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


def inspect_images(
    golden: np.ndarray, inspected: np.ndarray, include_debug: bool = True
) -> dict:
    """Run the full re-inspection. See module docstring for the contract."""
    prep = prepare_correction(golden, inspected)
    metrics = prep.metrics
    if not prep.ok:
        return _undeterminable(
            prep.reason_code or "UNDETERMINABLE",
            prep.reason or "correction not trustworthy",
            metrics,
        )

    golden = golden.astype(np.uint8)
    warp = prep.warp
    fit = prep.fit
    assert warp is not None and fit is not None
    i_in_g = prep.inspected_in_golden
    coverage_g = prep.coverage_g
    coverage_i = prep.coverage_i
    assert i_in_g is not None and coverage_g is not None and coverage_i is not None

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

    corrected_g = prep.corrected_in_golden
    assert corrected_g is not None
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
        "correction": correction_payload(prep),
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
