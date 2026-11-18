"""Independent evaluation of the re-inspection detector.

Independence guarantees
-----------------------
* The detector is called only with two numpy arrays
  (``inspect_images(golden, inspected)``); no metadata is passed.
* A static check verifies the detector package imports neither the
  generator/evaluator nor reads any manifest or ground-truth file.

Acceptance criteria
-------------------
* four-corner mapping error <= 0.5 px (max over corners, mean over samples)
* strict pixel F1 >= 0.90 on defect samples
* all defect / clean / reject samples take the expected status
* solid-color and periodic-texture reject cases return undeterminable
  (the system must never report pass for them)
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

# __file__ = <root>/tests/evaluate.py  ->  one dirname reaches the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.inspection.detector import inspect_images  # noqa: E402
from app.inspection.geometry import corner_mapping_error  # noqa: E402
from tests.synth.generator import (  # noqa: E402
    BASE_SEED,
    build_suite,
    write_samples,
)

CORNER_LIMIT_PX = 0.5
F1_LIMIT = 0.90
TOLERANT_DILATION_PX = 3

DETECTOR_DIR = os.path.join(ROOT, "app", "inspection")
FORBIDDEN_TOKENS = ("tests.synth", "generator", "manifest", "ground_truth",
                    "true_warp", "gt_mask")


def check_detector_independence() -> list[str]:
    """Static audit: detector source must not reference ground-truth sources."""
    violations = []
    for fname in sorted(os.listdir(DETECTOR_DIR)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(DETECTOR_DIR, fname)
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        for token in FORBIDDEN_TOKENS:
            if token in source:
                violations.append(f"{fname} references forbidden token '{token}'")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("open(", "cv2.imread")) or " open(" in line:
                violations.append(f"{fname} appears to read files: {stripped}")
    return violations


def _binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def f1_scores(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, dict]:
    p, t = _binary(pred), _binary(truth)
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, np.logical_not(t)).sum())
    fn = int(np.logical_and(np.logical_not(p), t).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0

    # Tolerant F1 with a 3 px boundary tolerance. A prediction counts if it
    # lies within 3 px of a truth pixel and vice versa; implemented as a
    # dilation-based match so small boundary shifts are not penalized.
    kernel = np.ones((2 * TOLERANT_DILATION_PX + 1,) * 2, np.uint8)
    pd = cv2.dilate(p, kernel)
    td = cv2.dilate(t, kernel)
    tp_p = int(np.logical_and(p, td).sum())   # predicted pixels matched by truth
    tp_t = int(np.logical_and(t, pd).sum())   # truth pixels matched by pred
    prec_t = tp_p / max(int(p.sum()), 1)
    rec_t = tp_t / max(int(t.sum()), 1)
    f1_t = 2 * prec_t * rec_t / (prec_t + rec_t) if prec_t + rec_t else 0.0
    return f1, f1_t, {"tp": tp, "fp": fp, "fn": fn,
                      "precision": precision, "recall": recall}


def evaluate(samples_dir: str | None = None) -> dict:
    suite = build_suite()
    if samples_dir:
        path = write_samples(suite, samples_dir)
        print(f"[eval] wrote {len(suite)} samples and manifest -> {path}")

    rows = []
    corner_errors, f1_strict, f1_tolerant = [], [], []
    status_failures, reason_failures = [], []

    for s in suite:
        # Only arrays cross the detector boundary.
        result = inspect_images(s.golden, s.inspected, include_debug=False)
        row = {
            "name": s.name,
            "expected": s.expect_status,
            "got": result["status"],
            "reason_code": result["reason_code"],
        }

        if result["status"] != s.expect_status:
            status_failures.append(
                f"{s.name}: expected {s.expect_status}, got {result['status']}"
                + (f" ({result['reason_code']})" if result["reason_code"] else "")
            )

        if s.expect_status == "undeterminable":
            if result["status"] == "pass":  # explicitly forbidden
                reason_failures.append(
                    f"{s.name}: undeterminable content was reported as PASS"
                )
            if s.expect_reason and result["reason_code"] != s.expect_reason:
                reason_failures.append(
                    f"{s.name}: expected reason {s.expect_reason}, "
                    f"got {result['reason_code']}"
                )
            rows.append(row)
            continue

        # Decidable samples: check geometric and exposure accuracy.
        warp_est = np.array(result["correction"]["warp_matrix"])  # golden->inspected
        ce = corner_mapping_error(warp_est, s.true_warp)
        row["max_corner_error_px"] = round(ce["max_corner_error_px"], 4)
        row["gain_est"] = round(result["correction"]["gain"], 4)
        row["gain_true"] = round(s.gain, 4)
        row["bias_est"] = round(result["correction"]["bias"], 3)
        row["bias_true"] = round(s.bias, 3)
        corner_errors.append(ce["max_corner_error_px"])

        if s.expect_status == "defect":
            from app.inspection.io_utils import decode_png_base64
            pred = decode_png_base64(result["defect_mask_png_base64"])
            f1, f1t, parts = f1_scores(pred, s.defect_mask)
            f1_strict.append(f1)
            f1_tolerant.append(f1t)
            row.update({
                "f1_strict": round(f1, 4),
                "f1_tolerant_3px": round(f1t, 4),
                "precision": round(parts["precision"], 4),
                "recall": round(parts["recall"], 4),
                "pred_pixels": int((pred > 0).sum()),
                "truth_pixels": int((s.defect_mask > 0).sum()),
            })
        elif result["defect_boxes"]:
            reason_failures.append(
                f"{s.name}: clean sample produced {len(result['defect_boxes'])} "
                "false-positive defect boxes"
            )
        rows.append(row)

    summary = {
        "seed": BASE_SEED,
        "n_samples": len(suite),
        "n_defect": len(f1_strict),
        "corner_error_max_of_means_px": float(np.mean(corner_errors)),
        "corner_error_worst_px": float(np.max(corner_errors)),
        "f1_strict_mean": float(np.mean(f1_strict)) if f1_strict else None,
        "f1_strict_min": float(np.min(f1_strict)) if f1_strict else None,
        "f1_tolerant_mean": float(np.mean(f1_tolerant)) if f1_tolerant else None,
        "status_failures": status_failures,
        "reason_failures": reason_failures,
        "independence_violations": check_detector_independence(),
    }
    summary["passed"] = (
        summary["corner_error_max_of_means_px"] <= CORNER_LIMIT_PX
        and summary["f1_strict_mean"] >= F1_LIMIT
        and not status_failures
        and not reason_failures
        and not summary["independence_violations"]
    )
    return {"summary": summary, "rows": rows}


def main() -> int:
    samples_dir = sys.argv[1] if len(sys.argv) > 1 else None
    report = evaluate(samples_dir)
    print(json.dumps(report["rows"], indent=2))
    s = report["summary"]
    print("\n================ EVALUATION SUMMARY ================")
    for key, value in s.items():
        print(f"{key}: {value}")
    print("====================================================")
    print("RESULT:", "PASS" if s["passed"] else "FAIL")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
