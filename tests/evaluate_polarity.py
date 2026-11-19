"""Independent evaluation of the tantalum-capacitor polarity checker.

Independence guarantees mirror ``tests/evaluate.py``: the analysis module is
called only with two numpy arrays plus two geometry boxes; a static audit
verifies the polarity/correction code under ``app/inspection/`` never
imports the generator or reads manifests / ground-truth files.

Acceptance criteria
-------------------
* every sample in :func:`build_basic_suite` gets its expected status
* no sample with an expected ``correct``/``reversed`` verdict is downgraded
  by more than the false-negative budget, and no ``undeterminable`` sample
  is ever reported as ``correct`` or ``reversed`` (the false-positive
  budget is zero for occlusion / ambiguity cases)
"""
from __future__ import annotations

import json
import os
import sys

# __file__ = <root>/tests/evaluate_polarity.py -> repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.inspection.polarity import (  # noqa: E402
    analyze_polarity,
    parse_box,
)
from tests.synth.polarity_generator import (  # noqa: E402
    build_basic_suite,
    make_polarity_sample,
    write_samples,
)

DETECTOR_DIR = os.path.join(ROOT, "app", "inspection")
FORBIDDEN_TOKENS = ("tests.synth", "polarity_generator", "generator",
                    "manifest", "ground_truth", "true_warp", "expect_status")
def check_detector_independence() -> list[str]:
    violations = []
    for fname in sorted(os.listdir(DETECTOR_DIR)):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(DETECTOR_DIR, fname), encoding="utf-8") as fh:
            source = fh.read()
        for token in FORBIDDEN_TOKENS:
            if token in source:
                violations.append(f"{fname} references forbidden token '{token}'")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("open(", "cv2.imread")) or " open(" in line:
                violations.append(f"{fname} appears to read files: {stripped}")
    return violations


def _extra_suite() -> list:
    """Extra cases beyond the basic suite (size edges, small shifts)."""
    return [
        make_polarity_sample(20, component_size=(32, 24)),
        make_polarity_sample(21, component_size=(32, 24), reversed_mount=True),
        make_polarity_sample(22, component_size=(64, 44)),
        make_polarity_sample(23, component_size=(64, 44), reversed_mount=True),
        make_polarity_sample(24, local_shift=(1.5, 0.5),
                             expect_override=("correct", None)),
        make_polarity_sample(25, local_shift=(1.0, -1.0), reversed_mount=True,
                             expect_override=("reversed", None)),
        make_polarity_sample(26, occlusion="mark", occlusion_size=30),
        make_polarity_sample(27, competing_evidence=True,
                             reversed_mount=True),
    ]


def evaluate(samples_dir: str | None = None) -> dict:
    suite = build_basic_suite() + _extra_suite()
    if samples_dir:
        path = write_samples(suite, samples_dir)
        print(f"[polarity-eval] wrote {len(suite)} samples -> {path}")

    rows, failures = [], []
    for s in suite:
        result = analyze_polarity(
            s.golden,
            s.inspected,
            parse_box(",".join(str(v) for v in s.component_box),
                      "component_box"),
            parse_box(",".join(str(v) for v in s.mark_box), "mark_box"),
            include_debug=False,
        )
        row = {
            "name": s.name,
            "expected": s.expect_status,
            "expected_reason": s.expect_reason,
            "got": result.status,
            "reason_code": result.reason_code,
            "local_displacement_px": (
                (result.local_alignment or {}).get("displacement_px")
            ),
        }
        if result.evidence:
            row["original_evidence"] = \
                result.evidence["original_end"]["evidence"]
            row["opposite_evidence"] = \
                result.evidence["opposite_end"]["evidence"]
        rows.append(row)

        if result.status != s.expect_status or \
                result.reason_code != s.expect_reason:
            failures.append(
                f"{s.name}: expected {s.expect_status}/{s.expect_reason}, "
                f"got {result.status}/{result.reason_code}"
            )
        if s.expect_status == "undeterminable" and \
                result.status in ("correct", "reversed"):
            failures.append(
                f"{s.name}: undeterminable content was given a directional "
                f"verdict ({result.status})"
            )

    summary = {
        "n_samples": len(suite),
        "n_correct": sum(1 for r in rows if r["got"] == "correct"),
        "n_reversed": sum(1 for r in rows if r["got"] == "reversed"),
        "n_undeterminable": sum(
            1 for r in rows if r["got"] == "undeterminable"),
        "failures": failures,
        "independence_violations": check_detector_independence(),
    }
    summary["passed"] = not failures and not summary["independence_violations"]
    return {"summary": summary, "rows": rows}


def main() -> int:
    samples_dir = sys.argv[1] if len(sys.argv) > 1 else None
    report = evaluate(samples_dir)
    print(json.dumps(report["rows"], indent=2))
    s = report["summary"]
    print("\n============= POLARITY EVALUATION SUMMARY =============")
    for key, value in s.items():
        print(f"{key}: {value}")
    print("=======================================================")
    print("RESULT:", "PASS" if s["passed"] else "FAIL")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
