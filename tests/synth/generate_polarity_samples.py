"""CLI: regenerate the polarity sample suite and its manifest.

Usage:
    python -m tests.synth.generate_polarity_samples [output_dir]

The manifest records ground-truth polarity, warp/gain/bias and the golden
component / mark boxes. It is written outside the application package and
is never read by the analyzer (enforced in tests/evaluate_polarity.py).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.synth.polarity_generator import (  # noqa: E402
    build_basic_suite,
    write_samples,
)


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "samples")
    suite = build_basic_suite()
    manifest_path = write_samples(suite, out)
    print(f"Wrote {len(suite)} polarity samples to {out}")
    print(f"Manifest (ground truth, detector never reads this): {manifest_path}")
    by_status: dict[str, int] = {}
    for s in suite:
        by_status[s.expect_status] = by_status.get(s.expect_status, 0) + 1
    print("Expected outcomes:", by_status)
    for s in suite:
        print(
            f"  {s.name}: component={s.component_box} mark={s.mark_box} "
            f"-> {s.expect_status}/{s.expect_reason}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
