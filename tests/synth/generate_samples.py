"""CLI: regenerate the fixed-seed synthetic sample suite and manifest.

Usage:
    python -m tests.synth.generate_samples [output_dir]

The manifest contains ground-truth warp/gain/bias and defect masks. It is
written *outside* the application package and is never read by the
detector (enforced statically in tests/evaluate.py).
"""
from __future__ import annotations

import os
import sys

# Allow running both as a module (`python -m tests.synth.generate_samples`)
# and as a script from the repository root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.synth.generator import build_suite, write_samples  # noqa: E402


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "samples")
    suite = build_suite()
    manifest_path = write_samples(suite, out)
    print(f"Wrote {len(suite)} samples to {out}")
    print(f"Manifest (ground truth, detector never reads this): {manifest_path}")
    by_status: dict[str, int] = {}
    for s in suite:
        by_status[s.expect_status] = by_status.get(s.expect_status, 0) + 1
    print("Expected outcomes:", by_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
