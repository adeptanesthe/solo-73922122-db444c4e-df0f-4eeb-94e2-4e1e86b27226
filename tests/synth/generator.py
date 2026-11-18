"""Fixed-seed synthetic PCB sample generator.

Generates pairs of 512x512 grayscale images with a known global affine
disturbance, linear exposure change, and added/removed block defects:

* translation |tx|, |ty| <= 8 px
* rotation |angle| <= 3 degrees
* gain in [0.8, 1.2], bias in [-15, +15]
* defect blocks covering 1%-10% of the frame, gray difference >= 40

Ground-truth (warp matrix, gain, bias, defect mask) is returned inside
:class:`SyntheticSample` and may be written to a manifest that lives
outside the detector package.  The detector never imports this module and
cannot see any of these parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 512
BASE_SEED = 20261117
INNER_MARGIN = 24          # defects stay clear of warp-induced borders
NOISE_SIGMA_GOLDEN = 3.0
NOISE_SIGMA_INSPECTED = 2.0


@dataclass
class SyntheticSample:
    name: str
    golden: np.ndarray            # uint8 512x512
    inspected: np.ndarray         # uint8 512x512
    defect_mask: np.ndarray       # uint8 0/255, inspected coordinates
    true_warp: np.ndarray         # 2x3 golden -> inspected
    gain: float
    bias: float
    expect_status: str            # "defect" | "pass" | "undeterminable"
    expect_reason: str | None
    meta: dict


def _rng(index: int) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + index)


def make_golden_board(rng: np.random.Generator) -> np.ndarray:
    """Random but deterministic PCB-like golden board (non-periodic)."""
    img = np.full((SIZE, SIZE), 90.0, np.float32)

    # Smooth low-frequency shading (random, non-repeating).
    small = rng.normal(0.0, 1.0, (16, 16)).astype(np.float32)
    shading = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    img += shading * 12.0

    # Copper-like traces: thin random rectangles.
    for _ in range(28):
        x0 = rng.integers(20, SIZE - 80)
        y0 = rng.integers(20, SIZE - 80)
        length = int(rng.integers(40, 200))
        thick = int(rng.integers(3, 9))
        gray = float(rng.integers(120, 210))
        if rng.random() < 0.5:
            x1 = min(SIZE, x0 + length)
            img[y0:y0 + thick, x0:x1] = gray
        else:
            y1 = min(SIZE, y0 + length)
            img[y0:y1, x0:x0 + thick] = gray

    # Pads / vias: filled circles and small squares.
    for _ in range(46):
        cx = int(rng.integers(20, SIZE - 20))
        cy = int(rng.integers(20, SIZE - 20))
        radius = int(rng.integers(4, 12))
        gray = float(rng.integers(30, 235))
        cv2.circle(img, (cx, cy), radius, gray, thickness=-1)
    for _ in range(24):
        s = int(rng.integers(8, 20))
        x0 = int(rng.integers(20, SIZE - 20 - s))
        y0 = int(rng.integers(20, SIZE - 20 - s))
        gray = float(rng.integers(30, 235))
        cv2.rectangle(img, (x0, y0), (x0 + s, y0 + s), gray, thickness=-1)

    img += rng.normal(0.0, NOISE_SIGMA_GOLDEN, (SIZE, SIZE)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _random_warp(rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
    angle = float(rng.uniform(-3.0, 3.0))
    tx = float(rng.uniform(-8.0, 8.0))
    ty = float(rng.uniform(-8.0, 8.0))
    warp = cv2.getRotationMatrix2D(((SIZE - 1) / 2.0, (SIZE - 1) / 2.0),
                                   angle, 1.0)
    warp[0, 2] += tx
    warp[1, 2] += ty
    return warp.astype(np.float64), angle, tx, ty


def _paint_defects(
    golden: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Paint added/removed blocks. Returns (defected image, golden-frame mask,
    per-block evidence). Total defect area is 1%-10% of the frame."""
    target_fraction = float(rng.uniform(0.01, 0.10))
    target_area = target_fraction * SIZE * SIZE
    n_blocks = int(rng.integers(1, 5))
    defected = golden.copy()
    mask = np.zeros((SIZE, SIZE), np.uint8)
    blocks: list[dict] = []
    bg_gray = float(np.median(golden[:20, :]))

    remaining = target_area
    for b in range(n_blocks):
        share = remaining if b == n_blocks - 1 else \
            remaining * float(rng.uniform(0.2, 0.6))
        side = int(np.clip(np.sqrt(share), 14, 120))
        w = int(np.clip(side * float(rng.uniform(0.7, 1.4)), 12, 140))
        h = int(np.clip(share / max(w, 1), 12, 140))
        x0 = int(rng.integers(INNER_MARGIN, SIZE - INNER_MARGIN - w))
        y0 = int(rng.integers(INNER_MARGIN, SIZE - INNER_MARGIN - h))
        local = golden[y0:y0 + h, x0:x0 + w].astype(np.float32)
        local_mean = float(local.mean())

        if rng.random() < 0.5:  # added block: forced high-contrast gray
            if local_mean > 127:
                g2 = float(rng.integers(0, 60))
            else:
                g2 = float(rng.integers(195, 256))
            kind = "added"
        else:                   # removed block: repaint with background
            g2 = bg_gray + float(rng.integers(-8, 9))
            kind = "removed"
        diff = abs(g2 - local_mean)
        if diff < 40:  # guarantee the required contrast
            g2 = float(np.clip(local_mean + rng.choice([-1.0, 1.0]) *
                               rng.integers(45, 90), 0, 255))
            diff = abs(g2 - local_mean)
            kind = "added"

        defected[y0:y0 + h, x0:x0 + w] = np.uint8(np.clip(g2, 0, 255))
        cv2.rectangle(mask, (x0, y0), (x0 + w - 1, y0 + h - 1), 255, -1)
        blocks.append({"kind": kind, "box": [x0, y0, w, h],
                       "gray_diff": float(diff)})
        remaining -= w * h
        if remaining <= 144:
            break
    return defected, mask, blocks


def _warp_forward(img: np.ndarray, warp: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Apply a golden->inspected forward warp.

    ``warp`` maps golden coordinates to inspected coordinates; OpenCV's
    standard warpAffine inverts it internally for back-sampling.
    """
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, warp.astype(np.float32), (SIZE, SIZE),
                          flags=flags, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def make_normal_sample(index: int, with_defect: bool) -> SyntheticSample:
    rng = _rng(index)
    golden = make_golden_board(rng)
    warp, angle, tx, ty = _random_warp(rng)
    gain = float(rng.uniform(0.8, 1.2))
    bias = float(rng.uniform(-15.0, 15.0))

    if with_defect:
        defected, mask_g, blocks = _paint_defects(golden, rng)
        min_diff = min(b["gray_diff"] for b in blocks)
    else:
        defected, mask_g, blocks = golden.copy(), np.zeros((SIZE, SIZE), np.uint8), []
        min_diff = None

    warped = _warp_forward(defected, warp)
    inspected = gain * warped.astype(np.float32) + bias
    inspected += rng.normal(0.0, NOISE_SIGMA_INSPECTED, (SIZE, SIZE)).astype(np.float32)
    inspected = np.clip(inspected, 0, 255).astype(np.uint8)

    mask_i = _warp_forward(mask_g, warp, nearest=True)
    # Defect pixels that fell outside the inspected frame cannot be judged.
    mask_i[warped == 0] = 0

    name = f"synth_{index:02d}_{'defect' if with_defect else 'clean'}"
    meta = {
        "name": name,
        "type": "normal",
        "angle_deg": angle, "tx_px": tx, "ty_px": ty,
        "gain": gain, "bias": bias,
        "defect_area_fraction": float((mask_i > 0).sum() / (SIZE * SIZE)),
        "min_gray_diff": min_diff,
        "blocks": blocks,
    }
    return SyntheticSample(
        name=name, golden=golden, inspected=inspected, defect_mask=mask_i,
        true_warp=warp, gain=gain, bias=bias,
        expect_status="defect" if with_defect else "pass",
        expect_reason=None, meta=meta,
    )


def make_solid_sample(index: int, noisy: bool) -> SyntheticSample:
    rng = _rng(10_000 + index)
    level = int(rng.integers(80, 180))
    golden = np.full((SIZE, SIZE), level, np.uint8)
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-10, 10))
    warp, angle, tx, ty = _random_warp(rng)
    inspected = np.clip(gain * golden.astype(np.float32) + bias, 0, 255)
    if noisy:
        golden = np.clip(golden + rng.normal(0, 1.0, (SIZE, SIZE)),
                         0, 255).astype(np.uint8)
        inspected = np.clip(
            inspected + rng.normal(0, 1.0, (SIZE, SIZE)), 0, 255
        ).astype(np.uint8)
    else:
        inspected = inspected.astype(np.uint8)
    name = f"reject_{index:02d}_solid{'_noisy' if noisy else ''}"
    meta = {"name": name, "type": "reject_solid",
            "angle_deg": angle, "tx_px": tx, "ty_px": ty,
            "gain": gain, "bias": bias}
    return SyntheticSample(
        name=name, golden=golden, inspected=inspected,
        defect_mask=np.zeros((SIZE, SIZE), np.uint8),
        true_warp=warp, gain=gain, bias=bias,
        expect_status="undeterminable",
        expect_reason="SOLID_COLOR", meta=meta,
    )


def make_periodic_sample(index: int, checker: bool) -> SyntheticSample:
    """Periodic texture shifted by exactly one period -> non-unique fit."""
    rng = _rng(20_000 + index)
    period = 16
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-10, 10))

    coords = np.arange(SIZE)
    if checker:
        xx, yy = np.meshgrid(coords, coords)
        pattern = np.where(((xx // period) + (yy // period)) % 2 == 0,
                           70, 180).astype(np.uint8)
        shift = (period, 0)
        kind = "checker"
    else:
        xx = np.tile(coords, (SIZE, 1))
        pattern = np.where((xx // period) % 2 == 0, 70, 180).astype(np.uint8)
        shift = (period, 0)
        kind = "stripe"
    pattern = np.clip(
        pattern + rng.normal(0, 1.5, (SIZE, SIZE)), 0, 255
    ).astype(np.uint8)
    golden = pattern

    # Full-period translation: an alternative alignment with identical content.
    warp = np.array([[1.0, 0.0, float(shift[0])],
                     [0.0, 1.0, float(shift[1])]])
    inspected = np.clip(gain * golden.astype(np.float32) + bias, 0, 255)
    inspected = np.clip(
        inspected + rng.normal(0, 1.5, (SIZE, SIZE)), 0, 255
    ).astype(np.uint8)
    # Forward mapping: inspected pixel position = golden position + shift.
    inspected = _warp_forward(inspected, warp, nearest=True)

    # A real defect is present too; it must not be reported because the
    # correction itself is ambiguous.
    cv2.rectangle(inspected, (60, 60), (110, 110), 255, -1)

    name = f"reject_{index:02d}_periodic_{kind}"
    meta = {"name": name, "type": "reject_periodic", "period_px": period,
            "shift_px": list(shift), "gain": gain, "bias": bias,
            "contains_defect_block": True}
    return SyntheticSample(
        name=name, golden=golden, inspected=inspected,
        defect_mask=np.zeros((SIZE, SIZE), np.uint8),
        true_warp=warp, gain=gain, bias=bias,
        expect_status="undeterminable",
        expect_reason="PERIODIC_TEXTURE", meta=meta,
    )


def build_suite(
    n_defect: int = 20, n_clean: int = 4
) -> list[SyntheticSample]:
    samples: list[SyntheticSample] = []
    idx = 0
    for _ in range(n_defect):
        samples.append(make_normal_sample(idx, with_defect=True))
        idx += 1
    for _ in range(n_clean):
        samples.append(make_normal_sample(idx, with_defect=False))
        idx += 1
    samples.append(make_solid_sample(0, noisy=False))
    samples.append(make_solid_sample(1, noisy=True))
    samples.append(make_periodic_sample(0, checker=False))
    samples.append(make_periodic_sample(1, checker=True))
    return samples


def write_samples(samples: list[SyntheticSample], out_dir: str) -> str:
    """Persist PNGs plus a manifest of ground truth (kept out of the app)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for s in samples:
        cv2.imwrite(os.path.join(out_dir, f"{s.name}__golden.png"), s.golden)
        cv2.imwrite(os.path.join(out_dir, f"{s.name}__inspected.png"),
                    s.inspected)
        cv2.imwrite(os.path.join(out_dir, f"{s.name}__gt_mask.png"),
                    s.defect_mask)
        entry = dict(s.meta)
        entry["true_warp"] = s.true_warp.tolist()
        entry["expect_status"] = s.expect_status
        entry["expect_reason"] = s.expect_reason
        manifest.append(entry)
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path
