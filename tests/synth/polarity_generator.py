"""Synthetic tantalum-capacitor polarity samples (fixed seed).

Generates a golden board containing one SMD tantalum capacitor with a
polarity mark at one end, and an inspected board where the part may be:

* mounted normally (mark at the golden position),
* mounted reversed (part rotated 180 degrees about its own centre, so the
  mark sits at the opposite end of the same footprint),

with a known global affine disturbance and linear exposure change applied
to the whole inspected board. Optional occluders (a flat fixture shadow),
a competing mark-like band at the far end, and reduced mark contrast
exercise the undeterminable paths.

The golden component box and golden mark box live in the sample metadata so
tests can post exactly the coordinates the API contract requires. The
detector never imports this module.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 512
BASE_SEED = 20261119
INNER_MARGIN = 40

BOARD_BASE = 92
CAP_BODY = 160
CAP_MARK = 36            # mark vs body contrast ~124 gray levels
MIN_CONTRAST_CASE_MARK = 140   # contrast 20 -> below the 40 requirement


@dataclass
class PolaritySample:
    name: str
    golden: np.ndarray
    inspected: np.ndarray
    component_box: list[int]          # golden coordinates
    mark_box: list[int]               # golden coordinates
    true_warp: np.ndarray             # golden -> inspected
    gain: float
    bias: float
    expect_status: str                # correct | reversed | undeterminable
    expect_reason: str | None
    meta: dict


def _rng(index: int) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + index)


def _board_background(rng: np.random.Generator) -> np.ndarray:
    """Non-periodic board texture (no capacitor yet)."""
    img = np.full((SIZE, SIZE), float(BOARD_BASE), np.float32)
    small = rng.normal(0.0, 1.0, (16, 16)).astype(np.float32)
    img += cv2.resize(small, (SIZE, SIZE),
                      interpolation=cv2.INTER_CUBIC) * 8.0
    for _ in range(24):
        x0 = int(rng.integers(10, SIZE - 90))
        y0 = int(rng.integers(10, SIZE - 90))
        length = int(rng.integers(50, 180))
        thick = int(rng.integers(3, 7))
        gray = float(rng.integers(105, 135))
        if rng.random() < 0.5:
            img[y0:y0 + thick, x0:x0 + length] = gray
        else:
            img[y0:y0 + length, x0:x0 + thick] = gray
    for _ in range(30):
        cx = int(rng.integers(15, SIZE - 15))
        cy = int(rng.integers(15, SIZE - 15))
        r = int(rng.integers(4, 10))
        cv2.circle(img, (cx, cy), r, float(rng.integers(40, 220)), -1)
    for _ in range(18):
        s = int(rng.integers(8, 18))
        x0 = int(rng.integers(15, SIZE - 15 - s))
        y0 = int(rng.integers(15, SIZE - 15 - s))
        cv2.rectangle(img, (x0, y0), (x0 + s, y0 + s),
                      float(rng.integers(40, 220)), -1)
    img += rng.normal(0.0, 2.0, (SIZE, SIZE)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_capacitor(board: np.ndarray, box: list[int], mark_gray: int,
                    body_gray: int) -> tuple[np.ndarray, list[int]]:
    """Draw a capacitor with a marked end-cap at its low-x / low-y end."""
    img = board.copy()
    x, y, w, h = box
    body = float(body_gray)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), body, -1)
    # Slight metallic sheen so the body is not perfectly flat.
    if w >= h:
        sheen = np.linspace(-8, 8, w, dtype=np.float32)
        img[y:y + h, x:x + w] = np.clip(
            body + sheen[None, :], 0, 255
        ).astype(np.uint8)
        band = max(4, int(round(0.14 * w)))
        mark = [x + 2, y + 2, band, h - 4]
    else:
        sheen = np.linspace(-8, 8, h, dtype=np.float32)
        img[y:y + h, x:x + w] = np.clip(
            body + sheen[:, None], 0, 255
        ).astype(np.uint8)
        band = max(4, int(round(0.14 * h)))
        mark = [x + 2, y + 2, w - 4, band]
    img[mark[1]:mark[1] + mark[3], mark[0]:mark[0] + mark[2]] = mark_gray
    return img, mark


def _opposite_box(box: list[int], component: list[int]) -> list[int]:
    """Reflect ``box`` through the component centre (180 degree rotation)."""
    cx = component[0] + component[2] / 2.0
    cy = component[1] + component[3] / 2.0
    mx = box[0] + box[2] / 2.0
    my = box[1] + box[3] / 2.0
    nx = int(round(2 * cx - mx - box[2] / 2.0))
    ny = int(round(2 * cy - my - box[3] / 2.0))
    return [nx, ny, box[2], box[3]]


def _rotate_part_in_place(img: np.ndarray, box: list[int]) -> np.ndarray:
    """Rotate the capacitor 180 degrees about its centre (reversed mount)."""
    x, y, w, h = box
    out = img.copy()
    out[y:y + h, x:x + w] = cv2.rotate(
        img[y:y + h, x:x + w], cv2.ROTATE_180
    )
    return out


def _paint_competing_band(img: np.ndarray, box: list[int],
                          mark: list[int], gray: int,
                          reversed_mount: bool) -> np.ndarray:
    """Add a mark-like band opposite the inspected part's *actual* mark.

    With a reversed mount the real mark sits at the golden far end, so the
    competing band goes to the golden near end (and vice versa). Painting
    both at the same end would not be a competing-evidence case.
    """
    out = img.copy()
    x, y, w, h = box
    band = mark[2] if w >= h else mark[3]
    if w >= h:
        if reversed_mount:
            out[y + 2:y + h - 2, x + 2:x + 2 + band] = gray
        else:
            out[y + 2:y + h - 2, x + w - 2 - band:x + w - 2] = gray
    else:
        if reversed_mount:
            out[y + 2:y + 2 + band, x + 2:x + w - 2] = gray
        else:
            out[y + h - 2 - band:y + h - 2, x + 2:x + w - 2] = gray
    return out


def _occlude(img: np.ndarray, rng: np.random.Generator,
             near: list[int], size: int = 34,
             color: int | None = None) -> np.ndarray:
    """Paste a flat fixture shadow over the ``near`` neighbourhood."""
    out = img.copy()
    cx, cy = near[0] + near[2] // 2, near[1] + near[3] // 2
    x0 = int(np.clip(cx - size // 2 + int(rng.integers(-4, 5)), 0,
                     SIZE - size))
    y0 = int(np.clip(cy - size // 2 + int(rng.integers(-4, 5)), 0,
                     SIZE - size))
    gray = float(rng.integers(100, 150)) if color is None else float(color)
    cv2.rectangle(out, (x0, y0), (x0 + size, y0 + size), gray, -1)
    return out


def _shift_component(img: np.ndarray, box: list[int],
                     shift: tuple[float, float]) -> np.ndarray:
    """Locally displace only the capacitor patch (placement error).

    The board around it stays put, so the global registration follows the
    board and the component shows a local offset.
    """
    x, y, w, h = box
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(SIZE, x + w + pad), min(SIZE, y + h + pad)
    patch = img[y0:y1, x0:x1].copy()
    out = img.copy()
    dx, dy = shift
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    moved = cv2.warpAffine(patch, warp, (patch.shape[1], patch.shape[0]),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    out[y0:y1, x0:x1] = moved
    return out


def _random_warp(rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
    angle = float(rng.uniform(-2.5, 2.5))
    tx = float(rng.uniform(-3.0, 3.0))
    ty = float(rng.uniform(-3.0, 3.0))
    warp = cv2.getRotationMatrix2D(((SIZE - 1) / 2.0, (SIZE - 1) / 2.0),
                                   angle, 1.0)
    warp[0, 2] += tx
    warp[1, 2] += ty
    return warp.astype(np.float64), angle, tx, ty


def _apply_warp(img: np.ndarray, warp: np.ndarray,
                nearest: bool = False) -> np.ndarray:
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, warp.astype(np.float32), (SIZE, SIZE),
                          flags=flags, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def make_polarity_sample(
    index: int,
    *,
    reversed_mount: bool = False,
    occlusion: str | None = None,
    mark_contrast: int = CAP_MARK,
    competing_evidence: bool = False,
    component_size: tuple[int, int] = (48, 32),
    orientation: str = "horizontal",
    occlusion_size: int = 34,
    occlusion_color: int | None = None,
    local_shift: tuple[float, float] | None = None,
    expect_override: tuple[str, str | None] | None = None,
) -> PolaritySample:
    """Build one golden/inspected pair with a known polarity outcome.

    ``occlusion`` is None, ``"mark"`` (shadow over the end that actually
    carries the mark in the inspected image -- for a reversed mount that is
    the golden *opposite* end, so the true mark is hidden) or ``"opposite"``
    (shadow over the end without the mark, which is how an occluder could
    masquerade as evidence).
    """
    rng = _rng(index)
    board = _board_background(rng)

    w, h = component_size
    if orientation == "vertical":
        w, h = h, w
    x = int(rng.integers(INNER_MARGIN, SIZE - INNER_MARGIN - w))
    y = int(rng.integers(INNER_MARGIN, SIZE - INNER_MARGIN - h))
    box = [x, y, w, h]

    # Golden: reference orientation, mark at the low-x / low-y end.
    golden, mark = _draw_capacitor(board, box, mark_contrast, CAP_BODY)

    # Inspected: golden part, optionally reversed, then occluded.
    inspected_board = golden.copy()
    if reversed_mount:
        inspected_board = _rotate_part_in_place(inspected_board, box)
    if competing_evidence:
        inspected_board = _paint_competing_band(
            inspected_board, box, mark, mark_contrast, reversed_mount
        )
    if occlusion == "mark":
        mark_end = (_opposite_box(mark, box) if reversed_mount else mark)
        inspected_board = _occlude(
            inspected_board, rng, mark_end, size=occlusion_size,
            color=occlusion_color,
        )
    elif occlusion == "opposite":
        inspected_board = _occlude(
            inspected_board, rng, _opposite_box(mark, box),
            size=occlusion_size, color=occlusion_color,
        )
    if local_shift is not None:
        inspected_board = _shift_component(
            inspected_board, box, local_shift
        )

    warp, angle, tx, ty = _random_warp(rng)
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-12.0, 12.0))
    warped = _apply_warp(inspected_board, warp)
    inspected = gain * warped.astype(np.float32) + bias
    inspected += rng.normal(0.0, 2.0, (SIZE, SIZE)).astype(np.float32)
    inspected = np.clip(inspected, 0, 255).astype(np.uint8)

    if expect_override is not None:
        expect_status, expect_reason = expect_override
    elif occlusion == "mark":
        expect_status, expect_reason = "undeterminable", "MARK_OCCLUDED"
    elif occlusion == "opposite" and reversed_mount:
        # The end that actually carries the mark is the golden-opposite end,
        # which this occluder covers -> no evidence at either end.
        expect_status, expect_reason = "undeterminable", "MARK_OCCLUDED"
    elif competing_evidence:
        expect_status, expect_reason = "undeterminable", "AMBIGUOUS_EVIDENCE"
    elif reversed_mount:
        expect_status, expect_reason = "reversed", None
    else:
        expect_status, expect_reason = "correct", None

    name = f"polarity_{index:02d}_" + (
        "reversed" if reversed_mount else "correct")
    if orientation == "vertical":
        name += "_vertical"
    if occlusion:
        name += f"_occl_{occlusion}"
    if competing_evidence:
        name += "_competing"
    if mark_contrast != CAP_MARK:
        name += f"_contrast{mark_contrast}"
    meta = {
        "name": name,
        "component_box": box,
        "mark_box": mark,
        "reversed_mount": reversed_mount,
        "occlusion": occlusion,
        "competing_evidence": competing_evidence,
        "mark_contrast": mark_contrast,
        "gain": gain,
        "bias": bias,
        "angle_deg": angle,
        "tx_px": tx,
        "ty_px": ty,
        "expect_status": expect_status,
        "expect_reason": expect_reason,
    }
    return PolaritySample(
        name=name,
        golden=golden,
        inspected=inspected,
        component_box=box,
        mark_box=mark,
        true_warp=warp,
        gain=gain,
        bias=bias,
        expect_status=expect_status,
        expect_reason=expect_reason,
        meta=meta,
    )


def build_basic_suite() -> list[PolaritySample]:
    """Deterministic suite covering the required decisions."""
    return [
        make_polarity_sample(0),
        make_polarity_sample(1, reversed_mount=True),
        make_polarity_sample(2, orientation="vertical"),
        make_polarity_sample(3, reversed_mount=True, orientation="vertical"),
        make_polarity_sample(4, occlusion="mark"),
        make_polarity_sample(5, occlusion="mark", reversed_mount=True),
        make_polarity_sample(6, occlusion="opposite"),
        make_polarity_sample(7, occlusion="opposite", reversed_mount=True),
        # A mark-coloured occluder over the far end of a correctly mounted
        # part looks like a second mark -> both ends strong -> ambiguous.
        make_polarity_sample(8, occlusion="opposite",
                             occlusion_color=CAP_MARK, occlusion_size=20,
                             expect_override=("undeterminable",
                                              "AMBIGUOUS_EVIDENCE")),
        make_polarity_sample(9, competing_evidence=True),
        make_polarity_sample(10, component_size=(36, 26)),
        make_polarity_sample(11, component_size=(58, 40), reversed_mount=True),
        make_polarity_sample(12, local_shift=(4.0, 0.0),
                             expect_override=(
                                 "undeterminable",
                                 "LOCAL_DISPLACEMENT_TOO_LARGE")),
        make_polarity_sample(
            13, mark_contrast=MIN_CONTRAST_CASE_MARK,
            expect_override=(
                "undeterminable", "MARK_CONTRAST_INSUFFICIENT"),
        ),
    ]


def write_samples(samples: list[PolaritySample], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for s in samples:
        cv2.imwrite(os.path.join(out_dir, f"{s.name}__golden.png"), s.golden)
        cv2.imwrite(os.path.join(out_dir, f"{s.name}__inspected.png"),
                    s.inspected)
        entry = dict(s.meta)
        entry["true_warp"] = s.true_warp.tolist()
        manifest.append(entry)
    path = os.path.join(out_dir, "polarity_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path
