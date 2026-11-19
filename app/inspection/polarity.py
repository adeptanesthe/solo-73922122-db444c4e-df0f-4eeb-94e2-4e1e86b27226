"""SMD tantalum-capacitor polarity verification.

The polarised end of a tantalum capacitor carries a mark. A board may be
populated with the part rotated 180 degrees, which puts the mark at the
opposite end of the component footprint. Because the part body is
essentially symmetric, "no mark at the golden-mark end" alone is *not*
evidence of reversed mounting: the mark might simply be occluded. A
reversed verdict therefore requires positive evidence at the opposite
end, and a correct verdict requires positive evidence at the original end
while the opposite end is clearly free of the mark.

The analysis runs in the golden frame on the shared, trust-gated
correction from :mod:`.correction`:

1. ``prepare_correction`` performs the content trust gates, robust ECC
   registration and exposure compensation. Any rejection is passed through
   unchanged (the polarity verdict is then ``undeterminable``).
2. Applicability gates: component long side 32..64 px, golden mark/body
   contrast >= 40 gray levels, component inside the overlap, local
   post-registration displacement <= 2 px.
3. Two probe regions are evaluated in the corrected inspected image: the
   golden mark box (original end) and that box reflected through the
   component centre (opposite end). Evidence combines the signed contrast
   against the component body and the fraction of mark-like pixels, and is
   discarded if the local gray level does not match the golden mark (which
   is how an arbitrary occluder is kept from masquerading as a mark).
4. A verdict is only issued when exactly one end has strong evidence and
   the other end is clearly weak, with a sufficient margin. Both strong or
   both weak -> ``undeterminable`` (``AMBIGUOUS_EVIDENCE`` /
   ``MARK_OCCLUDED``); never ``correct`` in doubt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .correction import correction_payload, prepare_correction
from .io_utils import IMAGE_SIZE, encode_png_base64

# ---- applicability envelope -------------------------------------------
COMPONENT_LONG_SIDE_MIN = 32
COMPONENT_LONG_SIDE_MAX = 64
MIN_MARK_CONTRAST = 40.0          # |median(mark) - median(body)| in golden
MAX_LOCAL_DISPLACEMENT_PX = 2.0
MIN_BODY_AREA_PX = 40             # body pixels left after excluding both ends

# ---- evidence thresholds ----------------------------------------------
STRONG_EVIDENCE = 0.60            # normalized mark presence at one end
WEAK_EVIDENCE_MAX = 0.35          # the other end must be below this
MIN_EVIDENCE_MARGIN = 0.30        # and the two must differ by at least this
MARK_FILL_MIN = 0.30              # golden mark box must be mostly mark
MIN_REFERENCE_CONTRAST_FILL = 0.30

# Local residual alignment: exhaustive gradient-domain search on the
# component border band. The band excludes the component interior (both
# marked ends), so neither the mark nor an occluder can bias it, while the
# four component edges provide the alignment signal.
LOCAL_RING_PX = 3
LOCAL_SEARCH_PX = 6               # must exceed the 2 px tolerance
MIN_RING_GRADIENT = 6.0           # mean Sobel magnitude required in the band
MIN_RING_PIXELS = 80

VERDICT_ZH = {
    "correct": "正装",
    "reversed": "反装",
    "undeterminable": "无法判定",
}


class PolarityInputError(ValueError):
    """Malformed boxes / geometry: an HTTP 400-level input error."""


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def clipped(self) -> "Box":
        x0 = max(0, self.x)
        y0 = max(0, self.y)
        x1 = min(IMAGE_SIZE, self.x1)
        y1 = min(IMAGE_SIZE, self.y1)
        return Box(x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def expanded(self, pad: int) -> "Box":
        return Box(self.x - pad, self.y - pad,
                   self.w + 2 * pad, self.h + 2 * pad)

    def mask(self, shape: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
             pad: int = 0) -> np.ndarray:
        b = self.expanded(pad).clipped()
        m = np.zeros(shape, np.uint8)
        if b.w > 0 and b.h > 0:
            m[b.y:b.y1, b.x:b.x1] = 255
        return m


@dataclass
class LocalAlignment:
    dx_px: float
    dy_px: float
    displacement_px: float
    method: str
    response: float | None
    reliable: bool


@dataclass
class ProbeEvidence:
    box: Box
    probe_median_gray: float
    body_median_gray: float
    contrast_gray: float
    normalized_contrast: float
    fill_ratio: float
    mark_gray_distance: float
    mark_gray_matches: bool
    evidence: float
    strong: bool


@dataclass
class PolarityResult:
    status: str                    # correct | reversed | undeterminable
    reason_code: str | None
    reason: str | None
    metrics: dict = field(default_factory=dict)
    correction: dict | None = None
    component: dict | None = None
    local_alignment: dict | None = None
    evidence: dict | None = None
    debug_png_base64: str | None = None


def parse_box(raw: str, name: str) -> Box:
    """Parse ``"x,y,w,h"`` (JSON-style ``[x, y, w, h]`` also accepted)."""
    if raw is None:
        raise PolarityInputError(f"{name} is required")
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4 or any(not p for p in parts):
        raise PolarityInputError(
            f"{name} must be four comma-separated integers 'x,y,w,h'"
        )
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise PolarityInputError(
            f"{name} must be four comma-separated integers 'x,y,w,h'"
        ) from exc
    if any(not np.isfinite(v) or abs(v - round(v)) > 1e-6 for v in values):
        raise PolarityInputError(f"{name} values must be integers")
    x, y, w, h = (int(round(v)) for v in values)
    return validate_box(Box(x, y, w, h), name)


def validate_box(box: Box, name: str) -> Box:
    if box.w <= 0 or box.h <= 0:
        raise PolarityInputError(f"{name} width and height must be positive")
    if box.x < 0 or box.y < 0 or box.x1 > IMAGE_SIZE or box.y1 > IMAGE_SIZE:
        raise PolarityInputError(
            f"{name} must lie fully inside the {IMAGE_SIZE}x{IMAGE_SIZE} frame"
        )
    return box


def _validate_boxes(component: Box, mark: Box) -> None:
    if mark.x < component.x or mark.y < component.y or \
            mark.x1 > component.x1 or mark.y1 > component.y1:
        raise PolarityInputError("mark_box must lie inside component_box")
    if mark.x == component.x and mark.y == component.y and \
            mark.x1 == component.x1 and mark.y1 == component.y1:
        raise PolarityInputError(
            "mark_box must not equal component_box; a body region is required"
        )
    opposite = _opposite_box(mark, component)
    if opposite.x < component.x or opposite.y < component.y or \
            opposite.x1 > component.x1 or opposite.y1 > component.y1:
        raise PolarityInputError(
            "mark_box is not positioned like an end mark: its 180-degree "
            "reflection does not lie inside component_box"
        )


def _opposite_box(box: Box, component: Box) -> Box:
    """Reflect ``box`` through the component centre (180 degree rotation)."""
    cx, cy = component.center
    mx, my = box.center
    nx = int(round(2 * cx - mx - box.w / 2.0))
    ny = int(round(2 * cy - my - box.h / 2.0))
    return Box(nx, ny, box.w, box.h)


def _body_mask(component: Box, probes: list[Box], pad: int = 1) -> np.ndarray:
    """Component interior excluding both probe regions (orientation neutral)."""
    body = component.mask()
    for probe in probes:
        body[probe.expanded(pad).clipped().mask() > 0] = 0
    return body


def _median(img: np.ndarray, mask: np.ndarray) -> float:
    vals = img[mask > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _fill_fraction(img: np.ndarray, mask: np.ndarray, body_median: float,
                   ref_contrast: float) -> float:
    vals = img[mask > 0].astype(np.float64) - body_median
    if vals.size == 0:
        return 0.0
    sign = 1.0 if ref_contrast >= 0 else -1.0
    mark_like = sign * vals >= 0.5 * abs(ref_contrast)
    return float(mark_like.mean())


def _probe_evidence(
    corrected: np.ndarray,
    probe: Box,
    body_mask: np.ndarray,
    body_median: float,
    ref_contrast: float,
    golden_mark_median: float,
    golden_fill: float,
) -> ProbeEvidence:
    probe_mask = probe.mask()
    probe_median = _median(corrected, probe_mask)
    contrast = probe_median - body_median
    normalized = contrast / ref_contrast if abs(ref_contrast) > 1e-6 else 0.0
    fill = _fill_fraction(corrected, probe_mask, body_median, ref_contrast)
    fill_ratio = fill / max(golden_fill, MIN_REFERENCE_CONTRAST_FILL)
    # An arbitrary occluder can have the right *contrast sign*; require the
    # local gray level to agree with the golden mark as well.
    mark_distance = abs(probe_median - golden_mark_median)
    tol = max(18.0, 0.40 * abs(ref_contrast))
    matches = mark_distance <= tol
    raw = min(float(np.clip(normalized, 0.0, 1.0)),
              float(np.clip(fill_ratio, 0.0, 1.0)))
    evidence = raw if matches else 0.0
    return ProbeEvidence(
        box=probe,
        probe_median_gray=round(probe_median, 3),
        body_median_gray=round(body_median, 3),
        contrast_gray=round(contrast, 3),
        normalized_contrast=round(float(normalized), 4),
        fill_ratio=round(float(fill_ratio), 4),
        mark_gray_distance=round(float(mark_distance), 3),
        mark_gray_matches=bool(matches),
        evidence=round(float(evidence), 4),
        strong=bool(evidence >= STRONG_EVIDENCE),
    )


def _estimate_local_alignment(
    golden: np.ndarray,
    corrected: np.ndarray,
    component: Box,
    probes: list[Box],
) -> LocalAlignment:
    """Residual post-registration shift at the component borders.

    The search uses the *gradient magnitude* of a narrow band straddling the
    component outline. The interior is excluded, so both possible mark ends
    (and any occluder on them) are outside the support, and the component
    edges themselves define the measurement. Exhaustive search plus a
    parabolic sub-pixel refinement avoids the multiple-peak failures of
    phase correlation on a small, structured patch.
    """
    ring = LOCAL_RING_PX
    search = LOCAL_SEARCH_PX
    win = component.expanded(ring + search + 2).clipped()
    if win.w < 2 * ring + 4 or win.h < 2 * ring + 4:
        return LocalAlignment(0.0, 0.0, float("inf"), "none", None, False)

    g = golden[win.y:win.y1, win.x:win.x1].astype(np.float32)
    c = corrected[win.y:win.y1, win.x:win.x1].astype(np.float32)

    band = np.zeros(g.shape, np.uint8)
    outer = component.expanded(ring).clipped()
    inner = component.expanded(-ring)
    band[outer.y - win.y:outer.y1 - win.y,
         outer.x - win.x:outer.x1 - win.x] = 1
    if inner.w > 0 and inner.h > 0:
        band[inner.y - win.y:inner.y1 - win.y,
             inner.x - win.x:inner.x1 - win.x] = 0
    for probe in probes:
        local = probe.expanded(2).clipped()
        x0 = max(0, local.x - win.x)
        y0 = max(0, local.y - win.y)
        x1 = min(g.shape[1], local.x1 - win.x)
        y1 = min(g.shape[0], local.y1 - win.y)
        if x1 > x0 and y1 > y0:
            band[y0:y1, x0:x1] = 0

    ys, xs = np.nonzero(band)
    if ys.size < MIN_RING_PIXELS:
        return LocalAlignment(0.0, 0.0, float("inf"), "none", None, False)

    g_grad = _gradient_magnitude(g)
    ring_mean = float(g_grad[ys, xs].mean())
    if ring_mean < MIN_RING_GRADIENT:
        return LocalAlignment(
            0.0, 0.0, float("inf"), "low_ring_texture",
            round(ring_mean, 3), False,
        )
    c_grad = _gradient_magnitude(c)

    h, w = g.shape
    scores: dict[tuple[int, int], float] = {}
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            if dx == 0 and dy == 0:
                shifted = c_grad
            else:
                mat = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
                shifted = cv2.warpAffine(
                    c_grad, mat, (w, h), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            diff = g_grad[ys, xs] - shifted[ys, xs]
            scores[(dx, dy)] = float(np.mean(np.abs(diff)))

    best = min(scores, key=scores.get)
    bx, by = best
    # Hitting the search boundary means the shift may be larger than the
    # range we can measure -> not trustworthy.
    at_boundary = abs(bx) == search or abs(by) == search

    def refine(axis: int) -> float:
        if axis == 0:
            a = scores.get((bx - 1, by))
            b = scores.get((bx + 1, by))
        else:
            a = scores.get((bx, by - 1))
            b = scores.get((bx, by + 1))
        c0 = scores[best]
        if a is None or b is None:
            return 0.0
        den = a - 2.0 * c0 + b
        if abs(den) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (a - b) / den, -1.0, 1.0))

    dx = bx + refine(0)
    dy = by + refine(1)
    disp = float(np.hypot(dx, dy))
    reliable = (not at_boundary) and disp <= float(search) - 1.0
    return LocalAlignment(
        dx_px=round(float(dx), 4),
        dy_px=round(float(dy), 4),
        displacement_px=round(disp, 4),
        method="gradient_ring_search",
        response=round(float(scores[best]), 4),
        reliable=reliable,
    )


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _correction_failure(prep, reason_code: str | None = None) -> PolarityResult:
    return PolarityResult(
        status="undeterminable",
        reason_code=reason_code or prep.reason_code or "UNDETERMINABLE",
        reason=prep.reason or "correction is not trustworthy",
        metrics=prep.metrics,
        correction=None,
    )


def analyze_polarity(
    golden: np.ndarray,
    inspected: np.ndarray,
    component_box: Box,
    mark_box: Box,
    include_debug: bool = True,
) -> PolarityResult:
    """Verify tantalum-capacitor polarity. See the module docstring."""
    if golden.shape != (IMAGE_SIZE, IMAGE_SIZE) or \
            inspected.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("both images must be 512x512")
    _validate_boxes(component_box, mark_box)
    golden = golden.astype(np.uint8)
    inspected = inspected.astype(np.uint8)

    # ---- shared trustworthy correction ------------------------------------
    prep = prepare_correction(golden, inspected)
    if not prep.ok:
        return _correction_failure(prep)
    metrics = prep.metrics
    corrected = prep.corrected_in_golden
    coverage_g = prep.coverage_g
    assert corrected is not None and coverage_g is not None

    long_side = max(component_box.w, component_box.h)
    short_side = min(component_box.w, component_box.h)
    metrics["component_long_side_px"] = long_side
    metrics["component_short_side_px"] = short_side
    if not (COMPONENT_LONG_SIDE_MIN <= long_side <= COMPONENT_LONG_SIDE_MAX):
        return PolarityResult(
            status="undeterminable",
            reason_code="COMPONENT_SIZE_OUT_OF_RANGE",
            reason=(
                f"component long side {long_side}px is outside the supported "
                f"{COMPONENT_LONG_SIDE_MIN}-{COMPONENT_LONG_SIDE_MAX}px range"
            ),
            metrics=metrics,
            correction=_correction_dict(prep),
        )

    opposite_box = _opposite_box(mark_box, component_box)
    probes = [mark_box, opposite_box]
    body_mask = _body_mask(component_box, probes)
    body_area = int((body_mask > 0).sum())
    metrics["component_body_area_px"] = body_area
    if body_area < MIN_BODY_AREA_PX:
        return PolarityResult(
            status="undeterminable",
            reason_code="COMPONENT_BODY_TOO_SMALL",
            reason="component body region left after excluding both ends is "
                   "too small to establish the reference contrast",
            metrics=metrics,
            correction=_correction_dict(prep),
        )

    # Both ends and the component must be inside the trusted overlap.
    coverage = (coverage_g > 0).astype(np.uint8) * 255
    coverage = cv2.erode(coverage, np.ones((5, 5), np.uint8), iterations=1)
    region = component_box.expanded(2).mask()
    if np.any(coverage[region > 0] == 0):
        return PolarityResult(
            status="undeterminable",
            reason_code="COMPONENT_NOT_IN_OVERLAP",
            reason="the component lies (partly) outside the trusted overlap "
                   "region and cannot be judged",
            metrics=metrics,
            correction=_correction_dict(prep),
        )

    # ---- golden mark reference / applicability contrast ------------------
    mark_mask = mark_box.mask()
    golden_mark_median = _median(golden, mark_mask)
    golden_body_median = _median(golden, body_mask)
    ref_contrast = golden_mark_median - golden_body_median
    golden_fill = _fill_fraction(
        golden, mark_mask, golden_body_median, ref_contrast
    )
    metrics["mark_reference"] = {
        "mark_median_gray": round(golden_mark_median, 3),
        "body_median_gray": round(golden_body_median, 3),
        "contrast_gray": round(ref_contrast, 3),
        "fill_fraction": round(golden_fill, 4),
    }
    if abs(ref_contrast) < MIN_MARK_CONTRAST:
        return PolarityResult(
            status="undeterminable",
            reason_code="MARK_CONTRAST_INSUFFICIENT",
            reason=(
                f"golden mark/body contrast {ref_contrast:+.1f} gray levels is "
                f"below the required {MIN_MARK_CONTRAST:.0f}"
            ),
            metrics=metrics,
            correction=_correction_dict(prep),
        )
    if golden_fill < MARK_FILL_MIN:
        return PolarityResult(
            status="undeterminable",
            reason_code="MARK_REFERENCE_UNRELIABLE",
            reason=(
                f"only {golden_fill:.0%} of the golden mark box is mark-like; "
                "the provided mark box does not describe a clear mark"
            ),
            metrics=metrics,
            correction=_correction_dict(prep),
        )

    # ---- local post-registration alignment -------------------------------
    local = _estimate_local_alignment(golden, corrected, component_box, probes)
    metrics["local_alignment"] = {
        "dx_px": local.dx_px,
        "dy_px": local.dy_px,
        "displacement_px": local.displacement_px,
        "method": local.method,
        "response": local.response,
        "reliable": local.reliable,
        "max_allowed_px": MAX_LOCAL_DISPLACEMENT_PX,
    }
    if not local.reliable:
        return PolarityResult(
            status="undeterminable",
            reason_code="LOCAL_ALIGNMENT_UNRELIABLE",
            reason="local displacement around the component could not be "
                   "measured reliably",
            metrics=metrics,
            correction=_correction_dict(prep),
            local_alignment=metrics["local_alignment"],
        )
    if local.displacement_px > MAX_LOCAL_DISPLACEMENT_PX:
        return PolarityResult(
            status="undeterminable",
            reason_code="LOCAL_DISPLACEMENT_TOO_LARGE",
            reason=(
                f"local displacement {local.displacement_px:.2f}px exceeds "
                f"{MAX_LOCAL_DISPLACEMENT_PX:.0f}px; the component is not "
                "aligned closely enough for a polarity decision"
            ),
            metrics=metrics,
            correction=_correction_dict(prep),
            local_alignment=metrics["local_alignment"],
        )

    # ---- mark evidence at both ends --------------------------------------
    body_median_c = _median(corrected, body_mask)
    orig = _probe_evidence(
        corrected, mark_box, body_mask, body_median_c, ref_contrast,
        golden_mark_median, golden_fill,
    )
    opp = _probe_evidence(
        corrected, opposite_box, body_mask, body_median_c, ref_contrast,
        golden_mark_median, golden_fill,
    )
    margin = abs(orig.evidence - opp.evidence)
    evidence = {
        "original_end": _probe_payload(orig),
        "opposite_end": _probe_payload(opp),
        "evidence_margin": round(float(margin), 4),
        "thresholds": {
            "strong": STRONG_EVIDENCE,
            "weak_max": WEAK_EVIDENCE_MAX,
            "min_margin": MIN_EVIDENCE_MARGIN,
        },
    }
    metrics["polarity_evidence"] = evidence

    status, code, reason = _decide(orig, opp, margin)
    debug_b64 = None
    if include_debug:
        debug_b64 = encode_png_base64(
            _debug_panel(golden, corrected, component_box, mark_box,
                         opposite_box, orig, opp, status)
        )
    return PolarityResult(
        status=status,
        reason_code=code,
        reason=reason,
        metrics=metrics,
        correction=_correction_dict(prep),
        component={
            "box": component_box.as_list(),
            "center": [round(c, 2) for c in component_box.center],
            "long_side_px": long_side,
            "short_side_px": short_side,
            "long_axis": "horizontal" if component_box.w >= component_box.h
                         else "vertical",
            "mark_box": mark_box.as_list(),
            "opposite_mark_box": opposite_box.as_list(),
        },
        local_alignment=metrics["local_alignment"],
        evidence=evidence,
        debug_png_base64=debug_b64,
    )


def _decide(
    orig: ProbeEvidence, opp: ProbeEvidence, margin: float
) -> tuple[str, str | None, str | None]:
    orig_strong = orig.strong
    opp_strong = opp.strong
    if orig_strong and opp_strong:
        return ("undeterminable", "AMBIGUOUS_EVIDENCE",
                "both ends show mark-consistent evidence; polarity is "
                "ambiguous")
    if orig_strong:
        if opp.evidence > WEAK_EVIDENCE_MAX or margin < MIN_EVIDENCE_MARGIN:
            return ("undeterminable", "AMBIGUOUS_EVIDENCE",
                    "the opposite end is not clearly free of the mark "
                    f"(evidence {opp.evidence:.2f}, margin {margin:.2f})")
        return ("correct", None, None)
    if opp_strong:
        if orig.evidence > WEAK_EVIDENCE_MAX or margin < MIN_EVIDENCE_MARGIN:
            return ("undeterminable", "AMBIGUOUS_EVIDENCE",
                    "the original end is not clearly free of the mark "
                    f"(evidence {orig.evidence:.2f}, margin {margin:.2f})")
        return ("reversed", None, None)
    if orig.evidence <= WEAK_EVIDENCE_MAX and opp.evidence <= WEAK_EVIDENCE_MAX:
        return ("undeterminable", "MARK_OCCLUDED",
                "no mark-consistent evidence at either end; the mark is "
                "occluded or absent and reversed mounting must not be "
                "inferred from a missing mark alone")
    return ("undeterminable", "AMBIGUOUS_EVIDENCE",
            "mark evidence is too close between the two ends "
            f"(original {orig.evidence:.2f}, opposite {opp.evidence:.2f})")


def _correction_dict(prep) -> dict | None:
    if not prep.ok:
        return None
    return correction_payload(prep)


def _probe_payload(p: ProbeEvidence) -> dict:
    return {
        "box": p.box.as_list(),
        "evidence": p.evidence,
        "strong": p.strong,
        "normalized_contrast": p.normalized_contrast,
        "fill_ratio": p.fill_ratio,
        "mark_gray_distance": p.mark_gray_distance,
        "mark_gray_matches": p.mark_gray_matches,
        "probe_median_gray": p.probe_median_gray,
        "body_median_gray": p.body_median_gray,
    }


def _debug_panel(
    golden: np.ndarray,
    corrected: np.ndarray,
    component: Box,
    mark: Box,
    opposite: Box,
    orig: ProbeEvidence,
    opp: ProbeEvidence,
    status: str,
) -> np.ndarray:
    def color(img: np.ndarray, labels: bool) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (component.x, component.y),
                      (component.x1 - 1, component.y1 - 1), (0, 255, 255), 1)
        cv2.rectangle(out, (mark.x, mark.y), (mark.x1 - 1, mark.y1 - 1),
                      (0, 255, 0), 1)
        cv2.rectangle(out, (opposite.x, opposite.y),
                      (opposite.x1 - 1, opposite.y1 - 1), (255, 255, 0), 1)
        if labels:
            cv2.putText(out, f"orig {orig.evidence:.2f}", (mark.x, mark.y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(out, f"opp {opp.evidence:.2f}",
                        (opposite.x, opposite.y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                        cv2.LINE_AA)
        return out

    top_left = color(golden, False)
    top_right = color(corrected, True)
    top = np.hstack([top_left, top_right])
    zoom_pad = 10
    win = component.expanded(zoom_pad).clipped()
    target_w = top.shape[1]
    target_h = top.shape[0]
    zoom_h = max(1, target_h // 2)
    zoom_w = max(1, target_w // 2)
    halves = []
    for img in (golden, corrected):
        patch = img[win.y:win.y1, win.x:win.x1]
        if patch.size == 0:
            patch = np.zeros((2, 2), np.uint8)
        scaled = cv2.resize(patch, (zoom_w, zoom_h),
                            interpolation=cv2.INTER_NEAREST)
        halves.append(cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR))
    bottom = np.hstack(halves)
    panel = np.vstack([top, bottom])
    # cv2.putText has no CJK glyphs; the Chinese verdict is in the JSON.
    label = f"verdict: {status}"
    cv2.putText(panel, label, (6, panel.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return panel
