"""FastAPI application for PCB golden-board re-inspection.

Endpoints
---------
``POST /api/v1/inspect``
    Multipart upload of two 512x512 grayscale PNG files (``golden`` and
    ``inspected``). Returns registration / exposure evidence plus the
    defect mask and bounding boxes in *inspected-image* coordinates.
``POST /api/v1/polarity``
    Multipart upload of the same two PNGs plus a component box and a mark
    box (both in **golden** coordinates). Returns a tantalum-capacitor
    polarity verdict: correct / reversed / undeterminable.
``GET  /health``
    Liveness probe.
``GET  /docs``
    Swagger UI with upload forms for both endpoints (Try it out).

A content-level inability to decide is HTTP 200 with
``status="undeterminable"`` (it is a valid inspection outcome). Bad input
is HTTP 400/422.
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .inspection.detector import inspect_images
from .inspection.io_utils import (
    IMAGE_SIZE,
    ImageValidationError,
    decode_gray_png,
)
from .inspection.polarity import (
    VERDICT_ZH,
    PolarityInputError,
    analyze_polarity,
    parse_box,
)

app = FastAPI(
    title="PCB Golden-Board Re-inspection API",
    version="1.1.0",
    description=(
        "Upload a 512x512 grayscale golden-board PNG and an inspected PNG. "
        "The service estimates a global affine registration and linear "
        "exposure compensation (robust to defects), then returns the defect "
        "mask / boxes in inspected-image coordinates. Ambiguous content "
        "(solid color, periodic texture, untrusted correction) returns "
        "status **undeterminable**, never pass.\n\n"
        "**Polarity** (`/api/v1/polarity`) verifies SMD tantalum-capacitor "
        "orientation from a component box and a polarity-mark box given in "
        "golden coordinates: it compares mark evidence at the original end "
        "with the 180-degree opposite end and only reports reversed when "
        "the opposite end positively shows the mark. An occluded mark or "
        "near-equal evidence is undeterminable, never correct."
    ),
)


class WarpParams(BaseModel):
    tx_px: float
    ty_px: float
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear_deg: float


class Correction(BaseModel):
    warp_matrix: list[list[float]] = Field(
        description="2x3 affine, maps golden coords to inspected coords"
    )
    warp_convention: str
    warp_params: WarpParams
    gain: float = Field(description="Fitted generation gain: I = gain*G + bias")
    bias: float = Field(description="Fitted generation bias")
    exposure_model: str
    correction_gain: float = Field(
        description="Multiplier that maps inspected intensities back to golden"
    )
    correction_bias: float = Field(
        description="Offset that maps inspected intensities back to golden"
    )
    robust_sigma: float
    ecc_correlation: float
    overlap_ratio: float


class PolarityProbe(BaseModel):
    box: list[int] = Field(description="[x, y, w, h] in golden coordinates")
    evidence: float
    strong: bool
    normalized_contrast: float
    fill_ratio: float
    mark_gray_distance: float
    mark_gray_matches: bool
    probe_median_gray: float
    body_median_gray: float


class PolarityEvidence(BaseModel):
    original_end: PolarityProbe
    opposite_end: PolarityProbe
    evidence_margin: float
    thresholds: dict


class PolarityComponent(BaseModel):
    box: list[int]
    center: list[float]
    long_side_px: int
    short_side_px: int
    long_axis: str
    mark_box: list[int]
    opposite_mark_box: list[int]


class PolarityResponse(BaseModel):
    status: str = Field(description="correct | reversed | undeterminable")
    verdict_zh: str = Field(description="正装 | 反装 | 无法判定")
    reason_code: str | None = Field(
        default=None,
        description=(
            "Set when undeterminable: MARK_OCCLUDED, AMBIGUOUS_EVIDENCE, "
            "LOCAL_ALIGNMENT_UNRELIABLE, LOCAL_DISPLACEMENT_TOO_LARGE, "
            "COMPONENT_SIZE_OUT_OF_RANGE, COMPONENT_NOT_IN_OVERLAP, "
            "MARK_CONTRAST_INSUFFICIENT, MARK_REFERENCE_UNRELIABLE, "
            "COMPONENT_BODY_TOO_SMALL, or a shared correction code "
            "(SOLID_COLOR, LOW_TEXTURE, PERIODIC_TEXTURE, ECC_FAILED, "
            "LOW_CORRELATION, IMPLAUSIBLE_WARP, INSUFFICIENT_OVERLAP, "
            "LOW_INLIER_RATIO, EXPOSURE_FIT_FAILED, IMPLAUSIBLE_EXPOSURE)"
        ),
    )
    reason: str | None = None
    metrics: dict
    correction: Correction | None = None
    component: PolarityComponent | None = None
    local_alignment: dict | None = None
    evidence: PolarityEvidence | None = None
    debug_png_base64: str | None = Field(
        default=None, description="Diagnostic panel (PNG, base64)"
    )


class InspectResponse(BaseModel):
    status: str = Field(
        description="pass | defect | undeterminable"
    )
    reason_code: str | None = Field(
        default=None,
        description=(
            "Set when undeterminable: SOLID_COLOR, LOW_TEXTURE, "
            "PERIODIC_TEXTURE, ECC_FAILED, LOW_CORRELATION, "
            "IMPLAUSIBLE_WARP, INSUFFICIENT_OVERLAP, LOW_INLIER_RATIO, "
            "EXPOSURE_FIT_FAILED, IMPLAUSIBLE_EXPOSURE"
        ),
    )
    reason: str | None = None
    metrics: dict
    correction: Correction | None = None
    defect_mask_png_base64: str | None = Field(
        default=None,
        description="PNG-encoded binary defect mask, inspected coordinates",
    )
    defect_boxes: list[list[int]] = Field(
        default_factory=list,
        description="[x, y, w, h] bounding boxes in inspected coordinates",
    )
    debug_png_base64: str | None = Field(
        default=None, description="2x2 diagnostic panel (PNG, base64)"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "image_size": IMAGE_SIZE}


@app.post("/api/v1/inspect", response_model=InspectResponse)
async def inspect(
    golden: UploadFile = File(..., description="512x512 grayscale golden PNG"),
    inspected: UploadFile = File(
        ..., description="512x512 grayscale inspected PNG"
    ),
    include_debug: bool = Form(
        default=True, description="Include the 2x2 diagnostic panel"
    ),
) -> dict:
    golden_bytes = await golden.read()
    inspected_bytes = await inspected.read()
    try:
        g = decode_gray_png(golden_bytes)
        i = decode_gray_png(inspected_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return inspect_images(g, i, include_debug=include_debug)


@app.post("/api/v1/polarity", response_model=PolarityResponse)
async def polarity(
    golden: UploadFile = File(..., description="512x512 grayscale golden PNG"),
    inspected: UploadFile = File(
        ..., description="512x512 grayscale inspected PNG"
    ),
    component_box: str = Form(
        ...,
        description="Component bounding box in GOLDEN coordinates: 'x,y,w,h'",
        examples=["200,220,48,32"],
    ),
    mark_box: str = Form(
        ...,
        description=(
            "Polarity-mark bounding box in GOLDEN coordinates: 'x,y,w,h'. "
            "Must lie inside component_box."
        ),
        examples=["203,223,8,26"],
    ),
    include_debug: bool = Form(
        default=True, description="Include the diagnostic panel"
    ),
) -> dict:
    golden_bytes = await golden.read()
    inspected_bytes = await inspected.read()
    try:
        g = decode_gray_png(golden_bytes)
        i = decode_gray_png(inspected_bytes)
        component = parse_box(component_box, "component_box")
        mark = parse_box(mark_box, "mark_box")
        result = analyze_polarity(
            g, i, component, mark, include_debug=include_debug
        )
    except (ImageValidationError, PolarityInputError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "status": result.status,
        "verdict_zh": VERDICT_ZH[result.status],
        "reason_code": result.reason_code,
        "reason": result.reason,
        "metrics": result.metrics,
        "correction": result.correction,
        "component": result.component,
        "local_alignment": result.local_alignment,
        "evidence": result.evidence,
        "debug_png_base64": result.debug_png_base64,
    }
    return payload
