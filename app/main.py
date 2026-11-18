"""FastAPI application for PCB golden-board re-inspection.

Endpoints
---------
``POST /api/v1/inspect``
    Multipart upload of two 512x512 grayscale PNG files (``golden`` and
    ``inspected``). Returns registration / exposure evidence plus the
    defect mask and bounding boxes in *inspected-image* coordinates.
``GET  /health``
    Liveness probe.
``GET  /docs``
    Swagger UI with an upload form (Try it out).

A content-level inability to decide is HTTP 200 with
``status="undeterminable"`` (it is a valid inspection outcome). Bad input
is HTTP 422/400.
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

app = FastAPI(
    title="PCB Golden-Board Re-inspection API",
    version="1.0.0",
    description=(
        "Upload a 512x512 grayscale golden-board PNG and an inspected PNG. "
        "The service estimates a global affine registration and linear "
        "exposure compensation (robust to defects), then returns the defect "
        "mask / boxes in inspected-image coordinates. Ambiguous content "
        "(solid color, periodic texture, untrusted correction) returns "
        "status **undeterminable**, never pass."
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
