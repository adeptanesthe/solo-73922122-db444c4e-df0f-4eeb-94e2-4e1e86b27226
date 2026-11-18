"""Functional tests for the re-inspection API and core pipeline."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inspection.detector import inspect_images
from app.inspection.io_utils import decode_png_base64, encode_png
from app.main import app
from tests.synth.generator import (
    make_normal_sample,
    make_periodic_sample,
    make_solid_sample,
)

client = TestClient(app)


def _upload(golden: np.ndarray, inspected: np.ndarray, debug: bool = True):
    files = {
        "golden": ("golden.png", io.BytesIO(encode_png(golden)), "image/png"),
        "inspected": ("inspected.png", io.BytesIO(encode_png(inspected)),
                      "image/png"),
    }
    return client.post(
        f"/api/v1/inspect?include_debug={'true' if debug else 'false'}",
        files=files,
    )


def test_health_and_docs():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_defect_sample_end_to_end():
    sample = make_normal_sample(0, with_defect=True)
    resp = _upload(sample.golden, sample.inspected)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "defect"
    assert len(body["defect_boxes"]) >= 1
    mask = decode_png_base64(body["defect_mask_png_base64"])
    assert mask.shape == (512, 512)
    # Defect mask must live inside the reported overlap region.
    corr = body["correction"]
    assert 0.8 <= corr["gain"] <= 1.2 and abs(corr["bias"]) <= 15
    assert corr["ecc_correlation"] >= 0.75
    assert body["debug_png_base64"] is not None
    # Non-overlap rule: no mask pixels on the frame border.
    assert mask[0, :].sum() == 0 and mask[:, 0].sum() == 0


def test_clean_sample_passes():
    sample = make_normal_sample(50, with_defect=False)
    body = _upload(sample.golden, sample.inspected, debug=False).json()
    assert body["status"] == "pass"
    assert body["defect_boxes"] == []


def test_solid_color_is_undeterminable_never_pass():
    sample = make_solid_sample(0, noisy=False)
    body = _upload(sample.golden, sample.inspected).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "SOLID_COLOR"
    assert body["correction"] is None


def test_periodic_texture_is_undeterminable_never_pass():
    sample = make_periodic_sample(1, checker=True)
    body = _upload(sample.golden, sample.inspected).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "PERIODIC_TEXTURE"
    # A defect block exists in the input, but must not be reported.
    assert body["defect_boxes"] == []


def test_rejects_wrong_size():
    small = np.zeros((256, 256), np.uint8)
    resp = _upload(small, small)
    assert resp.status_code == 400


def test_rejects_non_png_bytes():
    good = make_normal_sample(1, with_defect=True).golden
    resp = client.post(
        "/api/v1/inspect",
        files={
            "golden": ("a.png", io.BytesIO(b"not an image"), "image/png"),
            "inspected": ("b.png", io.BytesIO(encode_png(good)), "image/png"),
        },
    )
    assert resp.status_code == 400


def test_detector_never_judges_non_overlap():
    sample = make_normal_sample(2, with_defect=True)
    result = inspect_images(sample.golden, sample.inspected, include_debug=False)
    assert result["status"] in ("defect", "pass")
    mask = decode_png_base64(result["defect_mask_png_base64"])
    # Every predicted defect pixel must lie within the inspected frame's
    # valid coverage (the detector erodes the overlap border by 3px).
    assert mask.dtype == np.uint8 and mask.shape == (512, 512)
    border = np.zeros_like(mask)
    border[:3, :] = 255
    border[-3:, :] = 255
    border[:, :3] = 255
    border[:, -3:] = 255
    assert np.logical_and(mask > 0, border > 0).sum() == 0


def test_exposure_estimation_accuracy():
    sample = make_normal_sample(3, with_defect=True)
    result = inspect_images(sample.golden, sample.inspected, include_debug=False)
    assert result["status"] == "defect"
    assert abs(result["correction"]["gain"] - sample.gain) < 0.05
    assert abs(result["correction"]["bias"] - sample.bias) < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
