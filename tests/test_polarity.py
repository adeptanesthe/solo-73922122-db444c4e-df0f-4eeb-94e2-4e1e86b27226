"""Functional tests for the tantalum-capacitor polarity endpoint."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inspection.io_utils import decode_png_base64, encode_png
from app.inspection.polarity import (
    PolarityInputError,
    analyze_polarity,
    parse_box,
)
from app.main import app
from tests.synth.polarity_generator import (
    build_basic_suite,
    make_polarity_sample,
)

client = TestClient(app)


def _box(text_box: list[int]) -> str:
    return ",".join(str(v) for v in text_box)


def _upload(sample, component=None, mark=None, debug: bool = True):
    files = {
        "golden": ("golden.png", io.BytesIO(encode_png(sample.golden)),
                   "image/png"),
        "inspected": (
            "inspected.png", io.BytesIO(encode_png(sample.inspected)),
            "image/png",
        ),
    }
    data = {
        "component_box": _box(component or sample.component_box),
        "mark_box": _box(mark or sample.mark_box),
        "include_debug": "true" if debug else "false",
    }
    return client.post("/api/v1/polarity", files=files, data=data)


def _analyze(sample) -> dict:
    return analyze_polarity(
        sample.golden,
        sample.inspected,
        parse_box(_box(sample.component_box), "component_box"),
        parse_box(_box(sample.mark_box), "mark_box"),
        include_debug=False,
    )


def test_polarity_docs_and_schema():
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/api/v1/polarity" in spec["paths"]
    body = spec["components"]["schemas"][
        "Body_polarity_api_v1_polarity_post"
    ]
    assert set(body["required"]) == {
        "golden", "inspected", "component_box", "mark_box"
    }
    # The existing re-inspection endpoint stays available.
    assert "/api/v1/inspect" in spec["paths"]


def test_polarity_suite_end_to_end():
    failures = []
    for sample in build_basic_suite():
        resp = _upload(sample, debug=False)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] != sample.expect_status or \
                body["reason_code"] != sample.expect_reason:
            failures.append(
                f"{sample.name}: expected {sample.expect_status}/"
                f"{sample.expect_reason}, got {body['status']}/"
                f"{body['reason_code']}"
            )
        assert body["verdict_zh"] in ("正装", "反装", "无法判定")
    assert not failures, "\n".join(failures)


def test_correct_mount_is_reported_correct():
    sample = make_polarity_sample(0)
    body = _upload(sample).json()
    assert body["status"] == "correct"
    assert body["verdict_zh"] == "正装"
    assert body["reason_code"] is None
    assert body["evidence"]["original_end"]["strong"] is True
    assert body["evidence"]["opposite_end"]["strong"] is False
    assert body["component"]["long_side_px"] == 48
    assert body["debug_png_base64"] is not None


def test_reversed_mount_is_reported_reversed():
    sample = make_polarity_sample(1, reversed_mount=True)
    body = _upload(sample).json()
    assert body["status"] == "reversed"
    assert body["verdict_zh"] == "反装"
    assert body["evidence"]["opposite_end"]["strong"] is True
    assert body["evidence"]["original_end"]["strong"] is False


def test_original_end_without_mark_is_not_reversed_when_occluded():
    """The core rule: a missing mark at the original end is not reverse proof."""
    sample = make_polarity_sample(4, occlusion="mark")
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "MARK_OCCLUDED"
    assert body["evidence"]["original_end"]["strong"] is False
    assert body["evidence"]["opposite_end"]["strong"] is False


def test_occluded_mark_on_reversed_part_is_undeterminable():
    sample = make_polarity_sample(5, occlusion="mark", reversed_mount=True)
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "MARK_OCCLUDED"


def test_two_marks_are_ambiguous():
    sample = make_polarity_sample(9, competing_evidence=True)
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "AMBIGUOUS_EVIDENCE"


def test_small_mark_like_occluder_on_far_end_is_ambiguous():
    sample = make_polarity_sample(
        8, occlusion="opposite", occlusion_color=36, occlusion_size=20
    )
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "AMBIGUOUS_EVIDENCE"


def test_low_contrast_mark_is_undeterminable():
    sample = make_polarity_sample(11, mark_contrast=140)
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "MARK_CONTRAST_INSUFFICIENT"
    assert body["evidence"] is None


def test_local_displacement_above_two_px_is_undeterminable():
    sample = make_polarity_sample(12, local_shift=(4.0, 0.0))
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "LOCAL_DISPLACEMENT_TOO_LARGE"
    assert body["local_alignment"]["displacement_px"] > 2.0


def test_local_displacement_within_two_px_is_accepted():
    sample = make_polarity_sample(13, local_shift=(1.5, 0.5),
                                  expect_override=("correct", None))
    body = _upload(sample).json()
    assert body["status"] == "correct"
    assert body["local_alignment"]["displacement_px"] <= 2.0


@pytest.mark.parametrize(
    "component_size",
    [(30, 20), (70, 40), (20, 14)],
)
def test_component_size_outside_envelope(component_size):
    sample = make_polarity_sample(20, component_size=component_size)
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "COMPONENT_SIZE_OUT_OF_RANGE"


def test_solid_board_is_undeterminable():
    flat = np.full((512, 512), 120, np.uint8)
    from tests.synth.polarity_generator import PolaritySample
    sample = PolaritySample(
        name="flat", golden=flat, inspected=flat,
        component_box=[200, 200, 48, 32], mark_box=[202, 202, 8, 28],
        true_warp=np.eye(2, 3), gain=1.0, bias=0.0,
        expect_status="undeterminable", expect_reason="SOLID_COLOR",
        meta={},
    )
    body = _upload(sample).json()
    assert body["status"] == "undeterminable"
    assert body["reason_code"] == "SOLID_COLOR"
    assert body["correction"] is None


def test_resized_boxes_are_input_errors():
    sample = make_polarity_sample(0)
    small = np.zeros((256, 256), np.uint8)
    resp = _upload(sample)
    assert resp.status_code == 200  # sanity: valid baseline

    # Mark box outside the component box.
    resp = _upload(sample, mark=[100, 100, 8, 8])
    assert resp.status_code == 400

    # Component box outside the frame.
    resp = _upload(sample, component=[490, 200, 48, 32])
    assert resp.status_code == 400

    # Wrong image size.
    files = {
        "golden": ("g.png", io.BytesIO(encode_png(small)), "image/png"),
        "inspected": ("i.png", io.BytesIO(encode_png(small)), "image/png"),
    }
    resp = client.post(
        "/api/v1/polarity", files=files,
        data={"component_box": "10,10,48,32", "mark_box": "12,12,8,28"},
    )
    assert resp.status_code == 400


def test_parse_box_accepts_bracket_form_and_validation():
    assert parse_box("10,20,30,40", "b").as_list() == [10, 20, 30, 40]
    assert parse_box("[10, 20, 30, 40]", "b").as_list() == [10, 20, 30, 40]
    with pytest.raises(PolarityInputError):
        parse_box("10,20,30", "b")
    with pytest.raises(PolarityInputError):
        parse_box("10,20,0,40", "b")
    with pytest.raises(PolarityInputError):
        parse_box("500,20,30,40", "b")


def test_light_mark_polarity_is_detected():
    """A mark brighter than the body works (contrast sign is learned)."""
    from tests.synth.polarity_generator import (
        _apply_warp,
        _board_background,
        _draw_capacitor,
        _random_warp,
        _rng,
        _rotate_part_in_place,
    )

    for reversed_mount, expected in ((False, "correct"), (True, "reversed")):
        rng = _rng(7000)
        board = _board_background(rng)
        box = [260, 180, 48, 32]
        golden, mark = _draw_capacitor(board, box, 240, 160)
        inspected_board = (_rotate_part_in_place(golden, box)
                           if reversed_mount else golden.copy())
        warp, _, _, _ = _random_warp(rng)
        warped = _apply_warp(inspected_board, warp)
        inspected = np.clip(
            1.05 * warped.astype(np.float32) - 4, 0, 255
        ).astype(np.uint8)
        result = analyze_polarity(
            golden, inspected,
            parse_box(_box(box), "component_box"),
            parse_box(_box(mark), "mark_box"),
            include_debug=False,
        )
        assert result.status == expected, result.reason_code


def test_mark_box_must_describe_an_end_mark():
    sample = make_polarity_sample(0)
    cb = sample.component_box
    mb = sample.mark_box
    # A box near the middle of the body does not reflect into the component.
    middle = [cb[0] + cb[2] // 2 - mb[2] // 2, cb[1] + mb[3] // 2 - mb[1] + mb[1],
              mb[2], mb[3]]
    with pytest.raises(PolarityInputError):
        analyze_polarity(
            sample.golden, sample.inspected,
            parse_box(_box(cb), "component_box"),
            parse_box(_box(middle), "mark_box"),
            include_debug=False,
        )


def test_polarity_is_deterministic():
    sample = make_polarity_sample(3, reversed_mount=True)
    first = _analyze(sample)
    second = _analyze(sample)
    assert first.status == second.status == "reversed"
    assert first.evidence == second.evidence
    assert first.local_alignment == second.local_alignment


def test_debug_panel_is_a_png():
    sample = make_polarity_sample(0)
    result = analyze_polarity(
        sample.golden,
        sample.inspected,
        parse_box(_box(sample.component_box), "component_box"),
        parse_box(_box(sample.mark_box), "mark_box"),
        include_debug=True,
    )
    assert result.debug_png_base64 is not None
    img = decode_png_base64(result.debug_png_base64)
    assert img is not None and img.ndim in (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
