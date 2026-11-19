# PCB Golden-Board Re-inspection API

A FastAPI service that re-inspects a circuit-board image against a golden
reference. It removes the two nuisance factors a re-inspection camera
introduces — **global affine misalignment** and **linear exposure drift** —
and then localizes the *real* defects. When the correction is not
trustworthy (solid-color input, periodic texture with multiple alignments,
poor correlation, insufficient overlap), the service returns
`undeterminable` instead of ever reporting a false "pass".

It additionally verifies **SMD tantalum-capacitor polarity** on the same
trusted correction (see [Polarity](#polarity-tantalum-capacitors)). A
missing mark at the golden-mark end is never by itself treated as a
reversed part: the opposite (180°-rotated) end must positively show the
mark.

## What it does

1. **Trust gates** — reject solid-color / near-flat images and detect
   periodic textures whose autocorrelation admits an alternative alignment
   (the registration would be non-unique).
2. **Robust affine registration** — multi-scale ECC (`MOTION_AFFINE`) with
   residual-mask refits, so defect pixels are treated as outliers and
   cannot be absorbed into the warp.
3. **Plausibility / correlation / overlap gates** — wild warps, weak
   correlation and frames with too little overlap are refused.
4. **Robust linear exposure compensation** — Huber-IRLS fit of
   `I = gain * G + bias` on Gaussian-smoothed overlap pixels (smoothing
   removes the sub-pixel-registration slope bias); the inspected image is
   brought back to the golden scale by the inverse relation
   `I_corrected = (I - bias) / gain`.
5. **Defect detection** — gradient-adaptive residual thresholding plus
   same-sign coherence and area filters, evaluated only inside the eroded
   overlap. The mask and boxes are returned in **inspected-image
   coordinates**. Non-overlap pixels are never judged.

## Coordinate convention

The reported `warp_matrix` is a 2×3 affine mapping
**golden coordinates → inspected coordinates**
(`[x_i, y_i]^T = W [x_g, y_g, 1]^T`). This matches OpenCV 4.10's
`findTransformECC` output convention (verified empirically:
`warpAffine(inspected, W, WARP_INVERSE_MAP)` reconstructs the golden image).

## Running with Docker Compose

```bash
docker compose up --build
```

Then open **http://localhost:8000/docs** — Swagger UI exposes an upload
form for each endpoint:

- `GET /health` — liveness probe
- `POST /api/v1/inspect` — multipart upload of two 512×512 grayscale PNGs;
  the JSON result includes the base64 defect mask and a 2×2 debug panel.
- `POST /api/v1/polarity` — the same two PNGs plus `component_box` and
  `mark_box` (`"x,y,w,h"`, golden coordinates); the JSON result includes
  the verdict, the per-end evidence and a diagnostic panel.
- Inputs that are not 512×512 grayscale images, malformed boxes, boxes
  outside the frame, or a mark box that does not describe an end mark
  return HTTP 400.
- A content-level inability to decide is HTTP 200 with
  `status: "undeterminable"` and a machine-readable `reason_code`.

Bundled example PNGs (defect suite and polarity suite, with manifests)
are written under `samples/` by the generator CLIs and are mounted
read-only into the container.

### Response `status`

| value             | meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `pass`            | decidable, no defect found in the overlap region               |
| `defect`          | decidable, defect mask/boxes returned in inspected coordinates |
| `undeterminable`  | correction not trustworthy; **never** a false pass             |

`reason_code` for undeterminable: `SOLID_COLOR`, `LOW_TEXTURE`,
`PERIODIC_TEXTURE`, `ECC_FAILED`, `LOW_CORRELATION`, `IMPLAUSIBLE_WARP`,
`INSUFFICIENT_OVERLAP`, `LOW_INLIER_RATIO`, `EXPOSURE_FIT_FAILED`,
`IMPLAUSIBLE_EXPOSURE`.

## Polarity (tantalum capacitors)

`POST /api/v1/polarity` decides whether a polarised SMD tantalum capacitor
is mounted in the golden orientation (`correct` / 正装) or rotated 180°
(`reversed` / 反装), or refuses (`undeterminable` / 无法判定).

### Request

`multipart/form-data`:

| field           | type   | meaning                                            |
|-----------------|--------|----------------------------------------------------|
| `golden`        | file   | 512×512 grayscale golden PNG                       |
| `inspected`     | file   | 512×512 grayscale inspected PNG                    |
| `component_box` | string | `"x,y,w,h"` of the component, **golden** coords    |
| `mark_box`      | string | `"x,y,w,h"` of the polarity mark, **golden** coords |
| `include_debug` | bool   | include the diagnostic PNG (default `true`)        |

`mark_box` must lie strictly inside `component_box`.

### Applicability envelope (outside it: `undeterminable`, HTTP 200)

- component long side between **32 and 64 px** inclusive
  (`COMPONENT_SIZE_OUT_OF_RANGE` otherwise);
- golden mark vs body contrast **≥ 40 gray levels**
  (`MARK_CONTRAST_INSUFFICIENT`), and the mark box must be mostly
  mark-like (`MARK_REFERENCE_UNRELIABLE`);
- post-registration **local displacement ≤ 2 px** around the component
  (`LOCAL_DISPLACEMENT_TOO_LARGE`, or `LOCAL_ALIGNMENT_UNRELIABLE` when it
  cannot be measured);
- the whole component must lie inside the trusted overlap
  (`COMPONENT_NOT_IN_OVERLAP`).

### Decision rule

The contrast/evidence of the mark box and of the box reflected through the
component centre (the 180° opposite end) are measured in the corrected
inspected image. Evidence combines the signed contrast against the
component body with the mark-like fill fraction, and is zeroed unless the
local gray level matches the golden mark.

- `correct` — original end strong **and** opposite end clearly weak, with
  a sufficient margin;
- `reversed` — opposite end strong **and** original end clearly weak;
- `undeterminable` — otherwise:
  - `MARK_OCCLUDED` — neither end shows mark evidence (the mark is hidden);
    a missing mark alone is **never** reported as `reversed`;
  - `AMBIGUOUS_EVIDENCE` — both ends strong, or the evidence margin is too
    small (e.g. a mark-coloured occluder on the far end);
  - any shared correction failure (`SOLID_COLOR`, `PERIODIC_TEXTURE`,
    `LOW_CORRELATION`, …) is passed through unchanged.

### Response

```json
{
  "status": "reversed",
  "verdict_zh": "反装",
  "reason_code": null,
  "reason": null,
  "metrics": {"component_long_side_px": 48, "...": "..."},
  "component": {"box": [259, 80, 48, 32], "opposite_mark_box": [298, 80, 7, 28], "...": "..."},
  "local_alignment": {"displacement_px": 0.02, "method": "gradient_ring_search", "reliable": true},
  "evidence": {
    "original_end": {"box": [261, 82, 7, 28], "evidence": 0.0, "strong": false},
    "opposite_end": {"box": [298, 80, 7, 28], "evidence": 1.0, "strong": true},
    "evidence_margin": 1.0
  },
  "correction": {"warp_matrix": [[...]], "gain": 0.97, "...": "..."},
  "debug_png_base64": "..."
}
```

The correction block is identical to `/api/v1/inspect` (same gates, same
warp convention: golden → inspected).

## Local development (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Synthetic data and independent evaluation

Everything test-related lives under `tests/` and is import-isolated from
the detector.

```bash
# Regenerate the fixed-seed suite (20 defect + 4 clean + 4 reject cases)
python -m tests.synth.generate_samples samples

# Independent evaluation; writes samples + ground-truth manifest, then scores
python -m tests.evaluate samples

# API / functional tests
python -m pytest -q
```

### Synthetic envelope (fixed seed `20261117`)

- translation ≤ 8 px per axis, rotation ≤ 3°, gain 0.8–1.2, bias ±15
- added/removed blocks covering 1 %–10 % of the frame, gray difference ≥ 40
- extra reject cases: solid color (flat and lightly noisy) and periodic
  stripe/checker textures shifted by one full period (a real defect block
  is embedded but must not be reported because the alignment is ambiguous)

### Measured acceptance results

Running `python -m tests.evaluate samples` on the reference environment:

| criterion                              | limit   | measured              |
|----------------------------------------|---------|-----------------------|
| four-corner mapping error (mean, px)   | ≤ 0.5   | ~0.10                 |
| four-corner mapping error (worst, px)  | —       | ~0.21                 |
| strict pixel F1 (mean over 20 defects) | ≥ 0.90  | ~0.97                 |
| strict pixel F1 (worst case)           | —       | ~0.95                 |
| status correctness (28/28)             | all     | pass/defect/reject OK |
| solid & periodic reject cases          | reject  | undeterminable        |

### Polarity synthetic suite

```bash
# 14 fixed-seed polarity samples (correct / reversed / vertical / occluded /
# ambiguous / out-of-envelope contrast) + golden boxes in the manifest
python -m tests.synth.generate_polarity_samples samples

# Independent polarity evaluation (22 samples incl. size edges and <=2 px
# local shifts) with the same static independence audit
python -m tests.evaluate_polarity samples
```

Measured on the reference environment: 22/22 expected outcomes (7 correct,
6 reversed, 9 undeterminable; zero directional verdicts on any occlusion or
ambiguity sample).

### Detector independence guarantee

The detector is only ever called with two numpy arrays
(`inspect_images(golden, inspected)`; polarities with two arrays and two
boxes). `tests/evaluate.py` and `tests/evaluate_polarity.py` additionally
perform a static audit that fails if any module under `app/inspection/`
references the generator, a manifest, ground truth, or performs file I/O.
The detector cannot read the generation parameters or ground-truth masks.

## Repository layout

```
app/
  main.py                  FastAPI app, upload validation, both response schemas
  inspection/
    io_utils.py            PNG decode/encode, 512x512 grayscale validation
    geometry.py            robust multi-scale ECC, warp helpers, trust bounds
    trust.py               solid-color / low-texture / periodicity gates
    exposure.py            Huber-IRLS gain/bias fit, residual defect detection
    correction.py          shared trust gates + ECC + exposure (both endpoints)
    detector.py            re-inspection orchestration (array-only entry point)
    polarity.py            tantalum-capacitor polarity decision (array+boxes)
tests/
  synth/generator.py       fixed-seed synthetic board + defect generator
  synth/polarity_generator.py  fixed-seed capacitor polarity generator
  synth/generate_samples.py  CLI to persist the defect suite
  synth/generate_polarity_samples.py  CLI to persist the polarity suite
  evaluate.py              independent defect scorer + independence audit
  evaluate_polarity.py     independent polarity scorer + independence audit
  test_api.py              pytest API and re-inspection pipeline tests
  test_polarity.py         pytest API and polarity decision tests
Dockerfile, docker-compose.yml, requirements.txt
```
