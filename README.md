# PCB Golden-Board Re-inspection API

A FastAPI service that re-inspects a circuit-board image against a golden
reference. It removes the two nuisance factors a re-inspection camera
introduces — **global affine misalignment** and **linear exposure drift** —
and then localizes the *real* defects. When the correction is not
trustworthy (solid-color input, periodic texture with multiple alignments,
poor correlation, insufficient overlap), the service returns
`undeterminable` instead of ever reporting a false "pass".

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

Then open **http://localhost:8000/docs** — Swagger UI lets you upload the
two PNG files (`golden`, `inspected`) through the `POST /api/v1/inspect`
form and view the JSON result, including base64 defect mask and a 2×2
debug panel.

- `GET /health` — liveness probe
- `POST /api/v1/inspect` — multipart upload of two 512×512 grayscale PNGs
- Inputs that are not 512×512 grayscale images return HTTP 400.
- A content-level inability to decide is HTTP 200 with
  `status: "undeterminable"` and a machine-readable `reason_code`.

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

### Detector independence guarantee

The detector is only ever called with two numpy arrays
(`inspect_images(golden, inspected)`). `tests/evaluate.py` additionally
performs a static audit that fails if any module under `app/inspection/`
references the generator, a manifest, ground truth, or performs file I/O.
The detector cannot read the generation parameters or ground-truth masks.

## Repository layout

```
app/
  main.py                  FastAPI app, upload validation, response schema
  inspection/
    io_utils.py            PNG decode/encode, 512x512 grayscale validation
    geometry.py            robust multi-scale ECC, warp helpers, trust bounds
    trust.py               solid-color / low-texture / periodicity gates
    exposure.py            Huber-IRLS gain/bias fit, residual defect detection
    detector.py            end-to-end orchestration (array-only entry point)
tests/
  synth/generator.py       fixed-seed synthetic board + defect generator
  synth/generate_samples.py  CLI to persist the suite
  evaluate.py              independent scorer + detector independence audit
  test_api.py              pytest API and pipeline tests
Dockerfile, docker-compose.yml, requirements.txt
```
