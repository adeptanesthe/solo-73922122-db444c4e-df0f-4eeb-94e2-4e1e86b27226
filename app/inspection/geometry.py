"""Global affine registration via robust multi-scale ECC.

Coordinate convention (verified empirically against OpenCV 4.10 and the
official Python ECC tutorial)
-----------------------------------------------------------------------
* ``G`` = golden (template) image, frame ``g``
* ``I`` = inspected (input) image, frame ``i``
* ``findTransformECC(templateImage=G, inputImage=I)`` returns a warp ``W``
  in the *golden -> inspected* (forward) direction.  A controlled
  pure-translation experiment confirms::

        warpAffine(I, W, WARP_INVERSE_MAP) == G      (residual ~ 0)
        warpAffine(G, W)                    == I      (content moved by W)

  i.e. ``W`` maps golden coordinates to inspected coordinates.

Defects present in ``I`` (or missing features of ``G``) are treated as
outliers: ECC is re-run with an inlier mask rebuilt from residuals so the
fitting cannot absorb a real defect into the warp.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_SIZE = 512
PYRAMID_LEVELS = 2
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

# Plausibility bounds. The synthetic envelope draws |tx|,|ty|<=8 px plus a
# rotation (<=3 deg about the image center) whose induced translation can
# add ~11 px per axis, so a component can reach ~19 px. These bounds keep a
# safety margin while still rejecting wild/ambiguous warps.
MAX_TRANSLATION_PX = 20.0
MAX_ROTATION_DEG = 4.0
MAX_SCALE_LOG = 0.05          # ~5% scale deviation
MAX_SHEAR_DEG = 2.0

# A 16 px margin keeps black fill borders out of the alignment cost.
BORDER_MARGIN = 16


class RegistrationError(RuntimeError):
    """Raised when ECC cannot converge."""


@dataclass
class RegistrationResult:
    warp: np.ndarray                 # 2x3, golden -> inspected (forward)
    cc: float                        # final ECC correlation coefficient
    levels: list[float]              # CC per pyramid level
    iterations: list[int]            # ECC iterations per level


def _ecc_one_level(
    template: np.ndarray,
    moved: np.ndarray,
    warp: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, float, int]:
    """Run one ECC optimization. Returns (forward warp, correlation, iters)."""
    try:
        cc, warp_out = cv2.findTransformECC(
            template,
            moved,
            warp.astype(np.float32),
            cv2.MOTION_AFFINE,
            ECC_CRITERIA,
            mask,
            5,  # gaussFilterSize
        )
    except cv2.error as exc:  # pragma: no cover - depends on OpenCV build
        raise RegistrationError(f"ECC failed: {exc}") from exc
    if not np.isfinite(warp_out).all():
        raise RegistrationError("ECC produced a non-finite warp")
    return warp_out, float(cc), ECC_CRITERIA[1]


def render_inspected_in_golden(
    inspected: np.ndarray, warp: np.ndarray, nearest: bool = False
) -> np.ndarray:
    """Render an inspected-frame image into golden-frame geometry.

    The ECC warp is golden->inspected, so the inverse flag is required to
    back-sample inspected pixels into golden coordinates.
    """
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(
        inspected,
        warp.astype(np.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=interp | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def render_golden_in_inspected(
    golden: np.ndarray, warp: np.ndarray, nearest: bool = False
) -> np.ndarray:
    """Render a golden-frame image/mask into inspected-frame geometry."""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(
        golden,
        warp.astype(np.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _pyramids(g0: np.ndarray, i0: np.ndarray, mask0: np.ndarray):
    g_pyr, i_pyr, m_pyr = [g0], [i0], [mask0]
    for _ in range(1, PYRAMID_LEVELS):
        g_pyr.append(cv2.pyrDown(g_pyr[-1]))
        i_pyr.append(cv2.pyrDown(i_pyr[-1]))
        m_pyr.append(cv2.resize(m_pyr[-1], g_pyr[-1].shape[::-1],
                                interpolation=cv2.INTER_NEAREST))
    return g_pyr, i_pyr, m_pyr


def _normalize(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    sd = max(float(f.std()), 1e-6)
    return (f - float(f.mean())) / sd


def estimate_affine(
    golden: np.ndarray,
    inspected: np.ndarray,
    inlier_mask_inspected: np.ndarray | None = None,
) -> RegistrationResult:
    """Estimate the golden->inspected (forward) affine warp.

    ``inlier_mask_inspected`` (uint8, 255 = inlier) lives in inspected-frame
    geometry, which is the frame of ECC's ``inputImage`` mask.
    """
    assert golden.shape == inspected.shape == (IMAGE_SIZE, IMAGE_SIZE)
    size = IMAGE_SIZE

    if inlier_mask_inspected is None:
        inlier_mask_inspected = np.zeros((size, size), np.uint8)
        inlier_mask_inspected[
            BORDER_MARGIN:size - BORDER_MARGIN,
            BORDER_MARGIN:size - BORDER_MARGIN,
        ] = 255

    g0, i0 = _normalize(golden), _normalize(inspected)
    g_pyr, i_pyr, m_pyr = _pyramids(g0, i0, inlier_mask_inspected)

    warp = np.eye(2, 3, dtype=np.float32)
    ccs: list[float] = []
    iters: list[int] = []
    for level in range(PYRAMID_LEVELS - 1, -1, -1):
        if level != PYRAMID_LEVELS - 1:
            # Lift the coarser-level warp to this resolution.
            warp = warp.copy()
            warp[:, 2] *= 2.0
        warp, cc, n_iter = _ecc_one_level(
            g_pyr[level], i_pyr[level], warp, m_pyr[level]
        )
        ccs.append(cc)
        iters.append(n_iter)

        # Rebuild the inspected-frame inlier mask from full-resolution
        # residuals so defect pixels are excluded from subsequent passes.
        if level > 0:
            w_full = warp.copy()
            w_full[:, 2] *= float(size) / g_pyr[level].shape[0]
            i_in_g = render_inspected_in_golden(i0, w_full)
            coverage_g = render_inspected_in_golden(
                np.full((size, size), 255, np.uint8), w_full, nearest=True
            )
            diff = np.abs(g0 - i_in_g)
            valid = coverage_g > 0
            thr = max(3.0, float(np.median(diff[valid])) * 2.5)
            good_g = ((diff < thr) & valid).astype(np.uint8) * 255
            # Forward-project the golden-frame good mask into inspected frame.
            rebuilt_i = render_golden_in_inspected(good_g, w_full, nearest=True)
            rebuilt_i = cv2.bitwise_and(rebuilt_i, inlier_mask_inspected)
            g_pyr, i_pyr, m_pyr = _pyramids(g0, i0, rebuilt_i)

    return RegistrationResult(
        warp=warp.astype(np.float64),
        cc=ccs[-1],
        levels=ccs,
        iterations=iters,
    )


def robust_refit_with_mask(
    golden: np.ndarray,
    inspected: np.ndarray,
    inlier_mask_inspected: np.ndarray,
    initial_warp: np.ndarray,
) -> RegistrationResult:
    """Re-run ECC restricted to inlier pixels (defects cannot move the fit)."""
    g0 = _normalize(golden)
    i0 = _normalize(inspected)
    g_pyr, i_pyr, m_pyr = _pyramids(g0, i0, inlier_mask_inspected)

    warp = initial_warp.astype(np.float32).copy()
    ccs, iters = [], []
    for level in range(PYRAMID_LEVELS - 1, -1, -1):
        if level != PYRAMID_LEVELS - 1:
            warp = warp.copy()
            warp[:, 2] *= 2.0
        warp, cc, n_iter = _ecc_one_level(
            g_pyr[level], i_pyr[level], warp, m_pyr[level]
        )
        ccs.append(cc)
        iters.append(n_iter)
    return RegistrationResult(
        warp=warp.astype(np.float64), cc=ccs[-1], levels=ccs, iterations=iters
    )


def overlap_masks(
    warp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage masks (255 = overlap) in golden and inspected frames."""
    ones = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, np.uint8)
    coverage_g = render_inspected_in_golden(ones, warp, nearest=True)
    coverage_i = render_golden_in_inspected(ones, warp, nearest=True)
    return coverage_g, coverage_i


def invert_warp(warp: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine."""
    full = np.eye(3, dtype=np.float64)
    full[:2, :] = warp.astype(np.float64)
    return np.linalg.inv(full)[:2, :]


def decompose_affine(warp: np.ndarray) -> dict[str, float]:
    """Decompose a 2x3 affine via polar decomposition A = R @ P.

    ``R`` is the orthogonal rotation/reflection and ``P`` the symmetric
    positive-definite scale/shear part. A pure rotation therefore reports
    zero shear and unit principal scales.
    """
    a_mat = warp[:2, :2].astype(np.float64)
    tx, ty = float(warp[0, 2]), float(warp[1, 2])

    u, s, vt = np.linalg.svd(a_mat)
    # Reflection-safe polar rotation.
    sign = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    d = np.diag([1.0, float(sign)])
    r = u @ d @ vt
    rotation = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))

    # Shear: how far the mapped basis axes depart from 90 degrees.
    c0, c1 = a_mat[:, 0], a_mat[:, 1]
    cos_a = float(np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1)))
    cos_a = np.clip(cos_a, -1.0, 1.0)
    shear = float(np.degrees(np.arccos(cos_a) - np.pi / 2.0))

    return {
        "tx_px": tx,
        "ty_px": ty,
        "rotation_deg": rotation,
        "scale_x": float(s[0]),
        "scale_y": float(s[1]),
        "shear_deg": shear,
    }


def warp_plausible(warp: np.ndarray) -> tuple[bool, str | None]:
    """Reject wild warps (registration on ambiguous / degenerate content)."""
    p = decompose_affine(warp)
    if abs(p["tx_px"]) > MAX_TRANSLATION_PX or abs(p["ty_px"]) > MAX_TRANSLATION_PX:
        return False, "translation exceeds plausibility bound"
    if abs(p["rotation_deg"]) > MAX_ROTATION_DEG:
        return False, "rotation exceeds plausibility bound"
    if abs(np.log(max(p["scale_x"], 1e-6))) > MAX_SCALE_LOG or \
       abs(np.log(max(p["scale_y"], 1e-6))) > MAX_SCALE_LOG:
        return False, "scale exceeds plausibility bound"
    if abs(p["shear_deg"]) > MAX_SHEAR_DEG:
        return False, "shear exceeds plausibility bound"
    return True, None


def corner_mapping_error(
    warp_est: np.ndarray, warp_true: np.ndarray
) -> dict[str, float]:
    """Mapping error (px) of the four golden-image corners.

    Both warps use the golden->inspected (forward) convention.
    """
    corners = np.array(
        [[0, 0], [IMAGE_SIZE - 1, 0],
         [IMAGE_SIZE - 1, IMAGE_SIZE - 1], [0, IMAGE_SIZE - 1]],
        dtype=np.float64,
    )
    homog = np.concatenate([corners, np.ones((4, 1))], axis=1)
    mapped_est = homog @ warp_est.T
    mapped_true = homog @ warp_true.T
    err = np.linalg.norm(mapped_est - mapped_true, axis=1)
    return {
        "max_corner_error_px": float(err.max()),
        "mean_corner_error_px": float(err.mean()),
        "per_corner_error_px": [float(x) for x in err],
    }
