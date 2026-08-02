#!/usr/bin/env python3
"""Symbolic refinement: snap VLM-predicted curve control points onto the sketch ink.

This is the "symbolic" half of the neuro-symbolic split. The VLM (perception) says *what*
primitives are present and roughly where; this module fixes *precisely where* by fitting
each predicted primitive to the image ink with least squares.

CRITICAL: refinement reads ONLY the rendered PNG (its ink pixels). It never sees ground
truth. No function here takes a gt/label argument, and a GT-free self-check (distance from
the refined curve to nearest ink must not exceed the original's) guarantees refinement can
only ever improve — or leave unchanged — the fit to the actual image.

Depends only on numpy / scipy / PIL (no cv2 / skimage).
"""
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

from common import sample_curve

# ---- coordinate maps (exact inverse of render_dataset.py) -------------------
# render: px = (x+22)/44*384 ; py = (22-y)/44*384   (384x384, no margins)
INV = 44.0 / 384.0  # data units per pixel


def px_to_data(cols, rows):
    """Pixel (col, row) arrays -> (N, 2) data coordinates (y-axis un-flipped)."""
    return np.stack([cols * INV - 22.0, 22.0 - rows * INV], axis=1)


def load_ink(image_path, thresh=128):
    """Return (N, 2) ink-pixel positions in DATA coordinates."""
    im = np.array(Image.open(image_path).convert("L"))
    rows, cols = np.where(im < thresh)  # np.where gives (row, col)
    return px_to_data(cols.astype(float), rows.astype(float))


# ---- refinement config ------------------------------------------------------

DEFAULTS = {
    "band": 1.2,        # data units (~10px): ink within this of the predicted curve
    "min_inliers": 12,  # too few -> leave curve unchanged
    "max_rms": 1.0,     # fit residual RMS (data units) above which we reject the fit
    "n_samples": 96,    # points sampled along a curve for banding / residual
    "r_min": 0.2,       # circle/arc sanity bounds
    "r_max": 30.0,
}


# ---- shared helpers ---------------------------------------------------------

def _mean_ink_residual(curve, tree, n_samples):
    """Mean distance (data units) from points sampled along `curve` to nearest ink."""
    pts = sample_curve(curve, n=n_samples)
    d, _ = tree.query(pts)
    return float(d.mean())


def band_inliers(curve, tree, ink, band, n_samples):
    """Ink pixels within `band` of the sampled predicted curve (seeded by the VLM)."""
    pts = sample_curve(curve, n=n_samples)
    idx = tree.query_ball_point(pts, r=band)
    flat = [i for sub in idx for i in sub]
    if not flat:
        return np.empty((0, 2))
    return ink[np.unique(np.asarray(flat, dtype=int))]


# ---- per-type fits ----------------------------------------------------------

def _fit_line(P):
    """Total-least-squares line: returns (centroid, unit_dir, unit_normal, perp_rms)."""
    c = P.mean(0)
    Q = P - c
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    d = vt[0]
    n = vt[1]
    perp_rms = float(np.sqrt(((Q @ n) ** 2).mean()))
    return c, d, n, perp_rms


def _refine_line(cp, P, cfg):
    c, d, _, perp_rms = _fit_line(P)
    if perp_rms > cfg["max_rms"]:
        return None
    p0, p1 = np.asarray(cp[0], float), np.asarray(cp[1], float)
    # project predicted endpoints onto the fitted line (fixes perpendicular offset,
    # preserves the VLM's segment extent)
    q0 = c + np.dot(p0 - c, d) * d
    q1 = c + np.dot(p1 - c, d) * d
    return [q0.tolist(), q1.tolist()]


def _fit_circle_kasa(P):
    """Algebraic circle fit (Kasa): x^2+y^2 = A x + B y + C -> center, radius."""
    x, y = P[:, 0], P[:, 1]
    Amat = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    A, B, C = np.linalg.lstsq(Amat, b, rcond=None)[0]
    cx, cy = A / 2.0, B / 2.0
    r = np.sqrt(max(C + cx * cx + cy * cy, 1e-9))
    return np.array([cx, cy, r])


def _fit_circle(P):
    """Geometric circle fit refined from Kasa init: returns (cx, cy, r, rms)."""
    init = _fit_circle_kasa(P)

    def resid(p):
        return np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) - p[2]

    try:
        sol = least_squares(resid, init, method="lm")
        cx, cy, r = sol.x
        rms = float(np.sqrt((resid(sol.x) ** 2).mean()))
    except Exception:
        cx, cy, r = init
        rms = float(np.sqrt((resid(init) ** 2).mean()))
    return cx, cy, abs(r), rms


def _refine_circle(cp, P, cfg):
    cx, cy, r, rms = _fit_circle(P)
    if rms > cfg["max_rms"] or not (cfg["r_min"] <= r <= cfg["r_max"]):
        return None
    center = np.array([cx, cy])
    a, b = np.asarray(cp[0], float), np.asarray(cp[1], float)
    u = b - a
    nu = np.linalg.norm(u)
    u = u / nu if nu > 1e-9 else np.array([1.0, 0.0])  # keep predicted diameter direction
    return [(center - r * u).tolist(), (center + r * u).tolist()]


def _refine_arc(cp, P, cfg):
    cx, cy, r, rms = _fit_circle(P)
    if rms > cfg["max_rms"] or not (cfg["r_min"] <= r <= cfg["r_max"]):
        return None
    center = np.array([cx, cy])

    def project(p):
        v = np.asarray(p, float) - center
        nv = np.linalg.norm(v)
        return (center + r * v / nv if nv > 1e-9 else center + np.array([r, 0.0]))

    # project start/mid/end radially onto the fitted circle -> preserves angular
    # ordering so sample_curve's sweep-direction logic stays valid
    return [project(cp[0]).tolist(), project(cp[1]).tolist(), project(cp[2]).tolist()]


_REFINERS = {"line": _refine_line, "circle": _refine_circle, "arc": _refine_arc}


# ---- public API -------------------------------------------------------------

def refine_curve(curve, tree, ink, cfg=DEFAULTS):
    """Return a refined copy of `curve` (same type, same #control_points).

    Falls back to the original whenever refinement can't confidently improve the fit:
    too few inliers, high residual, degenerate radius, or a refined curve that fits the
    ink *worse* than the original (GT-free self-check).
    """
    t = curve["type"]
    refiner = _REFINERS.get(t)
    if refiner is None:
        return curve
    P = band_inliers(curve, tree, ink, cfg["band"], cfg["n_samples"])
    if len(P) < cfg["min_inliers"]:
        return curve
    new_cp = refiner(curve["control_points"], P, cfg)
    if new_cp is None:
        return curve
    cand = {"type": t, "control_points": new_cp}
    # self-check: never accept a refinement that fits the image worse
    if _mean_ink_residual(cand, tree, cfg["n_samples"]) <= \
       _mean_ink_residual(curve, tree, cfg["n_samples"]) + 1e-6:
        return cand
    return curve


def refine_design(pred_curves, image_path, cfg=DEFAULTS):
    """Refine every predicted curve of one design against its rendered PNG."""
    ink = load_ink(image_path)
    if len(ink) == 0:
        return list(pred_curves)
    tree = cKDTree(ink)
    return [refine_curve(c, tree, ink, cfg) for c in pred_curves]
