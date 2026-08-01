#!/usr/bin/env python3
"""Shared utilities for CAD sketch parsing: prompt, JSON parsing, metrics, data split.

The evaluation metrics here are the heart of the project. A predicted design and a
ground-truth design are both lists of curves; we score them by matching curves with
the Hungarian algorithm under an exact Earth Mover's Distance (EMD) between the point
sets sampled along each curve. EMD is a true metric (Chamfer is not), so design-to-
design distance is well behaved. Chamfer is kept only as a cheaper reference number.
"""
import json
import math
import re

import numpy as np

# ---- fixed prompt -----------------------------------------------------------

PROMPT = (
    "You are a CAD sketch parser. The image is an engineering sketch made of "
    "geometric primitives. Output ALL primitives as JSON.\n"
    "Each primitive is one of:\n"
    '  - "line":   control_points = [[x0,y0],[x1,y1]] (two endpoints)\n'
    '  - "circle": control_points = [[x0,y0],[x1,y1]] (two ends of a diameter)\n'
    '  - "arc":    control_points = [[xs,ys],[xm,ym],[xe,ye]] (start, mid, end)\n'
    "Coordinates are floats roughly in the range [-20, 20].\n"
    'Respond with ONLY a JSON object of the form '
    '{"curves": [ {"type": ..., "control_points": ...}, ... ]} and nothing else.'
)

VALID_TYPES = ("line", "circle", "arc")
N_POINTS_PER_TYPE = {"line": 2, "circle": 2, "arc": 3}


# ---- parsing model output ---------------------------------------------------

def parse_curves(text):
    """Extract a list of curves from raw model text. Returns [] on unrecoverable output.

    Tolerates markdown fences and leading/trailing prose by grabbing the first
    balanced {...} block, then validates each curve's type and control-point arity.
    """
    if not text:
        return []
    # strip common ```json fences
    text = re.sub(r"```(?:json)?", "", text)
    obj = _first_json_object(text)
    if obj is None:
        return []
    curves = obj.get("curves", []) if isinstance(obj, dict) else []
    clean = []
    for c in curves:
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        cp = c.get("control_points")
        if t not in VALID_TYPES or not isinstance(cp, list):
            continue
        if len(cp) != N_POINTS_PER_TYPE[t]:
            continue
        try:
            pts = [[float(p[0]), float(p[1])] for p in cp]
        except (TypeError, ValueError, IndexError):
            continue
        clean.append({"type": t, "control_points": pts})
    return clean


def _first_json_object(text):
    """Return the first balanced {...} parsed as JSON, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ---- sampling points along a curve ------------------------------------------

def _circle_from_3pts(p0, p1, p2):
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return (ux, uy), r


def sample_curve(curve, n=32):
    """Return an (n, 2) array of points sampled uniformly along the curve."""
    t = curve["type"]
    cp = np.asarray(curve["control_points"], dtype=float)
    if t == "line":
        s, e = cp[0], cp[1]
        ts = np.linspace(0, 1, n)[:, None]
        return s[None, :] * (1 - ts) + e[None, :] * ts
    if t == "circle":
        a, b = cp[0], cp[1]
        center = (a + b) / 2
        r = np.linalg.norm(a - b) / 2
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)], axis=1)
    if t == "arc":
        fit = _circle_from_3pts(cp[0], cp[1], cp[2])
        if fit is None:  # collinear -> treat as a line start->end
            s, e = cp[0], cp[2]
            ts = np.linspace(0, 1, n)[:, None]
            return s[None, :] * (1 - ts) + e[None, :] * ts
        (ux, uy), r = fit
        a0 = math.atan2(cp[0][1] - uy, cp[0][0] - ux)
        a1 = math.atan2(cp[1][1] - uy, cp[1][0] - ux)
        a2 = math.atan2(cp[2][1] - uy, cp[2][0] - ux)
        # unwrap so the mid angle lies between start and end
        def unwrap(a, ref):
            while a < ref:
                a += 2 * np.pi
            return a
        a1u = unwrap(a1, a0)
        a2u = unwrap(a2, a0)
        if a1u > a2u:  # mid not between -> go the other way
            a2u = unwrap(a2, a0) - 2 * np.pi if a2 < a0 else a2
            ang = np.linspace(a0, a2u, n)
        else:
            ang = np.linspace(a0, a2u, n)
        return np.stack([ux + r * np.cos(ang), uy + r * np.sin(ang)], axis=1)
    raise ValueError(f"unknown curve type {t}")


# ---- distances --------------------------------------------------------------

def _emd(pa, pb):
    """Exact EMD between two equal-weight point sets (squared-euclidean ground cost)."""
    import ot  # POT
    na, nb = len(pa), len(pb)
    a = np.full(na, 1.0 / na)
    b = np.full(nb, 1.0 / nb)
    M = ot.dist(pa, pb, metric="sqeuclidean")
    return float(ot.emd2(a, b, M))


def _chamfer(pa, pb):
    """Symmetric Chamfer distance between two point sets (mean nearest-neighbour)."""
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean()) / 2


def curve_distance(c1, c2, n=32, metric="emd"):
    pa = sample_curve(c1, n)
    pb = sample_curve(c2, n)
    return _emd(pa, pb) if metric == "emd" else _chamfer(pa, pb)


# ---- design-to-design scoring -----------------------------------------------

def match_designs(pred, gt, n=32, metric="emd"):
    """Hungarian match predicted curves to GT curves. Returns a dict of scores.

    matched_distance : mean curve distance over matched pairs (lower is better)
    type_accuracy    : fraction of matched pairs with identical primitive type
    count_error      : |#pred - #gt|
    count_correct    : 1.0 if counts equal else 0.0
    """
    from scipy.optimize import linear_sum_assignment
    res = {"count_error": abs(len(pred) - len(gt)),
           "count_correct": float(len(pred) == len(gt))}
    if len(pred) == 0 or len(gt) == 0:
        res.update(matched_distance=float("inf"), type_accuracy=0.0, n_matched=0)
        return res
    cost = np.zeros((len(pred), len(gt)))
    for i, cp in enumerate(pred):
        for j, cg in enumerate(gt):
            cost[i, j] = curve_distance(cp, cg, n, metric)
    ri, ci = linear_sum_assignment(cost)
    dists = cost[ri, ci]
    type_hits = sum(pred[i]["type"] == gt[j]["type"] for i, j in zip(ri, ci))
    res.update(matched_distance=float(dists.mean()),
               type_accuracy=type_hits / len(ri),
               n_matched=len(ri))
    return res


# ---- deterministic train/eval split -----------------------------------------

def load_index(index_path):
    rows = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_index(rows, n_eval=500):
    """Deterministic split: sort by cad_id, take the last n_eval as the eval set."""
    rows = sorted(rows, key=lambda r: r["cad_id"])
    return rows[:-n_eval], rows[-n_eval:]
