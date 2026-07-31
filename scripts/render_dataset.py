#!/usr/bin/env python3
"""Render CAD sketch designs (data_pool_5_per_signature.jsonl) into (PNG, JSON) pairs.

Each input line is a design with a list of curves:
  line   : control_points = [p0, p1]           -> straight segment
  circle : control_points = [a, b]              -> see --circle-mode
  arc    : control_points = [start, mid, end]   -> 3-point circular arc

Output per design:
  dataset/images/<cad_id>.png   the rendered sketch
  dataset/labels/<cad_id>.json  {"cad_id", "curves": [...] }  (== ground truth)
and a single dataset/index.jsonl mapping image -> label for training.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc as MplArc


# ---- geometry helpers -------------------------------------------------------

def circle_from_3pts(p0, p1, p2):
    """Center and radius of the circle through three points (None if collinear)."""
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


def draw_arc(ax, cp):
    """3-point arc: fit circle through the points, draw the sweep start->end via mid."""
    start, mid, end = cp
    fit = circle_from_3pts(start, mid, end)
    if fit is None:  # collinear -> degenerate to a line
        ax.plot([start[0], end[0]], [start[1], end[1]], color="black", lw=1.6)
        return
    (cx, cy), r = fit
    a0 = math.degrees(math.atan2(start[1] - cy, start[0] - cx))
    a1 = math.degrees(math.atan2(mid[1] - cy, mid[0] - cx))
    a2 = math.degrees(math.atan2(end[1] - cy, end[0] - cx))

    def norm(a):  # wrap into [0, 360)
        return a % 360

    a0n, a1n, a2n = norm(a0), norm(a1), norm(a2)
    # choose sweep direction so that the mid angle lies between start and end
    ccw = (norm(a1n - a0n) <= norm(a2n - a0n))
    theta1, theta2 = (a0, a2) if ccw else (a2, a0)
    patch = MplArc((cx, cy), 2 * r, 2 * r, angle=0, theta1=theta1, theta2=theta2,
                   color="black", lw=1.6)
    ax.add_patch(patch)


def draw_circle(ax, cp, mode):
    a, b = cp
    if mode == "diameter":
        center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        r = math.hypot(a[0] - b[0], a[1] - b[1]) / 2
    else:  # center-radius: a is center, b is a point on the circle
        center = a
        r = math.hypot(a[0] - b[0], a[1] - b[1])
    ax.add_patch(plt.Circle(center, r, fill=False, color="black", lw=1.6))


def render_design(curves, out_png, circle_mode, size_px=384):
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    for c in curves:
        cp = c["control_points"]
        t = c["type"]
        if t == "line":
            ax.plot([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], color="black", lw=1.6)
        elif t == "circle":
            draw_circle(ax, cp, circle_mode)
        elif t == "arc":
            draw_arc(ax, cp)
    ax.set_xlim(-22, 22)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# ---- comparison sheet for verifying circle semantics ------------------------

def circle_mode_compare(records, out_path):
    """Render the first design with a circle under both interpretations, side by side."""
    rec = next((r for r in records if any(c["type"] == "circle" for c in r["design"]["curves"])), None)
    if rec is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, mode in zip(axes, ("diameter", "center-radius")):
        for c in rec["design"]["curves"]:
            cp = c["control_points"]
            if c["type"] == "line":
                ax.plot([cp[0][0], cp[1][0]], [cp[0][1], cp[1][1]], color="black", lw=1.6)
            elif c["type"] == "circle":
                draw_circle(ax, cp, mode)
            elif c["type"] == "arc":
                draw_arc(ax, cp)
        ax.set_xlim(-22, 22)
        ax.set_ylim(-22, 22)
        ax.set_aspect("equal")
        ax.set_title(f"circle = {mode}")
        ax.axis("off")
    fig.suptitle(f"{rec['cad_id']}  (which looks coherent?)")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--limit", type=int, default=None, help="render only the first N designs")
    ap.add_argument("--circle-mode", choices=["diameter", "center-radius"], default="diameter")
    ap.add_argument("--compare-circle", action="store_true",
                    help="also emit a side-by-side sheet of both circle interpretations")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            records.append(json.loads(line))

    if args.compare_circle:
        p = circle_mode_compare(records, out / "circle_mode_compare.png")
        print(f"circle-mode comparison -> {p}")

    index = []
    for rec in records:
        cad_id = rec["cad_id"]
        curves = rec["design"]["curves"]
        png = out / "images" / f"{cad_id}.png"
        lbl = out / "labels" / f"{cad_id}.json"
        render_design(curves, png, args.circle_mode)
        lbl.write_text(json.dumps({"cad_id": cad_id, "curves": curves}, ensure_ascii=False))
        index.append({"image": str(png), "label": str(lbl), "cad_id": cad_id,
                      "n_curves": len(curves)})

    (out / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index))
    print(f"rendered {len(index)} designs -> {out}/")


if __name__ == "__main__":
    main()
