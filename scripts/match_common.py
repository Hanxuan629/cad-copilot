#!/usr/bin/env python3
"""Shared helpers for the line-matching task (task 2).

Given a set of already-drawn CAD curves and a set of TARGET curves (both parametric,
both in the same [-20,20] data frame from data/mrcad_raw_json), the task is: for each
CAD curve, which target curve does it correspond to? One-to-many is allowed (several CAD
curves may map to the same target).

This module provides:
  - iter_trials()      : stream trials out of the raw_json files
  - geometric_match()  : per-CAD-curve nearest target by chamfer (the GT / geometric baseline)
  - text_prompt()      : render a trial as a text matching prompt
  - parse_mapping()    : parse a model's "C0->T3" JSON mapping back to indices

Reuses scripts/common.py for the geometry (curve_distance, REJECT_TAU) — no reimplementation.
"""
import json
import re
from pathlib import Path

from common import curve_distance, REJECT_TAU

# verified raw_json files, in priority order (largest/cleanest first)
RAW_JSON_FILES = [
    "eval_verified_complete.json",
    "coverage_verified.json",
    "eval_verified_incomplete.json",
]


def iter_trials(data_dir, files=None):
    """Yield trials from the raw_json files. Each trial is the raw dict."""
    data_dir = Path(data_dir)
    for fn in (files or RAW_JSON_FILES):
        p = data_dir / fn
        if not p.exists():
            continue
        for trial in json.loads(p.read_text()):
            yield trial


def final_round_cad(trial):
    """CAD curves from the trial's last round with a non-empty execution design."""
    curves = []
    for r in trial.get("rounds", []):
        design = (r.get("execution") or {}).get("design") or {}
        if design.get("curves"):
            curves = design["curves"]  # last non-empty wins -> most complete
    return curves


def geometric_match(cad_curves, target_curves, tau=REJECT_TAU, metric="chamfer"):
    """For each CAD curve, index of the nearest same-viable target curve, or -1.

    This is both the GT-label constructor and the no-VLM geometric baseline. Nearest by
    curve_distance; rejected (mapped to -1) if the best distance exceeds tau. One-to-many
    is automatic: independent nearest per CAD curve, so several may pick the same target.
    Returns (match_indices, best_distances).
    """
    match, dists = [], []
    for c in cad_curves:
        best_j, best_d = -1, float("inf")
        for j, t in enumerate(target_curves):
            d = curve_distance(c, t, metric=metric)
            if d < best_d:
                best_d, best_j = d, j
        if best_d > tau:
            best_j = -1
        match.append(best_j)
        dists.append(best_d)
    return match, dists


# ---- prompt construction ----------------------------------------------------

def _fmt_curve(c):
    pts = ", ".join(f"[{x:.1f},{y:.1f}]" for x, y in c["control_points"])
    return f"{c['type']}({pts})"


def text_prompt(cad_curves, target_curves):
    """A text matching prompt: numbered CAD lines + target lines -> JSON mapping."""
    lines = ["You match drawn CAD curves to the target curves they correspond to.",
             "Coordinates are in a shared frame roughly [-20, 20].", "",
             "TARGET curves:"]
    for j, t in enumerate(target_curves):
        lines.append(f"  T{j}: {_fmt_curve(t)}")
    lines.append("")
    lines.append("DRAWN CAD curves:")
    for i, c in enumerate(cad_curves):
        lines.append(f"  C{i}: {_fmt_curve(c)}")
    lines += ["",
              "For EACH drawn curve C_i, output the single target T_j it best corresponds "
              "to (multiple C may map to the same T; use null if none).",
              'Respond with ONLY a JSON object like {"C0":"T3","C1":"T3","C2":null}.']
    return "\n".join(lines)


def parse_mapping(text, n_cad):
    """Parse a {"C0":"T3",...} mapping into a list of target indices (len n_cad, -1=none)."""
    out = [-1] * n_cad
    if not text:
        return out
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    obj = None
    while start != -1 and obj is None:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        ci = re.search(r"\d+", str(k))
        if not ci:
            continue
        i = int(ci.group())
        if not (0 <= i < n_cad):
            continue
        tj = re.search(r"\d+", str(v)) if v is not None else None
        out[i] = int(tj.group()) if tj else -1
    return out


def match_accuracy(pred_match, gt_match):
    """Top-1 accuracy: fraction of CAD curves whose predicted target == GT target."""
    if not gt_match:
        return float("nan")
    hits = sum(1 for p, g in zip(pred_match, gt_match) if p == g)
    return hits / len(gt_match)
