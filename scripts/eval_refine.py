#!/usr/bin/env python3
"""Score symbolic refinement: chamfer (and perception metrics) before vs after.

Reads the predictions captured by eval_model.py --dump-pred, refines each design against
its PNG (refine.refine_design, image-only), and scores both the raw predictions and the
refined predictions against ground truth with common.match_designs.

Ground truth is used ONLY here, for scoring — never inside refinement.

CPU-only (seconds to minutes on 200 designs) -> safe to run on the Head Node.

Expected story: geometry improves (mean matched chamfer drops) while perception is flat
(edit_norm / type_accuracy essentially unchanged).
"""
import argparse
import json
from pathlib import Path

from common import match_designs
from refine import refine_design, DEFAULTS


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(scores):
    """Aggregate a list of match_designs dicts into headline numbers."""
    md = [s["matched_distance"] for s in scores if s["matched_distance"] != float("inf")]
    return {
        "edit_norm": _mean([s["edit_norm"] for s in scores]),
        "type_accuracy": _mean([s["type_accuracy"] for s in scores]),
        "n_accepted": sum(s["n_accepted"] for s in scores),
        "chamfer": _mean(md),
        "n_finite": len(md),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions jsonl from eval_model --dump-pred")
    ap.add_argument("--metric", choices=["emd", "chamfer"], default="chamfer")
    ap.add_argument("--band", type=float, default=DEFAULTS["band"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", help="per-image/per-curve detail")
    ap.add_argument("--out", default=None, help="per-image before/after jsonl")
    args = ap.parse_args()

    cfg = dict(DEFAULTS, band=args.band)
    rows = [json.loads(l) for l in open(args.pred) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[refine-eval] {len(rows)} designs  metric={args.metric}  band={args.band}", flush=True)

    before_scores, after_scores = [], []
    dump = open(args.out, "w") if args.out else None
    n_changed_curves = 0
    n_total_curves = 0
    for k, row in enumerate(rows):
        pred = row["pred_curves"]
        gt = row["gt_curves"]
        refined = refine_design(pred, row["image"], cfg)
        n_total_curves += len(pred)
        n_changed_curves += sum(
            1 for a, b in zip(pred, refined) if a["control_points"] != b["control_points"])
        b = match_designs(pred, gt, metric=args.metric)
        a = match_designs(refined, gt, metric=args.metric)
        before_scores.append(b)
        after_scores.append(a)
        if dump is not None:
            dump.write(json.dumps({"cad_id": row["cad_id"],
                                   "before": b, "after": a,
                                   "refined_curves": refined}) + "\n")
        if args.verbose:
            print(f"  [{k+1}/{len(rows)}] {row['cad_id']}  "
                  f"chamfer {b['matched_distance']:.3f}->{a['matched_distance']:.3f}  "
                  f"edit_norm {b['edit_norm']:.2f}->{a['edit_norm']:.2f}  "
                  f"n_acc {b['n_accepted']}->{a['n_accepted']}", flush=True)
    if dump is not None:
        dump.close()

    B, A = summarize(before_scores), summarize(after_scores)
    print(f"\n{'metric':<22}{'pred-only':>12}{'pred+refine':>14}")
    print("-" * 48)
    for key, lab in [("chamfer", f"matched {args.metric} (↓)"),
                     ("edit_norm", "edit_norm (↓)"),
                     ("type_accuracy", "type_acc (↑)"),
                     ("n_accepted", "n_accepted (↑)")]:
        print(f"{lab:<22}{B[key]:>12.4f}{A[key]:>14.4f}")
    print(f"\ncurves refined: {n_changed_curves}/{n_total_curves} "
          f"({100*n_changed_curves/max(n_total_curves,1):.1f}%)")
    print(f"chamfer averaged over finite rows: before n={B['n_finite']}, after n={A['n_finite']}")


if __name__ == "__main__":
    main()
