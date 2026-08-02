#!/usr/bin/env python3
"""Build the CAD-curve -> target-curve matching dataset from data/mrcad_raw_json.

Each output row is one trial: its final-round CAD curves, the target curves, and the
GT match (per CAD curve, the index of the target it corresponds to, or -1). The GT is
constructed geometrically (nearest target by chamfer under REJECT_TAU) — the same call
doubles as the no-VLM baseline in eval_probe.py.

Skips trials with too few/many curves or an empty CAD design. CPU-only.
"""
import argparse
import json
from pathlib import Path

from match_common import iter_trials, final_round_cad, geometric_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/mrcad_raw_json")
    ap.add_argument("--out", default="data/match_cad2tgt.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap #trials (debug)")
    ap.add_argument("--min-curves", type=int, default=2)
    ap.add_argument("--max-curves", type=int, default=24)
    args = ap.parse_args()

    n_written = n_skipped = 0
    with open(args.out, "w") as fout:
        for trial in iter_trials(args.data_dir):
            cad = final_round_cad(trial)
            tgt = (trial.get("target") or {}).get("curves") or []
            if not (args.min_curves <= len(cad) <= args.max_curves) or not tgt:
                n_skipped += 1
                continue
            gt_match, gt_dists = geometric_match(cad, tgt)
            fout.write(json.dumps({
                "trial_id": trial.get("trial_id"),
                "target_id": trial.get("target_id"),
                "cad_curves": cad,
                "target_curves": tgt,
                "gt_match": gt_match,
                "gt_dists": [round(d, 3) if d != float("inf") else None for d in gt_dists],
            }) + "\n")
            n_written += 1
            if args.limit and n_written >= args.limit:
                break
    print(f"wrote {n_written} trials, skipped {n_skipped} -> {args.out}")


if __name__ == "__main__":
    main()
