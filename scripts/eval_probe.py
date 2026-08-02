#!/usr/bin/env python3
"""Score the zero-shot line-matching probes against the geometric GT.

The geometric matcher (match_common.geometric_match) is BOTH the GT-label source and the
no-model baseline, so by construction its accuracy is 1.0 — it is the ceiling the VLM is
measured against, not a competitor. We report:
  - geometric baseline : coverage (fraction of CAD curves confidently matched, != -1)
  - VLM-text / VLM-visual : top-1 accuracy vs GT, parse-success rate

Answers the user's question: how close does the VLM get to the geometry the GT encodes?
CPU-only.
"""
import argparse
import json

from match_common import match_accuracy


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def summarize_probe(rows):
    accs, parse_ok, n_curves = [], 0, 0
    for r in rows:
        pred, gt = r["pred_match"], r["gt_match"]
        accs.append(match_accuracy(pred, gt))
        # parse succeeded if at least one non -1 prediction, or model clearly emitted a map
        if any(p != -1 for p in pred):
            parse_ok += 1
        n_curves += len(gt)
    return {
        "trials": len(rows),
        "mean_acc": sum(accs) / len(accs) if accs else float("nan"),
        "parse_ok_rate": parse_ok / len(rows) if rows else float("nan"),
        "curves": n_curves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/match_cad2tgt.jsonl",
                    help="the built dataset (for geometric-baseline coverage)")
    ap.add_argument("--text", default=None, help="probe_text.jsonl")
    ap.add_argument("--visual", default=None, help="probe_visual.jsonl")
    args = ap.parse_args()

    # geometric baseline coverage (from the dataset's GT)
    data = load(args.data)
    total = sum(len(r["gt_match"]) for r in data)
    matched = sum(1 for r in data for x in r["gt_match"] if x != -1)
    print(f"{'method':<16}{'trials':>8}{'acc':>10}{'parse_ok':>10}")
    print("-" * 44)
    print(f"{'geometric(GT)':<16}{len(data):>8}{'1.000':>10}"
          f"{'':>10}   [coverage {matched}/{total} = {matched/total:.2f}]")

    for name, path in [("VLM-text", args.text), ("VLM-visual", args.visual)]:
        if not path:
            continue
        s = summarize_probe(load(path))
        print(f"{name:<16}{s['trials']:>8}{s['mean_acc']:>10.3f}{s['parse_ok_rate']:>10.3f}")


if __name__ == "__main__":
    main()
