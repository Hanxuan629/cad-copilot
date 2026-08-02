#!/usr/bin/env python3
"""Zero-shot probe: can stock Qwen3-VL-2B match CAD curves to target curves?

Two prompt modes, run on the SAME trials for a fair comparison:
  - text   : coordinates written into the prompt, no image
  - visual : CAD curves and target curves each rendered with numbered labels, side by side

Writes one jsonl per mode with the model's predicted mapping (per CAD curve -> target index),
for eval_probe.py to score against the geometric GT. GPU — run via SLURM.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_model import MODEL_ID, load_model
from match_common import text_prompt, parse_mapping
from common import sample_curve

VISUAL_INSTR = (
    "Two sketches are shown. RIGHT = TARGET curves (labelled T0, T1, ...). "
    "LEFT = DRAWN CAD curves (labelled C0, C1, ...). Coordinates share one frame.\n"
    "For EACH drawn curve C_i, output the single target T_j it best corresponds to "
    "(multiple C may map to the same T; null if none).\n"
    'Respond with ONLY a JSON object like {"C0":"T3","C1":"T3","C2":null}.'
)


@torch.no_grad()
def gen_text(model, processor, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.3)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]


@torch.no_grad()
def gen_visual(model, processor, image_path, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.3)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]


def _draw(ax, curves, prefix, title):
    for i, c in enumerate(curves):
        pts = sample_curve(c, 40)
        ax.plot(pts[:, 0], pts[:, 1], color="black", lw=1.3)
        mid = pts[len(pts) // 2]
        ax.text(mid[0], mid[1], f"{prefix}{i}", color="red", fontsize=9,
                ha="center", va="center")
    ax.set_xlim(-22, 22); ax.set_ylim(-22, 22); ax.set_aspect("equal")
    ax.set_title(title); ax.axis("off")


def render_pair(cad_curves, target_curves, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    _draw(axes[0], cad_curves, "C", "DRAWN (C)")
    _draw(axes[1], target_curves, "T", "TARGET (T)")
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/match_cad2tgt.jsonl")
    ap.add_argument("--mode", choices=["text", "visual"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-dir", default="results/probe_imgs")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[probe] mode={args.mode} n={len(rows)} -> {args.out}", flush=True)

    model, processor = load_model(None)  # stock model, no adapter
    if args.mode == "visual":
        Path(args.img_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(args.out, "w") as fout:
        for k, r in enumerate(rows):
            cad, tgt = r["cad_curves"], r["target_curves"]
            if args.mode == "text":
                raw = gen_text(model, processor, text_prompt(cad, tgt))
            else:
                png = Path(args.img_dir) / f"{r['trial_id']}.png"
                render_pair(cad, tgt, png)
                raw = gen_visual(model, processor, png, VISUAL_INSTR)
            pred = parse_mapping(raw, len(cad))
            fout.write(json.dumps({"trial_id": r["trial_id"], "pred_match": pred,
                                   "gt_match": r["gt_match"], "raw": raw}) + "\n")
            fout.flush()
            if k < 3 or (k + 1) % 25 == 0:
                print(f"  [{k+1}/{len(rows)}] {time.time()-t0:.0f}s "
                      f"pred[:6]={pred[:6]} gt[:6]={r['gt_match'][:6]}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
