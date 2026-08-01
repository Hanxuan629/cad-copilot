#!/usr/bin/env python3
"""Evaluate a Qwen3-VL model (base or LoRA-adapted) on CAD sketch -> JSON parsing.

Runs inference over the held-out eval split, parses each prediction into curves,
and scores it against ground truth with the EMD/Chamfer + type/count metrics in
common.py. Writes a per-sample JSONL and prints an aggregate summary.

V100 notes: load in fp16 (Volta has no bf16), attn_implementation="sdpa"
(no FlashAttention-2 on Volta). Never run this on the Head Node -- submit via SLURM.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from common import (PROMPT, parse_curves, match_designs, load_index, split_index)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def load_model(adapter_dir=None):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa",
        device_map="cuda",
    )
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()  # fold LoRA in for faster inference
    model.eval()
    return model, processor


@torch.no_grad()
def predict(model, processor, image_path, max_new_tokens=512):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n_prompt = inputs["input_ids"].shape[1]
    trimmed = out[:, n_prompt:]
    n_gen = trimmed.shape[1]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0], n_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="cap #eval samples (debug)")
    ap.add_argument("--metric", choices=["emd", "chamfer"], default="chamfer")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--verbose", action="store_true", help="print per-image timing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_index(Path(args.dataset) / "index.jsonl")
    _, eval_rows = split_index(rows, n_eval=args.n_eval)
    if args.limit:
        eval_rows = eval_rows[:args.limit]

    tag = "lora" if args.adapter else "base"
    out_path = args.out or f"eval_{tag}_{args.metric}.jsonl"
    print(f"[eval] model={tag} metric={args.metric} n={len(eval_rows)} -> {out_path}", flush=True)

    model, processor = load_model(args.adapter)

    agg = {"edit_distance": [], "edit_norm": [], "matched_distance": [],
           "type_accuracy": [], "count_correct": [], "count_error": [], "parse_fail": 0}
    t0 = time.time()
    with open(out_path, "w") as fout:
        for k, row in enumerate(eval_rows):
            label = json.loads(Path(row["label"]).read_text())
            gt = label["curves"]
            t_gen = time.time()
            raw, n_gen = predict(model, processor, Path(row["image"]).resolve(),
                                 max_new_tokens=args.max_new_tokens)
            gen_s = time.time() - t_gen
            pred = parse_curves(raw)
            if not pred:
                agg["parse_fail"] += 1
            t_score = time.time()
            score = match_designs(pred, gt, metric=args.metric)
            score_s = time.time() - t_score
            fout.write(json.dumps({"cad_id": row["cad_id"], "n_pred": len(pred),
                                   "n_gt": len(gt), **score}) + "\n")
            fout.flush()
            for key in ("edit_distance", "edit_norm", "type_accuracy",
                        "count_correct", "count_error"):
                agg[key].append(score[key])
            if score["matched_distance"] != float("inf"):
                agg["matched_distance"].append(score["matched_distance"])
            if args.verbose or k < 3:
                print(f"  [{k+1}/{len(eval_rows)}] gen={gen_s:.1f}s ntok={n_gen} "
                      f"score={score_s:.1f}s n_pred={len(pred)} n_gt={len(gt)}", flush=True)
            if (k + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(eval_rows)}  ({el/(k+1):.1f}s/img)", flush=True)

    n = len(eval_rows)
    def mean(x):
        return sum(x) / len(x) if x else float("nan")
    print("\n===== SUMMARY ({}) =====".format(tag))
    print(f"  samples              : {n}")
    print(f"  parse failures       : {agg['parse_fail']} ({100*agg['parse_fail']/n:.1f}%)")
    print("  -- perception axis --")
    print(f"  edit distance (ADDs) : {mean(agg['edit_distance']):.2f}  (lower better)")
    print(f"  edit distance (norm) : {mean(agg['edit_norm']):.3f}  (0=perfect structure)")
    print(f"  type accuracy        : {mean(agg['type_accuracy']):.3f}")
    print(f"  count-correct rate   : {mean(agg['count_correct']):.3f}")
    print(f"  mean count error     : {mean(agg['count_error']):.2f}")
    print("  -- geometry axis --")
    print(f"  mean matched {args.metric:<7} : {mean(agg['matched_distance']):.4f}  (accepted pairs; lower better)")
    print(f"  total time           : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
