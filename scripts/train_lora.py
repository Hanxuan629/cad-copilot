#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-VL-2B on CAD sketch -> JSON parsing.

Designed for the TC1 cluster's QoS limits: single V100, 6h wall-clock max. Training
therefore checkpoints often and RESUMES from the latest checkpoint automatically, so a
run that gets cut off at 6h can be continued by simply resubmitting the same job.

V100 notes: fp16 (no bf16 on Volta), attn_implementation="sdpa" (no FlashAttention-2),
gradient checkpointing on to fit the 32GB card. Never run on the Head Node -- use SLURM.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

from common import PROMPT, load_index, split_index

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


class SketchDataset(Dataset):
    """Yields chat-formatted (image + prompt -> JSON answer) supervised examples."""

    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        label = json.loads(Path(row["label"]).read_text())
        answer = json.dumps({"curves": label["curves"]}, ensure_ascii=False)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": str(Path(row['image']).resolve())},
                {"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        # full sequence (with answer) for labels; prompt-only length to mask the prompt
        full = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt")
        prompt_only = self.processor.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        item = {k: v[0] for k, v in full.items()}
        labels = item["input_ids"].clone()
        labels[: prompt_only["input_ids"].shape[1]] = -100  # mask the prompt tokens
        item["labels"] = labels
        return item


def collate(batch):
    """Pad a batch on the right; pad labels with -100 and input_ids with 0."""
    keys = batch[0].keys()
    out = {}
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    for k in keys:
        if k in ("input_ids", "attention_mask", "labels"):
            pad = -100 if k == "labels" else 0
            stacked = []
            for b in batch:
                t = b[k]
                if t.shape[0] < maxlen:
                    t = torch.cat([t, torch.full((maxlen - t.shape[0],), pad, dtype=t.dtype)])
                stacked.append(t)
            out[k] = torch.stack(stacked)
        else:
            # image grid / pixel values: keep as list-friendly stack when shapes match
            try:
                out[k] = torch.stack([b[k] for b in batch])
            except RuntimeError:
                out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out


def latest_checkpoint(output_dir):
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
    return str(ckpts[-1]) if ckpts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--output", default="checkpoints/lora-qwen3vl-2b")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="cap #train samples (debug)")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    rows = load_index(Path(args.dataset) / "index.jsonl")
    train_rows, _ = split_index(rows, n_eval=args.n_eval)
    if args.limit:
        train_rows = train_rows[:args.limit]
    print(f"[train] {len(train_rows)} training designs", flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa",
        device_map="cuda")
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # gradient checkpointing + PEFT: the input embeddings' output must require grad,
    # otherwise the checkpointed graph detaches and backward can't reach LoRA params.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True, bf16=False,                       # V100: fp16 only
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=SketchDataset(train_rows, processor),
        data_collator=collate,
    )

    resume = latest_checkpoint(args.output)
    if resume:
        print(f"[train] resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output)          # final adapter
    print(f"[train] saved adapter -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
