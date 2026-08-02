# CAD-Copilot

A neuro-symbolic multimodal agent for parsing CAD sketches into structured
vector geometry, then reconstructing them through an execute-and-verify loop.

## The idea

VLMs are strong at *perception* (what primitives are here, roughly where) but
weak at emitting *precise coordinates* — floats get shredded into tokens and
drift. So instead of asking one model to do everything, the task is decomposed
**along capability boundaries**:

| Stage | Component | Job |
|-------|-----------|-----|
| 1. Perceive | LoRA-finetuned **Qwen3-VL-2B** | sketch image → coarse primitives (type, count, rough location) |
| 2. Refine   | symbolic geometric solver | snap coordinates to precise values via least-squares / edge fitting |
| 3. Drive    | Claude Agent SDK | run the render → compare-to-target → feedback → re-solve loop |

The research angle is *where the neural/symbolic boundary should sit*, measured
with rigorous geometric metrics (EMD / Chamfer curve matching), not token
accuracy.

## Task A (current): CAD sketch → structured JSON

- **Input:** rendered CAD sketch PNG + a fixed prompt
- **Output:** structured JSON of all primitives (`line` / `circle` / `arc` + control points)
- **Ground truth:** `design.curves` from the source dataset
- **Eval:** EMD / Chamfer curve-matching error, primitive-type accuracy, primitive-count accuracy

## Layout

```
scripts/
  render_dataset.py   data_pool → (PNG, JSON) pairs
  common.py           fixed prompt, JSON parsing, EMD/Chamfer + Hungarian scoring, data split
  eval_model.py       run a model (base or LoRA) over the eval split; --dump-pred saves predictions
  train_lora.py       LoRA fine-tune (fp16, grad-checkpoint, checkpoint + auto-resume)
  refine.py           symbolic solver: snap predicted curves onto image ink (PNG-only, no GT)
  eval_refine.py      score chamfer before/after refinement (CPU)
slurm/
  eval_base.sh        baseline eval of stock Qwen3-VL-2B
  train_lora.sh       LoRA training (6h wall limit; resubmit to resume)
  eval_lora.sh        eval the LoRA-adapted model
  capture_lora.sh     capture LoRA predictions on a 200-image subset (for refinement)
```

Runs on the CCDS TC1 cluster (single Tesla V100 32GB, SLURM, 6h QoS). All GPU work is
submitted as SLURM jobs — fp16 + sdpa attention (Volta has no bf16 / FlashAttention-2).

## Status

- [x] `render_dataset.py` — data_pool → (PNG, JSON) pairs
- [x] baseline eval script + SLURM job
- [x] LoRA finetune script + SLURM job (checkpoint/resume for the 6h wall limit)
- [x] EMD / Chamfer evaluation (base vs LoRA, shared `eval_model.py`)
- [x] symbolic refinement stage (`refine.py`, `eval_refine.py`)
- [ ] agent execute-verify loop

## Results — the neuro-symbolic split, measured

Task A, Qwen3-VL-2B. Perception axis = edit_norm / type accuracy; geometry axis = chamfer
on accepted pairs. (Base/LoRA: 500-image eval; refinement: 200-image subset.)

| stage | edit_norm ↓ | type_acc ↑ | chamfer ↓ | what moved |
|---|---|---|---|---|
| base (stock VLM) | 0.998 | 0.002 | 2.90 | — |
| + LoRA | 0.612 | 0.388 | 2.94 | **perception**, not geometry |
| + symbolic refine | 0.599 | 0.401 | **2.48** | **geometry** (−16%), not perception |

**The thesis, empirically:** LoRA teaches the VLM *what to draw* (type accuracy 0.2%→39%)
but leaves geometric precision flat (chamfer 2.90→2.94). A symbolic solver that snaps the
predicted control points onto the sketch ink then owns *precisely where* (chamfer
2.94→2.48) while barely touching perception. The two gains are orthogonal — exactly the
capability boundary the design targets.

The refinement (`refine.py`) reads **only the rendered PNG** (never ground truth): it fits
each predicted line/circle/arc to nearby ink with least squares, and a GT-free self-check
(refined curve must fit the ink no worse than the original) guarantees it can only improve
or leave a curve unchanged. 66% of curves were refined; the rest fell back untouched.

See [plan.md](plan.md) for the full execution plan.
