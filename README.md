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
  eval_model.py       run a model (base or LoRA) over the eval split, print metrics
  train_lora.py       LoRA fine-tune (fp16, grad-checkpoint, checkpoint + auto-resume)
slurm/
  eval_base.sh        baseline eval of stock Qwen3-VL-2B
  train_lora.sh       LoRA training (6h wall limit; resubmit to resume)
  eval_lora.sh        eval the LoRA-adapted model
```

Runs on the CCDS TC1 cluster (single Tesla V100 32GB, SLURM, 6h QoS). All GPU work is
submitted as SLURM jobs — fp16 + sdpa attention (Volta has no bf16 / FlashAttention-2).

## Status

- [x] `render_dataset.py` — data_pool → (PNG, JSON) pairs
- [x] baseline eval script + SLURM job
- [x] LoRA finetune script + SLURM job (checkpoint/resume for the 6h wall limit)
- [x] EMD / Chamfer evaluation (base vs LoRA, shared `eval_model.py`)
- [ ] symbolic refinement stage
- [ ] agent execute-verify loop

## First results (Task A, Qwen3-VL-2B, 500-image held-out eval)

| metric | base | LoRA | |
|---|---|---|---|
| parse fail % | 19.6 | **2.5** | ↓ |
| edit_norm (structure) | 0.998 | **0.612** | ↓ better |
| type accuracy | 0.002 | **0.388** | ↑ ~190× |
| count error | 11.1 | 17.2 | ↑ worse (over-generates) |
| chamfer (accepted pairs) | 2.90 | 2.96 | ≈ unchanged |

**Takeaway — the neuro-symbolic case, empirically:** LoRA sharply improves *perception*
(primitive type, structure, output format) but leaves *geometric precision* essentially
flat (chamfer unchanged). The VLM learns **what to draw**, not **precisely where** — which
is exactly the boundary the symbolic refinement stage is meant to own. A secondary finding:
the tuned model tends to over-generate curves (count error rises even as structure improves).

See [plan.md](plan.md) for the full execution plan.
