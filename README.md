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

## Status

- [ ] `render_dataset.py` — data_pool → (PNG, JSON) pairs
- [ ] baseline eval of stock Qwen3-VL-2B
- [ ] LoRA finetune
- [ ] EMD evaluation (base vs LoRA)
- [ ] symbolic refinement stage
- [ ] agent execute-verify loop

See [plan.md](plan.md) for the full execution plan.
