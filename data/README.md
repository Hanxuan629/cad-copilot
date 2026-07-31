# Data

Raw data is **not** tracked in git (too large / licensing). This project's
Task A needs a single file:

- `data_pool_5_per_signature.jsonl` — ~25.7k CAD sketch designs. Each line:
  ```json
  {
    "cad_id": "...",
    "design": {"curves": [
      {"type": "line",   "control_points": [[x,y],[x,y]]},
      {"type": "circle", "control_points": [[cx,cy],[px,py]]},
      {"type": "arc",    "control_points": [[x1,y1],[mx,my],[x2,y2]]}
    ]},
    "scale_factor": 221.87
  }
  ```

## Getting the data onto a server

```bash
# from your laptop
scp path/to/data_pool_5_per_signature.jsonl user@server:~/cad-copilot/data/
```

Then render (PNG, JSON) training pairs:

```bash
python scripts/render_dataset.py \
    --input data/data_pool_5_per_signature.jsonl \
    --out dataset/ --limit 20   # drop --limit for the full set
```
