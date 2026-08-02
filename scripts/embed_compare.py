#!/usr/bin/env python3
"""Task 3 comparison: which embedding-distance best reflects GEOMETRIC design similarity?

The user wants an embedding distance that measures whether two CAD designs *look* alike.
We compare three distances against a geometric ground truth (chamfer via match_designs):

  1. geometric-AE : L2 between the trained set-autoencoder embeddings (cad_autoencoder.py)
  2. qwen-text    : cosine distance between Qwen3-Embedding vectors of the CAD code TEXT
  3. random       : shuffled control (sanity floor)

Metric = Spearman/Pearson correlation of each distance vs the true geometric distance over
a sample of design pairs. High correlation => that embedding captures geometric similarity.

The point: test whether an off-the-shelf TEXT embedding (qwen-text) can stand in for
geometric similarity, or whether you need the geometry-native AE. GPU (Qwen model).
"""
import argparse
import json
import itertools
from pathlib import Path

import numpy as np
import torch

from common import load_index, curve_distance, sample_curve

QWEN_EMB_ID = "Qwen/Qwen3-Embedding-0.6B"


# ---- CAD design -> code text (for the text embedding) -----------------------

def design_to_code(curves):
    """Serialize a design into a compact CAD-code string."""
    parts = []
    for c in curves:
        pts = " ".join(f"({x:.1f},{y:.1f})" for x, y in c["control_points"])
        parts.append(f"{c['type']} {pts}")
    return "; ".join(parts)


# ---- geometric ground-truth distance between two designs --------------------

def design_geom_distance(a, b, n=16):
    """Symmetric mean nearest-curve chamfer between two designs (geometry GT)."""
    if not a or not b:
        return float("nan")
    D = np.zeros((len(a), len(b)))
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            D[i, j] = curve_distance(ca, cb, n=n)
    return float((D.min(axis=1).mean() + D.min(axis=0).mean()) / 2)


# ---- AE embedding -----------------------------------------------------------

def ae_embeddings(designs, ckpt_path, device):
    from cad_autoencoder import AE, DesignSet
    from torch.utils.data import DataLoader
    ck = torch.load(ckpt_path, map_location=device)
    model = AE(emb=ck["emb"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    dl = DataLoader(DesignSet(designs), batch_size=128)
    embs = []
    with torch.no_grad():
        for feats, mask in dl:
            embs.append(model.encode(feats.to(device), mask.to(device)).cpu().numpy())
    return np.concatenate(embs)


# ---- Qwen3-Embedding on code text -------------------------------------------

def _last_token_pool(h, attn):
    left = attn[:, -1].sum() == attn.shape[0]
    if left:
        return h[:, -1]
    idx = attn.sum(1) - 1
    return h[torch.arange(h.shape[0]), idx]


def qwen_embeddings(texts, device, batch_size=16):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(QWEN_EMB_ID, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_EMB_ID, torch_dtype=torch.float16).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            emb = _last_token_pool(h, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, dim=1)
            out.append(emb.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--ae-ckpt", default="checkpoints/cad_ae.pt")
    ap.add_argument("--n-designs", type=int, default=300, help="designs to sample")
    ap.add_argument("--n-pairs", type=int, default=3000, help="random pairs for correlation")
    ap.add_argument("--out", default="results/embed_compare.json")
    args = ap.parse_args()

    from scipy.stats import spearmanr, pearsonr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_index(Path(args.dataset) / "index.jsonl")
    rows = rows[-args.n_designs:]  # eval-side designs (unseen-ish)
    designs = [json.loads(Path(r["label"]).read_text())["curves"] for r in rows]
    texts = [design_to_code(d) for d in designs]
    print(f"[embed-compare] {len(designs)} designs, {args.n_pairs} pairs", flush=True)

    # embeddings
    ae = ae_embeddings(designs, args.ae_ckpt, device)
    print("  AE embeddings done", flush=True)
    qw = qwen_embeddings(texts, device)
    print("  Qwen embeddings done", flush=True)

    # sample pairs (deterministic via fixed stride, no RNG)
    N = len(designs)
    pairs = list(itertools.combinations(range(N), 2))
    step = max(1, len(pairs) // args.n_pairs)
    pairs = pairs[::step][:args.n_pairs]

    geom, d_ae, d_qw, d_rand = [], [], [], []
    for k, (i, j) in enumerate(pairs):
        g = design_geom_distance(designs[i], designs[j])
        if not np.isfinite(g):
            continue
        geom.append(g)
        d_ae.append(float(np.linalg.norm(ae[i] - ae[j])))
        d_qw.append(float(1.0 - np.dot(qw[i], qw[j])))          # cosine distance
        d_rand.append(float(np.linalg.norm(ae[i] - ae[(j * 7 + 3) % N])))  # shuffled control
        if (k + 1) % 500 == 0:
            print(f"  pairs {k+1}/{len(pairs)}", flush=True)

    res = {}
    for name, d in [("geometric-AE", d_ae), ("qwen-text", d_qw), ("random", d_rand)]:
        sp = spearmanr(geom, d).correlation
        pe = pearsonr(geom, d)[0]
        res[name] = {"spearman": round(float(sp), 4), "pearson": round(float(pe), 4)}
    res["n_pairs"] = len(geom)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n===== embedding distance vs geometric GT (higher=better) =====")
    print(f"{'method':<16}{'spearman':>10}{'pearson':>10}")
    for name in ("geometric-AE", "qwen-text", "random"):
        print(f"{name:<16}{res[name]['spearman']:>10.3f}{res[name]['pearson']:>10.3f}")
    print(f"n_pairs={res['n_pairs']}  -> {args.out}")


if __name__ == "__main__":
    main()
