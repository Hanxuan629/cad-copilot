#!/usr/bin/env python3
"""Task 3: a CAD-design autoencoder → design embedding → embedding-distance similarity.

A CAD design is a variable-length set of curves (line/circle/arc). We encode each curve
as a fixed feature (type one-hot + sampled points), pool the set into one design embedding
(~256-d), and decode it back to reconstruct each curve's sampled points. Training目标 =
reconstruction (no similarity labels needed). Once trained, the L2 distance between two
designs' embeddings is a fast learned **similarity metric** — the deliverable.

Set-based (PointNet-style) so it's permutation-invariant over curves. Pure geometry,
no VLM. Small model (~few M params); trains fast on a V100.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common import sample_curve, load_index, split_index

TYPES = ["line", "circle", "arc"]
N_PTS = 16                      # points sampled per curve
MAX_CURVES = 24                # pad/truncate each design to this many curves
CURVE_FEAT = len(TYPES) + N_PTS * 2   # 3 + 32 = 35


def curve_to_feat(c):
    """Curve -> fixed (CURVE_FEAT,) vector: type one-hot + N_PTS sampled (x,y)."""
    oh = np.zeros(len(TYPES), dtype=np.float32)
    oh[TYPES.index(c["type"])] = 1.0
    pts = sample_curve(c, N_PTS).astype(np.float32).reshape(-1) / 22.0  # normalize ~[-1,1]
    return np.concatenate([oh, pts])


class DesignSet(Dataset):
    """Each item: (feats [MAX_CURVES, CURVE_FEAT], mask [MAX_CURVES])."""

    def __init__(self, designs):
        self.designs = designs

    def __len__(self):
        return len(self.designs)

    def __getitem__(self, i):
        curves = self.designs[i][:MAX_CURVES]
        feats = np.zeros((MAX_CURVES, CURVE_FEAT), dtype=np.float32)
        mask = np.zeros(MAX_CURVES, dtype=np.float32)
        for k, c in enumerate(curves):
            feats[k] = curve_to_feat(c)
            mask[k] = 1.0
        return torch.from_numpy(feats), torch.from_numpy(mask)


class AE(nn.Module):
    """PointNet-style set autoencoder over a design's curves."""

    def __init__(self, emb=256, hid=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(CURVE_FEAT, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU())
        self.to_emb = nn.Linear(hid, emb)
        # decoder: embedding -> per-slot curve features
        self.dec = nn.Sequential(
            nn.Linear(emb, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, MAX_CURVES * CURVE_FEAT))

    def encode(self, feats, mask):
        h = self.enc(feats)                                  # [B, C, hid]
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)  # masked mean
        return self.to_emb(pooled)                           # [B, emb]

    def forward(self, feats, mask):
        z = self.encode(feats, mask)
        recon = self.dec(z).view(-1, MAX_CURVES, CURVE_FEAT)
        return recon, z


def recon_loss(recon, feats, mask):
    """MSE over valid curve slots only (type logits + coords)."""
    m = mask.unsqueeze(-1)
    diff = (recon - feats) ** 2 * m
    return diff.sum() / m.sum().clamp(min=1) / CURVE_FEAT


def load_designs(index_path, labels_key="curves"):
    rows = load_index(index_path)
    designs = []
    for r in rows:
        lbl = json.loads(Path(r["label"]).read_text())
        designs.append(lbl[labels_key])
    return designs, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="checkpoints/cad_ae.pt")
    args = ap.parse_args()

    designs, _ = load_designs(Path(args.dataset) / "index.jsonl")
    if args.limit:
        designs = designs[:args.limit]
    # deterministic split (mirror split_index: sort not needed, designs already stable order)
    tr, ev = designs[:-args.n_eval], designs[-args.n_eval:]
    print(f"[ae] train={len(tr)} eval={len(ev)} emb={args.emb}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_dl = DataLoader(DesignSet(tr), batch_size=args.batch_size, shuffle=True, num_workers=2)
    ev_dl = DataLoader(DesignSet(ev), batch_size=args.batch_size, num_workers=2)
    model = AE(emb=args.emb).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for feats, mask in tr_dl:
            feats, mask = feats.to(dev), mask.to(dev)
            recon, _ = model(feats, mask)
            loss = recon_loss(recon, feats, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(feats)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for feats, mask in ev_dl:
                feats, mask = feats.to(dev), mask.to(dev)
                recon, _ = model(feats, mask)
                vl += recon_loss(recon, feats, mask).item() * len(feats)
        tl /= len(tr); vl /= len(ev)
        if vl < best:
            best = vl
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "emb": args.emb}, args.out)
        print(f"  epoch {ep+1}/{args.epochs}  train={tl:.5f}  val={vl:.5f}"
              f"{'  *best' if vl==best else ''}", flush=True)
    print(f"[ae] best val loss = {best:.5f}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
