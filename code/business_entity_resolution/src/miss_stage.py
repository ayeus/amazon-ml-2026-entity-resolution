import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
ms = pd.read_pickle(f"{WORK}/val_missed.pkl")
meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); val = set(meta[meta.split == "val"].i1)
key = lambda a, b, c: a.astype(np.int64) * 100_000_000_000 + b.astype(np.int64) * 10 + c
parts = []
for f in sorted(glob.glob(f"{WORK}/block_train/*.parquet")):
    B = pd.read_parquet(f, columns=["i1", "j", "src", "r_t", "r_n", "r_a"])
    B = B[B.i1.isin(val)]
    if len(B): parts.append(B)
B = pd.concat(parts, ignore_index=True)
bk = key(B.i1.values, B.j.values, B.src.values)
ms["in_block"] = np.isin(key(ms.i1.values, ms.j.values, ms.src.values), bk)
print("missed total", len(ms), "| present in blocking output (dropped by cheap filter):", int(ms.in_block.sum()), f"({ms.in_block.mean():.3f}) | never retrieved by blocking: {int((~ms.in_block).sum())}")
print("\nmisses NEVER retrieved by blocking, by category:")
print(ms[~ms.in_block].cat.value_counts().to_frame("n").assign(share=lambda x: (x.n / (~ms.in_block).sum()).round(3)).to_string())
print("\nmisses DROPPED by cheap filter, by category:")
print(ms[ms.in_block].cat.value_counts().to_string())
tot = 138614
print(f"\nblocking-only recall (val, pairs): {1 - (~ms.in_block).sum()/tot:.4f}   after cheap filter: {1 - len(ms)/tot:.4f}")
ms.to_pickle(f"{WORK}/val_missed2.pkl")
