"""Compare decision rules on validation predictions (val_pred3.parquet)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import score_selection, select_expected_f05

va = pd.read_parquet(f"{WORK}/val_pred3.parquet"); meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"]
p = va.p.values
print("recall ceiling", round(va.y.sum() / mva.n_true.sum(), 4), "| S1", len(mva))
res = {}
for t in np.arange(0.5, 0.86, 0.05):
    res[f"global thr {t:.2f}"] = score_selection(va, p >= t, mva)[0]
for s in (2, 3):   # per-source thresholds
    pass
best = (0, None)
for t2 in (0.55, 0.65, 0.75):
    for t3 in (0.55, 0.65, 0.75):
        thr = np.where(va.src_id.values == 2, t2, t3); m = score_selection(va, p >= thr, mva)[0]
        if m > best[0]: best = (m, (t2, t3))
res[f"per-source thr {best[1]}"] = best[0]
for mp in (0.0, 0.3, 0.6, 1.0):
    res[f"expected-F0.5 subset (miss_prior {mp})"] = score_selection(va, select_expected_f05(va, p, miss_prior=mp), mva)[0]
# margin rule: keep p>=t and drop candidates far below the S1's best
top = pd.Series(p).groupby(va.i1.values).transform("max").values
for t in (0.6, 0.7):
    for r in (0.5, 0.7):
        res[f"thr {t} & p>= {r}*top"] = score_selection(va, (p >= t) & (p >= r * top), mva)[0]
# one-owner conflict resolution among S1s inside the validation set only (partial: only ~1/11 of S1 are here)
for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"macro F0.5 {v:.5f}  {k}")
