"""Compare decision rules on the (test-like) validation predictions of the v3 matcher.

Rules: global threshold, per-country threshold, expected-F0.5 subset selection, address-cluster consistency (a record is
accepted only if its address cluster's mean probability also clears the bar), and one-owner resolution inside the
validation S1 set.  Every rule is chosen/evaluated on validation only.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import macro_f05, select_expected_f05


def entity_f(D, sel, meta):
    tp = pd.Series(D.y.values * sel).groupby(D.i1.values).sum(); n = pd.Series(sel.astype(int)).groupby(D.i1.values).sum()
    return macro_f05(tp.reindex(meta.i1.values).fillna(0).values, n.reindex(meta.i1.values).fillna(0).values, meta.n_true.values)[1]


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    V = pd.read_parquet(f"{WORK}/val_pred_v3{tag}.parquet")
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"].reset_index(drop=True)
    mva = mva[mva.i1.isin(V.i1.unique()) | (mva.n_true == 0)].reset_index(drop=True) if tag else mva
    ctry = pd.read_pickle(f"{WORK}/train_s1.pkl").country.to_numpy(dtype=object)[mva.i1.values]
    p = V.p.values
    res = {}
    for th in np.arange(0.40, 0.96, 0.05):
        res[f"global thr {th:.2f}"] = entity_f(V, p >= th, mva).mean()
    best_g = max((v, k) for k, v in res.items() if k.startswith("global"))
    # per-country threshold (chosen per country on validation)
    vc = pd.read_pickle(f"{WORK}/train_s1.pkl").country.to_numpy(dtype=object)[V.i1.values]
    sel = np.zeros(len(V), bool); chosen = {}
    for c in ("US", "India"):
        best = max(((entity_f(V[vc == c], p[vc == c] >= th, mva[ctry == c]).mean(), th) for th in np.arange(0.4, 0.96, 0.05)))
        chosen[c] = round(best[1], 2); sel[vc == c] = p[vc == c] >= best[1]
    res[f"per-country thr {chosen}"] = entity_f(V, sel, mva).mean()
    for mp in (0.0, 0.3, 0.6):
        res[f"expected-F0.5 subset (miss_prior {mp})"] = entity_f(V, select_expected_f05(V, p, miss_prior=mp), mva).mean()
    for k, v in sorted(res.items(), key=lambda x: -x[1])[:12]:
        print(f"macro F0.5 {v:.5f}  {k}")
    print(f"\nbest global: {best_g[1]} -> {best_g[0]:.5f}")
