"""Test-like validation: reweight validation S1 entities so the per-S1 mix of candidate pair types matches the mix
measured (label-free) on a test sample.  Entities are weighted, not pairs, so the metric stays a per-S1 macro average."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import macro_f05

CATS = ["A", "B", "C", "D"]


def pair_cat(D):
    name_sim = (D.jw >= 0.85) | (D.a_sub_b == 1) | (D.b_sub_a == 1)
    street = D.a_ixw >= 2; nc = D.num_conf == 1; extra = D.b_minus_a >= 1
    return np.select([name_sim & street & nc & extra, name_sim & street & nc, name_sim & street & ~nc], ["A", "B", "C"], "D")


def entity_profile(D):
    """per-S1 counts of each pair category (label-free)"""
    c = pd.Series(pair_cat(D), index=D.index)
    return pd.crosstab(D.i1.values, c.values).reindex(columns=CATS, fill_value=0)


def weights(val_prof, test_prof):
    """Weight val S1 by the ratio of test/val frequency of their (A+B distractor count) bucket and D-count bucket."""
    def key(P):
        ab = (P.A + P.B).clip(upper=4); d = P.D.clip(upper=6)
        return ab.astype(str) + "_" + d.astype(str)
    kv, kt = key(val_prof), key(test_prof)
    fv = kv.value_counts(normalize=True); ft = kt.value_counts(normalize=True)
    w = kv.map((ft / fv).fillna(0)).fillna(0)
    return w


def weighted_score(D, sel, meta, w):
    tp = pd.Series(D.y.values * sel).groupby(D.i1.values).sum()
    n = pd.Series(sel.astype(int)).groupby(D.i1.values).sum()
    tpv = tp.reindex(meta.i1.values).fillna(0).values; nv = n.reindex(meta.i1.values).fillna(0).values
    _, f = macro_f05(tpv, nv, meta.n_true.values)
    ww = w.reindex(meta.i1.values).fillna(0).values
    return float((f * ww).sum() / ww.sum()), float(f.mean())


if __name__ == "__main__":
    V = pd.read_parquet(f"{WORK}/ds4_val.parquet"); V["p"] = pd.read_parquet(f"{WORK}/val_pred_big.parquet").p.values
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"]
    T = pd.read_parquet(f"{WORK}/diag_test_sample.parquet")
    vp = entity_profile(V).reindex(mva.i1.values, fill_value=0)
    for c in (None, "US", "India", "France"):
        Tc = T if c is None else T[T.country == c]
        tp = entity_profile(Tc)
        w = weights(vp, tp)
        print(f"\n== weighted to test profile [{c or 'ALL test'}]  (effective val S1: {w.sum()**2/(w**2).sum():.0f})")
        print("   mean A+B per S1  val %.3f  test %.3f  reweighted-val %.3f" % ((vp.A + vp.B).mean(), (tp.A + tp.B).mean(), ((vp.A + vp.B) * w).sum() / w.sum()))
        for th in (0.6, 0.7, 0.8, 0.9, 0.95):
            ws, us = weighted_score(V, V.p.values >= th, mva, w)
            print(f"   thr {th:.2f}: plain val {us:.5f}   test-like {ws:.5f}")
