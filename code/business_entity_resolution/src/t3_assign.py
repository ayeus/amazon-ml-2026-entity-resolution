"""Task 3: per-S1 constrained assignment on top of the matcher's probabilities (label-free rules, tuned on validation).

rel_cut : keep a pair only if p >= thr AND p >= (best p of that S1) - margin        -> targets false positives
rescue  : S1 with no pair >= thr also gets its argmax when p_top >= t_low and
          p_top - p_2nd >= margin (a clear single winner)                            -> targets missed matches
Usage: python3 t3_assign.py <val_pred.parquet> <ds_meta.pkl>
"""
import sys, itertools
import numpy as np, pandas as pd
from decide import score_selection


def s1_context(i1, p):
    """Per-row: best p of its S1, 2nd-best p of its S1 (0 if none), whether the row is its S1's argmax."""
    o = np.lexsort((-p, i1)); i1s, ps = i1[o], p[o]
    first = np.r_[True, i1s[1:] != i1s[:-1]]
    grp = np.cumsum(first) - 1; start = np.flatnonzero(first)
    top = ps[start][grp]
    has2 = np.r_[start[1:], len(ps)] - start > 1
    second = np.where(has2, ps[np.minimum(start + 1, len(ps) - 1)], 0.0)[grp]
    is_top = np.zeros(len(ps), bool); is_top[start] = True
    inv = np.empty_like(o); inv[o] = np.arange(len(o))
    return top[inv], second[inv], is_top[inv]


def rule(p, top, second, is_top, i1, thr, margin=None, t_low=None, r_margin=None):
    sel = p >= thr
    if margin is not None:
        sel &= p >= top - margin
    if t_low is not None:
        empty = ~pd.Series(sel).groupby(i1).transform("any").to_numpy()
        sel |= empty & is_top & (p >= t_low) & (p - second >= r_margin)
    return sel


if __name__ == "__main__":
    D = pd.read_parquet(sys.argv[1]); meta = pd.read_pickle(sys.argv[2])
    meta = meta[meta.split == "val"].reset_index(drop=True)
    p = D.p.to_numpy(np.float64); i1 = D.i1.to_numpy()
    top, second, is_top = s1_context(i1, p)
    f = lambda sel: score_selection(D, sel, meta)[0]
    base = {t: f(p >= t) for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8)}
    bt = max(base, key=base.get)
    print("threshold only:", {k: round(v, 5) for k, v in base.items()}, "-> best", bt, round(base[bt], 5), flush=True)
    M = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    res = []
    for thr, m in itertools.product((0.5, 0.6, 0.65, 0.7), M):
        res.append(("rel_cut", thr, m, None, None, f(rule(p, top, second, is_top, i1, thr, margin=m))))
    for thr, tl, rm in itertools.product((0.65, 0.7), (0.2, 0.3, 0.4, 0.5), M):
        res.append(("rescue", thr, None, tl, rm, f(rule(p, top, second, is_top, i1, thr, t_low=tl, r_margin=rm))))
    R = pd.DataFrame(res, columns=["rule", "thr", "margin", "t_low", "r_margin", "val"])
    R["delta"] = R.val - base[bt]
    for r, g in R.groupby("rule"):
        print(f"\n== {r}: top 8 of {len(g)}"); print(g.sort_values("val", ascending=False).head(8).round(5).to_string(index=False))
    b = R.sort_values("val", ascending=False).iloc[0]
    print(f"\nBEST overall: {b.rule} thr {b.thr} margin {b.margin} t_low {b.t_low} r_margin {b.r_margin} -> {b.val:.5f} "
          f"(threshold-only best {base[bt]:.5f}, delta {b.delta:+.5f})")
