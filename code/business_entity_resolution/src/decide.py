"""Stage 4: metric + decision rules.

Metric: macro F_0.5 over ALL S1 entities.  Per entity F = 1.25*TP/(0.25*|T| + |P|) (== 1.25PR/(0.25P+R));
empty truth & empty prediction -> 1.0; exactly one of them empty -> 0.0.
"""
import numpy as np, pandas as pd


def f05_entity(pred, truth):
    pred, truth = set(pred), set(truth)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    return 1.25 * tp / (0.25 * len(truth) + len(pred))


def macro_f05(pred_tp, pred_n, n_true):
    """Vectorised. Arrays aligned per S1: TP count, |P|, |T|."""
    pred_tp = np.asarray(pred_tp, float); pred_n = np.asarray(pred_n, float); n_true = np.asarray(n_true, float)
    f = np.where((pred_n == 0) & (n_true == 0), 1.0,
                 np.where((pred_n == 0) | (n_true == 0), 0.0, 1.25 * pred_tp / (0.25 * n_true + pred_n + 1e-12)))
    return f.mean(), f


def score_selection(D, sel, meta):
    """D: candidate frame (i1, y); sel: boolean mask of selected rows; meta: frame with i1, n_true (ALL S1 in the eval set)."""
    tp = pd.Series(D.y.values * sel).groupby(D.i1.values).sum()
    n = pd.Series(sel.astype(int)).groupby(D.i1.values).sum()
    tpv = tp.reindex(meta.i1.values).fillna(0).values; nv = n.reindex(meta.i1.values).fillna(0).values
    return macro_f05(tpv, nv, meta.n_true.values)


def select_threshold(D, p, thr):
    return p >= thr


def select_expected_f05(D, p, miss_prior=0.0, min_p=0.0):
    """Per-S1 subset (top-m by probability, m>=0) maximising plug-in expected F_0.5, empty allowed.
    E[F_m] ~= 1.25*sum_{top m} p / (0.25*E|T| + m),  E|T| = sum_all p + miss_prior;  empty value = prod(1-p)."""
    df = pd.DataFrame({"i1": D.i1.values, "p": np.asarray(p, float), "row": np.arange(len(D))})
    df = df.sort_values(["i1", "p"], ascending=[True, False])
    g = df.groupby("i1")
    df["cs"] = g.p.cumsum(); df["m"] = g.cumcount() + 1
    et = g.p.transform("sum") + miss_prior
    df["ef"] = 1.25 * df.cs / (0.25 * et + df.m)
    lp = np.log1p(-np.clip(df.p.values, 0, 1 - 1e-9))
    df["pe"] = np.exp(pd.Series(lp, index=df.index).groupby(df.i1).transform("sum").values)  # P(no true match in cands)
    best = df.groupby("i1").ef.transform("max")
    bestm = df.assign(isb=(df.ef >= best - 1e-12)).groupby("i1").apply(lambda x: x.m[x.isb].iloc[0] if len(x) else 0, include_groups=False)
    bm = df.i1.map(bestm).values
    use = (df.pe.values * 1.0 < best.values)  # empty beats best subset if P(empty truth) >= best expected F
    sel = np.zeros(len(D), dtype=bool)
    ok = (df.m.values <= bm) & use & (df.p.values >= min_p)
    sel[df.row.values[ok]] = True
    return sel
