"""Task 6: macro F0.5 on the manual French audit with a 50/50 split (tune the threshold on one half, report the other).

Scoring = france_audit.score (predictions outside the audited neighbourhood count as FP; '?' pairs excluded), but per S1
and for a grid of thresholds.  Split = stratified by audit stratum; one fixed split (seed 0) plus the mean / std of the
held-out score over many random splits (50 S1 per half is very noisy: one S1 = 0.02).
Usage: python3 t6_audit.py name=work:scored [name=work:scored ...]
"""
import os, sys
import numpy as np, pandas as pd
from france_audit import load, DIAG, ROOT

GRID = np.round(np.arange(0.50, 0.981, 0.02), 2)
key = lambda a, b, c: np.asarray(a).astype(np.int64) * 100_000_000_000 + np.asarray(b).astype(np.int64) * 10 + np.asarray(c)


def audit_table(work):
    P = pd.read_csv(f"{DIAG}/france_audit_pairs.tsv", sep="\t", dtype={"label": str}, keep_default_na=False)
    d = {s: pd.read_pickle(f"{work}/test_s{s}.pkl") for s in (1, 2, 3)}
    rid = {s: pd.Series(np.arange(len(d[s])), index=d[s].entity_id.values) for s in (2, 3)}
    P["i1n"] = pd.Series(np.arange(len(d[1])), index=d[1].entity_id.values).reindex(P.s1_id.values).values
    P["jn"] = [rid[s].get(r, -1) for s, r in zip(P.src, P.rec_id)]
    P["k"] = key(P.i1n.values, P.jn.values, P.src.values)
    return P


OUTSIDE = {}


def per_s1_f(P, S, name=None):
    """Matrix [n_audited_S1, len(GRID)] of per-S1 F0.5 against the manual labels."""
    s1 = np.sort(P.i1n.unique()); pos = {v: i for i, v in enumerate(s1)}
    S = S[S.i1.isin(set(s1))]; sk = key(S.i1.values, S.j.values, S.src.values)
    lab = dict(zip(P.k.values, P.label.values))
    status = np.array([lab.get(k, "out") for k in sk])            # "1", "0", "?", or "out" (outside neighbourhood)
    if os.environ.get("BER_OUT_EXCLUDE"):                         # sensitivity: unlabelled predictions neither hurt nor help
        status = np.where(status == "out", "?", status)
    si = np.array([pos[v] for v in S.i1.values]); p = S.p.values
    if name:
        OUTSIDE[name] = (int(((p >= 0.9) & (status == "out")).sum()), int((p >= 0.9).sum()))
    nt = P[P.label == "1"].groupby("i1n").size().reindex(s1).fillna(0).values
    F = np.zeros((len(s1), len(GRID)))
    for g, t in enumerate(GRID):
        sel = p >= t
        tp = np.bincount(si[sel & (status == "1")], minlength=len(s1))
        npred = np.bincount(si[sel & (status != "?")], minlength=len(s1))
        F[:, g] = np.where((npred == 0) & (nt == 0), 1.0, np.where((npred == 0) | (nt == 0), 0.0, 1.25 * tp / (0.25 * nt + npred + 1e-12)))
    return s1, F


def splits(strata, n, seed):
    rng = np.random.RandomState(seed); a = np.zeros(len(strata), bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s); rng.shuffle(idx); a[idx[: len(idx) // 2 + (rng.rand() < 0.5) * (len(idx) % 2)]] = True
    return a


if __name__ == "__main__":
    models = [a.split("=", 1) for a in sys.argv[1:]]
    tables = {}
    for name, spec in models:
        work, scored = spec.split(":")
        work = f"{ROOT}/{work}"
        P = audit_table(work); _, S, _ = load(work, scored)
        s1, F = per_s1_f(P, S, name); tables[name] = F
        strata = P.drop_duplicates("i1n").set_index("i1n").stratum.reindex(s1).values
    print("predictions at p>=0.9 on audited S1 that fall OUTSIDE the labelled neighbourhood (counted as FP):",
          {n: f"{o}/{t}" for n, (o, t) in OUTSIDE.items()})
    print(f"audited S1 {len(s1)}; labelled pairs {int(P.label.isin(['0', '1']).sum())}, uncertain {int((P.label == '?').sum())}")
    print("full-audit macro F0.5 by threshold (agreement with manual labels):")
    print(pd.DataFrame({n: F.mean(0) for n, F in tables.items()}, index=GRID).round(4).iloc[::2].to_string())
    rows = []
    for name, F in tables.items():
        held, thrs = [], []
        for seed in range(200):
            a = splits(strata, len(s1), seed)
            for tune, test in ((a, ~a), (~a, a)):
                g = int(np.argmax(F[tune].mean(0))); held.append(F[test, g].mean()); thrs.append(GRID[g])
        a = splits(strata, len(s1), 0); g = int(np.argmax(F[a].mean(0)))
        rows.append(dict(model=name, seed0_thr=GRID[g], seed0_heldout=round(F[~a, g].mean(), 4),
                         heldout_mean=round(np.mean(held), 4), heldout_std=round(np.std(held), 4),
                         thr_median=float(np.median(thrs)), full_at_0_90=round(F[:, list(GRID).index(0.9)].mean(), 4)))
    print("\n50/50 split: tune threshold on one half, report the other (seed 0; and mean/std over 200 splits x 2 directions)")
    print(pd.DataFrame(rows).to_string(index=False))
    names = list(tables)
    if len(names) > 1:
        print("\npaired held-out difference vs", names[0], "(same splits, each model tuned on its own half):")
        for n in names[1:]:
            diffs = []
            for seed in range(200):
                a = splits(strata, len(s1), seed)
                for tune, test in ((a, ~a), (~a, a)):
                    g0 = int(np.argmax(tables[names[0]][tune].mean(0))); g1 = int(np.argmax(tables[n][tune].mean(0)))
                    diffs.append(tables[n][test, g1].mean() - tables[names[0]][test, g0].mean())
            diffs = np.array(diffs)
            print(f"  {n}: mean {diffs.mean():+.4f}, share of splits better {np.mean(diffs > 0):.2f}, worse {np.mean(diffs < 0):.2f}")
