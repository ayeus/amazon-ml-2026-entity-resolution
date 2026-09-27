"""French audit set (manual labels; France has no ground truth anywhere).

sample  : stratified France test S1 -> neighbourhood = model candidates U top-8 blocking records by cs_all.
          Writes diagnostics/france_audit_pairs.tsv (one row per pair, label column empty) and a readable batch file.
score   : given the labelled TSV (label 1 / 0 / ?), macro F0.5 over audited S1 for any scored-candidates folder
          (p >= thr), counting '?' pairs as excluded.  Reported as agreement with manual labels, NOT a French error rate.
Strata  : empty prediction | near-threshold (0.3<=p<0.9) | high-confidence only | groupe/participations/developpement
          in the neighbourhood | SNC/Cie/Ets/EI in a name.
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from decide import macro_f05

ROOT = os.environ.get("BER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DIAG = f"{ROOT}/diagnostics"
FR_WORDS = r"\b(groupe|participations|developpement|développement)\b"
FR_LEGAL = r"\b(snc|cie|ets|ei)\b"


def load(work, scored):
    d = {s: pd.read_pickle(f"{work}/test_s{s}.pkl") for s in (1, 2, 3)}
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{work}/{scored}_test/France_*.parquet"))], ignore_index=True)
    B = pd.concat([pd.read_parquet(f, columns=["i1", "j", "src", "cs_all"]) for f in sorted(glob.glob(f"{work}/block_test/France_*.parquet"))],
                  ignore_index=True)
    return d, S, B


def sample(work, scored, thr=0.70, seed=7):
    d, S, B = load(work, scored)
    rng = np.random.RandomState(seed)
    g = S.groupby("i1").p
    stats = pd.DataFrame({"n_acc": g.apply(lambda x: int((x >= thr).sum())), "n_near": g.apply(lambda x: int(((x >= 0.3) & (x < 0.9)).sum()))})
    nm = {s: d[s].name.str.lower() for s in (2, 3)}
    Bn = B.copy(); bn = np.empty(len(Bn), object)
    for s in (2, 3):
        mk = Bn.src.values == s; bn[mk] = nm[s].values[Bn.j.values[mk]]
    Bn["nm"] = pd.Series(bn, index=Bn.index).astype(str)
    fr_words = set(Bn.i1[Bn.nm.str.contains(FR_WORDS, regex=True)])
    s1nm = d[1].name.str.lower()
    fr_legal = set(np.flatnonzero(s1nm.str.contains(FR_LEGAL, regex=True).values)) & set(S.i1.unique())
    pools = {"empty prediction": set(stats.index[stats.n_acc == 0]),
             "near-threshold": set(stats.index[stats.n_near > 0]),
             "high-confidence only": set(stats.index[(stats.n_acc > 0) & (stats.n_near == 0)]),
             "groupe/participations/developpement": fr_words, "SNC/Cie/Ets/EI": fr_legal}
    want = {"empty prediction": 15, "near-threshold": 35, "high-confidence only": 15, "groupe/participations/developpement": 25, "SNC/Cie/Ets/EI": 10}
    chosen, stratum = [], {}
    for k, n in want.items():
        cand = sorted(pools[k] - set(chosen))
        pick = list(rng.choice(cand, min(n, len(cand)), replace=False))
        chosen += pick; stratum.update({int(i): k for i in pick})
    rows = []
    for i1 in chosen:
        cand = S[S.i1 == i1][["j", "src", "p"]]
        top = B[B.i1 == i1].sort_values("cs_all", ascending=False).head(8)[["j", "src"]]
        nb = pd.concat([cand, top]).drop_duplicates(["j", "src"])
        for _, r in nb.iterrows():
            b = d[int(r.src)].iloc[int(r.j)]
            rows.append(dict(i1=int(i1), stratum=stratum[int(i1)], s1_id=d[1].entity_id.values[i1], s1_name=d[1].name.values[i1],
                             s1_addr=d[1].addr.values[i1], rec_id=b.entity_id, src=int(r.src), rec_name=b["name"], rec_addr=b.addr,
                             p_v3=float(r.p) if not np.isnan(r.p) else np.nan, label="", why=""))
    P = pd.DataFrame(rows)
    P.to_csv(f"{DIAG}/france_audit_pairs.tsv", sep="\t", index=False)
    with open(f"{DIAG}/france_audit_read.txt", "w") as f:
        for k, (i1, G) in enumerate(P.groupby("i1", sort=False)):
            f.write(f"\n#{k} [{G.stratum.iloc[0]}] S1 {G.s1_id.iloc[0]}: {G.s1_name.iloc[0]} | {G.s1_addr.iloc[0]}\n")
            for q, r in enumerate(G.itertuples()):
                pv = "  -  " if np.isnan(r.p_v3) else f"{r.p_v3:.2f}"
                f.write(f"   {q:2d} S{r.src} p={pv}  {r.rec_name} | {r.rec_addr}\n")
    print(f"sampled {P.i1.nunique()} France S1, {len(P)} pairs:", P.drop_duplicates('i1').stratum.value_counts().to_dict())


def score(work, scored, thr):
    P = pd.read_csv(f"{DIAG}/france_audit_pairs.tsv", sep="\t", dtype={"label": str}, keep_default_na=False)
    d, S, _ = load(work, scored)
    rid = {s: pd.Series(np.arange(len(d[s])), index=d[s].entity_id.values) for s in (2, 3)}
    s1 = pd.Series(np.arange(len(d[1])), index=d[1].entity_id.values)
    P["i1n"] = s1.reindex(P.s1_id.values).values
    P["jn"] = [rid[s].get(r, -1) for s, r in zip(P.src, P.rec_id)]
    key = lambda a, b, c: a.astype(np.int64) * 100_000_000_000 + b.astype(np.int64) * 10 + c
    pred = set(key(S.i1.values[S.p.values >= thr], S.j.values[S.p.values >= thr], S.src.values[S.p.values >= thr]))
    P["pred"] = np.isin(key(P.i1n.values, P.jn.values, P.src.values), list(pred))
    L = P[P.label.isin(["0", "1"])].copy(); L["y"] = (L.label == "1")
    other = S[(S.p >= thr) & S.i1.isin(set(P.i1n))]                    # predictions outside the audited neighbourhood count as FP
    extra = other[~np.isin(key(other.i1.values, other.j.values, other.src.values), key(P.i1n.values, P.jn.values, P.src.values))]
    g = L.groupby("i1n")
    tp = g.apply(lambda x: int((x.y & x.pred).sum())); npred = g.pred.sum() + extra.groupby("i1").size().reindex(tp.index).fillna(0)
    nt = g.y.sum()
    m, f = macro_f05(tp.values, npred.values, nt.values)
    strat = L.drop_duplicates("i1n").set_index("i1n").stratum.reindex(tp.index).values
    out = {"macro_F0.5_vs_manual": round(m, 4), "S1": len(tp), "labelled_pairs": len(L), "uncertain_pairs": int((P.label == "?").sum())}
    for s in pd.unique(strat):
        out[s] = round(float(f[strat == s].mean()), 4)
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3], float(sys.argv[4]))
