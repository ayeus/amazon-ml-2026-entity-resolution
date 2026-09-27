import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from blocking import load_split, generate
from prep import DATA


def truth_pairs():
    g = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    g = g[g.matched_entity_ids != ""]
    g = g.assign(m=g.matched_entity_ids.str.split(",")).explode("m")
    return g[["source1_entity_id", "m"]]


if __name__ == "__main__":
    nS1 = int(sys.argv[1]); cap = float(sys.argv[2]); K = int(sys.argv[3])
    data = load_split("train")
    ids = {s: data[s][0].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    pos = {s: pd.Series(np.arange(len(ids[s])), index=ids[s]) for s in (1, 2, 3)}
    tp = truth_pairs()
    tp["i1"] = pos[1].reindex(tp.source1_entity_id.values).values
    tp["src"] = tp.m.str[1].astype(int)
    tp["j"] = -1
    for s in (2, 3):
        mk = tp.src.values == s
        tp.loc[mk, "j"] = pos[s].reindex(tp.m.values[mk]).values
    rng = np.random.RandomState(0)
    s1rows = np.sort(rng.choice(len(ids[1]), nS1, replace=False))
    ctry = data[1][0].country.to_numpy(dtype=object)
    for country in ("US", "India"):
        cand, idf = generate(data, country, s1rows, K2=K, K3=K, cap_frac=cap)
        m1 = s1rows[ctry[s1rows] == country]
        sel = tp[tp.i1.isin(m1)]
        key = lambda a, b, c: a.astype(np.int64) * 100_000_000_000 + b.astype(np.int64) * 10 + c
        ck = key(cand.i1.values, cand.j.values, cand.src.values)
        tk = key(sel.i1.values, sel.j.values, sel.src.values)
        hit = np.isin(tk, ck)
        print(f"{country}: S1={len(m1)} cand/S1={len(cand)/len(m1):.1f} pair recall={hit.mean():.4f}  ({hit.sum()}/{len(hit)})", flush=True)
        cand["hit"] = np.isin(ck, tk)
        print("cum recall by rank:", cand.groupby("rank").hit.sum().cumsum().div(len(sel)).round(4).to_dict(), flush=True)
