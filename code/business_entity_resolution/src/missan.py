"""Blocking diagnostics on a fixed S1-level validation split: recall by source/country, zero-cand, miss samples."""
import os, sys, zlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from blocking import load_split, generate
from evalblock import truth_pairs

def is_val(ids):  # deterministic S1-level split: 20% validation
    return np.array([zlib.crc32(("split" + i).encode()) % 5 == 0 for i in ids])

if __name__ == "__main__":
    nval = int(sys.argv[1]); cap = float(sys.argv[2]); Kt, Kn, Ka = map(int, sys.argv[3:6])
    data = load_split("train")
    ids = {s: data[s][0].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    pos = {s: pd.Series(np.arange(len(ids[s])), index=ids[s]) for s in (1, 2, 3)}
    tp = truth_pairs()
    tp["i1"] = pos[1].reindex(tp.source1_entity_id.values).values
    tp["src"] = tp.m.str[1].astype(int); tp["j"] = -1
    for s in (2, 3):
        mk = tp.src.values == s; tp.loc[mk, "j"] = pos[s].reindex(tp.m.values[mk]).values
    val = np.flatnonzero(is_val(ids[1]))
    print("val S1", len(val), "of", len(ids[1]), "(no overlap with train by construction)")
    rng = np.random.RandomState(1); s1rows = np.sort(rng.choice(val, nval, replace=False))
    ctry = data[1][0].country.to_numpy(dtype=object)
    key = lambda a, b, c: a.astype(np.int64) * 100_000_000_000 + b.astype(np.int64) * 10 + c
    misses = []
    for country in ("US", "India"):
        cand, _ = generate(data, country, s1rows, Kt=Kt, Kn=Kn, Ka=Ka, cap_frac=cap)
        m1 = s1rows[ctry[s1rows] == country]
        sel = tp[tp.i1.isin(m1)].copy()
        ck = key(cand.i1.values, cand.j.values, cand.src.values)
        sel["hit"] = np.isin(key(sel.i1.values, sel.j.values, sel.src.values), ck)
        print(country, "overall", round(sel.hit.mean(), 4), "| S2", round(sel[sel.src == 2].hit.mean(), 4),
              "| S3", round(sel[sel.src == 3].hit.mean(), 4), flush=True)
        has = np.isin(m1, cand.i1.unique()); tr = set(sel.i1)
        zero = m1[~has]
        print(f"  S1 with zero candidates: {len(zero)}/{len(m1)}; of those with true matches: {sum(z in tr for z in zero)}")
        cnt = cand.groupby("i1").size(); print("  cand/S1 quantiles", cnt.quantile([.5, .9, .99]).to_dict())
        ck2 = pd.Series(ck); sel["ch"] = ""
        misses.append(sel[~sel.hit].assign(country=country))
    ms = pd.concat(misses)
    ms.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "work", "misses.pkl"))
    rs = np.random.RandomState(0)
    for country in ("US", "India"):
        m = ms[ms.country == country]; print(f"\n===== {country} misses: {len(m)}")
        for _, r in m.sample(min(25, len(m)), random_state=0).iterrows():
            a = data[1][0].iloc[r.i1]; b = data[int(r.src)][0].iloc[r.j]
            A = pd.read_pickle if False else None
            print(f"S1 {a.core} | {a.atok}\n   S{r.src} {b.core} | {b.atok}")
