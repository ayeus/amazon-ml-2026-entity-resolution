"""Candidate-ceiling analysis on the fixed validation S1: where true pairs are lost (never blocked vs dropped by the
candidate filter), what they look like, and the oracle macro F0.5 under filter variants.  Read-only w.r.t. artifacts."""
import os, sys, glob, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from cheap_filter import cheap_feats, select
from make_big import truth_key
from decide import macro_f05

key = lambda a, b, c: a.astype(np.int64) * 100_000_000_000 + b.astype(np.int64) * 10 + c

if __name__ == "__main__":
    data = load_split("train", with_X=False); F.set_data(data)
    ids = {s: data[s][0].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); m = meta[meta.split == "val"].reset_index(drop=True); val = set(m.i1)
    parts = []
    for f in sorted(glob.glob(f"{WORK}/block_train/*.parquet")):
        b = pd.read_parquet(f); keep = b.i1.isin(val).values
        if keep.any():
            c = pd.read_parquet(f"{WORK}/comp_train/{os.path.basename(f)}")
            parts.append(pd.concat([b[keep].reset_index(drop=True), c[keep].reset_index(drop=True)], axis=1))
    T = pd.concat(parts, ignore_index=True)
    tk = truth_key(ids); T["y"] = np.isin(key(T.i1.values, T.j.values, T.src.values), tk)
    cheap = xgb.Booster(); cheap.load_model(f"{WORK}/cheap3_xgb.json"); cf = json.load(open(f"{WORK}/cheap3_feats.json"))
    T["pc"] = cheap.predict(xgb.DMatrix(cheap_feats(T)[cf].to_numpy(np.float32), feature_names=cf))
    n_true = m.n_true.values
    def oracle(sel):
        inc = T[sel].groupby("i1").y.sum().reindex(m.i1).fillna(0).values
        return inc.sum() / n_true.sum(), macro_f05(inc, inc, n_true)[0], sel.sum() / len(m)
    for nm, sel in [("blocking only (no filter)", np.ones(len(T), bool)), ("current filter tau .015 keep2", select(T.pc.values, T.i1.values, 0.015, 2))] + \
                   [(f"filter tau {t} keep{k}", select(T.pc.values, T.i1.values, t, k)) for t in (0.004, 0.008) for k in (2, 3, 4)]:
        r, o, c = oracle(sel); print(f"{nm:32s} recall {r:.4f}  oracle F0.5 {o:.5f}  cand/S1 {c:.2f}", flush=True)
    # typed misses
    tp = pd.DataFrame({"k": tk}); tp = tp[(tp.k // 100_000_000_000).isin(val)]
    blocked = set(key(T.i1.values, T.j.values, T.src.values)); kept = set(key(T.i1.values, T.j.values, T.src.values)[select(T.pc.values, T.i1.values, 0.015, 2)])
    tp["stage"] = np.where(~tp.k.isin(blocked), "never blocked", np.where(~tp.k.isin(kept), "dropped by filter", "kept"))
    print("\n", tp.stage.value_counts().to_string(), flush=True)
    ms = tp[tp.stage != "kept"].copy()
    ms["i1"] = ms.k // 100_000_000_000; ms["src"] = ms.k % 10; ms["j"] = (ms.k // 10) % 10_000_000_000
    d = {s: data[s][0] for s in (1, 2, 3)}
    a_core = d[1].core.values[ms.i1.values]; a_at = d[1].atok.values[ms.i1.values]
    b_core = np.empty(len(ms), object); b_at = np.empty(len(ms), object)
    for s in (2, 3):
        mk = ms.src.values == s; b_core[mk] = d[s].core.values[ms.j.values[mk]]; b_at[mk] = d[s].atok.values[ms.j.values[mk]]
    from rapidfuzz.process import cpdist
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz import fuzz
    ms["jw"] = cpdist(a_core, b_core, scorer=JaroWinkler.normalized_similarity, workers=-1)
    ms["atsr"] = cpdist(a_at, b_at, scorer=fuzz.token_set_ratio, workers=-1) / 100
    ms["noaddr"] = (b_at == "")
    cnt = pd.Series([x.replace(" ", "") for x in d[1].core.values]).value_counts()
    ms["name_mult"] = pd.Series([x.replace(" ", "") for x in a_core]).map(cnt).values
    ms["type"] = np.select([ms.noaddr & (ms.name_mult > 1), ms.noaddr, (ms.jw < 0.6) & (ms.atsr >= 0.6), ms.jw < 0.6, ms.atsr < 0.5],
                           ["no address, name shared by >1 S1", "no address, unique name", "alias/junk name, address ok", "name and address both weak", "name ok, address weak"], "name ok, address ok")
    print("\n", pd.crosstab(ms.type, ms.stage).to_string(), flush=True)
    ms.to_pickle(f"{WORK}/ceiling_misses.pkl")
