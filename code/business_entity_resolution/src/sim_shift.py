"""Simulate the test regime on validation: drop a fraction q of (non-validation) train S1 entities, keep ALL S2/S3
records (their records become orphans), recompute every S1-set-dependent quantity (cross-S1 competition, S1 name
statistics), and re-score the SAME 40k validation S1 with a given matcher."""
import os, sys, glob, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from competition import comp_features
from make_big import truth_key, process
from decide import score_selection

if __name__ == "__main__":
    q, model, seed = float(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
    data = load_split("train")
    n1 = len(data[1][0]); ids = {s: data[s][0].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"]; val = mva.i1.values
    rng = np.random.RandomState(seed)
    cand = np.setdiff1d(np.arange(n1), val)
    drop = rng.choice(cand, int(q * n1), replace=False)
    keep = np.ones(n1, bool); keep[drop] = False
    print(f"q={q}: dropped {len(drop)} S1 -> (S2+S3)/S1 = {(len(ids[2])+len(ids[3]))/keep.sum():.3f}", flush=True)
    F.S1_MASK = keep; F.set_data(data)
    tk = truth_key(ids)
    cheap = xgb.Booster(); cheap.load_model(f"{WORK}/cheap_xgb.json"); cf = json.load(open(f"{WORK}/cheap_feats.json"))
    cfg = json.load(open(f"{WORK}/cheap_cfg.json"))
    bst = xgb.Booster(); bst.load_model(f"{WORK}/{model}.json"); feats = json.load(open(f"{WORK}/model3_feats.json"))
    vset = set(val); out = []
    by_c = {}
    for f in sorted(glob.glob(f"{WORK}/block_train/*.parquet")):
        c = os.path.basename(f)[:-8].rsplit("_", 1)[0]; by_c.setdefault(c, []).append(f)
    for c, files in by_c.items():
        t = time.time()
        tab = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        tab = tab[keep[tab.i1.values]].reset_index(drop=True)          # dropped S1 no longer compete for records
        C = comp_features(tab)
        tab = pd.concat([tab, C], axis=1)
        tab = tab[np.isin(tab.i1.values, val)].reset_index(drop=True)
        X = process(tab, cheap, cf, cfg)
        X["p"] = bst.predict(xgb.DMatrix(X[feats]))
        key = X.i1.values.astype(np.int64) * 100_000_000_000 + X.j.values.astype(np.int64) * 10 + X.src_id.values
        X["y"] = np.isin(key, tk).astype(np.int8)
        out.append(X[["i1", "j", "src_id", "y", "p"]]); print(f"  {c}: {len(X)} rows {time.time()-t:.0f}s", flush=True)
    V = pd.concat(out, ignore_index=True)
    V.to_parquet(f"{WORK}/val_sim_q{q}_{model}.parquet")
    for th in np.arange(0.5, 0.96, 0.05):
        print(f"SIM q={q} {model} thr {th:.2f}: macro F0.5 {score_selection(V, V.p.values >= th, mva)[0]:.5f}", flush=True)
    s = V[V.p >= 0.7]
    print(f"pred matches/S1 {len(s)/len(mva):.3f} | frac p in [0.3,0.9] {((V.p>.3)&(V.p<.9)).mean():.3f} | FP pairs {int(((V.p>=0.7)&(V.y==0)).sum())}")
