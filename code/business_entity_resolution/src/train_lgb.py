"""Task 9: LightGBM on the t45 feature set vs the XGBoost models, and ensembles, on the same test-like validation.

Training rows: whole S1 entities with crc32(i1) % 10 < KEEP (default 7 -> ~70% of the 10.1M rows) from EVERY file (the
file order is India first, so taking the first files would skew the country mix).  Matrix preallocated (memory).
Writes model_lgb_t45.txt, val_pred_lgb_t45.parquet in WORK.  Run with BER_WORK=<root>/work_train.
"""
import os, sys, glob, time, zlib
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("BER_DERIVE", "t45")
from prep import WORK
from train_t2 import DS, derive, read, DERIVED
from train_v3 import report
from decide import score_selection
import json

KEEP = int(os.environ.get("BER_LGB_KEEP", "7"))


def keep_mask(i1):
    u = np.unique(i1); k = np.array([zlib.crc32(str(int(x)).encode()) % 10 < KEEP for x in u])
    return np.isin(i1, u[k])


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    feats = json.load(open(f"{WORK}/model_v3_t45_feats.json"))
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"].reset_index(drop=True)
    ctry = pd.read_pickle(f"{WORK}/train_s1.pkl").country.to_numpy(dtype=object)[mva.i1.values]
    files = sorted(glob.glob(f"{DS}/train_*.parquet"))
    masks = [keep_mask(pd.read_parquet(f, columns=["i1"]).i1.to_numpy()) for f in files]
    N = int(sum(m.sum() for m in masks)); X = np.empty((N, len(feats)), np.float32); y = np.empty(N, np.float32); a = 0
    t = time.time()
    for f, m in zip(files, masks):
        d = derive(read(f))[feats + ["y"]][m]
        X[a:a + len(d)] = d[feats].to_numpy(np.float32); y[a:a + len(d)] = d.y.to_numpy(np.float32); a += len(d)
    assert a == N
    print(f"train rows {N} of {sum(len(m) for m in masks)} ({KEEP}0% of S1), features {len(feats)}, loaded in {time.time()-t:.0f}s", flush=True)
    vfiles = sorted(glob.glob(f"{DS}/val_*.parquet"))
    va = pd.concat([derive(read(f))[feats + ["i1", "j", "src_id", "y"]] for f in vfiles], ignore_index=True)
    base = pd.read_parquet(f"{WORK}/val_pred_v3.parquet"); assert (base.j.values == va.j.values).all()
    params = dict(objective="binary", learning_rate=0.08, num_leaves=255, max_depth=-1, min_data_in_leaf=20, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, max_bin=255, num_threads=10, verbose=-1, seed=0)
    t = time.time()
    ds = lgb.Dataset(X, label=y, feature_name=feats, free_raw_data=True)
    bst = lgb.train(params, ds, rounds)
    del ds, X, y
    print(f"trained {rounds} rounds in {time.time()-t:.0f}s", flush=True)
    bst.save_model(f"{WORK}/model_lgb_t45.txt")
    p = bst.predict(va[feats].to_numpy(np.float32))
    va[["i1", "j", "src_id", "y"]].assign(p=p).to_parquet(f"{WORK}/val_pred_lgb_t45.parquet")
    report("LightGBM t45", va, p, mva, ctry)
    P = {n: pd.read_parquet(f"{WORK}/val_pred_v3{tg}.parquet").p.values for n, tg in [("base", ""), ("t4", "_t4"), ("t5", "_t5"), ("t45", "_t45")]}
    P["lgb"] = p
    D = va[["i1", "y"]]
    for combo in [("t45",), ("lgb",), ("t45", "lgb"), ("t4", "t5", "t45"), ("t4", "t5", "t45", "lgb"), ("t4", "t5", "lgb")]:
        q = np.mean([P[c] for c in combo], 0)
        r = {th: score_selection(D, q >= th, mva)[0] for th in (0.55, 0.6, 0.62, 0.65, 0.68, 0.7, 0.75)}
        b = max(r, key=r.get)
        print(f"ENSEMBLE mean{combo}: best {r[b]:.5f} @ {b} | " + " ".join(f"{k}:{v:.5f}" for k, v in r.items()), flush=True)
