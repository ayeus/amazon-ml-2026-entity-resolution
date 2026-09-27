"""Train the matcher on the big filtered train set; evaluate macro F0.5 on the same 40k validation S1."""
import os, sys, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import score_selection

DROP = ["y", "i1", "j", "src_id", "p"]

if __name__ == "__main__":
    rounds, eta, depth, tag = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    tr = pd.read_parquet(f"{WORK}/ds4_train.parquet"); va = pd.read_parquet(f"{WORK}/ds4_val.parquet")
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"]
    feats = json.load(open(f"{WORK}/model3_feats.json"))
    miss = [c for c in feats if c not in tr.columns]; assert not miss, miss
    print("train", tr.shape, "pos rate", round(tr.y.mean(), 3), "| val", va.shape, "| S1 in val", va.i1.nunique(), flush=True)
    params = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], tree_method="hist", max_depth=depth, eta=eta,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256)
    dtr = xgb.DMatrix(tr[feats], label=tr.y); dva = xgb.DMatrix(va[feats], label=va.y)
    t = time.time(); bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")], verbose_eval=100)
    p = bst.predict(dva)
    bst.save_model(f"{WORK}/model_{tag}.json")
    va["p"] = p; va[["i1", "j", "src_id", "y", "p"]].to_parquet(f"{WORK}/val_pred_{tag}.parquet")
    res = [(score_selection(va, p >= th, mva)[0], th) for th in np.arange(0.4, 0.86, 0.05)]
    for m, th in res:
        print(f"RESULT {tag} thr {th:.2f}: macro F0.5 {m:.5f}")
    print("BEST", max(res), f"| recall ceiling {va.y.sum()/mva.n_true.sum():.4f} | cand/S1 {len(va)/mva.shape[0]:.1f} | {time.time()-t:.0f}s")
