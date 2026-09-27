"""Stage 3b/4: train the pair matcher (XGBoost, Apache-2.0) and evaluate decision rules on the S1-level validation split."""
import os, sys, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import score_selection, select_expected_f05

DROP = ["y", "i1", "j", "src_id"]


def load(name):
    D = pd.read_parquet(f"{WORK}/ds_{name}.parquet")
    return D


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    tr, va = load("train"), load("val")
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl")
    mva = meta[meta.split == "val"]
    feats = [c for c in tr.columns if c not in DROP]
    print("train", tr.shape, "pos rate", tr.y.mean().round(4), "| val", va.shape, "pos", va.y.sum(), "feats", len(feats), flush=True)
    dtr = xgb.DMatrix(tr[feats], label=tr.y); dva = xgb.DMatrix(va[feats], label=va.y)
    params = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], tree_method="hist", max_depth=8, eta=0.08,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256)
    t = time.time()
    bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=50)
    print(f"trained {bst.best_iteration+1} rounds in {time.time()-t:.0f}s", flush=True)
    bst.save_model(f"{WORK}/model_xgb.json"); json.dump(feats, open(f"{WORK}/model_feats.json", "w"))
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    va["p"] = p
    print("val logloss/aucpr ok. Candidate recall on val (pairs):", va.y.sum(), "of", int(mva.n_true.sum()),
          "=", round(va.y.sum() / mva.n_true.sum(), 4))
    res = {}
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m, _ = score_selection(va, (p >= thr), mva); res[f"thr{thr}"] = m
    for mp in (0.0, 0.2, 0.5):
        m, _ = score_selection(va, select_expected_f05(va, p, miss_prior=mp), mva); res[f"expF_miss{mp}"] = m
    for k, v in res.items():
        print(f"macro F0.5 {k}: {v:.5f}")
    imp = pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False).head(15)
    print(imp.round(1).to_string())
