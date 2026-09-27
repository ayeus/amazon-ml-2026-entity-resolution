"""XGBoost vs LightGBM on identical data: same training-row subsample (whole files, i.e. whole S1 entities), same features,
same test-like validation; scored by macro F0.5 at each model's best validation threshold."""
import os, sys, glob, time, json
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from train_v3 import load, entity_f, DS, DSX

if __name__ == "__main__":
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    feats = json.load(open(f"{WORK}/model_v3_feats.json"))
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"].reset_index(drop=True)
    va = load("val", feats + ["i1", "j", "src_id", "y"])
    parts, n = [], 0
    for f in sorted(glob.glob(f"{DS}/train_*.parquet")):
        d = pd.concat([pd.read_parquet(f), pd.read_parquet(f"{DSX}/{os.path.basename(f)}")], axis=1)[feats + ["y"]]
        parts.append(d.astype(np.float32)); n += len(d)
        if n >= max_rows:
            break
    tr = pd.concat(parts, ignore_index=True); del parts
    X, y = tr[feats].to_numpy(np.float32), tr.y.to_numpy(np.float32); del tr
    Xv = va[feats].to_numpy(np.float32)
    print(f"train rows {len(X)} | val rows {len(Xv)} | features {len(feats)}", flush=True)
    res = {}
    t = time.time()
    b = xgb.train(dict(objective="binary:logistic", tree_method="hist", max_depth=8, eta=0.08, subsample=0.8, colsample_bytree=0.7,
                       min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256),
                  xgb.DMatrix(X, label=y, feature_names=feats), rounds)
    res["xgboost"] = (b.predict(xgb.DMatrix(Xv, feature_names=feats)), time.time() - t)
    t = time.time()
    m = lgb.train(dict(objective="binary", learning_rate=0.08, num_leaves=255, max_depth=-1, min_data_in_leaf=20, feature_fraction=0.7,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=10, max_bin=255, verbose=-1),
                  lgb.Dataset(X, label=y, feature_name=feats, free_raw_data=True), rounds)
    res["lightgbm"] = (m.predict(Xv), time.time() - t)
    for name, (p, secs) in res.items():
        best = max((entity_f(va, p >= th, mva).mean(), th) for th in np.arange(0.4, 0.96, 0.05))
        at70 = entity_f(va, p >= 0.7, mva).mean()
        print(f"RESULT {name}: best macro F0.5 {best[0]:.5f} @ {best[1]:.2f} | @0.70 {at70:.5f} | train {secs:.0f}s", flush=True)
