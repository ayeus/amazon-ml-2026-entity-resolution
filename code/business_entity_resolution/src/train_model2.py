"""Ablation + final matcher training on ds2_{train,val} (base + name-ambiguity + competition features)."""
import os, sys, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import score_selection, select_expected_f05
from features import EXTRA

DROP = ["y", "i1", "j", "src_id"]


def evalp(va, p, mva):
    best = max(((score_selection(va, p >= t, mva)[0], t) for t in np.arange(0.3, 0.96, 0.05)))
    return best


if __name__ == "__main__":
    mode = sys.argv[1]            # ablate | final
    rounds = int(sys.argv[2]); eta = float(sys.argv[3])
    tr = pd.read_parquet(f"{WORK}/ds2_train.parquet"); va = pd.read_parquet(f"{WORK}/ds2_val.parquet")
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"]
    allf = [c for c in tr.columns if c not in DROP]
    comp = [c for c in allf if c.startswith("comp_")]
    base = [c for c in allf if c not in EXTRA and c not in comp]
    sets = {"base": base, "base+name": base + EXTRA, "base+name+comp": base + EXTRA + comp} if mode == "ablate" else {"final": base + EXTRA + comp}
    params = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], tree_method="hist", max_depth=8, eta=eta,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256)
    for name, feats in sets.items():
        t = time.time()
        dtr = xgb.DMatrix(tr[feats], label=tr.y); dva = xgb.DMatrix(va[feats], label=va.y)
        bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")], verbose_eval=100)
        p = bst.predict(dva)
        m, thr = evalp(va, p, mva)
        print(f"RESULT {name}: {len(feats)} feats, {rounds} rounds eta {eta}, best global-threshold macro F0.5 = {m:.5f} @ thr {thr:.2f} ({time.time()-t:.0f}s)", flush=True)
        if mode == "final":
            bst.save_model(f"{WORK}/model2_xgb.json"); json.dump(feats, open(f"{WORK}/model2_feats.json", "w"))
            va["p"] = p; va[["i1", "j", "src_id", "y", "p"]].to_parquet(f"{WORK}/val_pred2.parquet")
            imp = pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False).head(20)
            print(imp.round(1).to_string())
