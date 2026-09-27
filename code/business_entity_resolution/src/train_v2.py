"""v2 training: cross-fitted cheap candidate filter -> context recompute on kept rows -> final matcher -> validation metrics."""
import os, sys, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from features import EXTRA, recontext
from cheap_filter import cheap_feats, select
from decide import score_selection

DROP = ["y", "i1", "j", "src_id", "p"]
CP = dict(objective="binary:logistic", tree_method="hist", max_depth=6, eta=0.2, subsample=0.8, colsample_bytree=0.8,
          min_child_weight=5, nthread=10)


def cheap_X(D):
    d = D.rename(columns={"src": "src01"})
    d["src"] = d["src_id"]
    return cheap_feats(d)


def fold_of(i1):
    return (pd.util.hash_array(np.asarray(i1, dtype=np.int64)) % 2).astype(int)


if __name__ == "__main__":
    tau_target_recall = float(sys.argv[1]) if len(sys.argv) > 1 else 0.997
    rounds_m, eta_m = int(sys.argv[2]), float(sys.argv[3])
    tr = pd.read_parquet(f"{WORK}/ds2_train.parquet"); va = pd.read_parquet(f"{WORK}/ds2_val.parquet")
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"]
    t = time.time()
    Xtr, Xva = cheap_X(tr), cheap_X(va)
    # cross-fitted cheap filter probabilities on train; full-train model for val/test
    fold = fold_of(tr.i1.values); ptr = np.zeros(len(tr), np.float32)
    for f in (0, 1):
        b = xgb.train(CP, xgb.DMatrix(Xtr[fold != f], label=tr.y[fold != f]), 120)
        ptr[fold == f] = b.predict(xgb.DMatrix(Xtr[fold == f]))
    cheap = xgb.train(CP, xgb.DMatrix(Xtr, label=tr.y), 120)
    cheap.save_model(f"{WORK}/cheap_xgb.json"); json.dump(list(Xtr.columns), open(f"{WORK}/cheap_feats.json", "w"))
    pva = cheap.predict(xgb.DMatrix(Xva))
    print(f"cheap filter trained ({time.time()-t:.0f}s)", flush=True)
    tot = va.y.sum(); n_s1 = va.i1.nunique()
    best = None
    for tau in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03):
        k = select(pva, va.i1.values, tau)
        rec = va.y[k].sum() / tot
        print(f"  tau {tau}: cand/S1 {k.sum()/n_s1:5.1f}  recall of in-blocking-cand true pairs {rec:.4f}", flush=True)
        if rec >= tau_target_recall:
            best = tau
    tau = best if best is not None else 0.0005
    json.dump({"tau": tau, "min_keep": 2}, open(f"{WORK}/cheap_cfg.json", "w"))
    print("chosen tau", tau, flush=True)
    ktr, kva = select(ptr, tr.i1.values, tau), select(pva, va.i1.values, tau)
    tr, va = tr[ktr].reset_index(drop=True), va[kva].reset_index(drop=True)
    print("kept train", len(tr), "val", len(va), f"({len(va)/n_s1:.1f} cand/S1)", flush=True)
    data = load_split("train"); F.set_data(data)
    t = time.time()
    tr = recontext(tr, tr.i1.values, tr.j.values, tr.src_id.values); va = recontext(va, va.i1.values, va.j.values, va.src_id.values)
    print(f"context recomputed ({time.time()-t:.0f}s)", flush=True)
    tr.to_parquet(f"{WORK}/ds3_train.parquet"); va.to_parquet(f"{WORK}/ds3_val.parquet")
    feats = [c for c in tr.columns if c not in DROP]
    params = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], tree_method="hist", max_depth=8, eta=eta_m,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256)
    dtr = xgb.DMatrix(tr[feats], label=tr.y); dva = xgb.DMatrix(va[feats], label=va.y)
    t = time.time(); bst = xgb.train(params, dtr, rounds_m, evals=[(dva, "val")], verbose_eval=100)
    p = bst.predict(dva)
    bst.save_model(f"{WORK}/model3_xgb.json"); json.dump(feats, open(f"{WORK}/model3_feats.json", "w"))
    va["p"] = p; va[["i1", "j", "src_id", "y", "p"]].to_parquet(f"{WORK}/val_pred3.parquet")
    res = [(score_selection(va, p >= th, mva)[0], th) for th in np.arange(0.3, 0.96, 0.05)]
    for m, th in res:
        print(f"RESULT v2 thr {th:.2f}: macro F0.5 {m:.5f}")
    print("BEST", max(res), f"| recall ceiling {va.y.sum()/mva.n_true.sum():.4f} | cand/S1 {len(va)/n_s1:.1f} | {time.time()-t:.0f}s")
    print(pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False).head(20).round(1).to_string())
