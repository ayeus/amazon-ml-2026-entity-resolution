"""Train matcher with stage-2 features (word identity, address clusters, number geometry) and evaluate:
plain validation, per-country, and test-like validation (entities reweighted to test's alt-cluster profile)."""
import os, sys, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import score_selection, macro_f05
import wordstats as ws

NEW = ws.PAIR_NAMES + ws.CLUSTER_NAMES


def load(name):
    D = pd.read_parquet(f"{WORK}/ds4_{name}.parquet"); X = pd.read_parquet(f"{WORK}/ds5x_{name}.parquet")
    assert len(D) == len(X)
    return pd.concat([D, X], axis=1)


def alt_profile(D):
    alt = ((D.cl_is_alt == 1) & (D.cl_size >= 2)).astype(int)
    return pd.Series(alt.values).groupby(D.i1.values).sum().clip(upper=4)


def per_entity_f(D, sel, meta):
    tp = pd.Series(D.y.values * sel).groupby(D.i1.values).sum(); n = pd.Series(sel.astype(int)).groupby(D.i1.values).sum()
    return macro_f05(tp.reindex(meta.i1.values).fillna(0).values, n.reindex(meta.i1.values).fillna(0).values, meta.n_true.values)[1]


if __name__ == "__main__":
    rounds, eta, depth, tag = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    tr, va = load("train"), load("val")
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"].reset_index(drop=True)
    feats = json.load(open(f"{WORK}/model3_feats.json")) + NEW
    json.dump(feats, open(f"{WORK}/model5_feats.json", "w"))
    params = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], tree_method="hist", max_depth=depth, eta=eta,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256)
    t = time.time()
    bst = xgb.train(params, xgb.DMatrix(tr[feats], label=tr.y), rounds, evals=[(xgb.DMatrix(va[feats], label=va.y), "val")], verbose_eval=100)
    bst.save_model(f"{WORK}/model_{tag}.json")
    p = bst.predict(xgb.DMatrix(va[feats])); va["p"] = p
    va[["i1", "j", "src_id", "y", "p"]].to_parquet(f"{WORK}/val_pred_{tag}.parquet")
    print(f"trained {rounds} rounds in {time.time()-t:.0f}s", flush=True)
    # test sample profile
    T = pd.concat([pd.read_parquet(f"{WORK}/diag_test_sample.parquet"), pd.read_parquet(f"{WORK}/diag_test_sample_x.parquet")], axis=1)
    T["p"] = bst.predict(xgb.DMatrix(T[feats]))
    vprof = alt_profile(va).reindex(mva.i1.values).fillna(0).astype(int)
    ctry = pd.read_pickle(f"{WORK}/train_s1.pkl").country.to_numpy(dtype=object)[mva.i1.values]
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        f = per_entity_f(va, p >= th, mva)
        line = f"RESULT {tag} thr {th:.2f}: val {f.mean():.5f} (US {f[ctry=='US'].mean():.5f}, India {f[ctry=='India'].mean():.5f})"
        for c in ("US", "India", "France"):
            tp_ = alt_profile(T[T.country == c]).value_counts(normalize=True); vp_ = vprof.value_counts(normalize=True)
            w = vprof.map((tp_ / vp_).fillna(0)).fillna(0).values
            line += f" | test-like[{c}] {(f * w).sum() / w.sum():.5f}"
        print(line, flush=True)
    alt = (T.cl_is_alt == 1) & (T.cl_size >= 2)
    va_alt = (va.cl_is_alt == 1) & (va.cl_size >= 2)
    print(f"\nalt-cluster pairs: val true-rate {va.y[va_alt].mean():.3f}, val pred>=0.7 {(va.p[va_alt]>=0.7).mean():.3f} | test pred>=0.7 by country:",
          T[alt].groupby("country").p.apply(lambda x: round((x >= 0.7).mean(), 3)).to_dict())
    print("test predicted matches/S1 at 0.7:", T.groupby("country").apply(lambda g: round((g.p >= 0.7).sum() / g.i1.nunique(), 3), include_groups=False).to_dict())
    imp = pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False)
    print("\ntop features:"); print(imp.head(20).round(1).to_string())
    print("\nnew features gain rank:", {k: int(np.flatnonzero(imp.index == k)[0]) + 1 for k in NEW if k in imp.index})
