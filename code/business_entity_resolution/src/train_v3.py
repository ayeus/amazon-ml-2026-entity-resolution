"""Train the matcher on the augmented (test-like) dataset and evaluate on the augmented validation S1.

Also runs the CALIBRATION CHECK: the previously submitted model (model_s5, public leaderboard 0.951) is scored on the
same validation rows.  If the augmented validation is a faithful test proxy, that score should land near 0.95.
"""
import os, sys, glob, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import macro_f05
import wordstats as ws
from features import NEW_CHANNEL_COLS

TAG = os.environ.get("BER_DS_TAG", "")          # "_smoke" for a dry run
DS, DSX = f"{WORK}/ds6{TAG}", f"{WORK}/ds6x{TAG}"


def load(nm, cols):
    fs = sorted(glob.glob(f"{DS}/{nm}_*.parquet"))
    parts = []
    for f in fs:
        a = pd.read_parquet(f); b = pd.read_parquet(f"{DSX}/{os.path.basename(f)}")
        assert len(a) == len(b), f"row mismatch {f}"
        parts.append(pd.concat([a, b], axis=1)[[c for c in cols if c in a.columns or c in b.columns]])
    return pd.concat(parts, ignore_index=True)


class FileIter(xgb.DataIter):
    """Streams the training files one at a time as float32 numpy (memory-efficient QuantileDMatrix construction;
    also avoids the pandas-3 DataFrame path that xgboost's iterator does not accept)."""

    def __init__(self, files, feats):
        self.files, self.feats, self.i = files, feats, 0
        super().__init__(cache_prefix=None)

    def next(self, input_data):
        if self.i == len(self.files):
            return False
        f = self.files[self.i]
        d = pd.concat([pd.read_parquet(f), pd.read_parquet(f"{DSX}/{os.path.basename(f)}")], axis=1)
        input_data(data=d[self.feats].to_numpy(np.float32), label=d["y"].to_numpy(np.float32), feature_names=self.feats)
        self.i += 1
        return True

    def reset(self):
        self.i = 0


def entity_f(D, sel, meta):
    tp = pd.Series(D.y.values * sel).groupby(D.i1.values).sum(); n = pd.Series(sel.astype(int)).groupby(D.i1.values).sum()
    return macro_f05(tp.reindex(meta.i1.values).fillna(0).values, n.reindex(meta.i1.values).fillna(0).values, meta.n_true.values)[1]


def report(tag, D, p, meta, ctry):
    best = (0, 0)
    for th in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        f = entity_f(D, p >= th, meta)
        best = max(best, (f.mean(), th))
        print(f"RESULT {tag} thr {th:.2f}: val {f.mean():.5f}  (US {f[ctry == 'US'].mean():.5f}, India {f[ctry == 'India'].mean():.5f})", flush=True)
    return best


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"].reset_index(drop=True)
    ctry = pd.read_pickle(f"{WORK}/train_s1.pkl").country.to_numpy(dtype=object)[mva.i1.values]
    from features import BASE_FEATS
    feats = list(BASE_FEATS) + ws.PAIR_NAMES + ws.CLUSTER_NAMES + list(NEW_CHANNEL_COLS)
    OLD = f"{WORK}/../work"
    calib = os.path.exists(f"{OLD}/model_s5.json")          # optional check against the previously submitted model
    s5_feats = json.load(open(f"{OLD}/model5_feats.json")) if calib else []
    need = sorted(set(feats) | set(s5_feats) | {"i1", "j", "src_id", "y"})
    va = load("val", need)
    if TAG:
        keep = mva.i1.isin(va.i1.unique()).values; mva = mva[keep].reset_index(drop=True); ctry = ctry[keep]
    feats = [c for c in feats if c in va.columns]
    json.dump(feats, open(f"{WORK}/model_v3{TAG}_feats.json", "w"))
    print(f"val {va.shape} ({va.i1.nunique()} S1 with candidates) | features {len(feats)}", flush=True)
    print(f"val candidate pair recall ceiling: {va.y.sum() / mva.n_true.sum():.4f}  | cand/S1 {len(va) / len(mva):.2f}", flush=True)

    # calibration check with the submitted model (leaderboard 0.951)
    b5 = (float("nan"), None); p5 = np.full(len(va), np.nan, np.float32)
    if calib:
      s5 = xgb.Booster(); s5.load_model(f"{WORK}/../work/model_s5.json")
      # model_s5 must see word features computed with ITS OWN word statistics (learned before augmentation)
      from enrich5 import arrays, features as stage2_features
      import pickle
      old_stats = pickle.load(open(f"{WORK}/../work/word_stats_full.pkl", "rb"))
      X5 = stage2_features(va[["i1", "j", "src_id", "jw"]], arrays("train"), old_stats)
      v5 = va.copy(); v5[list(X5.columns)] = X5.values
      p5 = s5.predict(xgb.DMatrix(v5[s5_feats])); del v5, X5
      b5 = report("CALIBRATION model_s5 (leaderboard 0.951)", va, p5, mva, ctry)
    import gc; gc.collect()

    params = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], tree_method="hist", max_depth=8, eta=0.08,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256)
    t = time.time()
    dtr = xgb.QuantileDMatrix(FileIter(sorted(glob.glob(f"{DS}/train_*.parquet")), feats), max_bin=256)
    print(f"train rows {dtr.num_row()} (streamed), features {dtr.num_col()}", flush=True)
    dva = xgb.DMatrix(va[feats].to_numpy(np.float32), label=va.y.to_numpy(np.float32), feature_names=feats)
    bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")], verbose_eval=100)
    bst.save_model(f"{WORK}/model_v3{TAG}.json")
    p = bst.predict(dva)
    va.assign(p=p, p5=p5)[["i1", "j", "src_id", "y", "p", "p5"]].to_parquet(f"{WORK}/val_pred_v3{TAG}.parquet")
    print(f"trained {rounds} rounds in {time.time()-t:.0f}s", flush=True)
    b3 = report("NEW model_v3", va, p, mva, ctry)
    print(f"\nSUMMARY: model_s5 best {b5[0]:.5f} @ {b5[1]}, model_v3 best {b3[0]:.5f} @ {b3[1]}", flush=True)
    open(f"{WORK}/threshold{TAG}.txt", "w").write(f"{b3[1]:.2f}\n")      # decision threshold chosen on validation only
    imp = pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False)
    print(imp.head(25).round(1).to_string())
