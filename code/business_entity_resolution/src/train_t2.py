"""Rebuild matcher (train_v3 on ds6/ds6x) plus a set of derived columns, selected with BER_DERIVE (t2 | t4).

t2: "number agrees" product column (Task 2).  t4: S1 anchor-profile counts (Task 4b), label-free, per S1 candidate set.
Writes model_v3_<set>.json / _feats.json / val_pred_v3_<set>.parquet / threshold_<set>.txt in WORK, so the rebuild
model (model_v3.json) is never overwritten.  Run with BER_WORK=<root>/work_train.
"""
import os, sys, glob, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
import wordstats as ws
from features import BASE_FEATS, NEW_CHANNEL_COLS
from train_v3 import DS, DSX, load, FileIter, report

SET = os.environ.get("BER_DERIVE", "t2")
MTAG = os.environ.get("BER_MODEL_TAG", f"_{SET}")


def derive_t2(d):
    """exact house-number match (n_rel==1) x address-token containment >= 0.8 (a_ain = |A∩B| / |A| address tokens)."""
    d["nagree_ain8"] = ((d["n_rel"].to_numpy() == 1) & (d["a_ain"].to_numpy() >= 0.8)).astype(np.float32)
    return d


def derive_t4(d):
    """S1 anchor profile: how many strong (address-backed, name-similar) candidates the S1 already has from the same /
    the other source as B, and how many other address-less candidates from B's source compete inside the S1's set.
    Needs every candidate of an S1 in the same frame (verified: no S1 spans two dataset files)."""
    src = d["src_id"].to_numpy(); i1 = d["i1"].to_numpy()
    strong = ((d["b_noaddr"].to_numpy() == 0) & (d["a_jac"].to_numpy() >= 0.5) & (d["jw"].to_numpy() >= 0.85)).astype(np.float32)
    noad = (d["b_noaddr"].to_numpy() == 1).astype(np.float32)
    t = pd.DataFrame({"i1": i1, "s2": strong * (src == 2), "s3": strong * (src == 3), "n2": noad * (src == 2), "n3": noad * (src == 3)})
    g = t.groupby("i1")[["s2", "s3", "n2", "n3"]].transform("sum")
    is2 = src == 2
    d["anc_same"] = np.where(is2, g.s2 - t.s2, g.s3 - t.s3).astype(np.float32)
    d["anc_other"] = np.where(is2, g.s3, g.s2).astype(np.float32)
    d["noad_same"] = np.where(is2, g.n2 - t.n2, g.n3 - t.n3).astype(np.float32)
    return d


from ngram import NG_NAMES
# set -> (derive fn, derived columns, row-aligned extra dataset folders read next to ds6/ds6x)
DERIVE = {"base": (lambda d: d, [], []), "t2": (derive_t2, ["nagree_ain8"], []), "t4": (derive_t4, ["anc_same", "anc_other", "noad_same"], []),
          "t5": (lambda d: d, NG_NAMES, [f"{DS}n"]),
          "t45": (derive_t4, ["anc_same", "anc_other", "noad_same"] + NG_NAMES, [f"{DS}n"])}
derive, DERIVED, EXTRA = DERIVE[SET]


def read(f):
    b = os.path.basename(f)
    return pd.concat([pd.read_parquet(f), pd.read_parquet(f"{DSX}/{b}")] + [pd.read_parquet(f"{x}/{b}") for x in EXTRA], axis=1)


class DerivedIter(FileIter):
    def next(self, input_data):
        if self.i == len(self.files):
            return False
        f = self.files[self.i]
        d = derive(read(f))
        input_data(data=d[self.feats].to_numpy(np.float32), label=d["y"].to_numpy(np.float32), feature_names=self.feats)
        self.i += 1
        return True


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"].reset_index(drop=True)
    ctry = pd.read_pickle(f"{WORK}/train_s1.pkl").country.to_numpy(dtype=object)[mva.i1.values]
    base = list(BASE_FEATS) + ws.PAIR_NAMES + ws.CLUSTER_NAMES + list(NEW_CHANNEL_COLS)
    va = load("val", sorted(set(base) | {"i1", "j", "src_id", "y", "b_noaddr"}))
    if EXTRA:
        ex = pd.concat([pd.concat([pd.read_parquet(f"{x}/{os.path.basename(f)}") for x in EXTRA], axis=1)
                        for f in sorted(glob.glob(f"{DS}/val_*.parquet"))], ignore_index=True)
        assert len(ex) == len(va); va = pd.concat([va, ex], axis=1)
    va = derive(va)
    feats = [c for c in base if c in va.columns] + DERIVED
    assert feats[:-len(DERIVED)] == json.load(open(f"{WORK}/model_v3_feats.json")), "base features differ from rebuild model"
    json.dump(feats, open(f"{WORK}/model_v3{MTAG}_feats.json", "w"))
    print(f"val {va.shape} | features {len(feats)} (+{DERIVED})", flush=True)

    old = pd.read_parquet(f"{WORK}/val_pred_v3.parquet")          # baseline: the rebuild model on the same rows
    assert (old.i1.values == va.i1.values).all() and (old.j.values == va.j.values).all()
    b0 = report("BASELINE rebuild model_v3", va, old.p.values, mva, ctry)

    params = dict(objective="binary:logistic", eval_metric=["logloss", "aucpr"], tree_method="hist", max_depth=8, eta=0.08,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0, nthread=10, max_bin=256)
    t = time.time()
    dtr = xgb.QuantileDMatrix(DerivedIter(sorted(glob.glob(f"{DS}/train_*.parquet")), feats), max_bin=256)
    print(f"train rows {dtr.num_row()} (streamed), features {dtr.num_col()}", flush=True)
    dva = xgb.DMatrix(va[feats].to_numpy(np.float32), label=va.y.to_numpy(np.float32), feature_names=feats)
    bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")], verbose_eval=100)
    bst.save_model(f"{WORK}/model_v3{MTAG}.json")
    p = bst.predict(dva)
    va.assign(p=p)[["i1", "j", "src_id", "y", "p"]].to_parquet(f"{WORK}/val_pred_v3{MTAG}.parquet")
    print(f"trained {rounds} rounds in {time.time()-t:.0f}s", flush=True)
    b1 = report(f"NEW model_v3{MTAG}", va, p, mva, ctry)
    print(f"\nSUMMARY: baseline best {b0[0]:.5f} @ {b0[1]}, {MTAG} best {b1[0]:.5f} @ {b1[1]}", flush=True)
    open(f"{WORK}/threshold{MTAG}.txt", "w").write(f"{b1[1]:.2f}\n")
    imp = pd.Series(bst.get_score(importance_type="gain")).sort_values(ascending=False)
    print("gain rank of derived:", {c: (int(imp.index.get_loc(c)) + 1 if c in imp else None) for c in DERIVED})
    print(imp.head(15).round(1).to_string())
