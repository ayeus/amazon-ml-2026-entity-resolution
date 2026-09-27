"""Big training set: apply the cheap filter to the FULL train blocking tables (all non-validation S1), compute pair features
on the kept candidates only, label from ground truth.  Validation rows are built through the identical code path."""
import os, sys, glob, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from cheap_filter import cheap_feats, select
from evalblock import truth_pairs
from make_dataset import is_val


def truth_key(ids):
    pos = {s: pd.Series(np.arange(len(ids[s])), index=ids[s]) for s in (1, 2, 3)}
    tp = truth_pairs()
    tp["i1"] = pos[1].reindex(tp.source1_entity_id.values).values
    tp["src"] = tp.m.str[1].astype(int); tp["j"] = -1
    for s in (2, 3):
        mk = tp.src.values == s; tp.loc[mk, "j"] = pos[s].reindex(tp.m.values[mk]).values
    return tp.i1.values.astype(np.int64) * 100_000_000_000 + tp.j.values.astype(np.int64) * 10 + tp.src.values


def process(tab, cheap, cf, cfg, feats_keep=None):
    pc = cheap.predict(xgb.DMatrix(cheap_feats(tab)[cf].to_numpy(np.float32), feature_names=cf))
    cand = tab[select(pc, tab.i1.values, cfg["tau"], cfg["min_keep"])].reset_index(drop=True)
    X = F.pair_features(cand)
    E = F.extra_features(cand.i1.to_numpy(), cand.j.to_numpy(), cand.src.to_numpy())
    X = pd.concat([X, E, cand[[c for c in cand.columns if c.startswith("comp_") or c.startswith("cs_")]].reset_index(drop=True)], axis=1)
    X["i1"] = cand.i1.values; X["j"] = cand.j.values; X["src_id"] = cand.src.values
    return X


if __name__ == "__main__":
    data = load_split("train"); F.set_data(data)
    ids = {s: data[s][0].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    tk = truth_key(ids)
    val_mask = is_val(ids[1])
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); val_rows = set(meta[meta.split == "val"].i1.values)
    cheap = xgb.Booster(); cheap.load_model(f"{WORK}/cheap_xgb.json"); cf = json.load(open(f"{WORK}/cheap_feats.json"))
    cfg = json.load(open(f"{WORK}/cheap_cfg.json"))
    tr_parts, va_parts = [], []
    for bf in sorted(glob.glob(f"{WORK}/block_train/*.parquet")):
        name = os.path.basename(bf); t = time.time()
        B = pd.read_parquet(bf); C = pd.read_parquet(f"{WORK}/comp_train/{name}")
        tab = pd.concat([B, C], axis=1); del B, C
        for part, mask in ((tr_parts, ~val_mask[tab.i1.values]), (va_parts, np.isin(tab.i1.values, list(val_rows)))):
            sub = tab[mask].reset_index(drop=True)
            if len(sub) == 0:
                continue
            X = process(sub, cheap, cf, cfg)
            key = X.i1.values.astype(np.int64) * 100_000_000_000 + X.j.values.astype(np.int64) * 10 + X.src_id.values
            X["y"] = np.isin(key, tk).astype(np.int8)
            part.append(X)
        print(f"{name}: train rows so far {sum(len(x) for x in tr_parts)}, val rows {sum(len(x) for x in va_parts)}, {time.time()-t:.0f}s", flush=True)
    for nm, parts in (("val", va_parts), ("train", tr_parts)):
        D = pd.concat(parts, ignore_index=True)
        D.to_parquet(f"{WORK}/ds4_{nm}.parquet"); print(nm, D.shape, "pos", D.y.sum(), flush=True)
