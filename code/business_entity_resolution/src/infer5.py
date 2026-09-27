"""Test inference with the stage-2 matcher: block chunk + competition -> cheap filter (final candidate set) -> base pair
features -> word/cluster/number features (word stats learned on TRAIN only) -> probability.  One chunk at a time
(memory-bounded), resumable, atomic writes to work/scored5_{split}/."""
import os, sys, glob, time, json, pickle
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from make_big import process
from enrich5 import arrays, features as stage2


def run(split, model):
    data = load_split(split); F.set_data(data); A = arrays(split)
    cheap = xgb.Booster(); cheap.load_model(f"{WORK}/cheap_xgb.json"); cf = json.load(open(f"{WORK}/cheap_feats.json"))
    cfg = json.load(open(f"{WORK}/cheap_cfg.json"))
    bst = xgb.Booster(); bst.load_model(f"{WORK}/{model}.json"); feats = json.load(open(f"{WORK}/model5_feats.json"))
    stats = pickle.load(open(f"{WORK}/word_stats_full.pkl", "rb"))
    out_dir = f"{WORK}/scored5_{split}"; os.makedirs(out_dir, exist_ok=True)
    for bf in sorted(glob.glob(f"{WORK}/block_{split}/*.parquet")):
        name = os.path.basename(bf); out = f"{out_dir}/{name}"
        if os.path.exists(out):
            continue
        t = time.time()
        tab = pd.concat([pd.read_parquet(bf), pd.read_parquet(f"{WORK}/comp_{split}/{name}")], axis=1)
        n_full = len(tab)
        X = process(tab, cheap, cf, cfg); del tab
        S2 = stage2(X[["i1", "j", "src_id", "jw"]], A, stats)
        X = pd.concat([X.reset_index(drop=True), S2.reset_index(drop=True)], axis=1)
        p = bst.predict(xgb.DMatrix(X[feats])).astype(np.float32)
        res = X[["i1", "j", "src_id"]].rename(columns={"src_id": "src"}).assign(p=p)
        res.to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)
        print(f"{split} {name}: {n_full} -> {len(res)} cands, {time.time()-t:.0f}s", flush=True)
        del X, S2, res


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
