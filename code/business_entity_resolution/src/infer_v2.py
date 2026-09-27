"""Stage 5 (final): per-chunk inference on a split from the saved full-blocking tables.
block chunk + competition chunk -> cheap filter (=candidate set) -> pair features -> matcher probability.
Writes work/scored_{split}/{country}_{k}.parquet with the kept candidates only (i1, j, src, p).  Resumable + atomic."""
import os, sys, glob, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from cheap_filter import cheap_feats, select


def run(split):
    data = load_split(split); t = time.time(); F.set_data(data); print("pool stats", round(time.time() - t), "s", flush=True)
    cheap = xgb.Booster(); cheap.load_model(f"{WORK}/cheap_xgb.json"); cf = json.load(open(f"{WORK}/cheap_feats.json"))
    cfg = json.load(open(f"{WORK}/cheap_cfg.json"))
    bst = xgb.Booster(); bst.load_model(f"{WORK}/{os.environ.get('BER_MODEL', 'model3_xgb')}.json"); feats = json.load(open(f"{WORK}/model3_feats.json"))
    os.makedirs(f"{WORK}/scored_{split}", exist_ok=True)
    for bf in sorted(glob.glob(f"{WORK}/block_{split}/*.parquet")):
        name = os.path.basename(bf)
        out = f"{WORK}/scored_{split}/{name}"
        if os.path.exists(out):
            continue
        t = time.time()
        B = pd.read_parquet(bf); C = pd.read_parquet(f"{WORK}/comp_{split}/{name}")
        assert len(B) == len(C), "block/comp misaligned"
        tab = pd.concat([B, C], axis=1)
        pc = cheap.predict(xgb.DMatrix(cheap_feats(tab)[cf]))
        keep = select(pc, tab.i1.values, cfg["tau"], cfg["min_keep"])
        cand = tab[keep].reset_index(drop=True)
        n_full = len(tab); del tab, B, C
        X = F.pair_features(cand)
        E = F.extra_features(cand.i1.to_numpy(), cand.j.to_numpy(), cand.src.to_numpy())
        X = pd.concat([X, E, cand[[c for c in cand.columns if c.startswith("comp_") or c.startswith("cs_")]].reset_index(drop=True)], axis=1)
        p = bst.predict(xgb.DMatrix(X[feats])).astype(np.float32)
        res = cand[["i1", "j", "src"]].assign(p=p)
        res.to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)
        print(f"{split} {name}: {n_full} -> {len(cand)} cands ({len(cand)/max(cand.i1.nunique(),1):.1f}/S1), {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
