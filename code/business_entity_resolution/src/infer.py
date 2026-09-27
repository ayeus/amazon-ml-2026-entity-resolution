"""Stage 5: chunked inference on a split (test).  block -> features -> model probability, per S1 chunk.

Writes work/{split}_scored_{country}_{k}.parquet (i1, j, src, p, + few columns).  Resumable: existing chunks are skipped.
"""
import os, sys, time, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split, generate
from features import set_data, pair_features

KT, KN, KA, CAP = 15, 12, 12, 3e-3


def run(split, chunk_s1=80_000, limit=None):
    data = load_split(split); set_data(data)
    bst = xgb.Booster(); bst.load_model(f"{WORK}/model_xgb.json")
    feats = json.load(open(f"{WORK}/model_feats.json"))
    ctry = data[1][0].country.to_numpy(dtype=object)
    for country in sorted(set(ctry)):
        rows = np.flatnonzero(ctry == country)
        if limit:
            rows = rows[:limit]
        for k, a in enumerate(range(0, len(rows), chunk_s1)):
            out = f"{WORK}/{split}_scored_{country}_{k}.parquet"
            if os.path.exists(out):
                continue
            t = time.time()
            cand, _ = generate(data, country, rows[a:a + chunk_s1], Kt=KT, Kn=KN, Ka=KA, cap_frac=CAP)
            F = pair_features(cand)
            cand = cand.assign(p=bst.predict(xgb.DMatrix(F[feats])).astype(np.float32))
            cand.to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)  # atomic
            print(f"{split} {country} chunk {k}: {len(rows[a:a+chunk_s1])} S1, {len(cand)} pairs, {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
