"""Final-submission merge: one scored folder for make_outputs.py.

Chunks of countries that occur in the TRAINING data (derived from train_s1.pkl, not hard-coded) take p = mean of the
chosen score columns of the rebuild-configuration folder (final: p_t4, p_t5, p = mean of three XGBoost matchers);
chunks of countries absent from training take p from the v3-configuration folder (v3 matcher + house-number fix).
Block chunks are written per country ('<country>_<k>.parquet'). Both test preparations list records in the raw-file
order, so (i1, j, src) refer to the same records in both folders.
Usage: python3 make_hybrid.py <out_dir> <train_country_scored_dir> <cols> <unseen_country_scored_dir> <train_s1.pkl>
"""
import os, sys, glob
import numpy as np, pandas as pd

if __name__ == "__main__":
    out, seen_dir, cols, unseen_dir, train_s1 = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4], sys.argv[5]
    seen = set(pd.read_pickle(train_s1).country.unique())
    country = lambda f: os.path.basename(f)[:-len(".parquet")].rsplit("_", 1)[0]
    os.makedirs(out, exist_ok=True)
    n = {"train-country": 0, "unseen-country": 0}
    for f in sorted(glob.glob(f"{seen_dir}/*.parquet")):
        if country(f) not in seen:
            continue
        d = pd.read_parquet(f)
        d = d[["i1", "j", "src"]].assign(p=np.mean([d[c].to_numpy(np.float64) for c in cols], 0).astype(np.float32))
        b = os.path.basename(f); d.to_parquet(f"{out}/{b}.tmp"); os.replace(f"{out}/{b}.tmp", f"{out}/{b}"); n["train-country"] += 1
    for f in sorted(glob.glob(f"{unseen_dir}/*.parquet")):
        if country(f) in seen:
            continue
        b = os.path.basename(f); pd.read_parquet(f)[["i1", "j", "src", "p"]].to_parquet(f"{out}/{b}.tmp"); os.replace(f"{out}/{b}.tmp", f"{out}/{b}")
        n["unseen-country"] += 1
    print(f"{out}: {n} chunks; training countries {sorted(seen)} use mean of {cols}")
