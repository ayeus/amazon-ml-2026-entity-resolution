"""Add extra name-ambiguity features + cross-S1 competition features to the existing train/val pair sets (no recompute of
the expensive base pair features)."""
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
from features import set_data, extra_features
from competition import build_all

if __name__ == "__main__":
    build_all("train")
    data = load_split("train"); t = time.time(); set_data(data); print("pool stats", round(time.time() - t), "s", flush=True)
    ctry = {c: None for c in ("US", "India")}
    for name in ("val", "train"):
        D = pd.read_parquet(f"{WORK}/ds_{name}.parquet")
        E = extra_features(D.i1.to_numpy(), D.j.to_numpy(), D.src_id.to_numpy())
        keep = set(D.i1.unique())
        comps = []
        for cf in sorted(glob.glob(f"{WORK}/comp_train/*.parquet")):
            b = pd.read_parquet(cf.replace("comp_train", "block_train"), columns=["i1", "j", "src", "cs_all", "cs_name", "cs_addr"])
            m = b.i1.isin(keep).to_numpy()
            if m.any():
                c = pd.read_parquet(cf)[m.nonzero()[0].tolist()] if False else pd.read_parquet(cf).loc[m]
                comps.append(pd.concat([b.loc[m].reset_index(drop=True), c.reset_index(drop=True)], axis=1))
        C = pd.concat(comps, ignore_index=True).rename(columns={"src": "src_id"})
        D = pd.concat([D.reset_index(drop=True), E], axis=1)
        n0 = len(D)
        D = D.merge(C, on=["i1", "j", "src_id"], how="left")
        assert len(D) == n0 and D.comp_all_rk.notna().all(), "competition join lost rows"
        D.to_parquet(f"{WORK}/ds2_{name}.parquet")
        print(name, D.shape, flush=True)
