"""Label-free diagnosis: do clear-cut test pairs get lower p than clear-cut validation pairs? Which features shift?"""
import os, sys, glob, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from make_big import process

cheap = xgb.Booster(); cheap.load_model(f"{WORK}/cheap_xgb.json"); cf = json.load(open(f"{WORK}/cheap_feats.json"))
cfg = json.load(open(f"{WORK}/cheap_cfg.json"))
bst = xgb.Booster(); bst.load_model(f"{WORK}/model_big.json"); feats = json.load(open(f"{WORK}/model3_feats.json"))
data = load_split("test"); F.set_data(data)
ctry = data[1][0].country.to_numpy(dtype=object)
parts = []
for f in sorted(glob.glob(f"{WORK}/block_test/*_0.parquet")):          # first chunk per country = ~100k S1 each
    name = os.path.basename(f)
    tab = pd.concat([pd.read_parquet(f), pd.read_parquet(f"{WORK}/comp_test/{name}")], axis=1)
    keep = np.unique(tab.i1.values)[:20000]
    tab = tab[np.isin(tab.i1.values, keep)].reset_index(drop=True)
    X = process(tab, cheap, cf, cfg); X["p"] = bst.predict(xgb.DMatrix(X[feats])); X["country"] = ctry[X.i1.values]
    parts.append(X)
T = pd.concat(parts, ignore_index=True)
V = pd.read_parquet(f"{WORK}/ds4_val.parquet"); V["p"] = pd.read_parquet(f"{WORK}/val_pred_big.parquet").p.values
V["country"] = "VAL-" + pd.read_pickle(f"{WORK}/train_s1.pkl").country.values[V.i1.values]
T.to_parquet(f"{WORK}/diag_test_sample.parquet")
clear = lambda D: (D.ceq == 1) & (D.a_bin >= 0.8) & (D.num_conf == 0)
print("CLEAR-CUT pairs (identical compact name, >=80% of B address tokens in S1, no number conflict):")
for nm, D in list(V.groupby("country")) + list(T.groupby("country")):
    c = D[clear(D)]
    tag = f" true-rate={c.y.mean():.4f}" if "y" in c and nm.startswith("VAL") else ""
    print(f"  {nm:10s} n={len(c):>7d} share={len(c)/len(D):.3f}  p<0.7: {(c.p<0.7).mean():.4f}  p<0.5: {(c.p<0.5).mean():.4f}{tag}")
print("\nper-S1: #clear-cut candidates vs #predicted (p>=0.7):")
for nm, D in list(V.groupby("country")) + list(T.groupby("country")):
    g = D.assign(cl=clear(D), pr=D.p >= 0.7).groupby("i1")[["cl", "pr"]].sum()
    print(f"  {nm:10s} clear/S1={g.cl.mean():.3f} pred/S1={g.pr.mean():.3f} cand/S1={len(D)/len(g):.2f}")
cols = ["n_cand", "r_t", "comb_rk", "s_all_rk", "comp_n_suitors", "comp_all_rk", "sib_nm_n", "sib_ad_n", "e_nm1_cnt", "e_nmB_cnt", "e_known", "b_noaddr", "jw", "a_bin"]
print("\nfeature means (VAL vs TEST):")
print(pd.concat({nm: D[cols].mean() for nm, D in list(V.groupby("country")) + list(T.groupby("country"))}, axis=1).round(3).to_string())
