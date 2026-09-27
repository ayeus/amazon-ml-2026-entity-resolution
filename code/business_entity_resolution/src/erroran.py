"""Error analysis on the validation split: where is macro-F0.5 lost?"""
import os, sys, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import macro_f05, score_selection
va = pd.read_parquet(f"{WORK}/ds_val.parquet"); meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"].reset_index(drop=True)
bst = xgb.Booster(); bst.load_model(f"{WORK}/model_xgb.json"); feats = json.load(open(f"{WORK}/model_feats.json"))
va["p"] = bst.predict(xgb.DMatrix(va[feats]))
va[["i1", "j", "src_id", "y", "p"]].to_parquet(f"{WORK}/val_pred.parquet")
THR = 0.7
sel = (va.p >= THR).values
m, f = score_selection(va, sel, mva); mva["f"] = f
print("macro F0.5", round(m, 5), "| entities", len(mva))
tp = pd.Series(va.y.values * sel).groupby(va.i1.values).sum().reindex(mva.i1).fillna(0).values
np_ = pd.Series(sel.astype(int)).groupby(va.i1.values).sum().reindex(mva.i1).fillna(0).values
mva["tp"], mva["npred"] = tp, np_
mva["cands_true"] = va.groupby("i1").y.sum().reindex(mva.i1).fillna(0).values
mva["kind"] = np.select([mva.n_true == 0, mva.npred == 0, mva.tp == mva.n_true, mva.tp == mva.npred],
                        ["singleton", "missed_all", "perfect_recall", "perfect_prec"], "mixed")
g = mva.groupby("kind").agg(n=("f", "size"), meanF=("f", "mean"), lost=("f", lambda x: (1 - x).sum()))
g["share_of_total_loss"] = (g.lost / g.lost.sum()).round(3); g["frac_entities"] = (g.n / len(mva)).round(4)
print(g.round(4).to_string())
mva["f_lt1"] = mva.f < 0.999
print("\nby country:"); print(mva.groupby("country").f.agg(["size", "mean"]).round(4).to_string())
print("\nby n_true:"); print(mva.groupby(mva.n_true.clip(upper=8)).f.agg(["size", "mean"]).round(4).to_string())
sing = mva[mva.n_true == 0]
print("\nsingleton entities: predicted-empty (correct) =", round((sing.npred == 0).mean(), 4))
ok = mva[mva.n_true > 0]
print("non-singleton: pred-empty =", round((ok.npred == 0).mean(), 4), "| recall-in-cands =", round(ok.cands_true.sum() / ok.n_true.sum(), 4))
va["sel"] = sel
fp = va[(va.sel) & (va.y == 0)]; fn = va[(~va.sel) & (va.y == 1)]
print("\nFP pairs", len(fp), "FN pairs (in cands)", len(fn), "| TP", int(((va.sel) & (va.y == 1)).sum()))
print("FP by src", fp.src_id.value_counts().to_dict(), "| FN by src", fn.src_id.value_counts().to_dict())
mva.to_pickle(f"{WORK}/val_entity_eval.pkl")
