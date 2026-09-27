"""Test inference for a train_t2.py variant model (BER_DERIVE = t2 | t4 | t5 | t45): same chunk pipeline as infer_v3
(block chunk + competition -> cheap filter -> base/extra/competition -> word/cluster/number features), then the variant's
derived columns and, when needed, the n-gram features, then p.  Scores every test chunk (optionally only chunk-name
prefixes listed in BER_COUNTRIES, for diagnostics).  Resumable, atomic writes to <WORK>/scored6<tag>_test.
Run: BER_WORK=work_test BER_MODEL_DIR=work_train BER_DERIVE=t45 python3 infer_t.py
"""
import os, sys, glob, time, json, pickle
import multiprocessing as mp
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from make_big import process
from enrich5 import arrays, features as stage2_features
import ngram as G
from train_t2 import derive, DERIVED, MTAG
import numfix
numfix.apply(os.environ.get("BER_NUMFIX", ""))              # Task 8: house-number parsing fix for number geometry

MD = os.environ.get("BER_MODEL_DIR", WORK)
OTAG = os.environ.get("BER_OUT_TAG", MTAG)                   # output folder tag (defaults to the model tag)
CD = os.environ.get("BER_CHEAP_DIR", MD)                      # candidate filter (defines the candidate set)
SD = os.environ.get("BER_STATS_DIR", MD)                      # word statistics that belong to the model
# extra models scored in the same pass (their features must be a subset of this set's columns): "name:tag,name:tag"
EXTRA_MODELS = [x.split(":") for x in os.environ.get("BER_EXTRA_MODELS", "").split(",") if x]
COUNTRIES = tuple(x for x in os.environ.get("BER_COUNTRIES", "").split(",") if x)   # empty = all chunks

if __name__ == "__main__":
    split = "test"
    use_ng = any(c in G.NG_NAMES for c in DERIVED)
    S = G.strings(WORK, split) if use_ng else None
    pool = mp.get_context("fork").Pool(8) if use_ng else None     # forked before the big feature tables are loaded
    data = load_split(split, with_X=False); t = time.time(); F.set_data(data); print(f"pool stats {time.time()-t:.0f}s", flush=True)
    A = arrays(split)
    cheap = xgb.Booster(); cheap.load_model(f"{CD}/cheap3_xgb.json")
    cf = json.load(open(f"{CD}/cheap3_feats.json")); cfg = json.load(open(f"{CD}/cheap3_cfg.json"))
    bst = xgb.Booster(); bst.load_model(f"{MD}/model_v3{MTAG}.json"); feats = json.load(open(f"{MD}/model_v3{MTAG}_feats.json"))
    stats = pickle.load(open(f"{SD}/word_stats6_full.pkl", "rb"))
    extra = []
    for nm, tg in EXTRA_MODELS:
        if nm.startswith("lgb"):                                  # LightGBM model file model_<tg>.txt (Task 9)
            import lightgbm as lgb
            b = lgb.Booster(model_file=f"{MD}/model_{tg}.txt"); extra.append((nm, b, b.feature_name()))
            continue
        b = xgb.Booster(); b.load_model(f"{MD}/model_v3{tg}.json"); extra.append((nm, b, json.load(open(f"{MD}/model_v3{tg}_feats.json"))))
    out_dir = f"{WORK}/scored6{OTAG}_{split}"; os.makedirs(out_dir, exist_ok=True)
    files = [f for f in sorted(glob.glob(f"{WORK}/block_{split}/*.parquet")) if not COUNTRIES or os.path.basename(f).startswith(COUNTRIES)]
    print(f"numfix={os.environ.get('BER_NUMFIX', '') or 'off'} | model {MD}/model_v3{MTAG}.json: {len(feats)} features | cheap filter {CD} | word stats {SD} | {len(files)} chunks {COUNTRIES or 'all'} -> {out_dir}", flush=True)
    for bf in files:
        name = os.path.basename(bf); out = f"{out_dir}/{name}"
        if os.path.exists(out):
            continue
        t = time.time()
        tab = pd.concat([pd.read_parquet(bf), pd.read_parquet(f"{WORK}/comp_{split}/{name}")], axis=1); n_full = len(tab)
        X = process(tab, cheap, cf, cfg); del tab
        S2 = stage2_features(X[["i1", "j", "src_id", "jw"]], A, stats)
        X = pd.concat([X.reset_index(drop=True), S2.reset_index(drop=True)], axis=1)
        if use_ng:
            X = pd.concat([X, G.ngram_features(X.i1.to_numpy(), X.j.to_numpy(), X.src_id.to_numpy(), S, pool)], axis=1)
        X = derive(X)
        p = bst.predict(xgb.DMatrix(X[feats].to_numpy(np.float32), feature_names=feats)).astype(np.float32)
        O = X[["i1", "j", "src_id"]].rename(columns={"src_id": "src"}).assign(p=p)
        for nm, b, fs in extra:
            Xe = X[fs].to_numpy(np.float32)
            O[f"p_{nm}"] = (b.predict(Xe) if nm.startswith("lgb") else b.predict(xgb.DMatrix(Xe, feature_names=fs))).astype(np.float32)
        O.to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)
        print(f"{split} {name}: {n_full} blocked -> {len(X)} candidates ({len(X)/max(X.i1.nunique(),1):.2f}/S1), {time.time()-t:.0f}s", flush=True)
    if pool:
        pool.close()
