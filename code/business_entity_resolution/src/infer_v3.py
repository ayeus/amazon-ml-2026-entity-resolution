"""Test inference for the v3 (augmentation-trained) pipeline.  Test tables live in BER_WORK (work2); models, cheap
filter and word statistics are read from BER_MODEL_DIR (work3).  block chunk + competition -> cheap filter (final
candidate set) -> base/extra/competition features -> word/cluster/number features -> probability.  Chunk-wise,
resumable, atomic writes to <WORK>/scored6_test/."""
import os, sys, glob, time, json, pickle
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
from make_big import process
from enrich5 import arrays, features as stage2_features
import wordstats, numfix

MD = os.environ.get("BER_MODEL_DIR", WORK)
TAG = os.environ.get("BER_DS_TAG", "")                       # "_smoke": use smoke models, write to a separate folder
OTAG = os.environ.get("BER_OUT_TAG", TAG)                    # output folder tag (defaults to TAG)
SMOKE_CHUNKS = int(os.environ.get("BER_SMOKE_CHUNKS", "0"))
# House-number parsing fix (numfix.py) for S1 of countries ABSENT from the training data (derived from train_s1.pkl,
# not hard-coded; same rule as make_outputs --thr-unseen). Training countries keep the parser the model was trained with.
NUMFIX = os.environ.get("BER_NUMFIX", "")
TRAIN_WORK = os.environ.get("BER_TRAIN_WORK", MD)

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    data = load_split(split, with_X=False); t = time.time(); F.set_data(data); print(f"pool stats {time.time()-t:.0f}s", flush=True)
    A = arrays(split)
    cheap = xgb.Booster(); cheap.load_model(f"{MD}/cheap3{TAG}_xgb.json")
    cf = json.load(open(f"{MD}/cheap3{TAG}_feats.json")); cfg = json.load(open(f"{MD}/cheap3{TAG}_cfg.json"))
    bst = xgb.Booster(); bst.load_model(f"{MD}/model_v3{TAG}.json"); feats = json.load(open(f"{MD}/model_v3{TAG}_feats.json"))
    stats = pickle.load(open(f"{MD}/word_stats6{TAG}_full.pkl", "rb"))
    out_dir = f"{WORK}/scored6{OTAG}_{split}"; os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f"{WORK}/block_{split}/*.parquet"))
    orig_nums = wordstats._nums
    seen = set(pd.read_pickle(f"{TRAIN_WORK}/train_s1.pkl").country.unique()) if NUMFIX else set()
    for bf in (files[:SMOKE_CHUNKS] if SMOKE_CHUNKS else files):
        name = os.path.basename(bf); out = f"{out_dir}/{name}"
        if os.path.exists(out):
            continue
        t = time.time()
        country = name[:-len(".parquet")].rsplit("_", 1)[0]          # block chunks are written per country
        fix = bool(NUMFIX) and country not in seen
        wordstats._nums = numfix.make_nums(NUMFIX) if fix else orig_nums
        tab = pd.concat([pd.read_parquet(bf), pd.read_parquet(f"{WORK}/comp_{split}{TAG}/{name}")], axis=1); n_full = len(tab)
        X = process(tab, cheap, cf, cfg); del tab
        S2 = stage2_features(X[["i1", "j", "src_id", "jw"]], A, stats)
        X = pd.concat([X.reset_index(drop=True), S2.reset_index(drop=True)], axis=1)
        p = bst.predict(xgb.DMatrix(X[feats].to_numpy(np.float32), feature_names=feats)).astype(np.float32)
        X[["i1", "j", "src_id"]].rename(columns={"src_id": "src"}).assign(p=p).to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)
        print(f"{split} {name}: {n_full} blocked -> {len(X)} candidates ({len(X)/max(X.i1.nunique(),1):.2f}/S1), "
              f"numfix={'on' if fix else 'off'}, {time.time()-t:.0f}s", flush=True)
