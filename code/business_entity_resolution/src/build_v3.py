"""Test-like (augmented) dataset build on the work3 train blocking tables.

Disjoint S1 groups (split by Source-1 entity):
  VAL   = the fixed 40k validation S1 of ds_meta.pkl (their neighbourhoods now contain synthetic sibling duplicates, so
          this validation set has the test structure)
  CHEAP = 150k non-validation S1, used ONLY to train the candidate filter
  MATCH = 1.0M non-validation S1, used to train the matcher
Stages (resumable; each appends a line to ds6/manifest.jsonl):
  cheap   -> cheap3_{xgb.json,feats.json,cfg.json}: filter trained on CHEAP, tau chosen on VAL for a recall target
  dataset -> ds6/{train,val}_<chunk>.parquet: cheap-filtered candidates + base/extra/competition features + labels
  stage2  -> ds6x/{train,val}_<chunk>.parquet + word_stats6_full.pkl: word identity (cross-fitted on MATCH), address
             clusters, number geometry
"""
import os, sys, glob, time, json, pickle
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split
import features as F
import wordstats as ws
from cheap_filter import cheap_feats, select
from make_big import truth_key, process
from make_dataset import is_val
from enrich5 import arrays, features as stage2_features

CHEAP_P = dict(objective="binary:logistic", tree_method="hist", max_depth=6, eta=0.2, subsample=0.8, colsample_bytree=0.8,
               min_child_weight=5, nthread=10)
RECALL_TARGET = float(os.environ.get("BER_CHEAP_RECALL", "0.997"))
TAG = os.environ.get("BER_DS_TAG", "")                       # e.g. "_smoke" for a dry run into separate folders
SMOKE_CHUNKS = int(os.environ.get("BER_SMOKE_CHUNKS", "0"))   # >0: only use the first N blocking chunks
DS, DSX = f"{WORK}/ds6{TAG}", f"{WORK}/ds6x{TAG}"
MODEL_TAG = TAG


def log(stage, **kw):
    kw.update(stage=stage, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(f"{DS}/manifest.jsonl", "a") as f:
        f.write(json.dumps(kw, default=float) + "\n")
    print(stage, kw, flush=True)


def chunks():
    files = sorted(glob.glob(f"{WORK}/block_train/*.parquet"))
    if SMOKE_CHUNKS:
        files = files[:SMOKE_CHUNKS]
    for bf in files:
        name = os.path.basename(bf)
        yield name, pd.concat([pd.read_parquet(bf), pd.read_parquet(f"{WORK}/comp_train{TAG}/{name}")], axis=1)


def ensure_meta(ids1):
    """The fixed validation S1 set: identical RNG sequence to make_dataset.py (150k train draw, then 40k val draw)."""
    f = f"{WORK}/ds_meta.pkl"
    if os.path.exists(f):
        return
    from evalblock import truth_pairs
    val = is_val(ids1); rng = np.random.RandomState(7)
    tr_rows = rng.choice(np.flatnonzero(~val), 150000, replace=False); va_rows = rng.choice(np.flatnonzero(val), 40000, replace=False)
    tp = truth_pairs(); pos = pd.Series(np.arange(len(ids1)), index=ids1)
    ntrue = pd.Series(pos.reindex(tp.source1_entity_id.values).values).value_counts()
    meta = pd.DataFrame({"i1": np.concatenate([tr_rows, va_rows]), "split": ["train"] * len(tr_rows) + ["val"] * len(va_rows)})
    meta["n_true"] = ntrue.reindex(meta.i1.values).fillna(0).astype(int).values
    meta.to_pickle(f)


def groups(ids1):
    ensure_meta(ids1)
    meta = pd.read_pickle(f"{WORK}/ds_meta.pkl")
    val = meta[meta.split == "val"].i1.values
    non_val = np.flatnonzero(~is_val(ids1))
    rng = np.random.RandomState(2026)
    cheap = rng.choice(non_val, 150_000, replace=False)
    rest = np.setdiff1d(non_val, cheap)
    match = rng.choice(rest, min(len(rest), int(os.environ.get("BER_MATCH_S1", "1000000"))), replace=False)
    g = np.full(len(ids1), -1, np.int8); g[val] = 0; g[cheap] = 1; g[match] = 2
    assert not (set(val) & set(cheap)) and not (set(val) & set(match)) and not (set(cheap) & set(match)), "S1 group leakage"
    return g, meta[meta.split == "val"].reset_index(drop=True)


def key_of(i1, j, src):
    return i1.astype(np.int64) * 100_000_000_000 + j.astype(np.int64) * 10 + src


def stage_cheap(g, tk, mva):
    Xs, ys, gs, i1s = [], [], [], []
    for name, tab in chunks():
        sub = tab[np.isin(g[tab.i1.values], (0, 1))].reset_index(drop=True)
        X = cheap_feats(sub).astype(np.float32)
        Xs.append(X); ys.append(np.isin(key_of(sub.i1.values, sub.j.values, sub.src.values), tk).astype(np.int8))
        gs.append(g[sub.i1.values]); i1s.append(sub.i1.values)
    X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys); gg = np.concatenate(gs); i1 = np.concatenate(i1s)
    tr = gg == 1; va = gg == 0
    b = xgb.train(CHEAP_P, xgb.DMatrix(X[tr], label=y[tr]), 120)
    p = b.predict(xgb.DMatrix(X[va]))
    if SMOKE_CHUNKS:
        mva = mva[mva.i1.isin(np.unique(i1[va]))]
    n_true_val = int(mva.n_true.sum()); in_block = int(y[va].sum()); nS1 = len(mva)
    res = []
    for tau in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06):
        k = select(p, i1[va], tau)
        res.append((tau, float(k.sum() / nS1), float(y[va][k].sum() / in_block), float(y[va][k].sum() / n_true_val)))
        print(f"  tau {tau}: cand/S1 {res[-1][1]:.2f}  kept/in-blocking true {res[-1][2]:.4f}  overall pair recall {res[-1][3]:.4f}", flush=True)
    ok = [r for r in res if r[2] >= RECALL_TARGET]
    tau = max(ok)[0] if ok else res[0][0]
    b.save_model(f"{WORK}/cheap3{MODEL_TAG}_xgb.json"); json.dump(list(X.columns), open(f"{WORK}/cheap3{MODEL_TAG}_feats.json", "w"))
    json.dump({"tau": tau, "min_keep": 2}, open(f"{WORK}/cheap3{MODEL_TAG}_cfg.json", "w"))
    log("cheap", tau=tau, blocking_pair_recall_val=in_block / n_true_val, val_cand_per_S1_before=float(va.sum() / nS1),
        sweep=res, rows_train=int(tr.sum()), rows_val=int(va.sum()))


def stage_dataset(g, tk):
    cheap = xgb.Booster(); cheap.load_model(f"{WORK}/cheap3{MODEL_TAG}_xgb.json")
    cf = json.load(open(f"{WORK}/cheap3{MODEL_TAG}_feats.json")); cfg = json.load(open(f"{WORK}/cheap3{MODEL_TAG}_cfg.json"))
    n_tr = n_va = 0
    for name, tab in chunks():
        if os.path.exists(f"{DS}/train_{name}") and os.path.exists(f"{DS}/val_{name}"):
            continue
        t = time.time()
        sub = tab[np.isin(g[tab.i1.values], (0, 2))].reset_index(drop=True); del tab
        X = process(sub, cheap, cf, cfg)
        X["y"] = np.isin(key_of(X.i1.values, X.j.values, X.src_id.values), tk).astype(np.int8)
        gi = g[X.i1.values]
        for nm, m in (("train", gi == 2), ("val", gi == 0)):
            f = f"{DS}/{nm}_{name}"; X[m].reset_index(drop=True).to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
        n_tr += int((gi == 2).sum()); n_va += int((gi == 0).sum())
        print(f"  dataset {name}: {len(sub)} blocked rows -> {len(X)} kept, {time.time()-t:.0f}s", flush=True)
    log("dataset", train_rows_added=n_tr, val_rows_added=n_va)


def stage2():
    A = arrays("train")
    cols = ["i1", "j", "src_id", "y", "jw"]
    tr_files = sorted(glob.glob(f"{DS}/train_*.parquet"))
    D = pd.concat([pd.read_parquet(f, columns=cols) for f in tr_files], ignore_index=True)
    a_core = A[1][0][D.i1.values]; b_core = np.empty(len(D), object)
    for s in (2, 3):
        m = D.src_id.values == s; b_core[m] = A[s][0][D.j.values[m]]
    fold = (pd.util.hash_array(D.i1.values.astype(np.int64)) % 2).astype(int)
    st = {f: ws.learn(a_core[fold != f], b_core[fold != f], D.y.values[fold != f]) for f in (0, 1)}
    full = ws.learn(a_core, b_core, D.y.values)
    pickle.dump(full, open(f"{WORK}/word_stats6{MODEL_TAG}_full.pkl", "wb")); del D, a_core, b_core
    for f in sorted(glob.glob(f"{DS}/*_*.parquet")):
        out = f"{DSX}/{os.path.basename(f)}"
        if os.path.exists(out):
            continue
        d = pd.read_parquet(f, columns=cols)
        if os.path.basename(f).startswith("train_"):
            fo = (pd.util.hash_array(d.i1.values.astype(np.int64)) % 2).astype(int)
            X = stage2_features(d, A, (fo, st))
        else:
            X = stage2_features(d, A, full)
        X.to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)
    log("stage2", files=len(glob.glob(f"{DSX}/*.parquet")), extra_words=len(full["ex_n"]))


if __name__ == "__main__":
    os.makedirs(DS, exist_ok=True); os.makedirs(DSX, exist_ok=True)
    stages = sys.argv[1].split(",")
    data = load_split("train", with_X=False)
    ids = {s: data[s][0].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    g, mva = groups(ids[1])
    if "cheap" in stages or "dataset" in stages:
        tk = truth_key(ids)
    if "cheap" in stages or "dataset" in stages:
        t = time.time(); F.set_data(data); print(f"pool stats {time.time()-t:.0f}s", flush=True)   # strings for cheap sims
    if "cheap" in stages:
        stage_cheap(g, tk, mva)
    if "dataset" in stages:
        stage_dataset(g, tk)
    if "stage2" in stages:
        stage2()
