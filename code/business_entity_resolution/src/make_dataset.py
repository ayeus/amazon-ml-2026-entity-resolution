"""Build labelled candidate+feature sets from the TRAIN files with an S1-level split (train / validation)."""
import os, sys, time, zlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from blocking import load_split, generate
from features import set_data, pair_features
from evalblock import truth_pairs
from prep import WORK


def is_val(ids):
    return np.array([zlib.crc32(("split" + i).encode()) % 5 == 0 for i in ids])


def build(data, s1_rows, tp_key, Kt, Kn, Ka, cap, tag):
    ctry = data[1][0].country.to_numpy(dtype=object)
    parts = []
    for country in sorted(set(ctry[s1_rows])):
        cand, _ = generate(data, country, s1_rows, Kt=Kt, Kn=Kn, Ka=Ka, cap_frac=cap)
        t = time.time(); F = pair_features(cand); print(f"  {tag} {country} features {time.time()-t:.0f}s", flush=True)
        key = cand.i1.values.astype(np.int64) * 100_000_000_000 + cand.j.values.astype(np.int64) * 10 + cand.src.values
        F["y"] = np.isin(key, tp_key).astype(np.int8)
        F["i1"] = cand.i1.values; F["j"] = cand.j.values; F["src_id"] = cand.src.values
        parts.append(F)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    ntr, nva = int(sys.argv[1]), int(sys.argv[2])
    Kt, Kn, Ka, cap = 15, 12, 12, 3e-3
    data = load_split("train"); set_data(data)
    ids = {s: data[s][0].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    pos = {s: pd.Series(np.arange(len(ids[s])), index=ids[s]) for s in (1, 2, 3)}
    tp = truth_pairs()
    tp["i1"] = pos[1].reindex(tp.source1_entity_id.values).values
    tp["src"] = tp.m.str[1].astype(int); tp["j"] = -1
    for s in (2, 3):
        mk = tp.src.values == s; tp.loc[mk, "j"] = pos[s].reindex(tp.m.values[mk]).values
    tp_key = tp.i1.values.astype(np.int64) * 100_000_000_000 + tp.j.values.astype(np.int64) * 10 + tp.src.values
    ntrue = tp.groupby("i1").size()
    val = is_val(ids[1]); rng = np.random.RandomState(7)
    tr_rows = rng.choice(np.flatnonzero(~val), ntr, replace=False); va_rows = rng.choice(np.flatnonzero(val), nva, replace=False)
    assert not set(tr_rows) & set(va_rows), "leakage: S1 in both train and validation"
    meta = pd.DataFrame({"i1": np.concatenate([tr_rows, va_rows]), "split": ["train"] * len(tr_rows) + ["val"] * len(va_rows)})
    meta["n_true"] = ntrue.reindex(meta.i1.values).fillna(0).astype(int).values
    meta["country"] = data[1][0].country.to_numpy(dtype=object)[meta.i1.values]
    meta.to_pickle(f"{WORK}/ds_meta.pkl")
    for name, rows in (("val", va_rows), ("train", tr_rows)):
        t = time.time(); D = build(data, np.sort(rows), tp_key, Kt, Kn, Ka, cap, name)
        D.to_parquet(f"{WORK}/ds_{name}.parquet"); print(name, D.shape, "pos", D.y.sum(), f"{time.time()-t:.0f}s", flush=True)
