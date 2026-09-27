"""Stage-2 enrichment: word-identity evidence + address-cluster propagation + number geometry.
Writes ds5x_{train,val}.parquet (row-aligned with ds4_*), word_stats_full.pkl (for test), and the enriched test sample."""
import os, sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
import wordstats as ws


def arrays(split):
    d = {s: pd.read_pickle(f"{WORK}/{split}_s{s}.pkl") for s in (1, 2, 3)}
    return {s: (d[s].core.to_numpy(dtype=object), d[s].atok.to_numpy(dtype=object)) for s in (1, 2, 3)}


def side(A, D):
    a_core = A[1][0][D.i1.values]; a_at = A[1][1][D.i1.values]
    b_core = np.empty(len(D), object); b_at = np.empty(len(D), object)
    for s in (2, 3):
        m = D.src_id.values == s
        b_core[m] = A[s][0][D.j.values[m]]; b_at[m] = A[s][1][D.j.values[m]]
    return a_core, a_at, b_core, b_at


def features(D, A, stats_for_rows):
    """stats_for_rows: either one stats dict, or (fold array, {fold: stats})."""
    a_core, a_at, b_core, b_at = side(A, D)
    if isinstance(stats_for_rows, dict):
        W = ws.pair_word_features(a_core, b_core, stats_for_rows)
    else:
        fold, st = stats_for_rows
        W = pd.DataFrame(np.zeros((len(D), len(ws.PAIR_NAMES)), np.float32), columns=ws.PAIR_NAMES)
        for f, s in st.items():
            m = fold == f
            W.loc[m, :] = ws.pair_word_features(a_core[m], b_core[m], s).values
    W["jw"] = D.jw.values
    W = ws.cluster_features(W, D.i1.values, a_at, b_at)
    return W.drop(columns=["jw"])


if __name__ == "__main__":
    A = arrays("train")
    t = time.time()
    tr = pd.read_parquet(f"{WORK}/ds4_train.parquet", columns=["i1", "j", "src_id", "y", "jw"])
    va = pd.read_parquet(f"{WORK}/ds4_val.parquet", columns=["i1", "j", "src_id", "y", "jw"])
    a_core, _, b_core, _ = side(A, tr)
    fold = (pd.util.hash_array(tr.i1.values.astype(np.int64)) % 2).astype(int)
    st = {f: ws.learn(a_core[fold != f], b_core[fold != f], tr.y.values[fold != f]) for f in (0, 1)}
    full = ws.learn(a_core, b_core, tr.y.values)
    pickle.dump(full, open(f"{WORK}/word_stats_full.pkl", "wb"))
    print(f"word stats learned ({time.time()-t:.0f}s): {len(full['ex_n'])} extra-words, {len(full['ms_n'])} missing-words", flush=True)
    t = time.time(); Xv = features(va, A, full); Xv.to_parquet(f"{WORK}/ds5x_val.parquet"); print("val", Xv.shape, f"{time.time()-t:.0f}s", flush=True)
    t = time.time(); Xt = features(tr, A, (fold, st)); Xt.to_parquet(f"{WORK}/ds5x_train.parquet"); print("train", Xt.shape, f"{time.time()-t:.0f}s", flush=True)
    # enriched test sample for test-like diagnostics
    T = pd.read_parquet(f"{WORK}/diag_test_sample.parquet")
    Xs = features(T, arrays("test"), full); Xs.to_parquet(f"{WORK}/diag_test_sample_x.parquet"); print("test sample", Xs.shape, flush=True)
