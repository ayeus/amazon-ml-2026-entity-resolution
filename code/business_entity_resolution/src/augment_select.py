"""Self-contained selection of 'sibling' non-match records for augment.py (depends only on normalised TRAIN strings and
the TRAIN ground truth).

1. Join every Source-1 entity to original S2/S3 records that share a (country, name token, street token) key.
2. Branch vocabulary: among joined pairs where the record's name is the S1 name plus EXACTLY ONE added word, the added
   words whose match rate is ~0 (e.g. "north", "group", "holdings") — noise words ("services", "center") keep high rates.
3. A sibling of S1 entity a is a joined record b that
     - is NOT a true match of a (train ground truth)
     - has a house number different from a's number while a's own number is confirmed by another joined record
     - has a similar name (JaroWinkler >= 0.85 or token containment, at most 2 added words)
     - has a similar address (token-set ratio >= 0.75), i.e. it would be a strong candidate of a
Outputs (next to <out>): siblings parquet (i1, j, src_id) and branch_vocab.json; joined pairs are cached in pairs.parquet.
"""
import os, sys, time, json
import numpy as np, pandas as pd
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
import wordstats as ws
from make_big import truth_key

MAX_KEY = 50          # keys shared by more records than this are too common to indicate a sibling
BATCH = 1_000_000


def explode_keys(df, n_rows):
    """Vectorised (country|name-token|street-token) keys, hashed to int64, for rows [0, n_rows)."""
    out = []
    for lo in range(0, n_rows, BATCH):
        hi = min(lo + BATCH, n_rows); rows = np.arange(lo, hi)
        nt = pd.DataFrame({"row": rows, "tok": pd.Series(df.core.values[lo:hi]).str.split().values}).explode("tok")
        nt = nt[nt.tok.notna() & (nt.tok.str.len() >= 3)]
        nt = nt[nt.groupby("row").cumcount() < 2]
        st = pd.DataFrame({"row": rows, "tok": pd.Series(df.atok.values[lo:hi]).str.split().values}).explode("tok")
        st = st[st.tok.notna() & st.tok.str.isalpha() & (st.tok.str.len() >= 4)].drop_duplicates(["row", "tok"])
        st = st[st.groupby("row").cumcount() < 6]
        m = nt.merge(st, on="row", suffixes=("_n", "_s"))
        key = pd.Series(df.country.values[m.row.values]) + "|" + m.tok_n.values + "|" + m.tok_s.values
        out.append(pd.DataFrame({"key": pd.util.hash_array(key.to_numpy(object)).view(np.int64), "row": m.row.values}))
    return pd.concat(out, ignore_index=True)


def _strings(dfs, i1, j, src, col):
    a = dfs[1][col].values[i1]; b = np.empty(len(i1), object)
    for s in (2, 3):
        m = src == s; b[m] = dfs[s][col].values[j[m]]
    return a, b


def joined_pairs(dfs, n_orig, ids):
    """Key join -> cheap vectorised similarities for all pairs (batched) -> keep address-similar pairs -> heavier
    pair features only on that subset (memory-bounded)."""
    t = time.time()
    k1 = explode_keys(dfs[1], len(dfs[1]))
    pairs = []
    for s in (2, 3):
        kb = explode_keys(dfs[s], n_orig[s])
        cnt = kb.key.value_counts(); kb = kb[kb.key.map(cnt).values <= MAX_KEY]
        m = k1.merge(kb, on="key", suffixes=("1", "b"))
        pairs.append(pd.DataFrame({"i1": m.row1.values, "j": m.rowb.values, "src": np.int8(s)}).drop_duplicates())
        del kb, m
    del k1
    P = pd.concat(pairs, ignore_index=True); del pairs
    print(f"  key join: {len(P)} (S1, record) pairs sharing a name+street key ({time.time()-t:.0f}s)", flush=True)
    kw = dict(workers=-1, dtype=np.float32)
    atsr = np.empty(len(P), np.float32); jw = np.empty(len(P), np.float32)
    i1v, jv, sv = P.i1.values, P.j.values, P.src.values
    for lo in range(0, len(P), 10 * BATCH):
        hi = min(lo + 10 * BATCH, len(P))
        a, b = _strings(dfs, i1v[lo:hi], jv[lo:hi], sv[lo:hi], "atok"); atsr[lo:hi] = cpdist(a, b, scorer=fuzz.token_set_ratio, **kw) / 100
        a, b = _strings(dfs, i1v[lo:hi], jv[lo:hi], sv[lo:hi], "core"); jw[lo:hi] = cpdist(a, b, scorer=JaroWinkler.normalized_similarity, **kw)
    P["atsr"] = atsr; P["jw"] = jw
    P["y"] = np.isin(i1v.astype(np.int64) * 100_000_000_000 + jv.astype(np.int64) * 10 + sv, truth_key(ids))
    n_all, n_true_all = len(P), int(P.y.sum())
    P = P[P.atsr >= 0.75].reset_index(drop=True)          # siblings sit on the same street at a nearby number
    print(f"  address-similar pairs kept: {len(P)} of {n_all} ({time.time()-t:.0f}s)", flush=True)
    n_extra = np.empty(len(P), np.int16); a_sub = np.empty(len(P), bool); b_sub = np.empty(len(P), bool)
    for lo in range(0, len(P), BATCH):
        hi = min(lo + BATCH, len(P))
        a, b = _strings(dfs, P.i1.values[lo:hi], P.j.values[lo:hi], P.src.values[lo:hi], "core")
        sa = [set(x.split()) for x in a]; sb = [set(x.split()) for x in b]
        n_extra[lo:hi] = [len(y - x) for x, y in zip(sa, sb)]
        a_sub[lo:hi] = [x <= y for x, y in zip(sa, sb)]; b_sub[lo:hi] = [y <= x for x, y in zip(sa, sb)]
    P["n_extra"], P["a_sub_b"], P["b_sub_a"] = n_extra, a_sub, b_sub
    a, b = _strings(dfs, P.i1.values, P.j.values, P.src.values, "atok")
    P["n_rel"], _, _ = ws.number_geometry(a, b)
    P["s1_support"] = (P.n_rel == 1).groupby(P.i1).transform("sum").values
    print(f"  pair features done ({time.time()-t:.0f}s)", flush=True)
    return P, n_all, n_true_all


def branch_vocab(P, dfs, min_n=100, max_rate=0.01):
    one = P[P.a_sub_b & (P.n_extra == 1)]
    a_c = dfs[1].core.values[one.i1.values]; b_c = np.empty(len(one), object)
    for s in (2, 3):
        m = one.src.values == s; b_c[m] = dfs[s].core.values[one.j.values[m]]
    words = [next(iter(set(y.split()) - set(x.split()))) for x, y in zip(a_c, b_c)]
    R = pd.DataFrame({"w": words, "y": one.y.values}).groupby("w").y.agg(["size", "mean"])
    R = R[(R["size"] >= min_n) & (R["mean"] <= max_rate)]
    return sorted(w for w in R.index if w.isascii() and w.isalpha() and len(w) >= 3)


if __name__ == "__main__":
    out = sys.argv[1]; out_dir = os.path.dirname(out) or "."
    dfs = {s: pd.read_pickle(f"{WORK}/train_s{s}.pkl")[["entity_id", "core", "atok", "country"]] for s in (1, 2, 3)}
    # only ORIGINAL records can be siblings (synthetic rows, if present, are appended with ids like S2-A...)
    n_orig = {s: int((~dfs[s].entity_id.str.contains("-A", regex=False)).sum()) for s in (2, 3)}
    cache = f"{out_dir}/pairs.parquet"
    if os.path.exists(cache):
        P = pd.read_parquet(cache); n_all = n_true_all = -1
    else:
        ids = {s: dfs[s].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
        P, n_all, n_true_all = joined_pairs(dfs, n_orig, ids)
        P.to_parquet(cache)
    vocab = branch_vocab(P, dfs)
    json.dump(vocab, open(f"{out_dir}/branch_vocab.json", "w"))
    print(f"branch vocabulary: {len(vocab)} words, e.g. {vocab[:25]}", flush=True)
    sel = P[(~P.y) & (P.n_rel >= 2) & (P.s1_support > 0) & ((P.jw >= 0.85) | P.a_sub_b | P.b_sub_a)
            & (P.n_extra <= 2) & (P.atsr >= 0.75)]
    sel = sel.drop_duplicates(["j", "src"])[["i1", "j", "src"]].rename(columns={"src": "src_id"})
    sel.to_parquet(out)
    print(f"selected {len(sel)} sibling non-match records (joined pairs {n_all}, true matches {n_true_all}; "
          f"address-similar {len(P)})", flush=True)
