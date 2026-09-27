"""Task 5: character n-gram similarity features for (S1, B) candidate pairs.

name  : compact core name (legal words removed, spaces removed) -> 3-gram Jaccard / set-cosine / containment of A in B,
        4-gram Jaccard  (catches concatenated / domain-style names and in-word typos that token features miss)
addr  : address tokens joined by spaces -> 3-gram Jaccard / set-cosine (-1 when B has no address)
Train side: python3 ngram.py <work dir> <ds dir name>   -> writes a row-aligned <ds>n/ folder (one parquet per file).
"""
import os, sys, glob, math
import multiprocessing as mp
import numpy as np, pandas as pd

NG_NAMES = ["ng_n3j", "ng_n3c", "ng_n3a", "ng_n4j", "ng_a3j", "ng_a3c"]


def _grams(s, n):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _sim(a, b):
    if not a or not b:
        return 0.0, 0.0, 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i), i / math.sqrt(len(a) * len(b)), i / len(a)


def pair_block(args):
    an, bn, aa, ba = args
    out = np.empty((len(an), len(NG_NAMES)), np.float32)
    for k in range(len(an)):
        x, y = an[k].replace(" ", ""), bn[k].replace(" ", "")
        j3, c3, a3 = _sim(_grams(x, 3), _grams(y, 3))
        j4, _, _ = _sim(_grams(x, 4), _grams(y, 4))
        if ba[k]:
            aj, ac, _ = _sim(_grams(f" {aa[k]} ", 3), _grams(f" {ba[k]} ", 3))
        else:
            aj = ac = -1.0
        out[k] = (j3, c3, a3, j4, aj, ac)
    return out


def strings(work, split):
    d = {s: pd.read_pickle(f"{work}/{split}_s{s}.pkl") for s in (1, 2, 3)}
    return {s: (d[s].core.to_numpy(dtype=object), d[s].atok.to_numpy(dtype=object)) for s in (1, 2, 3)}


def ngram_features(i1, j, src, S, pool, shard=50_000):
    """Row-aligned n-gram features for candidate arrays (i1, j, src in {2,3}); S = strings(); pool = mp pool."""
    an = S[1][0][i1]; aa = S[1][1][i1]
    bn = np.empty(len(i1), object); ba = np.empty(len(i1), object)
    for s in (2, 3):
        mk = src == s; bn[mk] = S[s][0][j[mk]]; ba[mk] = S[s][1][j[mk]]
    jobs = [(an[a:a + shard], bn[a:a + shard], aa[a:a + shard], ba[a:a + shard]) for a in range(0, len(i1), shard)]
    res = pool.map(pair_block, jobs, chunksize=1)
    return pd.DataFrame(np.concatenate(res) if res else np.zeros((0, len(NG_NAMES)), np.float32), columns=NG_NAMES)


if __name__ == "__main__":
    work, ds = sys.argv[1], sys.argv[2]
    S = strings(work, "train")
    out_dir = f"{work}/{ds}n"; os.makedirs(out_dir, exist_ok=True)
    with mp.get_context("fork").Pool(8) as pool:
        for f in sorted(glob.glob(f"{work}/{ds}/*.parquet")):
            out = f"{out_dir}/{os.path.basename(f)}"
            if os.path.exists(out):
                continue
            c = pd.read_parquet(f, columns=["i1", "j", "src_id"])
            F = ngram_features(c.i1.to_numpy(), c.j.to_numpy(), c.src_id.to_numpy(), S, pool)
            assert len(F) == len(c)
            F.to_parquet(out + ".tmp"); os.replace(out + ".tmp", out)
            print(os.path.basename(f), len(F), flush=True)
