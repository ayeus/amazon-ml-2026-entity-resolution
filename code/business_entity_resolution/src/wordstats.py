"""Word-identity and address-cluster evidence (stage-2 features).

1. Word evidence: for every candidate pair, the words B adds to S1's name ("extra") and the words it drops
   ("missing").  Each word carries a smoothed match rate learned from TRAIN-split labelled pairs only
   (cross-fitted for the training rows themselves).  Words never seen in training and not a typo of an S1 word are
   counted separately ("unknown"), which is how unseen-country branch words (e.g. French "groupe") are caught.
2. Address-cluster propagation: within one S1's candidate set, records sharing the same house-number key belong to
   the same physical address and usually the same real entity.  Each record receives its cluster's worst word evidence,
   so a clean-looking duplicate of a branch record ("holdings" next to "... Downtown") inherits the branch signal.
3. Number geometry: numeric relation between S1's and B's house numbers (equal / typo / truncation / unrelated).
"""
import os, sys
import numpy as np, pandas as pd
import multiprocessing as mp
from collections import Counter
from rapidfuzz.distance import Levenshtein, JaroWinkler

PRIOR_N = 20.0
MIN_KNOWN = 20


def learn(a_core, b_core, y):
    """Counts of (n, positives) for extra and missing words.  a_core/b_core: arrays of normalised name strings."""
    ex_n, ex_p, ms_n, ms_p = Counter(), Counter(), Counter(), Counter()
    for a, b, l in zip(a_core, b_core, y.astype(int).tolist()):
        sa, sb = set(a.split()), set(b.split())
        for t in sb - sa:
            ex_n[t] += 1; ex_p[t] += l
        for t in sa - sb:
            ms_n[t] += 1; ms_p[t] += l
    base = float(np.mean(y))
    return dict(ex_n=ex_n, ex_p=ex_p, ms_n=ms_n, ms_p=ms_p, base=base)


def _rate(n, p, base):
    return (p + PRIOR_N * base) / (n + PRIOR_N)


def _is_typo(t, others):
    for o in others:
        if abs(len(o) - len(t)) <= 2 and (Levenshtein.distance(t, o) <= 1 or JaroWinkler.normalized_similarity(t, o) >= 0.9):
            return True
    return False


_S = {}


def _pair_block(args):
    a_arr, b_arr = args
    st = _S["st"]; base = st["base"]
    exn, exp_, msn, msp = st["ex_n"], st["ex_p"], st["ms_n"], st["ms_p"]
    out = np.zeros((len(a_arr), len(PAIR_NAMES)), np.float32)
    for k, (a, b) in enumerate(zip(a_arr, b_arr)):
        ta, tb = a.split(), b.split(); sa, sb = set(ta), set(tb)
        ex = sb - sa; ms = sa - sb
        ex_r, ex_unk, ex_typo = [], 0, 0
        for t in ex:
            n = exn.get(t, 0)
            if _is_typo(t, sa):
                ex_typo += 1; continue
            if n < MIN_KNOWN:
                ex_unk += 1; continue
            ex_r.append(_rate(n, exp_.get(t, 0), base))
        ms_r, ms_unk = [], 0
        for t in ms:
            if _is_typo(t, sb):
                continue
            n = msn.get(t, 0)
            if n < MIN_KNOWN:
                ms_unk += 1; continue
            ms_r.append(_rate(n, msp.get(t, 0), base))
        out[k] = (min(ex_r) if ex_r else 1.0, float(np.mean(ex_r)) if ex_r else 1.0, len(ex_r), ex_unk, ex_typo,
                  min(ms_r) if ms_r else 1.0, len(ms_r), ms_unk)
    return out


PAIR_NAMES = ["w_ex_min", "w_ex_mean", "w_ex_n", "w_ex_unk", "w_ex_typo", "w_ms_min", "w_ms_n", "w_ms_unk"]


def pair_word_features(a_core, b_core, stats, nproc=10, shard=100_000):
    _S["st"] = stats
    jobs = [(a_core[i:i + shard], b_core[i:i + shard]) for i in range(0, len(a_core), shard)]
    with mp.get_context("fork").Pool(nproc) as p:
        res = p.map(_pair_block, jobs, chunksize=1)
    return pd.DataFrame(np.concatenate(res) if res else np.zeros((0, len(PAIR_NAMES)), np.float32), columns=PAIR_NAMES)


def _nums(s):
    return [int(t) for t in s.split() if t.isdigit() and len(t) <= 9]


def number_geometry(a_atok, b_atok):
    """rel: 0 missing, 1 equal, 2 typo(edit<=1), 3 truncation/prefix/suffix, 4 unrelated; dist: min |a-b| (log1p)."""
    rel = np.zeros(len(a_atok), np.float32); dist = np.full(len(a_atok), -1.0, np.float32); bkey = np.empty(len(a_atok), object)
    for k, (a, b) in enumerate(zip(a_atok, b_atok)):
        A, B = _nums(a), _nums(b)
        bkey[k] = " ".join(str(x) for x in sorted(set(B)))
        if not A or not B:
            continue
        sa, sb = set(A), set(B)
        if sa & sb:
            rel[k] = 1; dist[k] = 0; continue
        r = 4
        for x in sa:
            xs = str(x)
            for z in sb:
                zs = str(z)
                if Levenshtein.distance(xs, zs) <= 1:
                    r = min(r, 2)
                elif xs.startswith(zs) or xs.endswith(zs) or zs.startswith(xs) or zs.endswith(xs):
                    r = min(r, 3)
        rel[k] = r
        dist[k] = np.log1p(min(abs(x - z) for x in sa for z in sb))
    return rel, dist, bkey


CLUSTER_NAMES = ["n_rel", "n_dist", "cl_size", "cl_agree", "s1_support", "cl_ex_min", "cl_ex_unk", "cl_ex_n",
                 "cl_jw_min", "cl_other_size", "cl_is_alt"]


def cluster_features(F, i1, a_atok, b_atok):
    """F must hold w_ex_min, w_ex_unk, w_ex_n, jw (row-aligned).  Adds number-geometry + cluster-propagated evidence."""
    rel, dist, bkey = number_geometry(a_atok, b_atok)
    F["n_rel"] = rel; F["n_dist"] = dist
    has = np.array([k != "" for k in bkey])
    agree = (rel == 1)
    F["s1_support"] = pd.Series(agree.astype(np.float32)).groupby(i1).transform("sum").values
    key = pd.util.hash_array(bkey.astype(object)) ^ (i1.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    key = np.where(has, key, np.uint64(0))
    k = pd.Series(key)
    g = F.groupby(key)
    size = k.groupby(key).transform("size").values.astype(np.float32)
    F["cl_size"] = np.where(has, size, 0)
    F["cl_agree"] = np.where(has, agree, -1).astype(np.float32)
    for src, dst, how in (("w_ex_min", "cl_ex_min", "min"), ("w_ex_unk", "cl_ex_unk", "max"), ("w_ex_n", "cl_ex_n", "max"),
                          ("jw", "cl_jw_min", "min")):
        v = g[src].transform(how).values
        F[dst] = np.where(has, v, F[src].values)
    # number of candidates of this S1 that sit in a DIFFERENT non-empty number cluster
    n_has = pd.Series(has.astype(np.float32)).groupby(i1).transform("sum").values
    F["cl_other_size"] = np.where(has, n_has - size, n_has)
    F["cl_is_alt"] = ((rel >= 2) & has & (F["s1_support"].values > 0)).astype(np.float32)
    return F
