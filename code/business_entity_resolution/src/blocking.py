"""Stage 2: multi-channel candidate generation (blocking) with an IDF-weighted inverted index.

Name tokens hash into the lower half of the feature space, address tokens into the upper half, so three
retrieval channels are available from one sparse matrix: NAME, ADDR and ALL (= NAME + ADDR).  For every
Source-1 entity and every source (S2, S3) we take the top-K of each channel and union them.  The NAME channel
finds records whose address is missing/noisy, the ADDR channel finds records with a junk/alias name at the
right address, ALL handles the general case.  Channel scores/ranks are kept as model features.
Tokens whose document frequency exceeds `cap_frac * pool` are ignored for retrieval (bounded work per entity).
"""
import os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK, NF, HALF

KEEP = ["entity_id", "country", "core", "legal", "atok"]
CH = "tnaxcye"   # + e: name score among records WITHOUT an address (few ties there)  # retrieval channels: t=name+addr, n=name, a=address, x=composite name-token|address-token
_G = {}


def load_split(split, with_X=True):
    out = {}
    for i in (1, 2, 3):
        df = pd.read_pickle(f"{WORK}/{split}_s{i}.pkl")[KEEP]
        f = f"{WORK}/{split}_s{i}_X.npz"
        X = sp.load_npz(f).tocsr() if (with_X and os.path.exists(f)) else None   # only blocking needs X
        out[i] = (df, X)
    return out


TIES = os.environ.get("BER_TIES", "0") == "1"   # keep all records tied with the K-th score (capped at 3K)


TIE_CH = os.environ.get("BER_TIE_CH", "tnaxcye")   # channels that keep boundary ties (when BER_TIES=1)


def _topk(S, K, ties=True):
    """Row-wise top-K of CSR -> arrays (row, col, val, rank).  With BER_TIES=1 records tied with the K-th best score are
    kept too (up to 3K): tied records are indistinguishable at this stage, so the filter/matcher should decide."""
    R, C, V, RK = [], [], [], []
    if K <= 0:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float32), np.zeros(0, np.int16))
    ip, ind, dat = S.indptr, S.indices, S.data
    for r in range(S.shape[0]):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        d = dat[a:b]
        if b - a > K:
            sel = np.argpartition(-d, K - 1)[:K]
            if TIES and ties:
                kth = d[sel].min()
                tied = np.flatnonzero(d >= kth)
                sel = tied if len(tied) <= 3 * K else tied[np.argsort(-d[tied], kind="stable")[:3 * K]]
        else:
            sel = np.arange(b - a)
        sel = sel[np.argsort(-d[sel], kind="stable")]
        R.append(np.full(len(sel), r, dtype=np.int64)); C.append(ind[a:b][sel].astype(np.int64))
        V.append(d[sel]); RK.append(np.arange(len(sel), dtype=np.int16))
    if not R:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float32), np.zeros(0, np.int16))
    return np.concatenate(R), np.concatenate(C), np.concatenate(V).astype(np.float32), np.concatenate(RK)


def _lookup(S, r, c):
    if len(r) == 0:
        return np.zeros(0, np.float32)
    return np.asarray(S[r, c]).ravel().astype(np.float32)


def _work(args):
    lo, hi = args
    g = _G
    X1 = g["X1"][lo:hi]
    Sn = (X1 @ g["wn"]).tocsr(); Sa = (X1 @ g["wa"]).tocsr(); Sx = (X1 @ g["wx"]).tocsr()
    selfn = np.asarray(Sn.sum(axis=1)).ravel().astype(np.float32); selfa = np.asarray(Sa.sum(axis=1)).ravel().astype(np.float32)
    parts = []
    for src, PT in ((2, g["PT2"]), (3, g["PT3"])):
        Mn = (Sn @ PT).tocsr(); Ma = (Sa @ PT).tocsr(); Mt = (Mn + Ma).tocsr(); Mx = (Sx @ PT).tocsr()
        keys = {}
        Mc = (Mn @ g["invB"][src][0]).tocsr(); My = (Mx @ g["invB"][src][2]).tocsr()  # cosine-style: favour short/clean B
        Me = (Mn @ g["noaddr"][src]).tocsr()                                             # name score, no-address records only
        for ch, M, K in (("t", Mt, g["Kt"]), ("n", Mn, g["Kn"]), ("a", Ma, g["Ka"]), ("x", Mx, g["Kx"]),
                         ("c", Mc, g["Kc"]), ("y", My, g["Kc"]), ("e", Me, g["Ke"])):
            r, c, v, rk = _topk(M, K, ch in TIE_CH)
            keys[ch] = (r, c, rk)
        allr = np.concatenate([keys[k][0] for k in CH]); allc = np.concatenate([keys[k][1] for k in CH])
        code = allr * (1 << 30) + allc
        u, first = np.unique(code, return_index=True)
        r = allr[first]; c = allc[first]
        d = {"i1": r + lo, "j": c, "src": np.full(len(r), src, np.int8),
             "s_all": _lookup(Mt, r, c), "s_name": _lookup(Mn, r, c), "s_addr": _lookup(Ma, r, c), "s_x": _lookup(Mx, r, c)}
        bn = g["selfB"][src][0][c]; ba = g["selfB"][src][1][c]
        an = selfn[r]; aa = selfa[r]
        d["cs_name"] = (d["s_name"] / np.sqrt(an * bn + 1e-6)).astype(np.float32)
        d["cs_addr"] = (d["s_addr"] / np.sqrt(aa * ba + 1e-6)).astype(np.float32)
        d["cs_all"] = (d["s_all"] / np.sqrt((an + aa) * (bn + ba) + 1e-6)).astype(np.float32)
        for k in CH:  # rank of each pair within each channel (99 = not retrieved by that channel)
            rk = np.full(len(r), 99, np.int16)
            kc = keys[k][0] * (1 << 30) + keys[k][1]
            idx = np.searchsorted(u, kc)
            tmp = np.full(len(u), 99, np.int16); tmp[idx] = keys[k][2]
            d["r_" + k] = tmp
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


def make_ctx(data, country, Kt=15, Kn=12, Ka=12, cap_frac=3e-3, Kx=None):
    """Build the (country-level) pool structures once; returns ctx used by generate_rows."""
    df1, X1 = data[1]
    m1 = np.flatnonzero(df1.country.values == country)
    pools = {}
    for s in (2, 3):
        df, X = data[s]
        rows = np.flatnonzero(df.country.values == country)
        pools[s] = (rows, X[rows])
    Xp = sp.vstack([pools[2][1], pools[3][1]]).tocsr()
    N = Xp.shape[0]
    dfc = np.bincount(Xp.indices, minlength=NF).astype(np.float64)
    idf = np.log((N + 1) / (dfc + 1)).astype(np.float32)
    cap = max(30, int(cap_frac * N))
    w = np.where((dfc > 0) & (dfc <= cap), idf, 0).astype(np.float32)
    wn = w.copy(); wn[HALF:] = 0
    wa = w.copy(); wa[:HALF] = 0; wa[2 * HALF:] = 0
    wx = w.copy(); wx[:2 * HALF] = 0
    capx = int(os.environ.get("BER_CAPX", "0"))   # composite keys are meant to be rare: optional tighter cap
    if capx > 0:
        wx[dfc > capx] = 0
    selfB = {s: (np.asarray(pools[s][1] @ wn).ravel().astype(np.float32), np.asarray(pools[s][1] @ wa).ravel().astype(np.float32)) for s in (2, 3)}
    invB = {}; noaddr = {}
    for s in (2, 3):
        noaddr[s] = sp.diags((np.asarray(pools[s][1] @ wa).ravel() == 0).astype(np.float32))
        bx = np.asarray(pools[s][1] @ wx).ravel()
        invB[s] = tuple(sp.diags((1.0 / np.sqrt(v + 1.0)).astype(np.float32)) for v in (selfB[s][0], selfB[s][1], bx))
    _G.update(dict(X1all=X1, wn=sp.diags(wn), wa=sp.diags(wa), wx=sp.diags(wx), Kt=Kt, Kn=Kn, Ka=Ka,
                   Kx=Kx if Kx is not None else int(os.environ.get("BER_KX", "10")), Kc=int(os.environ.get("BER_KC", "8")),
                   Ke=int(os.environ.get("BER_KE", "0")), noaddr=noaddr, invB=invB, selfB=selfB,
                   PT2=pools[2][1].T.tocsr(), PT3=pools[3][1].T.tocsr()))
    return dict(m1=m1, pools=pools, N=N, country=country)


def generate_rows(ctx, s1_rows=None, nproc=10, chunk=2000):
    """Candidates for the given S1 rows (default: all S1 of the country)."""
    m1 = ctx["m1"] if s1_rows is None else np.intersect1d(ctx["m1"], s1_rows)
    pools, country = ctx["pools"], ctx["country"]
    _G["X1"] = _G["X1all"][m1]
    jobs = [(a, min(a + chunk, len(m1))) for a in range(0, len(m1), chunk)]
    t = time.time()
    with mp.get_context("fork").Pool(nproc) as p:
        res = p.map(_work, jobs, chunksize=1)
    out = pd.concat(res, ignore_index=True)
    print(f"  blocking {country}: {len(m1)} S1 x {ctx['N']} pool -> {len(out)/max(len(m1),1):.1f} cand/S1 in {time.time()-t:.0f}s", flush=True)
    out["i1"] = m1[out.i1.values]
    j = np.empty(len(out), dtype=np.int64)
    for s in (2, 3):
        mk = out.src.values == s
        j[mk] = pools[s][0][out.j.values[mk]]
    out["j"] = j
    return out


def generate(data, country, s1_rows=None, Kt=15, Kn=12, Ka=12, cap_frac=3e-3, nproc=10, chunk=2000):
    """Backward-compatible: build ctx and generate for the given rows."""
    ctx = make_ctx(data, country, Kt, Kn, Ka, cap_frac)
    return generate_rows(ctx, s1_rows, nproc, chunk), None
