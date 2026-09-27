"""Cross-S1 competition features (label-free).  Every S2/S3 record belongs to at most one S1 entity, so for a candidate
pair (S1 a, record B) it matters how strongly *other* S1 entities also claim B.  Uses the pool-normalised cosine scores
(cs_all / cs_name / cs_addr) from full forward blocking of ALL S1 of the split.
"""
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK

CH = ("all", "name", "addr")


def _channel(key, v):
    """best_other, gap, rank of this row within its key group for value v (higher = better)."""
    n = len(key)
    order = np.lexsort((-v, key))
    ks = key[order]; vs = v[order]
    start = np.ones(n, dtype=bool); start[1:] = ks[1:] != ks[:-1]
    sidx = np.flatnonzero(start)
    gid = np.cumsum(start) - 1
    pos = np.arange(n) - sidx[gid]                     # 0-based rank in group
    top1 = vs[sidx]
    has2 = np.zeros(len(sidx), dtype=bool)
    gsize = np.diff(np.append(sidx, n))
    has2 = gsize > 1
    top2 = np.zeros(len(sidx), dtype=np.float32)
    top2[has2] = vs[sidx[has2] + 1]
    best_other = np.where(pos == 0, top2[gid], top1[gid]).astype(np.float32)
    out_bo = np.empty(n, np.float32); out_rk = np.empty(n, np.float32); out_sz = np.empty(n, np.float32)
    out_bo[order] = best_other; out_rk[order] = pos + 1; out_sz[order] = gsize[gid]
    return out_bo, out_rk, out_sz


def comp_features(tab):
    """tab: DataFrame with i1, j, src, cs_all, cs_name, cs_addr (ALL S1 of one country)."""
    key = tab.src.to_numpy().astype(np.int64) * (1 << 32) + tab.j.to_numpy().astype(np.int64)
    out = {}
    for c in CH:
        v = tab["cs_" + c].to_numpy().astype(np.float32)
        bo, rk, sz = _channel(key, v)
        out[f"comp_{c}_other"] = bo; out[f"comp_{c}_gap"] = v - bo; out[f"comp_{c}_rk"] = rk
        if c == "all":
            out["comp_n_suitors"] = sz
    return pd.DataFrame(out)


def build_all(split, parts=None):
    """For each country: compute competition features over ALL block chunks and write row-aligned per-chunk comp files.
    Records are hash-partitioned so every (src, j) group lies inside one partition: results are identical to a single
    global pass (same groups, same row order for ties) while peak memory drops ~parts-fold."""
    P = parts or int(os.environ.get("BER_COMP_PARTS", "4"))
    tag = os.environ.get("BER_DS_TAG", "")                  # smoke runs write to a separate folder
    limit = int(os.environ.get("BER_SMOKE_CHUNKS", "0"))
    bdir, cdir = f"{WORK}/block_{split}", f"{WORK}/comp_{split}{tag}"
    os.makedirs(cdir, exist_ok=True)
    by_c = {}
    files = sorted(glob.glob(f"{bdir}/*.parquet"))
    for f in (files[:limit] if limit else files):
        c, k = os.path.basename(f)[:-8].rsplit("_", 1); by_c.setdefault(c, []).append((int(k), f))
    for c, lst in by_c.items():
        lst.sort()
        if all(os.path.exists(f"{cdir}/{c}_{k}.parquet") for k, _ in lst):
            continue
        t = time.time()
        chunks = [pd.read_parquet(f, columns=["i1", "j", "src", "cs_all", "cs_name", "cs_addr"]) for _, f in lst]
        part_of = [(ch.j.to_numpy().astype(np.int64) * 31 + ch.src.to_numpy().astype(np.int64)) % P for ch in chunks]
        cols = None
        for p in range(P):
            idx = [np.flatnonzero(po == p) for po in part_of]
            tab = pd.concat([ch.iloc[ix] for ch, ix in zip(chunks, idx)], ignore_index=True)
            C = comp_features(tab); cols = list(C.columns); C = C.to_numpy(np.float32)
            a = 0
            for (k, _), ix in zip(lst, idx):
                np.save(f"{cdir}/.{c}_{k}_p{p}.npy", C[a:a + len(ix)]); a += len(ix)
            del tab, C
        for (k, _), ch, po in zip(lst, chunks, part_of):
            o = np.empty((len(ch), len(cols)), np.float32)
            for p in range(P):
                tmp = f"{cdir}/.{c}_{k}_p{p}.npy"; o[po == p] = np.load(tmp); os.remove(tmp)
            f = f"{cdir}/{c}_{k}.parquet"; pd.DataFrame(o, columns=cols).to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
        print(f"comp {split} {c}: {sum(len(ch) for ch in chunks)} rows, {P} partitions, {time.time()-t:.0f}s", flush=True)
        del chunks, part_of


if __name__ == "__main__":
    build_all(sys.argv[1])
