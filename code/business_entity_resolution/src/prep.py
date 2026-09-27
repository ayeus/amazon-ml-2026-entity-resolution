"""Stage 1: read raw TSVs, normalise, and build hashed sparse token matrices."""
import os, sys, zlib, time
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from normalize import norm_record, tokens_for_index

Q = 1 << 22
NF = 3 * Q
MASK = NF - 1
HALF = Q  # kept for import compatibility: size of one channel region
NAME_PREF = ('n:', 'np:', 'nc:', 'nc3:', 'g:', 'ne:')
ROOT = os.environ.get("BER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "sr", "dataset"))
WORK = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))


def _chunk(args):
    names, addrs = args
    n = len(names)
    cores, legals, atoks_s = [], [], []
    indptr = np.zeros(n + 1, dtype=np.int64)
    idx = []
    for i in range(n):
        core, legal, at = norm_record(names[i], addrs[i])
        cores.append(" ".join(core)); legals.append(" ".join(legal)); atoks_s.append(" ".join(at))
        toks = set((zlib.crc32(t.encode()) & (Q - 1)) + (0 if t.startswith(NAME_PREF) else (2 * Q if t.startswith('nx:') else Q)) for t in tokens_for_index(core, at))
        idx.extend(toks)
        indptr[i + 1] = indptr[i] + len(toks)
    return cores, legals, atoks_s, np.array(idx, dtype=np.int32), indptr


def prep_file(path, out_prefix, nproc=10, chunk=20000):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    with Pool(nproc) as p:
        res = p.map(_chunk, jobs, chunksize=1)
    cores = sum((r[0] for r in res), []); legals = sum((r[1] for r in res), [])
    atoks = sum((r[2] for r in res), [])
    ind = np.concatenate([r[3] for r in res])
    ptrs, off = [], 0
    for r in res:
        ptrs.append(r[4][:-1] + off); off += r[4][-1]
    ptrs.append(np.array([off])); indptr = np.concatenate(ptrs)
    X = sp.csr_matrix((np.ones(len(ind), dtype=np.float32), ind, indptr), shape=(len(df), NF))
    out = pd.DataFrame({"entity_id": df.entity_id, "country": df.country, "name": df.business_name,
                        "addr": df.business_address, "core": cores, "legal": legals, "atok": atoks})
    out.to_pickle(out_prefix + ".pkl")
    sp.save_npz(out_prefix + "_X.npz", X, compressed=False)
    return len(df)


if __name__ == "__main__":
    split = sys.argv[1]
    os.makedirs(WORK, exist_ok=True)
    for i in (1, 2, 3):
        t = time.time()
        n = prep_file(f"{DATA}/{split}/{split}_source{i}.tsv", f"{WORK}/{split}_s{i}")
        print(split, i, n, f"{time.time()-t:.0f}s", flush=True)
