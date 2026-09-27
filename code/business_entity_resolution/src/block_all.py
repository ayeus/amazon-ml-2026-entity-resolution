"""Full forward blocking for ALL S1 of a split -> resumable per-chunk parquet tables (used for cross-S1 competition
features and as the exact candidate set later fed to the matcher)."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from blocking import load_split, make_ctx, generate_rows

KT, KN, KA = (int(os.environ.get(k, d)) for k, d in (("BER_KT", "15"), ("BER_KN", "12"), ("BER_KA", "12")))
CAP = 3e-3
CHUNK = 100_000

if __name__ == "__main__":
    split = sys.argv[1]
    data = load_split(split)
    ctry = data[1][0].country.to_numpy(dtype=object)
    os.makedirs(f"{WORK}/block_{split}", exist_ok=True)
    for country in sorted(set(ctry)):
        rows = np.flatnonzero(ctry == country)
        todo = [(k, a) for k, a in enumerate(range(0, len(rows), CHUNK)) if not os.path.exists(f"{WORK}/block_{split}/{country}_{k}.parquet")]
        if not todo:
            continue
        ctx = make_ctx(data, country, KT, KN, KA, CAP)
        for k, a in todo:
            t = time.time()
            out = generate_rows(ctx, rows[a:a + CHUNK])
            f = f"{WORK}/block_{split}/{country}_{k}.parquet"
            out.to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
            print(f"{split} {country} chunk {k}/{len(rows)//CHUNK}: {len(out)} rows {time.time()-t:.0f}s", flush=True)
