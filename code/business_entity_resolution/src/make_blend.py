"""Cross-family blend for US/India: rebuild candidates scored by mean(XGB t45, LightGBM t45); where v3 also scored the same
(S1, record) pair, p = W*p_v3 + (1-W)*that mean (validation: W=0.2 -> 0.98211 @0.62 vs 0.98201 without v3).
The candidate set is the rebuild's (every row scored); France copied from a v3 scored folder.  work2 and work_test tables
are row-aligned, so (i1, j, src) keys are comparable.
Usage: python3 make_blend.py <out_name> <W> <france_scored_dir>
"""
import os, sys, glob
import numpy as np, pandas as pd

ROOT = os.environ.get("BER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

if __name__ == "__main__":
    out_name, W, fr_dir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    key = lambda d: d.i1.to_numpy(np.int64) * 100_000_000_000 + d.j.to_numpy(np.int64) * 10 + d.src.to_numpy(np.int64)
    v3 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/work2/scored6_test/*.parquet"))
                    if not os.path.basename(f).startswith("France")], ignore_index=True)
    vmap = pd.Series(v3.p.to_numpy(np.float64), index=key(v3)); del v3
    out = f"{ROOT}/work_test/{out_name}_test"; os.makedirs(out, exist_ok=True)
    n_all = n_has = 0
    for f in sorted(glob.glob(f"{ROOT}/work_test/scored6_t45lgb_test/*.parquet")):
        d = pd.read_parquet(f); xl = (d.p.to_numpy(np.float64) + d.p_lgb.to_numpy(np.float64)) / 2
        pv = np.array(vmap.reindex(key(d)).to_numpy(), dtype=np.float64, copy=True); has = ~np.isnan(pv)
        p = np.where(has, W * np.nan_to_num(pv) + (1 - W) * xl, xl).astype(np.float32)
        b = os.path.basename(f); d[["i1", "j", "src"]].assign(p=p).to_parquet(f"{out}/{b}.tmp"); os.replace(f"{out}/{b}.tmp", f"{out}/{b}")
        n_all += len(d); n_has += int(has.sum())
    for f in sorted(glob.glob(f"{fr_dir}/France_*.parquet")):
        b = os.path.basename(f); pd.read_parquet(f)[["i1", "j", "src", "p"]].to_parquet(f"{out}/{b}.tmp"); os.replace(f"{out}/{b}.tmp", f"{out}/{b}")
    print(f"{out}: US/India rows {n_all}, with a v3 score {n_has} ({n_has / max(n_all, 1):.1%}), W={W}; France from {fr_dir}")
