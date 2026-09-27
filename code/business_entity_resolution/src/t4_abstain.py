"""Task 4a: precision abstention for the no-address / shared-name bucket (label-free segment, threshold tuned on val).

Segment = candidate record B has no address AND the S1 compact name is shared by >= 2 S1 (e_nm1_cnt >= 2).
Rows in the segment use threshold thr_seg; all other rows keep the global threshold.
"""
import os, sys
import numpy as np, pandas as pd
from decide import score_selection

R = os.environ.get("BER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


def attrs(D):
    """b_noaddr / S1 name multiplicity / S1-no-address from the ORIGINAL train records (synthetic rows: has address)."""
    s1 = pd.read_pickle(f"{R}/work_train/train_s1.pkl")
    comp = s1.core.str.replace(" ", "", regex=False); nm1 = comp.map(comp.value_counts()).to_numpy()
    bno = np.zeros(len(D), bool)
    for s in (2, 3):
        o = pd.read_pickle(f"{R}/work_orig/train_s{s}.pkl"); at = (o.atok.to_numpy(dtype=object) == "")
        mk = (D.src_id.values == s) & (D.j.values < len(o)); bno[mk] = at[D.j.values[mk]]
    return bno, nm1[D.i1.values], (s1.atok.to_numpy(dtype=object) == "")[D.i1.values]


if __name__ == "__main__":
    for name, vp, mp, thr in [("REBUILD", f"{R}/work_train/val_pred_v3.parquet", f"{R}/work_train/ds_meta.pkl", 0.65),
                              ("V3", f"{R}/work4/val_pred_v3.parquet", f"{R}/work4/ds_meta.pkl", 0.70)]:
        D = pd.read_parquet(vp); meta = pd.read_pickle(mp); meta = meta[meta.split == "val"].reset_index(drop=True)
        bno, nm1, a_no = attrs(D); p = D.p.to_numpy(); y = D.y.to_numpy()
        base = score_selection(D, p >= thr, meta)[0]
        print(f"\n##### {name} baseline {base:.5f} @ {thr}")
        for sname, seg in [("B no-addr & name shared>=2", bno & (nm1 >= 2)), ("B no-addr & unique name", bno & (nm1 == 1)),
                           ("B no-addr (any)", bno), ("S1 no-addr & name shared>=2", a_no & (nm1 >= 2))]:
            sel = p >= thr
            print(f"  [{sname}] rows {seg.sum()}, pos {y[seg].sum()}, TP {(sel & seg & (y == 1)).sum()}, "
                  f"FN {(~sel & seg & (y == 1)).sum()}, FP {(sel & seg & (y == 0)).sum()}")
            out = []
            for ts in (0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99):
                t = np.where(seg, ts, thr)
                out.append(f"{ts}:{score_selection(D, p >= t, meta)[0] - base:+.5f}")
            print("    seg thr -> delta: " + "  ".join(out))
