import os, sys, zlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from decide import score_selection
from evalblock import truth_pairs

va = pd.read_parquet(f"{WORK}/val_pred3.parquet"); meta = pd.read_pickle(f"{WORK}/ds_meta.pkl"); mva = meta[meta.split == "val"]
print("ORACLE macro F0.5 (select exactly the true pairs present in my candidates):", round(score_selection(va, va.y.values == 1, mva)[0], 5))
print("model at thr 0.65:", round(score_selection(va, va.p.values >= 0.65, mva)[0], 5))
d = {s: pd.read_pickle(f"{WORK}/train_s{s}.pkl") for s in (1, 2, 3)}
ids = {s: d[s].entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
pos = {s: pd.Series(np.arange(len(ids[s])), index=ids[s]) for s in (1, 2, 3)}
tp = truth_pairs(); tp["i1"] = pos[1].reindex(tp.source1_entity_id.values).values
tp = tp[tp.i1.isin(set(mva.i1))].copy(); tp["src"] = tp.m.str[1].astype(int); tp["j"] = -1
for s in (2, 3):
    mk = tp.src.values == s; tp.loc[mk, "j"] = pos[s].reindex(tp.m.values[mk]).values
key = lambda a, b, c: a.astype(np.int64) * 100_000_000_000 + b.astype(np.int64) * 10 + c
tp["inc"] = np.isin(key(tp.i1.values, tp.j.values, tp.src.values), key(va.i1.values, va.j.values, va.src_id.values))
ms = tp[~tp.inc].copy()
print("true pairs", len(tp), "missed by blocking+filter", len(ms), f"({len(ms)/len(tp):.4f})")
def g(col, s, idx): return d[s][col].values[idx]
A_core = d[1].core.values[ms.i1.values]; A_at = d[1].atok.values[ms.i1.values]
B_core = np.empty(len(ms), object); B_at = np.empty(len(ms), object)
for s in (2, 3):
    mk = ms.src.values == s; B_core[mk] = d[s].core.values[ms.j.values[mk]]; B_at[mk] = d[s].atok.values[ms.j.values[mk]]
ms["b_noaddr"] = (B_at == "")
ms["name_eq"] = np.array([a.replace(" ", "") == b.replace(" ", "") for a, b in zip(A_core, B_core)])
from rapidfuzz.distance import JaroWinkler
ms["name_jw"] = [JaroWinkler.normalized_similarity(a, b) for a, b in zip(A_core, B_core)]
ms["addr_ov"] = [len(set(a.split()) & set(b.split())) / max(len(set(b.split())), 1) if b else -1 for a, b in zip(A_at, B_at)]
c1 = d[1].core.str.replace(" ", "").value_counts()
ms["s1_name_mult"] = pd.Series(A_core).str.replace(" ", "").map(c1).values
ms["country"] = d[1].country.values[ms.i1.values]
print("\nB has no address:", round(ms.b_noaddr.mean(), 3), "| name identical:", round(ms.name_eq.mean(), 3),
      "| name identical & no address:", round((ms.b_noaddr & ms.name_eq).mean(), 3))
print("S1 name multiplicity among S1 (compact name shared by k S1):", ms.s1_name_mult.clip(upper=5).value_counts(normalize=True).sort_index().round(3).to_dict())
ms["cat"] = np.select([ms.b_noaddr & ms.name_eq, ms.b_noaddr & (ms.name_jw >= 0.85), ms.b_noaddr,
                       (ms.name_jw < 0.6) & (ms.addr_ov >= 0.5), ms.name_jw < 0.6, ms.addr_ov < 0.3],
                      ["noaddr_name_identical", "noaddr_name_similar", "noaddr_name_differs", "junk_name_addr_matches", "junk_name_addr_weak", "name_ok_addr_weak"], "other")
print("\nmissed pairs by category:"); print(ms.cat.value_counts().to_frame("n").assign(share=lambda x: (x.n / len(ms)).round(3)).to_string())
print("\nmissed by country:", ms.country.value_counts().to_dict(), "| by src:", ms.src.value_counts().to_dict())
ms.to_pickle(f"{WORK}/val_missed.pkl")
