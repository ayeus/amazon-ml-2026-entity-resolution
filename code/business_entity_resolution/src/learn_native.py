"""Learn native-script token -> Latin token map from TRAIN-split S1 positive pairs only (no external data).

For every true pair whose S2/S3 name contains non-Latin tokens, count co-occurrence with the S1 name tokens
(core+legal).  A native token maps to the S1 token with the highest count if support>=MIN_SUP and
confidence (count / occurrences) >= MIN_CONF.
"""
import os, sys, json, zlib, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK
from evalblock import truth_pairs

MIN_SUP, MIN_CONF = 3, 0.5


def is_val(ids):
    return np.array([zlib.crc32(("split" + i).encode()) % 5 == 0 for i in ids])


def nonlatin(t):
    return not t.isascii()


if __name__ == "__main__":
    dfs = {s: pd.read_pickle(f"{WORK}/train_s{s}.pkl") for s in (1, 2, 3)}
    tp = truth_pairs()
    s1 = dfs[1]
    s1_tr = set(s1.entity_id.values[~is_val(s1.entity_id.values)])
    tp = tp[tp.source1_entity_id.isin(s1_tr)]
    s1_tok = dict(zip(s1.entity_id.values, (s1.core + " " + s1.legal).str.split()))
    other = {}
    for s in (2, 3):
        d = dfs[s]
        mk = d.core.str.contains(r"[^\x00-\x7f]", regex=True).values
        other.update(dict(zip(d.entity_id.values[mk], (d.core.values[mk] + " " + d.legal.values[mk]))))
    co = collections.defaultdict(collections.Counter); occ = collections.Counter(); lf = collections.Counter()
    for a, b in zip(tp.source1_entity_id.values, tp.m.values):
        nm = other.get(b)
        if nm is None:
            continue
        nt = [t for t in nm.split() if nonlatin(t)]
        lt = set(s1_tok[a])
        if nt:
            lf.update(lt)
        for t in set(nt):
            occ[t] += 1
            for l in lt:
                co[t][l] += 1
    mp = {}
    for t, c in co.items():
        l = max(c, key=lambda x: 2 * c[x] / (occ[t] + lf[x])); n = c[l]
        if n >= MIN_SUP and n / occ[t] >= MIN_CONF:
            mp[t] = l
    tot = sum(occ.values()); cov = sum(occ[t] for t in mp)
    print(f"native tokens seen {len(occ)} mapped {len(mp)}; occurrence coverage {cov/tot:.3f}")
    for t in list(mp)[:15]:
        print(t, "->", mp[t], co[t][mp[t]], occ[t])
    json.dump(mp, open(f"{WORK}/native_map.json", "w"), ensure_ascii=False)
