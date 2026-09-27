"""Unseen-country (France) post-processing rules on top of model probabilities (strings only, no labels, no external data).
R1 reject: the record adds a sibling/branch word to the S1 name (Groupe, Participations, Developpement, Distribution,
   Holding, International, France, or the train-learned English branch vocabulary) AND its house number differs from the
   S1's  -> reject it and every other record of this S1 sharing that house-number key (the sibling's cluster).
R2 accept: same compact core name, same house number, and >=70% of the record's address tokens in the S1 address.
Writes a scored folder with adjusted p (1.0 accepted / 0.0 rejected) for unseen-country S1 only."""
import os, sys, glob, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import wordstats as ws

FR_BRANCH = {"groupe", "group", "participations", "developpement", "distribution", "holding", "holdings", "international", "france"}

if __name__ == "__main__":
    work, src_scored, dst_scored, train_countries = sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4].split(","))
    vocab = FR_BRANCH | set(json.load(open(sys.argv[5]))) if len(sys.argv) > 5 else FR_BRANCH
    d = {s: pd.read_pickle(f"{work}/test_s{s}.pkl") for s in (1, 2, 3)}
    os.makedirs(f"{work}/{dst_scored}_test", exist_ok=True)
    stats = {"rows": 0, "R1_rejected": 0, "R2_accepted": 0}
    for f in sorted(glob.glob(f"{work}/{src_scored}_test/*.parquet")):
        S = pd.read_parquet(f); out = f"{work}/{dst_scored}_test/{os.path.basename(f)}"
        unseen = ~np.isin(d[1].country.values[S.i1.values], list(train_countries))
        if unseen.any():
            U = S[unseen].copy()
            a_core = d[1].core.values[U.i1.values]; a_at = d[1].atok.values[U.i1.values]
            b_core = np.empty(len(U), object); b_at = np.empty(len(U), object)
            for s in (2, 3):
                m = U.src.values == s; b_core[m] = d[s].core.values[U.j.values[m]]; b_at[m] = d[s].atok.values[U.j.values[m]]
            rel, _, bkey = ws.number_geometry(a_at, b_at)
            added = [set(y.split()) - set(x.split()) for x, y in zip(a_core, b_core)]
            branch = np.array([bool(ad & vocab) for ad in added])
            U["rel"], U["bkey"], U["branch_alt"] = rel, bkey, branch & (rel >= 2)
            # propagate to the record's house-number cluster within the same S1
            ck = U.i1.astype(str) + "|" + U.bkey.astype(str)
            bad = set(ck[U.branch_alt & (U.bkey != "")])
            r1 = U.branch_alt.values | (ck.isin(bad).values & (rel >= 2))
            same = np.array([x.replace(" ", "") == y.replace(" ", "") and x != "" for x, y in zip(a_core, b_core)])
            aov = np.array([len(set(y.split()) & set(x.split())) / max(len(set(y.split())), 1) if y else 0 for x, y in zip(a_at, b_at)])
            r2 = same & (rel == 1) & (aov >= 0.7) & ~r1
            p = U.p.values.copy(); p[r1] = 0.0; p[r2] = 1.0
            S.loc[unseen, "p"] = p
            stats["R1_rejected"] += int(r1.sum()); stats["R2_accepted"] += int((r2 & (U.p.values < 1.0)).sum())
        stats["rows"] += len(S)
        S.to_parquet(out)
    print(stats)
