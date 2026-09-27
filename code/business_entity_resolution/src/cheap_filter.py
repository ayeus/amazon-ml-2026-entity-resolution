"""Stage 1.5: cheap learned candidate filter on blocking-only signals (scores, ranks, cross-S1 competition).
Output = the final candidate set that is fed to the matcher (candidate_pairs.tsv)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK

BASE = ["s_all", "s_name", "s_addr", "cs_all", "cs_name", "cs_addr", "r_t", "r_n", "r_a", "src"]
OPT = ["s_x", "r_x", "r_c", "r_y", "r_e"]   # new retrieval channels (composite + cosine-normalised)
COMP = ["comp_all_other", "comp_all_gap", "comp_all_rk", "comp_name_other", "comp_name_gap", "comp_name_rk",
        "comp_addr_other", "comp_addr_gap", "comp_addr_rk", "comp_n_suitors"]


def _cheap_sims(D):
    """Vectorised (C++, multithreaded) name/address similarities for every blocking candidate.  Uses the normalised strings
    registered by features.set_data(); returns None if they are not available."""
    import features as F
    from rapidfuzz.process import cpdist
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz import fuzz
    if not F._D:
        return None
    i1 = D["i1"].to_numpy(); j = D["j"].to_numpy(); src = D["src"].to_numpy()
    a_core = F._D[1][0][i1]; a_at = F._D[1][2][i1]
    b_core = np.empty(len(D), object); b_at = np.empty(len(D), object)
    for s in (2, 3):
        m = src == s
        b_core[m] = F._D[s][0][j[m]]; b_at[m] = F._D[s][2][j[m]]
    kw = dict(workers=-1, dtype=np.float32)
    return {"c_jw": cpdist(a_core, b_core, scorer=JaroWinkler.normalized_similarity, **kw),
            "c_tsr": cpdist(a_core, b_core, scorer=fuzz.token_set_ratio, **kw) / 100,
            "c_atsr": cpdist(a_at, b_at, scorer=fuzz.token_set_ratio, **kw) / 100}


def cheap_feats(D):
    """D needs: i1, src (int 2/3), s_*, cs_*, r_*, comp_*.  Adds within-S1 rank/gap over the FULL blocking candidate set."""
    X = pd.DataFrame({c: D[c].to_numpy() for c in BASE[:-1]})
    X["src"] = (D["src"].to_numpy() == 3).astype(np.float32)
    for c in OPT:
        if c in D:
            X[c] = D[c].to_numpy().astype(np.float32)
    for c in COMP:
        X[c] = D[c].to_numpy()
    g = D["i1"].to_numpy()
    for c in ("cs_all", "cs_name", "cs_addr"):
        gb = X[c].groupby(g)
        X[c + "_gap"] = gb.transform("max").to_numpy() - X[c].to_numpy()
        X[c + "_rk"] = gb.rank(ascending=False, method="min").to_numpy().astype(np.float32)
    X["n_full"] = pd.Series(g).groupby(g).transform("size").to_numpy().astype(np.float32)
    sims = _cheap_sims(D)
    if sims is not None:
        for k, v in sims.items():
            X[k] = v
        X["c_jw_gap"] = pd.Series(sims["c_jw"]).groupby(g).transform("max").to_numpy() - sims["c_jw"]
    return X


def select(p, i1, tau, min_keep=2):
    """keep p>=tau, and always the top `min_keep` per S1 (so every S1 with candidates keeps something)."""
    rk = pd.Series(p).groupby(i1).rank(ascending=False, method="first").to_numpy()
    return (p >= tau) | (rk <= min_keep)
