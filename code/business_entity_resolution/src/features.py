"""Stage 3a: pair features for (S1 entity, S2/S3 candidate).

All features are language/country agnostic.  String similarities use RapidFuzz (MIT) on the normalised strings
produced by normalize.py.  Context features are computed per S1 candidate set (never use labels).
"""
import os, sys
import numpy as np, pandas as pd
import multiprocessing as mp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

_D = {}
NEW_CHANNEL_COLS = ("s_x", "r_x", "r_c", "r_y", "r_e")


def set_data(data):
    """data = {1: (df, X), 2: ..., 3: ...} from blocking.load_split"""
    _D.clear()
    for s in (1, 2, 3):
        df = data[s][0]
        _D[s] = (df.core.to_numpy(dtype=object), df.legal.to_numpy(dtype=object), df.atok.to_numpy(dtype=object))
    _pool_stats()


def _acr(a_tokens, cb):
    return 1.0 if len(a_tokens) >= 2 and cb == "".join(t[0] for t in a_tokens) else 0.0


def _row(ac, al, aa, bc, bl, ba):
    ta = ac.split(); tb = bc.split(); sa = set(ta); sb = set(tb)
    ca = ac.replace(" ", ""); cb = bc.replace(" ", "")
    ia = sa & sb
    ua = set(aa.split()); ub = set(ba.split())
    da = {t for t in ua if t[0].isdigit()}; db = {t for t in ub if t[0].isdigit()}
    ix = ua & ub
    ixw = {t for t in ix if not t[0].isdigit()}
    nb, na = len(ub), len(ua)
    la = set(al.split()); lb = set(bl.split())
    return (
        JaroWinkler.normalized_similarity(ac, bc), Levenshtein.normalized_similarity(ac, bc),
        fuzz.token_set_ratio(ac, bc) / 100, fuzz.token_sort_ratio(ac, bc) / 100, fuzz.partial_ratio(ac, bc) / 100,
        Levenshtein.normalized_similarity(ca, cb), JaroWinkler.normalized_similarity(ca, cb), fuzz.partial_ratio(ca, cb) / 100,
        float(ac == bc), float(ca == cb),
        len(ia) / max(len(sb), 1), len(ia) / max(len(sa), 1), len(ia) / max(len(sa | sb), 1),
        len(sb - sa), len(sa - sb), len(ta), len(tb), len(ca) - len(cb),
        _acr(ta, cb), _acr(tb, ca), float(len(tb) > 0 and sb <= sa), float(len(ta) > 0 and sa <= sb),
        float(bool(la) and la == lb), float(not lb), float(bool(la & lb)),
        float(ta[:1] == tb[:1]), float(ta[-1:] == tb[-1:]),
        # address
        float(nb == 0), na, nb, len(ix), len(ix) / max(nb, 1), len(ix) / max(na, 1), len(ix) / max(len(ua | ub), 1),
        len(ixw), len(da & db), float(bool(da) and bool(db) and not (da & db)), float(bool(db)), float(bool(da & db)),
        fuzz.token_set_ratio(aa, ba) / 100 if nb else -1.0, fuzz.partial_ratio(aa, ba) / 100 if nb else -1.0,
        fuzz.token_sort_ratio(aa, ba) / 100 if nb else -1.0,
        float(nb > 0 and ba.split()[-1] in ua),
    )


NAMES = ["jw", "lev", "tsr", "tsort", "pr", "clev", "cjw", "cpr", "eq", "ceq", "b_in_a", "a_in_b", "jac", "b_minus_a",
         "a_minus_b", "lta", "ltb", "clen_d", "acr_a", "acr_b", "b_sub_a", "a_sub_b", "legal_eq", "b_nolegal", "legal_ov",
         "first_eq", "last_eq",
         "b_noaddr", "ata", "atb", "a_ix", "a_bin", "a_ain", "a_jac", "a_ixw", "num_ix", "num_conf", "b_hasnum", "num_hit",
         "a_tsr", "a_pr", "a_tsort", "state_hit"]


def _shard(args):
    i1, j, src = args
    out = np.empty((len(i1), len(NAMES)), dtype=np.float32)
    A = _D[1]
    for k in range(len(i1)):
        B = _D[int(src[k])]
        out[k] = _row(A[0][i1[k]], A[1][i1[k]], A[2][i1[k]], B[0][j[k]], B[1][j[k]], B[2][j[k]])
    return out


def pair_features(cand, nproc=10, shard=200_000):
    """cand: DataFrame with i1, j, src (+ channel columns). Returns feature DataFrame (same row order)."""
    n = len(cand)
    i1 = cand.i1.to_numpy(); j = cand.j.to_numpy(); src = cand.src.to_numpy()
    jobs = [(i1[a:a + shard], j[a:a + shard], src[a:a + shard]) for a in range(0, n, shard)]
    with mp.get_context("fork").Pool(nproc) as p:
        res = p.map(_shard, jobs, chunksize=1)
    F = pd.DataFrame(np.concatenate(res) if res else np.zeros((0, len(NAMES)), np.float32), columns=NAMES)
    assert len(F) == n, "feature stage must preserve every candidate row"
    for c in ("s_all", "s_name", "s_addr"):
        F[c] = cand[c].to_numpy()
    for c in ("r_t", "r_n", "r_a"):
        F[c] = cand[c].to_numpy().astype(np.float32)
    for c in NEW_CHANNEL_COLS:            # composite (x) and cosine-normalised (c, y) retrieval channels, if blocked with them
        if c in cand:
            F[c] = cand[c].to_numpy().astype(np.float32)
    F["src"] = (src == 3).astype(np.float32)
    # B-side keys for sibling (noisy-duplicate) grouping
    bname = np.empty(n, dtype=object); baddr = np.empty(n, dtype=object)
    for s in (2, 3):
        mk = src == s
        bname[mk] = _D[s][0][j[mk]]; baddr[mk] = _D[s][2][j[mk]]
    add_context(F, i1, bname, baddr)
    return F


def _h(i1, strs):
    return pd.util.hash_array(np.asarray(strs, dtype=object)) ^ (i1.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))


def add_context(F, i1, bname, baddr):
    """Per-S1 candidate-set context: rank / gap to best / sibling-cluster agreement.  Never uses labels."""
    g = i1
    F["n_cand"] = pd.Series(g).groupby(g).transform("size").astype(np.float32).values
    for c in ("s_all", "s_name", "s_addr", "jw", "tsr", "clev", "a_jac", "a_tsr"):
        gb = F[c].groupby(g)
        mx = gb.transform("max").values
        F[c + "_mx"] = mx
        F[c + "_gap"] = mx - F[c].values
        F[c + "_rk"] = gb.rank(ascending=False, method="min").values.astype(np.float32)
    F["comb"] = F.jw + F.a_jac.clip(lower=0) + F.s_all / (F.s_all_mx + 1e-6)
    gb = F["comb"].groupby(g)
    F["comb_gap"] = gb.transform("max").values - F["comb"].values
    F["comb_rk"] = gb.rank(ascending=False, method="min").values.astype(np.float32)
    # margin between best and 2nd best combined evidence in the S1's candidate set
    tmp = pd.DataFrame({"g": g, "c": F["comb"].values})
    top2 = tmp.sort_values(["g", "c"], ascending=[True, False]).groupby("g").c.nth(1)
    sec = pd.Series(top2.values, index=tmp.sort_values(["g", "c"], ascending=[True, False]).groupby("g").g.nth(1).values)
    F["comb_2nd"] = pd.Series(g).map(sec).fillna(0).values.astype(np.float32)
    # sibling agreement: candidates with the same compact name / same address string are duplicates of one real entity
    cname = np.array([x.replace(" ", "") for x in bname], dtype=object)
    for tag, keys in (("nm", cname), ("ad", baddr)):
        h = pd.Series(_h(g, keys))
        F[f"sib_{tag}_n"] = h.groupby(h.values).transform("size").astype(np.float32).values
        F[f"sib_{tag}_comb_mx"] = F["comb"].groupby(h.values).transform("max").values
        F[f"sib_{tag}_comb_mean"] = F["comb"].groupby(h.values).transform("mean").values
    F.loc[baddr == "", ["sib_ad_n", "sib_ad_comb_mx", "sib_ad_comb_mean"]] = [0, 0, 0]
    # cluster evidence: similarity of every candidate to the S1's best ("anchor") candidate (noisy duplicates of one entity)
    anchor = F["comb"].groupby(g).transform("idxmax").to_numpy().astype(np.int64)
    an_jw = np.empty(len(F), np.float32); an_ts = np.empty(len(F), np.float32); an_ad = np.empty(len(F), np.float32)
    an_eq = np.empty(len(F), np.float32)
    for k in range(len(F)):
        q = anchor[k]
        if q == k:
            an_jw[k] = 1.0; an_ts[k] = 1.0; an_ad[k] = 1.0; an_eq[k] = 1.0
            continue
        an_jw[k] = JaroWinkler.normalized_similarity(bname[k], bname[q]); an_ts[k] = fuzz.token_set_ratio(bname[k], bname[q]) / 100
        an_ad[k] = fuzz.token_set_ratio(baddr[k], baddr[q]) / 100 if baddr[k] and baddr[q] else -1.0
        an_eq[k] = float(baddr[k] == baddr[q] and bool(baddr[k]))
    F["an_jw"], F["an_ts"], F["an_ad"], F["an_eq"] = an_jw, an_ts, an_ad, an_eq
    F["an_comb"] = F["comb"].values[anchor]
    # how the S1's best candidate compares to this candidate's cluster best
    F["cluster_gap"] = F.groupby(g)["comb"].transform("max").values - np.maximum(F.sib_nm_comb_mx, F.sib_ad_comb_mx)


_E = {}
S1_MASK = None


def _pool_stats():
    """Label-free name statistics of the split being processed (S1 vocabulary / name multiplicities)."""
    from collections import Counter
    c1 = _D[1][0]
    if S1_MASK is not None:   # simulate a smaller S1 set (dropped S1 entities contribute no statistics)
        c1 = np.where(S1_MASK, c1, "\x00dropped")
    vocab = Counter(t for x in c1 for t in x.split())
    comp1 = pd.Series([x.replace(" ", "") for x in c1])
    cnt1 = comp1.value_counts()
    _E["nm1_cnt"] = comp1.map(cnt1).to_numpy(dtype=np.float32)
    def tokstats(x):
        ts = x.split()
        if not ts:
            return (0.0, 0.0)
        l = [np.log1p(vocab.get(t, 0)) for t in ts]
        return (min(l), sum(l) / len(l))
    tk = np.array([tokstats(x) for x in c1], dtype=np.float32)
    _E["tk1_min"], _E["tk1_mean"] = tk[:, 0], tk[:, 1]
    compB = {s: pd.Series([x.replace(" ", "") for x in _D[s][0]]) for s in (2, 3)}
    poolcnt = pd.concat([compB[2], compB[3]]).value_counts()
    for s in (2, 3):
        _E[f"nmB_cnt{s}"] = compB[s].map(poolcnt).to_numpy(dtype=np.float32)
        _E[f"nmB_s1_{s}"] = compB[s].map(cnt1).fillna(0).to_numpy(dtype=np.float32)
        kn = np.empty(len(_D[s][0]), np.float32); nu = np.empty(len(_D[s][0]), np.float32)
        for i, x in enumerate(_D[s][0]):
            ts = x.split()
            k = sum(1 for t in ts if t in vocab)
            kn[i] = k / len(ts) if ts else 1.0; nu[i] = len(ts) - k
        _E[f"known{s}"], _E[f"nunk{s}"] = kn, nu


EXTRA = ["e_nm1_cnt", "e_tk1_min", "e_tk1_mean", "e_nmB_cnt", "e_nmB_s1", "e_known", "e_nunk", "e_allunk"]


def extra_features(i1, j, src):
    n = len(i1); out = np.zeros((n, len(EXTRA)), np.float32)
    out[:, 0] = _E["nm1_cnt"][i1]; out[:, 1] = _E["tk1_min"][i1]; out[:, 2] = _E["tk1_mean"][i1]
    for s in (2, 3):
        mk = src == s
        jj = j[mk]
        out[mk, 3] = _E[f"nmB_cnt{s}"][jj]; out[mk, 4] = _E[f"nmB_s1_{s}"][jj]
        out[mk, 5] = _E[f"known{s}"][jj]; out[mk, 6] = _E[f"nunk{s}"][jj]
        out[mk, 7] = ((_E[f"known{s}"][jj] == 0) & (_E[f"nunk{s}"][jj] > 0)).astype(np.float32)
    return pd.DataFrame(out, columns=EXTRA)


ROWLEVEL = None


def row_cols():
    return NAMES + ["s_all", "s_name", "s_addr", "r_t", "r_n", "r_a", "src"]


def recontext(F, i1, j, src):
    """Drop previously computed per-S1 context columns and recompute them on the (filtered) candidate set."""
    keep = [c for c in F.columns if c in set(row_cols()) or c in EXTRA or c.startswith("comp_") or c in ("y", "i1", "j", "src_id", "p")]
    F = F[keep].reset_index(drop=True)
    bname = np.empty(len(F), dtype=object); baddr = np.empty(len(F), dtype=object)
    for s in (2, 3):
        mk = src == s
        bname[mk] = _D[s][0][j[mk]]; baddr[mk] = _D[s][2][j[mk]]
    add_context(F, i1, bname, baddr)
    return F


# Base matcher feature list (pair + context + name-statistics + competition + blocking features).
BASE_FEATS = ["jw", "lev", "tsr", "tsort", "pr", "clev", "cjw", "cpr", "eq", "ceq", "b_in_a", "a_in_b", "jac", "b_minus_a", "a_minus_b", "lta", "ltb", "clen_d", "acr_a", "acr_b", "b_sub_a", "a_sub_b", "legal_eq", "b_nolegal", "legal_ov", "first_eq", "last_eq", "b_noaddr", "ata", "atb", "a_ix", "a_bin", "a_ain", "a_jac", "a_ixw", "num_ix", "num_conf", "b_hasnum", "num_hit", "a_tsr", "a_pr", "a_tsort", "state_hit", "s_all", "s_name", "s_addr", "r_t", "r_n", "r_a", "src", "e_nm1_cnt", "e_tk1_min", "e_tk1_mean", "e_nmB_cnt", "e_nmB_s1", "e_known", "e_nunk", "e_allunk", "comp_all_other", "comp_all_gap", "comp_all_rk", "comp_n_suitors", "comp_name_other", "comp_name_gap", "comp_name_rk", "comp_addr_other", "comp_addr_gap", "comp_addr_rk", "n_cand", "s_all_mx", "s_all_gap", "s_all_rk", "s_name_mx", "s_name_gap", "s_name_rk", "s_addr_mx", "s_addr_gap", "s_addr_rk", "jw_mx", "jw_gap", "jw_rk", "tsr_mx", "tsr_gap", "tsr_rk", "clev_mx", "clev_gap", "clev_rk", "a_jac_mx", "a_jac_gap", "a_jac_rk", "a_tsr_mx", "a_tsr_gap", "a_tsr_rk", "comb", "comb_gap", "comb_rk", "comb_2nd", "sib_nm_n", "sib_nm_comb_mx", "sib_nm_comb_mean", "sib_ad_n", "sib_ad_comb_mx", "sib_ad_comb_mean", "an_jw", "an_ts", "an_ad", "an_eq", "an_comb", "cluster_gap"]
