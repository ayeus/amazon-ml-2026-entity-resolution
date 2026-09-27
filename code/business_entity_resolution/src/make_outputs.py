"""Stage 6: turn scored candidates into the two submission files (atomic write) and print sanity counts.

matching_results.tsv : selected pairs (p >= thr; optional one-owner conflict resolution per S2/S3 record)
candidate_pairs.tsv  : the exact candidate set that was fed to the matcher (after the cheap filter)
"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK, ROOT


def write_atomic(path, lines):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    if os.path.getsize(tmp) == 0:
        os.remove(tmp); raise RuntimeError("refusing to write a 0-byte output")
    os.replace(tmp, path)


def lists_by_s1(n1, i1, ids):
    """Comma-joined id list per S1 row index (empty string when none); input arrays already ordered by (i1, -p)."""
    out = [""] * n1
    if len(i1) == 0:
        return out
    starts = np.flatnonzero(np.r_[True, i1[1:] != i1[:-1]]); ends = np.r_[starts[1:], len(i1)]
    for s, e in zip(starts, ends):
        out[i1[s]] = ",".join(ids[s:e])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--resolve", action="store_true", help="keep each S2/S3 record only for its highest-probability S1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scored", default="scored", help="scored-candidates folder prefix (scored / scored5)")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--blank-country", default=None, help="diagnostic only: predict empty for every S1 of this country")
    ap.add_argument("--thr-unseen", type=float, default=None,
                    help="threshold for S1 whose country never occurs in the TRAINING data (derived from the train files, "
                         "not hard-coded); chosen on the manual audit set because no labels exist for such countries")
    ap.add_argument("--train-work", default=None, help="work dir holding train_s1.pkl (to derive the set of training countries)")
    a = ap.parse_args()
    ids = {s: pd.read_pickle(f"{WORK}/{a.split}_s{s}.pkl").entity_id.to_numpy(dtype=object) for s in (1, 2, 3)}
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{WORK}/{a.scored}_{a.split}/*.parquet"))], ignore_index=True)
    n1 = len(ids[1])
    print("scored candidate rows", len(S), "S1 with candidates", S.i1.nunique(), "of", n1, flush=True)
    S["cid"] = np.where(S.src.values == 2, ids[2][np.minimum(S.j.values, len(ids[2]) - 1)], ids[3][np.minimum(S.j.values, len(ids[3]) - 1)])
    S = S.sort_values(["i1", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    cand_lists = lists_by_s1(n1, S.i1.values, S.cid.values)
    thr_row = np.full(len(S), a.thr, dtype=np.float32)
    if a.thr_unseen is not None:
        seen = set(pd.read_pickle(f"{a.train_work or WORK}/train_s1.pkl").country.unique())
        c1 = pd.read_pickle(f"{WORK}/{a.split}_s1.pkl").country.to_numpy(dtype=object)
        unseen = ~np.isin(c1[S.i1.values], list(seen))
        thr_row[unseen] = a.thr_unseen
        print(f"training countries {sorted(seen)}; {int(unseen.sum())} candidate rows of unseen-country S1 use threshold {a.thr_unseen}", flush=True)
    sel = S[S.p.values >= thr_row]
    if a.blank_country:
        c1 = pd.read_pickle(f"{WORK}/{a.split}_s1.pkl").country.to_numpy(dtype=object)
        sel = sel[c1[sel.i1.values] != a.blank_country]
        print(f"DIAGNOSTIC: blanked all {a.blank_country} S1 ({(c1 == a.blank_country).sum()})", flush=True)
    if a.resolve:
        best = sel.groupby(["src", "j"]).p.transform("max")
        nb = len(sel); sel = sel[sel.p >= best]
        sel = sel.drop_duplicates(["src", "j"], keep="first")
        print(f"conflict resolution removed {nb - len(sel)} of {nb} selected pairs", flush=True)
    match_lists = lists_by_s1(n1, sel.i1.values, sel.cid.values)
    os.makedirs(a.outdir, exist_ok=True)
    write_atomic(os.path.join(a.outdir, "candidate_pairs.tsv"), ["source1_entity_id\tcandidate_entity_ids"] + [f"{i}\t{c}" for i, c in zip(ids[1], cand_lists)])
    write_atomic(os.path.join(a.outdir, "matching_results.tsv"), ["source1_entity_id\tmatched_entity_ids"] + [f"{i}\t{c}" for i, c in zip(ids[1], match_lists)])
    nm = sum(1 for x in match_lists if x)
    print(f"matching_results: {nm}/{n1} S1 with >=1 match ({n1-nm} empty), {len(sel)} pairs | candidates: {len(S)} pairs, {len(S)/n1:.1f}/S1")
