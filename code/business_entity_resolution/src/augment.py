"""Training-data augmentation: give sibling businesses the multi-record clusters that the test data has.

Measured (label-free) on test vs train: look-alike "sibling" businesses (S1 name + a branch word, at a nearby house
number) exist in both, with the same number-offset and branch-word statistics, but in train such a sibling has ONE
record while in test it has 2-3 noisy records (cluster sizes 1/2/3 = ~52/38/9% US, 67/26/7% India).

For every TRAINING sibling record that is a labelled non-match (y=0) at a different house number from the S1's
supported number and with a similar name, we add k synthetic duplicates (k drawn from the test cluster-size profile)
at the same address, built only from provided training strings:
  name    = the S1 raw name, optionally + a branch word (vocabulary learned from train: extra words whose train match
            rate is ~0), optionally a legal-suffix change, case change, typo
  address = the sibling's raw address with case / number-format variation
Synthetic records are non-matches by construction (duplicates of a known non-match), so the ground truth is unchanged.
Output: a separate data directory (raw data is never modified).
"""
import os, sys, random, re, shutil, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prep import WORK, DATA

K_EXTRA_P = {0: 0.45, 1: 0.42, 2: 0.13}   # extra duplicates per sibling cluster (-> sizes 1/2/3 ~ test profile)
LEGAL = {"US": ["LLC", "Inc", "Inc.", "Corp", "Co", "Corporation", "L.L.C.", "LP"],
         "India": ["Pvt Ltd", "Private Limited", "Ltd", "LLP", "Pvt. Ltd.", "Limited"]}


def branch_vocab(path=None):
    if path and os.path.exists(path):
        return json.load(open(path))
    R = pd.read_pickle(f"{WORK}/extra_word_stats.pkl")
    R = R[(R.n >= 300) & (R.true_rate <= 0.01)]
    return sorted(w for w in R.index if w.isascii() and w.isalpha() and len(w) >= 3)


def typo(s, rng):
    words = s.split()
    idx = [i for i, w in enumerate(words) if len(w) >= 5]
    if not idx:
        return s
    i = rng.choice(idx); w = list(words[i]); k = rng.randrange(1, len(w) - 1)
    if rng.random() < 0.5:
        w[k], w[k + 1] = w[k + 1], w[k]
    else:
        del w[k]
    words[i] = "".join(w)
    return " ".join(words)


def name_variant(s1_name, country, vocab, rng):
    n = s1_name
    if rng.random() < 0.55:
        bw = rng.choice(vocab).title()
        n = f"{n} {bw}" if rng.random() < 0.8 else f"{bw} {n}"
    if rng.random() < 0.35:
        n = f"{n} {rng.choice(LEGAL.get(country, LEGAL['US']))}"
    if rng.random() < 0.25:
        n = typo(n, rng)
    r = rng.random()
    return n.upper() if r < 0.3 else (n.lower() if r < 0.4 else n)


def addr_variant(addr, rng):
    a = addr
    if rng.random() < 0.15:
        a = re.sub(r"^(\d+)", lambda m: "#" + m.group(1), a)
    if rng.random() < 0.1:
        a = re.sub(r"^(\d+)", lambda m: "00" + m.group(1), a)
    r = rng.random()
    return a.upper() if r < 0.35 else (a.title() if r < 0.5 else a)


if __name__ == "__main__":
    out_dir = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(seed); nrng = np.random.RandomState(seed)
    sel_path = sys.argv[3] if len(sys.argv) > 3 else None
    vocab = branch_vocab(os.path.join(os.path.dirname(sel_path), "branch_vocab.json") if sel_path else None); print(f"branch vocabulary learned from train: {len(vocab)} words, e.g. {vocab[:12]}", flush=True)
    raw = {i: pd.read_csv(f"{DATA}/train/train_source{i}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3) for i in (1, 2, 3)}
    sel_file = sys.argv[3] if len(sys.argv) > 3 else None   # parquet (i1, j, src_id) from augment_select.py
    rows = []
    for nm in (() if sel_file else ("train", "val")):
        D = pd.read_parquet(f"{WORK}/ds4_{nm}.parquet", columns=["i1", "j", "src_id", "y", "jw", "a_sub_b", "b_sub_a"])
        X = pd.read_parquet(f"{WORK}/ds5x_{nm}.parquet", columns=["n_rel", "s1_support"])
        D = pd.concat([D, X], axis=1)
        sib = D[(D.y == 0) & (D.n_rel >= 2) & (D.s1_support > 0) & ((D.jw >= 0.85) | (D.a_sub_b == 1) | (D.b_sub_a == 1))]
        rows.append(sib[["i1", "j", "src_id"]])
    sib = (pd.read_parquet(sel_file) if sel_file else pd.concat(rows, ignore_index=True)).drop_duplicates(["j", "src_id"])
    print(f"sibling non-match records eligible for cloning: {len(sib)}", flush=True)
    k = nrng.choice(list(K_EXTRA_P), size=len(sib), p=list(K_EXTRA_P.values()))
    s1 = raw[1]
    new = {2: [], 3: []}; seq = 0
    for (i1, j, src), kk in zip(sib[["i1", "j", "src_id"]].itertuples(index=False), k):
        if kk == 0:
            continue
        a = s1.iloc[int(i1)]; b = raw[int(src)].iloc[int(j)]
        for _ in range(int(kk)):
            tgt = int(src) if rng.random() < 0.5 else (5 - int(src))      # duplicates may land in either source
            seq += 1
            new[tgt].append((f"S{tgt}-A{seq:08d}", name_variant(a.business_name, a.country, vocab, rng),
                             addr_variant(b.business_address, rng), b.country))
    os.makedirs(f"{out_dir}/train", exist_ok=True)
    shutil.copy(f"{DATA}/train/train_source1.tsv", f"{out_dir}/train/train_source1.tsv")
    shutil.copy(f"{DATA}/train/train_ground_truth.tsv", f"{out_dir}/train/train_ground_truth.tsv")
    for i in (2, 3):
        add = pd.DataFrame(new[i], columns=["entity_id", "business_name", "business_address", "country"])
        out = pd.concat([raw[i], add], ignore_index=True)
        out.to_csv(f"{out_dir}/train/train_source{i}.tsv", sep="\t", index=False, quoting=3)
        print(f"source{i}: {len(raw[i])} original + {len(add)} synthetic sibling duplicates", flush=True)
    for i in (2, 3):
        print(pd.DataFrame(new[i][:4]).to_string(header=False, index=False))
