# Business Entity Resolution – reproducible pipeline

Normalisation → test-like training-data augmentation (built only from training records) → multi-channel blocking →
cross-S1 competition features → learned candidate filter → ~150-feature XGBoost pair matcher → validation-tuned threshold.
CPU only. Only the provided data is used: no external data, registries, geocoders, APIs or internet sources.
Model: XGBoost (Apache-2.0), far below the 8B-parameter limit. String similarity: RapidFuzz (MIT).

## Environment
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
macOS: XGBoost needs an OpenMP runtime — `brew install libomp`, or set `DYLD_FALLBACK_LIBRARY_PATH` to a directory that
contains `libomp.dylib` (e.g. `<site-packages>/torch/lib`). Do not launch through `nice`/`nohup` on macOS: they strip
`DYLD_*` variables. Linux wheels bundle OpenMP. Tested with Python 3.13, 10 CPU cores, 16 GB RAM (peak ~12 GB).

## Reproduce the final submission (public leaderboard 0.970178)
```
BER_DATA=/path/to/dataset bash run_final.sh           # dataset/ contains train/ and test/ (TSV, as provided)
```
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` under the workspace root (`BER_ROOT`, default: two
levels above this folder). `run_final.sh` runs `run_pipeline.sh` twice and merges:
- **S1 of countries present in training (US, India)** — "rebuild" configuration (tie-inclusive blocking + no-address
  channel, French legal forms, 1.6M training S1), scored by the **mean of three XGBoost matchers** (`train_t2.py`:
  t4 = + S1 anchor-profile counts, t5 = + character 3/4-gram similarities from `ngram.py`, t45 = both), threshold 0.65;
- **S1 of countries absent from training (France)** — default "v3" configuration with the house-number parsing fix
  (`numfix.py`), threshold 0.90.
The split into the two groups is derived from `train_s1.pkl` (no country list in the code). Runtime ≈ 9 h (10 cores,
16 GB). Every step is resumable (finished chunk files are skipped; all writes are temp-file + atomic rename).

Single-configuration run (v3 + house-number fix for unseen countries; leaderboard 0.970168, 5.87 candidates/S1, ≈ 4 h):
`BER_DATA=/path/to/dataset bash run_pipeline.sh`.

**Final candidate set** (`output/candidate_pairs.tsv` = exactly the rows the matchers scored): 11,717,048 pairs,
6.76 per S1 (US 6.49, India 7.22, France 6.03; median 6–7, 99th percentile 12–15, max 39); no S1 without candidates;
reduction ratio vs all S1×(S2∪S3) pairs 0.9999993. Blocking + filter keep ≈ 98.6% of true pairs on validation.
Validate: `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <dataset>/test --check-ids`.

## Pipeline (what `run_pipeline.sh` runs)
| step | script | what it does |
|---|---|---|
| 1 | `prep.py`, `learn_native.py` | normalise names/addresses; learn a native-script→Latin token map from TRAIN positives only; re-normalise |
| 2 | `augment_select.py`, `augment.py` | find "sibling" non-match records of TRAIN S1 entities (same street, similar name, different house number) and a branch-word vocabulary from TRAIN labels; add 0–2 synthetic noisy duplicates per sibling (non-matches by construction) into a separate copy of the train files — makes training data carry the multi-record sibling clusters that the test data has |
| 3 | `prep.py` | normalise augmented train and test into hashed sparse token matrices (name / address / composite regions) |
| 4 | `block_all.py`, `competition.py` | per country, per source, union of top-K records over six retrieval channels; cross-S1 competition features (each S2/S3 record has at most one owner) |
| 5 | `build_v3.py` | S1-level split (fixed 40k validation S1, 150k S1 for the candidate filter, 1.0M S1 for the matcher, disjoint); candidate filter (blocking signals + vectorised name/address similarities, threshold kept ≥99.7% of blocked true pairs on validation); pair features; word-identity (cross-fitted), address-cluster and number-geometry features |
| 5 | `train_v3.py` | XGBoost matcher (streamed QuantileDMatrix); macro-F0.5 on validation; writes `threshold.txt` |
| 6 | `infer_v3.py` | same filter + features + model on every test chunk; S1 of countries absent from training use the house-number parsing fix (`BER_NUMFIX=fr`) |
| 7 | `make_outputs.py` | one row per S1; matches ⊆ candidates; threshold from validation (0.70 v3) and 0.90 for countries absent from training; atomic writes |

Extra steps in `run_final.sh`: `ngram.py` (n-gram features for train/validation pairs) → `train_t2.py` ×3 (t4, t5, t45; 800
rounds; validation macro-F0.5 0.98157 / 0.98153 / 0.98166, mean of the three 0.98184 vs 0.98136 base) → `infer_t.py`
(all three models in one test pass) → `make_hybrid.py` (merge by training/unseen country) → `make_outputs.py`.

## Modules
- `normalize.py` — folding, legal-suffix separation, abbreviation canonicalisation, number extraction, learned native map.
- `blocking.py` — inverted-index retrieval with channels: combined IDF, name, address, composite name|address keys,
  length-normalised name and composite scores. Tokens above 0.3% document frequency are ignored (bounded work per S1).
- `competition.py` — how strongly other S1 entities claim the same record (hash-partitioned, memory-bounded, exact).
- `cheap_filter.py` — candidate filter features and selection rule (always keeps the top 2 per S1).
- `features.py` — string similarities, token overlaps, number conflicts, missingness, per-S1 context, name statistics.
- `wordstats.py` — word-identity evidence, address clusters, number geometry.
- `decide.py` — exact macro F0.5 (empty truth & empty prediction = 1.0) and decision rules; `decide_v3.py` compares rules.
- `numfix.py` — house-number parsing fix: day+French month(+year) tokens of date-named streets ("Allée du 8 Mai 1945")
  are not house numbers; "12bis/ter/quater" → 12.
- `ngram.py` — compact-name 3/4-gram Jaccard / cosine / containment and address 3-gram similarities.
- `train_t2.py` / `infer_t.py` — matcher variants with derived columns (t4 anchor profile, t5 n-grams, t45 both) and their
  test scoring; `make_hybrid.py` — final merge.

Experiments kept for the record (not used by `run_pipeline.sh` / `run_final.sh`): `train_lgb.py`, `compare_models.py`
(LightGBM; needs `lightgbm`), `t3_assign.py`, `t4_abstain.py`, `t6_audit.py`, `france_audit.py`, `france_rules.py`,
`make_blend.py`, `ceiling.py`, `probe_ceiling.py`, `missan.py`, and earlier-stage scripts (`train5.py`, `infer5.py`, …).

## Tests
`python3 -m pytest tests` — metric on hand-built cases (incl. the statement's 0.714 example), split leakage, candidate-filter
minimum, competition features, output format.

## Rules compliance
- No external data. The augmentation only recombines provided TRAIN strings with a vocabulary learned from TRAIN labels.
- The validation split is by Source-1 entity and asserted disjoint; thresholds are chosen on validation only.
- Country is an open-set string (test contains France): no country whitelists or one-hot features.
