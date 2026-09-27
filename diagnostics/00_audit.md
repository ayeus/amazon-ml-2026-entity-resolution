# 00 Audit (Step 0)

## Environment (probed)
- macOS (Darwin 25.6), 10 CPUs, 16 GiB RAM, 67 GiB free disk (85% used). Python 3.13.5.
- Data is LOCAL: `student_resource 2/dataset/{train,test}/*.tsv` (symlinked as `sr/`). No Drive mount (`/Volumes` = Macintosh HD only).
- Installed: duckdb 1.3.2, pyarrow 21, pandas 3.0.2, numpy 2.3.5, scipy 1.16.1, scikit-learn 1.7.1, xgboost 3.1.3, RapidFuzz 3.14.6 (I installed), faiss-cpu, torch, sentence-transformers.
- NOT installed: lightgbm, catboost.

## CONTEXT.md and prior artifacts: NOT FOUND
`find / -iname CONTEXT.md -o -iname '*.parquet'` returned nothing. There are no candidate-set files, no Parquet, no val split.
Consequently these claims cannot be verified and are treated as non-existent: train 18,903,790 / val 3,348,934 / test 11,408,860 candidate rows.
Step-0 items 3 (verify existing candidate sets) is therefore N/A; the only candidate generation is the prototype I wrote this session.

## Raw data (measured, python/pandas, this session)
| file | data rows |
|---|---|
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| train ground truth | 2,206,821 (one row per S1; S1 ids unique, 0 dup) |
| test S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 |
- No 0-byte files. Country counts: train US/India only; test also France (259,452 S1 / 703,378 S2 / 731,615 S3).
- Singleton rate (S1 with no true match), train: 123,247 / 2,206,821 = 5.58%.
- True matches per S1 (train): 0:123,247 1:119,157 2:375,212 3:530,841 4:484,115 5:321,957 6:164,868 7:63,968 8:18,680 9:4,205 10:534 11:37.
- 7,638,365 true pairs; all matched S2/S3 ids are UNIQUE, i.e. each S2/S3 record belongs to at most one S1 (S1 may still have many matches). 73.4% of S2 and 74.6% of S3 records are matched to some S1.
  This does not contradict "not one-to-one" (S1->many is allowed); it is a structural constraint usable in decision/context features.

## Blocking prototype (my code: src/normalize.py, prep.py, blocking.py, evalblock.py)
Measured with `python3 src/evalblock.py 20000 <cap> <K>`: random 20,000 train S1 (seed 0; 12,042 US, 7,958 India), truth = all S2+S3 true pairs, pool = ALL train S2/S3 of that country. Pair recall:
| cap_frac | K per source | US | India | cand/S1 |
|---|---|---|---|---|
| 1e-4 | 15 | 0.9064 | 0.8761 | ~29 |
| 1e-3 | 40 | 0.9663 | 0.9253 | ~80 |
| 1e-2 | 40 | 0.9827 | 0.9360 | ~80 |
Blocking runtime is small (1-17 s per country for 20k S1).

## Verdict
- CORRECT/keep: normalisation + hashed-token inverted-index blocking design; runtime is fine; the one-to-one-per-S2/S3 finding.
- RISKY: recall is BELOW the 97% gate, India clearly weakest (0.936 best). IDF/df cap statistics are computed from the S2/S3 pool, not "train S1 only" (unsupervised pool stats; at test time they come from the test pool) - flag for your decision.
- NOT YET MEASURED: S2 vs S3 recall split, fraction of S1 with zero candidates (and how many of those have true matches), candidates-per-S1 distribution, recall on a proper S1-level validation split (I have no train/val split yet), miss analysis by noise type.
- BROKEN/none. DISCARD: nothing.
- Missing: Devanagari transliteration is not implemented (13% of India S2 and 7.5% of India S3 names are Devanagari, likely the India miss source; unverified).
- I wrote the blocking prototype BEFORE this audit was requested. No feature/model/decision code exists.
