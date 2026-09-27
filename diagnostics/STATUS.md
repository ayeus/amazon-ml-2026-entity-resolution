# Status & resume plan (updated 2026-09-27 ~13:45 IST, paused until 15:00)

Deadline: final submission by **~2026-09-28 00:45 IST**. Submissions: **17 total, 5 used, 12 left** (user clarified 16:50).

## Leaderboard history (public)
| # | model | score |
|---|---|---|
| 1 | big XGBoost (base features), thr 0.70 | 0.930429 |
| 2 | France blanked (diagnostic) | 0.806 → France ≈ 0.89, US+India ≈ 0.94 |
| 3 | stage-2 `model_s5` (word identity + address clusters), thr 0.70 | 0.950981 |
| 4 | v3: augmented training + new blocking + name-sim filter, thr 0.70 | 0.96936 |
| 5 | **v3 + unseen-country threshold 0.90** | **0.970129** (files in `output/`) |

v3 test-like validation (US+India) 0.9805 → implied **France ≈ 0.91** (costs ~0.011 on LB). NEXT: France (label-free analysis: region vs department names, French branch words, address formats).

## Key findings (measured)
- Test has "sibling businesses" (S1 name + branch word at a nearby number) with 2–3 records each; train has 1. This is the val→test gap.
- Augmentation with synthetic sibling duplicates built only from train records reproduces the test-sized drop on validation
  (smoke, India: `model_s5` 0.968 clean → 0.939 augmented) and a model trained on it wins big on that test-like validation
  (smoke: 0.9499 → 0.9733, 1 chunk of training data only).
- Candidate filter + cheap name/address similarities (RapidFuzz cpdist): ~6.6 cand/S1 at 99.8% kept vs ~12 without.
- Clean, reproducible sibling selection (`augment_select.py`): 116.9M key-joined pairs → 12.06M address-similar →
  **1.25M siblings**, branch vocabulary of **57 words** (group, holdings, north…west, metro, downtown, …; excludes noise
  words services/center/partners). `augment.py` → **852,805** synthetic records.

## Directories
- `work2/`: TEST prep + blocking (new channels, 19 chunks) — done; `scored5_test/` = scores of the 0.951 model.
- `work4/`: clean final chain (train side): `sel/` (selection, vocab, cached pairs), prep, blocking, datasets, models.
- `work/`: small files for the calibration check (`model_s5.json`, `model5_feats.json`, `word_stats_full.pkl`), `ds_meta.pkl`.

## Running now: `work4/run_chain.sh` (log `work4/chain.log`), one heavy step at a time
augment ✔ → prep ✔ → block train (started 11:39, ~65 min) → competition train+test → build_v3 (cheap, dataset, stage2) →
train_v3 700 rounds (+ calibration check with model_s5). Then: `infer_v3.py test` (BER_WORK=work2, BER_MODEL_DIR=work4) →
`make_outputs.py --scored scored6 --thr $(cat work4/threshold.txt)` → validator `--check-ids` → intermediate submission.

## Environment notes
- XGBoost needs `DYLD_FALLBACK_LIBRARY_PATH=<python>/site-packages/torch/lib`; never launch via `nice`/`nohup` (they strip DYLD vars).
- 16 GB RAM: one heavy job at a time.
