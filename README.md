# Amazon ML Challenge 2026 – Business Entity Resolution

Match each Source-1 business record to its duplicates in Sources 2 and 3 (US, India, France), scored by macro F0.5 per
Source-1 entity. Final public leaderboard score: **0.970178**.

Approach: normalisation → test-like training-data augmentation (built only from training records) → multi-channel
blocking → cross-S1 competition features → learned candidate filter → ~150-feature XGBoost pair matcher →
validation-tuned thresholds. CPU only, no external data.

## Layout
| Path | Contents |
|---|---|
| `code/business_entity_resolution/` | Reproducible pipeline (`run_final.sh`, `run_pipeline.sh`, `src/`, `tests/`). Its README covers setup, every step and rules compliance. |
| `final_models/` | XGBoost matchers, candidate filters and word statistics behind the final submission. |
| `diagnostics/` | Data audit, research notes, status log, France audit labels. |
| `amazon_ml_entity_resolution_audit_preprocessing (4).ipynb`, `blocking_recall_recovery_92.py` | Early exploration: data audit and blocking-recall work. |

## Reproduce
```
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
BER_DATA=/path/to/dataset bash run_final.sh   # dataset/ holds train/ and test/ TSVs as provided
```
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` at the repo root (~9 h on 10 cores / 16 GB).

## Not in this repo
- The competition dataset (not redistributable) and the augmented training data derived from it.
- Intermediate work directories and submission outputs (~16 GB in total); `run_final.sh` regenerates the final output.
