#!/usr/bin/env bash
# End-to-end reproduction: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# Usage:  BER_DATA=/path/to/dataset  bash run_pipeline.sh        (dataset/ must contain train/ and test/)
# CPU only. Peak RAM ~12 GB. Runs one heavy step at a time. Every step is resumable (existing chunk outputs are skipped).
set -euo pipefail
cd "$(dirname "$0")/src"
ROOT="${BER_ROOT:-$(cd ../../.. && pwd)}"
export BER_ROOT="$ROOT"
export BER_DATA="${BER_DATA:-$ROOT/sr/dataset}"          # raw data, never modified
# Default = "v3" configuration (alone: leaderboard 0.970168 with the unseen-country threshold 0.90 and the house-number
# parsing fix for countries absent from training, 5.87 candidates per S1). The FINAL submission (0.970178) is produced by
# ../run_final.sh, which runs this script twice (v3 and "rebuild" configurations) and merges them.
# Rebuild configuration: BER_TIES=1 BER_TIE_CH=nae BER_KE=5 BER_FR_NORM=1 BER_MATCH_S1=1600000 BER_ROUNDS=800.
export BER_KX="${BER_KX:-10}" BER_KC="${BER_KC:-6}" BER_KE="${BER_KE:-0}" BER_TIES="${BER_TIES:-0}"   # blocking channels
export BER_COMP_PARTS="${BER_COMP_PARTS:-8}" BER_MATCH_S1="${BER_MATCH_S1:-1000000}" BER_FR_NORM="${BER_FR_NORM:-0}"
ROUNDS="${BER_ROUNDS:-700}"
# macOS without `brew install libomp`: point this at any directory containing libomp.dylib (e.g. torch/lib)
export DYLD_FALLBACK_LIBRARY_PATH="${BER_LIBOMP_DIR:-${DYLD_FALLBACK_LIBRARY_PATH:-}}"   # BER_LIBOMP_DIR survives protected launchers
W0="$ROOT/work_orig"; WT="$ROOT/work_train"; WX="$ROOT/work_test"; AUG="$ROOT/aug_data"
OUT="${BER_OUT:-$ROOT/output}"
mkdir -p "$W0" "$WT" "$WX" "$OUT"

echo "[1/7] normalise original train; learn native-script->Latin map from TRAIN positives; re-normalise"
BER_WORK="$W0" python3 prep.py train
BER_WORK="$W0" python3 learn_native.py
export BER_NATIVE_MAP="$W0/native_map.json"
BER_WORK="$W0" python3 prep.py train

echo "[2/7] test-like augmentation built only from training records (sibling duplicates, non-matches by construction)"
BER_WORK="$W0" python3 augment_select.py "$W0/siblings.parquet"
BER_WORK="$W0" python3 augment.py "$AUG" 0 "$W0/siblings.parquet"

echo "[3/7] normalise augmented train and test"
BER_DATA="$AUG" BER_WORK="$WT" python3 prep.py train
BER_WORK="$WX" python3 prep.py test

echo "[4/7] blocking for every S1 + cross-S1 competition features"
BER_WORK="$WT" python3 block_all.py train
BER_WORK="$WX" python3 block_all.py test
BER_WORK="$WT" python3 competition.py train
BER_WORK="$WX" python3 competition.py test

echo "[5/7] candidate filter, labelled datasets (S1-level split), word/cluster features, matcher"
BER_WORK="$WT" python3 build_v3.py cheap,dataset,stage2
BER_WORK="$WT" python3 train_v3.py "$ROUNDS"
if [ "${BER_STOP_AFTER_TRAIN:-0}" = 1 ]; then echo "stop after training (BER_STOP_AFTER_TRAIN=1)"; exit 0; fi

echo "[6/7] test inference (house-number parsing fix 'fr' for S1 of countries absent from training)"
BER_WORK="$WX" BER_MODEL_DIR="$WT" BER_TRAIN_WORK="$WT" BER_NUMFIX="${BER_NUMFIX:-fr}" python3 infer_v3.py test

echo "[7/7] submission files (threshold chosen on validation: $(cat "$WT/threshold.txt"); countries absent from training: 0.90)"
BER_WORK="$WX" python3 make_outputs.py --thr "$(cat "$WT/threshold.txt")" --thr-unseen 0.90 --train-work "$WT" --scored "${BER_FINAL_SCORED:-scored6}" --outdir "$OUT"
echo "done -> $OUT/{matching_results.tsv,candidate_pairs.tsv}"
echo "validate with: python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <dataset>/test --check-ids"
