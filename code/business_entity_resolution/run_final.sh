#!/usr/bin/env bash
# Reproduces the FINAL submission (public leaderboard 0.970178): raw TSVs -> output/{matching_results,candidate_pairs}.tsv
#   S1 of countries present in training (US, India): "rebuild" configuration (tie-inclusive blocking + no-address channel,
#       French legal forms, 1.6M training S1) scored by the MEAN of three XGBoost matchers trained on the same data with
#       extra features: t4 = S1 anchor-profile counts, t5 = character n-gram similarities, t45 = both; threshold 0.65
#       (chosen on the test-like validation split).
#   S1 of countries absent from training (France): default "v3" configuration + house-number parsing fix; threshold 0.90.
# Which S1 belong to which group is derived from train_s1.pkl (no country list in the code).
# Usage:  BER_DATA=/path/to/dataset  bash run_final.sh      (dataset/ contains train/ and test/)   Runtime ~9 h, peak ~12 GB.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${BER_ROOT:-$(cd "$HERE/../.." && pwd)}"
: "${BER_DATA:?set BER_DATA to the dataset folder that contains train/ and test/}"
export BER_DATA
export DYLD_FALLBACK_LIBRARY_PATH="${BER_LIBOMP_DIR:-${DYLD_FALLBACK_LIBRARY_PATH:-}}"
A="$ROOT/final_v3"; B="$ROOT/final_rebuild"; OUT="${BER_OUT:-$ROOT/output}"

echo "[A] v3 configuration (full pipeline; its France scores use the house-number fix)"
BER_ROOT="$A" BER_OUT="$A/output" bash "$HERE/run_pipeline.sh"

echo "[B] rebuild configuration up to the base matcher (datasets ds6/ds6x, candidate filter, word statistics)"
BER_ROOT="$B" BER_OUT="$B/output" BER_TIES=1 BER_TIE_CH=nae BER_KE=5 BER_FR_NORM=1 BER_MATCH_S1=1600000 BER_ROUNDS=800 \
  BER_STOP_AFTER_TRAIN=1 bash "$HERE/run_pipeline.sh"

cd "$HERE/src"
echo "[C] character n-gram features for every train/validation pair"
python3 ngram.py "$B/work_train" ds6
echo "[D] three matchers: t4 (anchor profile), t5 (n-grams), t45 (both)"
for S in t4 t5 t45; do BER_WORK="$B/work_train" BER_DERIVE=$S python3 train_t2.py 800; done
echo "[E] test scoring with t45 (+ t4, t5 in the same pass)"
BER_WORK="$B/work_test" BER_MODEL_DIR="$B/work_train" BER_DERIVE=t45 BER_EXTRA_MODELS="t4:_t4,t5:_t5" python3 infer_t.py
echo "[F] merge: training countries = mean(t4, t5, t45); countries absent from training = v3 + house-number fix"
python3 make_hybrid.py "$B/work_test/scored6_final_test" "$B/work_test/scored6_t45_test" p_t4,p_t5,p \
  "$A/work_test/scored6_test" "$B/work_train/train_s1.pkl"
echo "[G] submission files"
BER_WORK="$B/work_test" python3 make_outputs.py --thr 0.65 --thr-unseen 0.90 --train-work "$B/work_train" \
  --scored scored6_final --outdir "$OUT"
echo "done -> $OUT/{matching_results.tsv,candidate_pairs.tsv}"
