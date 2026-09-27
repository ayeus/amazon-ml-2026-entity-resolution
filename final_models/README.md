# Models behind the final submission (public LB 0.970178)

| Folder | Used for | Files |
|---|---|---|
| `rebuild_usindia/` | US + India S1 (training countries) | `model_v3_t4.json`, `model_v3_t5.json`, `model_v3_t45.json` (XGBoost; final p = mean of the three, threshold 0.65), their `_feats.json`, candidate filter `cheap3_*`, word statistics `word_stats6_full.pkl`; `model_v3.json` = rebuild base model |
| `v3_france/` | S1 from countries absent from training (France) | `model_v3.json` (XGBoost v3, threshold 0.90), `cheap3_*`, `word_stats6_full.pkl`, `native_map*.json`; scored with the house-number parsing fix (`src/numfix.py`, mode `fr`) |

Reproduction (see `code/business_entity_resolution/README.md`):
prep + blocking + competition for the test split, then `src/infer_t.py` (BER_DERIVE=t45, BER_EXTRA_MODELS=t4:_t4,t5:_t5)
for US/India, `src/infer_t.py` with BER_NUMFIX=fr and the v3 model/filter/word statistics for France,
`src/make_hybrid.py` (p = mean of p_t4, p_t5, p), `src/make_outputs.py --thr 0.65 --thr-unseen 0.90`.
