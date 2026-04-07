# Experiment Log

## Benchmark Scores

| # | Date | Model | Features | Val Score | Public LB | Notes |
|---|------|-------|----------|-----------|-----------|-------|
| 1 | 16/03 | LightGBM v1 (baseline) | Raw + 3 lag (wrong groupby) | 0.10803 | — | Early stop at round 10 |
| 2 | 18/03 | LightGBM v2 (fixed pipeline) | Raw + lag(1,2,3) + rolling(3,5) + diff | H1:0.031, H3:0.052, H10:0.112, H25:0.141 | — | lr=0.05, num_leaves=63 |
| 3 | 18/03 | LightGBM v3 (optimized) | + EWM + ratio + pctchg | H1:0.037, H3:0.058, H10:0.075, H25:0.157 | 0.1838 | lr=0.01, num_leaves=127 |
| 4 | 19/03 | LightGBM v4 (20-seed + retrain-on-all) | Interactions + TargetEnc + CS-norm + Cyclical + Lag/Roll/EWM/Diff/Rank (~170 feats) | H1:0.080, H3:0.140, H10:0.222, H25:0.272 (Agg: 0.2353) | 0.2612 | 20 seeds, retrain-on-all, clipping, L1/L2 reg |
| 5 | 20/03 | LightGBM v4.1 (+ feature selection) | Top 50-65 features per-horizon (importance-based) | H1:0.078, H3:0.139, H10:0.219, H25:0.275 (Agg: 0.2357) | 0.2566 ❌ | Feature selection decreased score vs v4 |
| 6 | 25/03 | LGB+CAT v5 (Ridge stacking, 5-fold CV) | V4 feats + time_phase + lifecycle + momentum + roll_min/max + target enc `code`, percentile clip (~150+ feats) | H1:0.047, H3:0.057, H10:0.107, H25:0.135 (Agg: 0.116) | — | Ridge stacking LGB+CAT, skip CAT H=10,25, percentile clip [p0.5,p99.5] |
| 7 | 26/03 | LGB v6 (Hybrid — Calibration + Concat + FreqEnc) | 190 feats, 15 LGB seeds, no CAT, linear calibration, train+test concat, freq encoding | H1:0.064, H3:0.110, H10:0.226, H25:0.297 (Agg: 0.248) | 0.2462 | Pure LGB 15-seed, ~4h runtime, details below |
| 8 | 30/03 | LGB v7 | ~170 feats, 15 seeds, pseudo-targets + target lags + recency weighting + top-100 prune + LGB native API + per-horizon scale grid | H1:0.872837, H3:0.833179, H10:0.751853, H25:0.760920 (Agg raw: 0.769606, scaled: 0.775646) | 0.5914 | vs V6: REMOVE linear calibration → REPLACE with per-horizon scale grid; ADD pseudo-target/target-lag/recency weighting/feature pruning/native cats; Public LB 0.2462→0.5914 |
| 9 | 03/04 | LGB v8 | ~200 feats, 7 seeds, super_proxy + pseudo bp/bq + extra_trees + lr=0.015 + top-120 + per-code scale | H1:0.897435 (×1.07), H3:0.905957 (×1.10), H10:0.898513 (×1.05), H25:0.884580 (×1.05) (Agg raw: 0.882857, scaled: 0.884091) | **0.7511** 🏆 | vs V7: ADD super_proxy/pseudo bp,bq/PSEUDO_OFFSETS/per-code scale; REMOVE N_SEEDS/pseudo_al_lag1; seeds 15→7; top 100→120; +0.16 vs 0.5914 |

## Notes
- Score range: 0 (worst) → 1 (best)
- Public LB uses 25% of test data
- Private LB uses remaining 75%
- **Host confirmed**: score > 0.5 = data leak, Private LB ≠ Public LB. "Clean" max ~0.35-0.40.
- **Notebook prize requirement**: must run < 6 hours
- v2 → v3: lr 0.05→0.01, num_leaves 63→127, added EWM/ratio/pct_change features
- v3 → v4 (inspired by top scorer 0.2359):
  - **20-seed ensemble**: Train 20 random models instead of 1 or 3
  - **Retrain-on-All**: Find best_iteration → retrain on ALL data (train+val) → predict test
  - **Target Encoding**: mean(y_target) by sub_category and sub_code
  - **Feature Interactions**: al-am, al/am, cg-by, s*al, s*am, s*cg
  - **Rolling std**: Added rolling std alongside rolling mean
  - **Feature rank**: rank(pct=True) by ts_index
  - **Cyclical time**: sin/cos(2π·ts/100)
  - **Regularization**: L1=0.1, L2=10.0
  - **Clipping**: Clip predictions by quantile [0.005, 0.995]
  - **Hyperparams**: lr=0.015, num_leaves=90, min_child_samples=200
  - **Val split**: ts_index=3500
- v4 → v4.1:
  - **Feature Selection**: Train 1 probe model → get feature_importance → keep top 50-65 features
  - ❌ Result: decreased score → reverted
- v4 → v5:
  - **CatBoost**: Added CatBoost (ordered boosting, categoricals-friendly) instead of XGBoost
  - **Ridge Stacking**: Ridge(alpha=1.0) learns optimal weights LGB+CAT from OOF
  - **5-Fold TimeSeriesSplit**: Replaced single hold-out with 5 time folds
  - **Per-Fold Target Encoding**: Prevent target leakage
  - **New Features**: time_phase, lifecycle, momentum, roll_min/max
  - **Anti-leakage**: shift(1) before rolling on target-derived features
  - **LGBM Params**: num_leaves 90→127, L1 0.1→2.0, extra_trees, path_smooth
  - **Skip CatBoost** for H=10, H=25 (score = 0)
  - **Added target encoding `code`**: Each asset code has its own behavior
  - **Post-Processing**: Percentile clip [p0.5, p99.5]
- v5 → v6:
  - **Removed CatBoost entirely**: Pure LGB, focused all budget on 15 seeds
  - **Calibration**: Linear `a×pred + b` on OOF
  - **Train+Test concat**: Concat before building lags → test rows get lags from end of train
  - **Frequency Encoding**: Added freq encoding for code/sub_code/sub_category (rule-safe, no target)
  - **Hold-out validation**: Replaced 5-fold CV with simple split ts_index=3500
  - **15 LGB seeds**: Scaled up from 7 seeds
  - **Estimated runtime**: ~4h (< 6h limit)
  - **Kaggle Discussion insight**: Host confirms score > 0.5 uses data leak. Private LB will neutralize.
- v6 → v7:
  - **Removed linear calibration**: Replaced with per-horizon scale grid `SCALE_GRID=[0.95..1.30]`
  - **Added pseudo-target features**: `feature_al/am/cg/s` shifted by `OPTIMAL_SHIFTS[horizon]`
  - **Added target lag features**: `y_lag1`, `y_lag3`, `y_diff1`, `y_expand_mean` (shift(1) to avoid leakage)
  - **Added recency weighting**: `RECENCY_FACTOR=42.0`
  - **Added feature pruning**: top-100 features by gain importance
  - **Added native categorical**: LightGBM `categorical_feature` parameter
  - **Added LGB Native API + Skill Metric**: `lgb.Dataset`/`lgb.train` + `feval=lgb_skill_metric`
  - **Result**: Val 0.248→0.770 scaled, Public LB 0.2462→0.5914
- v7 → v8:
  - **Added pseudo offsets**: `PSEUDO_OFFSETS=[-2,-1,1]` — nearby shifts around optimal (~12 new features)
  - **Added pseudo interactions**: `al-am diff/ratio/prod`, `al-cg diff/ratio`, `pseudo_al_lag1`, `pseudo_am_diff` (~7 new features)
  - **Replaced per-horizon → per-code scaling**: Optimize scale per `code` on val (min 30 samples, fallback global)
  - **Finer scale grid**: 8→17 values, range 0.90-1.40
  - **Reduced seeds**: 15→7
  - **Increased feature budget**: top-100→top-120
