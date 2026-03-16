# Experiment Log

## Benchmark Scores

| # | Date | Model | Features | Val Score | Public LB | Notes |
|---|------|-------|----------|-----------|-----------|-------|
| 1 | 16/03 | LightGBM v1 (baseline) | Raw + 3 lag (wrong groupby) | 0.10803 | — | Early stop ở round 10 |
| 2 | | | | | | |

## Notes
- Score range: 0 (worst) → 1 (best)
- Public LB uses 25% of test data
- Private LB uses remaining 75%
