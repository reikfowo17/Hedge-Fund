import os

# ── Environment Detection ──
IS_KAGGLE = os.path.exists('/kaggle/input')

# ── Data Paths ──
if IS_KAGGLE:
    TRAIN_PATH = "/kaggle/input/competitions/ts-forecasting/train.parquet"
    TEST_PATH  = "/kaggle/input/competitions/ts-forecasting/test.parquet"
    OUTPUT_DIR = "/kaggle/working"
else:
    TRAIN_PATH = "data/train.parquet"
    TEST_PATH  = "data/test.parquet"
    OUTPUT_DIR = "submissions"

# ── Constants ──
HORIZONS = [1, 3, 10, 25]
TARGET = "y_target"
WEIGHT = "weight"
VAL_THRESHOLD = 3500

# ── Column Groups ──
CAT_COLS   = ["code", "sub_code", "sub_category"]
META_COLS  = ["id", "code", "sub_code", "sub_category", "horizon", "ts_index"]
GROUP_COLS = ["code", "sub_code", "sub_category", "horizon"]

# ── Feature Engineering Config ──
KEY_FEATURES   = ["feature_al", "feature_am", "feature_cg", "feature_by", "mean_al_bp"]
EXTRA_FEATURES = ["feature_s"]
LAG_STEPS      = [1, 3, 5, 10, 25]
ROLLING_WINDOWS = [5, 10, 20]

# Optimal feature shifts per horizon (from EDA: feature_al is #1 strongest signal)
# feature shifted by -h means "value from h steps ago" = causal pseudo-target
OPTIMAL_SHIFTS = {1: -1, 3: -3, 10: -10, 25: -25}

# Recency weighting: up-weight recent observations during training
USE_RECENCY_WEIGHTING = True
RECENCY_FACTOR = 42.0  # max multiplier at latest ts_index

# Feature pruning: keep top N by importance
TOP_FEATURES = 100

# Scale grid for per-horizon optimization
SCALE_GRID = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]

# ── Seeds ──
N_SEEDS = 15
SEEDS = [42, 2024, 12345, 99, 420, 777, 1337, 2025, 7, 11, 314, 617, 888, 999, 5555]

# ── LightGBM Params ──
LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "None",
    "boosting_type":    "gbdt",
    "learning_rate":    0.015,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 200,
    "feature_fraction": 0.60,
    "bagging_fraction": 0.75,
    "bagging_freq":     5,
    "lambda_l1":        2.0,
    "lambda_l2":        10.0,
    "extra_trees":      True,
    "path_smooth":      1.0,
    "verbosity":        -1,
    "n_jobs":           -1,
}
