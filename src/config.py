import os

# ── Environment Detection ──
IS_KAGGLE = os.path.exists('/kaggle/input')

# ── Data Paths ──
if IS_KAGGLE:
    TRAIN_PATH = "/kaggle/input/datasets/heon29/hegde-fund/train.parquet"
    TEST_PATH  = "/kaggle/input/datasets/heon29/hegde-fund/test.parquet"
    OUTPUT_DIR = "/kaggle/working"
else:
    TRAIN_PATH = "data/train.parquet"
    TEST_PATH  = "data/test.parquet"
    OUTPUT_DIR = "submissions"

# ── Constants ──
SEED = 42
HORIZONS = [1, 3, 10, 25]
TARGET = "y_target"
WEIGHT = "weight"

# ── Column Groups ──
CAT_COLS   = ["code", "sub_code", "sub_category"]
META_COLS  = ["id", "code", "sub_code", "sub_category", "horizon", "ts_index"]
GROUP_COLS = ["code", "sub_code", "sub_category", "horizon"]

# ── Feature Engineering Config ──
# Top features to create lags for (update after EDA / feature importance)
TOP_LAG_FEATURES = [
    "feature_al", "feature_v", "feature_bp",
    "feature_a", "feature_b", "feature_c",
]
LAG_STEPS = [1, 2, 3, 5]
ROLLING_WINDOWS = [3, 5, 10]

# ── LightGBM Base Params ──
LGBM_BASE_PARAMS = {
    "objective":       "regression",
    "metric":          "rmse",
    "boosting_type":   "gbdt",
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq":    5,
    "lambda_l1":       0.1,
    "lambda_l2":       0.1,
    "seed":            SEED,
    "verbosity":       -1,
    "n_jobs":          -1,
}
