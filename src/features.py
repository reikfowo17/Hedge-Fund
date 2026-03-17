import pandas as pd
import numpy as np
from config import GROUP_COLS


def create_lag_features(df, feature_cols, lags=[1, 2, 3]):
    df = df.sort_values(GROUP_COLS + ["ts_index"])
    grouped = df.groupby(GROUP_COLS, observed=True)
    
    for col in feature_cols:
        col_series = grouped[col]
        for lag in lags:
            df[f"{col}_lag{lag}"] = col_series.shift(lag)
    
    return df


def create_rolling_features(df, feature_cols, windows=[3, 5]):
    df = df.sort_values(GROUP_COLS + ["ts_index"])
    grouped = df.groupby(GROUP_COLS, observed=True)
    
    for col in feature_cols:
        col_series = grouped[col]
        for w in windows:
            df[f"{col}_rmean{w}"] = col_series.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{col}_rstd{w}"] = col_series.transform(lambda x: x.rolling(w, min_periods=1).std())
    
    return df


def create_ewm_features(df, feature_cols, spans=[3, 7, 14]):
    """Exponential Weighted Mean — cho trọng số lớn hơn với dữ liệu gần đây."""
    df = df.sort_values(GROUP_COLS + ["ts_index"])
    grouped = df.groupby(GROUP_COLS, observed=True)
    
    for col in feature_cols:
        col_series = grouped[col]
        for span in spans:
            df[f"{col}_ewm{span}"] = col_series.transform(
                lambda x: x.ewm(span=span, min_periods=1).mean()
            )
    
    return df


def create_diff_features(df, feature_cols):
    df = df.sort_values(GROUP_COLS + ["ts_index"])
    grouped = df.groupby(GROUP_COLS, observed=True)
    
    for col in feature_cols:
        df[f"{col}_diff1"] = grouped[col].diff(1)
        df[f"{col}_diff2"] = grouped[col].diff(2)
        df[f"{col}_pctchg1"] = grouped[col].pct_change(1)
    
    return df


def create_ratio_features(df, feature_cols):
    """Tạo ratio giữa các feature quan trọng."""
    if len(feature_cols) < 2:
        return df
    
    for i in range(min(3, len(feature_cols))):
        for j in range(i + 1, min(4, len(feature_cols))):
            col_a, col_b = feature_cols[i], feature_cols[j]
            denom = df[col_b].replace(0, np.nan)
            df[f"ratio_{col_a}_{col_b}"] = (df[col_a] / denom).fillna(0)
    
    return df


def create_cross_sectional_features(df, feature_cols):
    groups = ["sub_category", "horizon", "ts_index"]
    means = df.groupby(groups, observed=True)[feature_cols].transform("mean")
    stds  = df.groupby(groups, observed=True)[feature_cols].transform("std")
    
    for col in feature_cols:
        df[f"{col}_dev_subcat"] = df[col] - means[col]
        # Z-score within group
        std_col = stds[col].replace(0, np.nan)
        df[f"{col}_zscore_subcat"] = ((df[col] - means[col]) / std_col).fillna(0)
    
    return df


def engineer_features(train_df, test_df, lag_features=None, lags=[1, 2, 3],
                      rolling_features=None, windows=[3, 5],
                      diff_features=None, cross_features=None):

    # Mark train vs test
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["_is_train"] = True
    test_df["_is_train"]  = False
    
    # Concat
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_df = all_df.sort_values(GROUP_COLS + ["ts_index"]).reset_index(drop=True)
    
    # Apply features
    if lag_features:
        print(f"Creating lag features ({len(lag_features)} cols × {len(lags)} lags)...")
        all_df = create_lag_features(all_df, lag_features, lags)
    
    if rolling_features:
        print(f"Creating rolling features ({len(rolling_features)} cols × {len(windows)} windows)...")
        all_df = create_rolling_features(all_df, rolling_features, windows)
        
        # EWM features on same columns
        ewm_spans = [3, 7, 14]
        print(f"Creating EWM features ({len(rolling_features)} cols × {len(ewm_spans)} spans)...")
        all_df = create_ewm_features(all_df, rolling_features, ewm_spans)
    
    if diff_features:
        print(f"Creating diff + pct_change features ({len(diff_features)} cols)...")
        all_df = create_diff_features(all_df, diff_features)
    
    # Ratio features (top 4 features)
    if lag_features and len(lag_features) >= 2:
        print(f"Creating ratio features...")
        all_df = create_ratio_features(all_df, lag_features[:4])
    
    if cross_features:
        print(f"Creating cross-sectional features ({len(cross_features)} cols)...")
        all_df = create_cross_sectional_features(all_df, cross_features)
    
    # Fill NaN from lag/rolling/diff (first rows of each group)
    new_cols = [c for c in all_df.columns 
                if any(s in c for s in ["_lag", "_rmean", "_rstd", "_diff", "_dev_",
                                        "_ewm", "_pctchg", "_zscore", "ratio_"])]
    for col in new_cols:
        all_df[col] = all_df[col].fillna(0)
    
    # Replace infinities
    all_df = all_df.replace([np.inf, -np.inf], 0)
    
    # Split back
    train_out = all_df[all_df["_is_train"]].drop(columns=["_is_train"]).reset_index(drop=True)
    test_out  = all_df[~all_df["_is_train"]].drop(columns=["_is_train"]).reset_index(drop=True)
    
    print(f"Feature engineering complete! Train: {train_out.shape}, Test: {test_out.shape}")
    return train_out, test_out
