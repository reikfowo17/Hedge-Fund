import pandas as pd
import numpy as np
from config import GROUP_COLS


def create_lag_features(df, feature_cols, lags=[1, 2, 3]):
    df = df.sort_values(GROUP_COLS + ["ts_index"])
    
    for col in feature_cols:
        for lag in lags:
            col_name = f"{col}_lag{lag}"
            df[col_name] = df.groupby(GROUP_COLS)[col].shift(lag)
    
    return df


def create_rolling_features(df, feature_cols, windows=[3, 5]):
    df = df.sort_values(GROUP_COLS + ["ts_index"])
    
    for col in feature_cols:
        for w in windows:
            df[f"{col}_rmean{w}"] = df.groupby(GROUP_COLS)[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_rstd{w}"] = df.groupby(GROUP_COLS)[col].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
    
    return df


def create_diff_features(df, feature_cols):
    df = df.sort_values(GROUP_COLS + ["ts_index"])
    
    for col in feature_cols:
        df[f"{col}_diff1"] = df.groupby(GROUP_COLS)[col].diff(1)
    
    return df


def create_cross_sectional_features(df, feature_cols):
    for col in feature_cols:
        group_mean = df.groupby(["sub_category", "horizon", "ts_index"])[col].transform("mean")
        df[f"{col}_dev_subcat"] = df[col] - group_mean
    
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
    
    if diff_features:
        print(f"Creating diff features ({len(diff_features)} cols)...")
        all_df = create_diff_features(all_df, diff_features)
    
    if cross_features:
        print(f"Creating cross-sectional features ({len(cross_features)} cols)...")
        all_df = create_cross_sectional_features(all_df, cross_features)
    
    # Fill NaN from lag/rolling/diff (first rows of each group)
    new_cols = [c for c in all_df.columns 
                if any(s in c for s in ["_lag", "_rmean", "_rstd", "_diff", "_dev_"])]
    for col in new_cols:
        all_df[col] = all_df[col].fillna(0)
    
    # Split back
    train_out = all_df[all_df["_is_train"]].drop(columns=["_is_train"]).reset_index(drop=True)
    test_out  = all_df[~all_df["_is_train"]].drop(columns=["_is_train"]).reset_index(drop=True)
    
    print(f"Feature engineering complete! Train: {train_out.shape}, Test: {test_out.shape}")
    return train_out, test_out
