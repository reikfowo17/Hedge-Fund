import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt


def plot_feature_importance(model, top_n=30, figsize=(10, 12)):
    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=figsize)
    top = importance.head(top_n)
    ax.barh(range(len(top)), top["importance"].values)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importance (Gain)")
    ax.set_xlabel("Importance (Gain)")
    plt.tight_layout()
    plt.show()
    
    return importance


def print_data_summary(df, name="DataFrame"):
    print(f"\n{'='*50}")
    print(f"Summary: {name}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    if "ts_index" in df.columns:
        print(f"ts_index range: {df['ts_index'].min()} → {df['ts_index'].max()}")
    
    if "horizon" in df.columns:
        print(f"Horizons: {sorted(df['horizon'].unique())}")
    
    for col in ["code", "sub_code", "sub_category"]:
        if col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")
    
    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\nMissing values ({len(missing)} columns):")
        for col, count in missing.items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print("\nNo missing values!")


def check_submission(submission_df, test_df):
    errors = []
    
    # Check columns
    if list(submission_df.columns) != ["id", "prediction"]:
        errors.append(f"Expected columns ['id', 'prediction'], got {list(submission_df.columns)}")
    
    # Check row count
    if len(submission_df) != len(test_df):
        errors.append(f"Row count: {len(submission_df)} vs expected {len(test_df)}")
    
    # Check NaN
    nan_count = submission_df["prediction"].isnull().sum()
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN predictions")
    
    # Check ID match
    if set(submission_df["id"]) != set(test_df["id"]):
        missing = set(test_df["id"]) - set(submission_df["id"])
        extra   = set(submission_df["id"]) - set(test_df["id"])
        if missing:
            errors.append(f"{len(missing)} missing IDs")
        if extra:
            errors.append(f"{len(extra)} extra IDs")
    
    if errors:
        print("SUBMISSION VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("Submission looks valid!")
        print(f"  Rows: {len(submission_df):,}")
        print(f"  Prediction range: [{submission_df['prediction'].min():.4f}, "
              f"{submission_df['prediction'].max():.4f}]")
        print(f"  Prediction mean: {submission_df['prediction'].mean():.4f}")
        return True
