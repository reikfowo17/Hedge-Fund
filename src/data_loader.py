import pandas as pd
import numpy as np
from config import TRAIN_PATH, TEST_PATH, CAT_COLS, GROUP_COLS

def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object or str(col_type) == "category":
            continue
        
        c_min, c_max = df[col].min(), df[col].max()
        
        if str(col_type).startswith("int"):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start_mem:.1f}MB → {end_mem:.1f}MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def load_data(reduce_memory=True):
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")
    
    #  1. Native categorical 
    for col in CAT_COLS:
        # Combine categories so train & test share the same category list
        all_cats = pd.CategoricalDtype(
            categories=sorted(set(train_df[col].unique()) | set(test_df[col].unique()))
        )
        train_df[col] = train_df[col].astype(all_cats)
        test_df[col]  = test_df[col].astype(all_cats)
    
    #  2. Sort by group + time 
    sort_cols = GROUP_COLS + ["ts_index"]
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
    test_df  = test_df.sort_values(sort_cols).reset_index(drop=True)
    
    #  3. Forward fill missing values 
    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    
    for col in feature_cols:
        if train_df[col].isnull().any():
            # Forward fill within each entity-horizon group
            train_df[col] = train_df.groupby(GROUP_COLS)[col].ffill()
            # Fallback: fill remaining NaN with column median (from train)
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val)
            
            if col in test_df.columns and test_df[col].isnull().any():
                test_df[col] = test_df.groupby(GROUP_COLS)[col].ffill()
                test_df[col] = test_df[col].fillna(median_val)  # Use train median
    
    #  4. Reduce memory 
    if reduce_memory:
        print("Reducing memory...")
        train_df = reduce_mem_usage(train_df, verbose=True)
        test_df  = reduce_mem_usage(test_df, verbose=True)
    
    print("Data loading complete!")
    return train_df, test_df


def get_feature_columns(df):
    exclude = {"id", "y_target", "weight"}
    return [c for c in df.columns if c not in exclude]


def time_split(df, split_ts_index=3000):
    train_part = df[df["ts_index"] <= split_ts_index].copy()
    val_part   = df[df["ts_index"] >  split_ts_index].copy()
    print(f"Time split at ts_index={split_ts_index}:")
    print(f"  Train: {train_part.shape}, Val: {val_part.shape}")
    return train_part, val_part
