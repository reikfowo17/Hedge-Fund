import pandas as pd
import numpy as np
import gc
from config import GROUP_COLS, KEY_FEATURES, EXTRA_FEATURES, LAG_STEPS, ROLLING_WINDOWS, VAL_THRESHOLD


def compute_target_encoding_stats(train_path):
    tmp = pd.read_parquet(train_path, columns=['sub_category', 'sub_code', 'y_target', 'ts_index'])
    train_only = tmp[tmp.ts_index <= VAL_THRESHOLD]

    stats = {
        'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
        'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
        'global_mean': float(train_only['y_target'].mean()),
    }

    del tmp, train_only
    gc.collect()
    return stats


def build_features(data, target_stats):
    df = data.copy()
    group_cols = GROUP_COLS

    # ── 1. Feature Interactions ──
    if 'feature_al' in df.columns and 'feature_am' in df.columns:
        df['d_al_am'] = df['feature_al'] - df['feature_am']
        df['r_al_am'] = df['feature_al'] / (df['feature_am'] + 1e-7)
    if 'feature_cg' in df.columns and 'feature_by' in df.columns:
        df['d_cg_by'] = df['feature_cg'] - df['feature_by']

    if 'feature_s' in df.columns:
        if 'feature_al' in df.columns:
            df['s_al_prod'] = df['feature_s'] * df['feature_al']
        if 'feature_am' in df.columns:
            df['s_am_prod'] = df['feature_s'] * df['feature_am']
        if 'feature_cg' in df.columns:
            df['s_cg_prod'] = df['feature_s'] * df['feature_cg']

    # ── 2. Target Encoding ──
    for c in ['sub_category', 'sub_code']:
        if c in df.columns:
            df[c + '_enc'] = df[c].map(target_stats[c]).fillna(target_stats['global_mean']).astype(np.float32)

    # ── 3. Cross-sectional normalization ──
    cs_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by']
    if 'd_al_am' in df.columns:
        cs_cols.append('d_al_am')
    for col in cs_cols:
        if col in df.columns:
            g = df.groupby('ts_index')[col]
            g_mean = g.transform('mean')
            g_std = g.transform('std') + 1e-7
            df[col + '_cs'] = ((df[col] - g_mean) / g_std).astype(np.float32)
            del g_mean, g_std

    # ── 4. Cyclical time ──
    df['t_sin'] = np.sin(2 * np.pi * df['ts_index'] / 100.0).astype(np.float32)
    df['t_cos'] = np.cos(2 * np.pi * df['ts_index'] / 100.0).astype(np.float32)

    # ── 5. Time-series engine (lags, rolling, ewm, diff, rank) ──
    df = df.sort_values(group_cols + ['ts_index'])

    target_cols = [c for c in KEY_FEATURES if c in df.columns]
    for ef in EXTRA_FEATURES:
        if ef in df.columns:
            target_cols.append(ef)

    grouped = df.groupby(group_cols, sort=False)

    for col in target_cols:
        # Lags
        for lag in LAG_STEPS:
            df[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)

        # Rolling mean + std
        for w in ROLLING_WINDOWS:
            df[f'{col}_roll_mean_{w}'] = grouped[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            ).astype(np.float32)
            df[f'{col}_roll_std_{w}'] = grouped[col].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            ).astype(np.float32)

        # EWM
        df[f'{col}_ewm_10'] = grouped[col].transform(
            lambda x: x.ewm(span=10, min_periods=1).mean()
        ).astype(np.float32)

        # Diff
        df[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)

        # Rank (cross-sectional)
        df[f'{col}_rank'] = df.groupby('ts_index')[col].rank(pct=True).astype(np.float32)

    gc.collect()

    # ── 6. Fill NaN/Inf ──
    df = df.fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)

    return df


def get_feature_columns(df):
    exclude = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    return [c for c in df.columns if c not in exclude]
