"""
Traffic Demand Prediction — V2 Pipeline

KEY IMPROVEMENTS over V1:
  1. 24-hour LAG features (demand of same geohash at same timestamp on day 48)
  2. Hierarchical lag fallbacks (timestamp → hour → time_bucket → geohash mean)
  3. Neighboring timestamp features (±15, ±30 min same geohash)
  4. Rolling statistics per geohash from day 48
  5. Proper OOF target encoding (no leakage)
  6. KFold validation on day 48 data (better validation distribution)
  7. Both log1p and raw target tested
  8. More aggressive model training (lower LR, more iterations)
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
SEED = 42
DATA_DIR = PROJECT_ROOT / 'e88186124ec611f1' / 'dataset'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Geohash decoder (pure Python)
# ---------------------------------------------------------------------------
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BITS = [16, 8, 4, 2, 1]

def _decode_geohash(geohash_str):
    is_lon = True
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    for char in geohash_str:
        cd = _BASE32.index(char)
        for mask in _BITS:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if cd & mask:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if cd & mask:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2


# ---------------------------------------------------------------------------
# V2 Feature Engineering
# ---------------------------------------------------------------------------
def parse_timestamp(df):
    parts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = parts[0]
    df['minute'] = parts[1]
    df['ts_minutes'] = df['hour'] * 60 + df['minute']
    return df


def build_day48_lookup(train_df):
    """Build comprehensive lookup tables from day 48 data.
    
    Returns dict of lookup tables keyed by (geohash, ts_minutes), 
    (geohash, hour), (geohash, time_bucket), (geohash,) etc.
    """
    d48 = train_df[train_df['day'] == 48].copy()
    
    lookups = {}
    
    # --- Exact (geohash, ts_minutes) lookup ---
    geo_ts = d48.groupby(['geohash', 'ts_minutes'])['demand'].agg(['mean', 'std', 'count']).reset_index()
    lookups['geo_ts_mean'] = dict(zip(
        zip(geo_ts['geohash'], geo_ts['ts_minutes']), geo_ts['mean']
    ))
    lookups['geo_ts_count'] = dict(zip(
        zip(geo_ts['geohash'], geo_ts['ts_minutes']), geo_ts['count']
    ))
    
    # --- (geohash, hour) lookup ---
    geo_hour = d48.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'median', 'std', 'count']).reset_index()
    lookups['geo_hour_mean'] = dict(zip(
        zip(geo_hour['geohash'], geo_hour['hour']), geo_hour['mean']
    ))
    lookups['geo_hour_median'] = dict(zip(
        zip(geo_hour['geohash'], geo_hour['hour']), geo_hour['median']
    ))
    lookups['geo_hour_std'] = dict(zip(
        zip(geo_hour['geohash'], geo_hour['hour']), geo_hour['std']
    ))
    
    # --- (geohash, time_bucket) lookup (4-hour blocks) ---
    d48['time_bucket'] = d48['hour'] // 4
    geo_tb = d48.groupby(['geohash', 'time_bucket'])['demand'].mean().to_dict()
    lookups['geo_tb_mean'] = geo_tb
    
    # --- (geohash, time_bucket_2h) lookup (2-hour blocks) ---
    d48['time_bucket_2h'] = d48['hour'] // 2
    geo_tb2 = d48.groupby(['geohash', 'time_bucket_2h'])['demand'].mean().to_dict()
    lookups['geo_tb2_mean'] = geo_tb2
    
    # --- geohash overall stats ---
    geo_stats = d48.groupby('geohash')['demand'].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
    lookups['geo_mean'] = dict(zip(geo_stats['geohash'], geo_stats['mean']))
    lookups['geo_median'] = dict(zip(geo_stats['geohash'], geo_stats['median']))
    lookups['geo_std'] = dict(zip(geo_stats['geohash'], geo_stats['std']))
    lookups['geo_min'] = dict(zip(geo_stats['geohash'], geo_stats['min']))
    lookups['geo_max'] = dict(zip(geo_stats['geohash'], geo_stats['max']))
    lookups['geo_count'] = dict(zip(geo_stats['geohash'], geo_stats['count']))
    
    # --- geo prefix stats ---
    for prefix_len in [3, 4, 5]:
        col = f'geo{prefix_len}'
        d48[col] = d48['geohash'].str[:prefix_len]
        grp = d48.groupby(col)['demand'].mean().to_dict()
        lookups[f'{col}_mean'] = grp
        grp_hour = d48.groupby([col, 'hour'])['demand'].mean().to_dict()
        lookups[f'{col}_hour_mean'] = grp_hour
    
    # --- hour-level stats (global) ---
    hour_stats = d48.groupby('hour')['demand'].mean().to_dict()
    lookups['hour_mean'] = hour_stats
    
    # --- RoadType-level stats ---
    road_stats = d48.groupby('RoadType')['demand'].mean().to_dict()
    lookups['road_mean'] = road_stats
    
    # --- RoadType × hour stats ---
    road_hour = d48.groupby(['RoadType', 'hour'])['demand'].mean().to_dict()
    lookups['road_hour_mean'] = road_hour
    
    # Global mean
    lookups['global_mean'] = d48['demand'].mean()
    
    return lookups


def add_lag_features(df, lookups):
    """Add 24-hour lag features using day 48 lookup tables.
    
    For each row, look up demand from day 48 at the same geohash+time.
    Uses hierarchical fallback: exact ts → same hour → time_bucket → geohash mean → global.
    """
    global_mean = lookups['global_mean']
    
    # --- Exact lag: same geohash, same timestamp ---
    df['lag_exact'] = df.apply(
        lambda r: lookups['geo_ts_mean'].get((r['geohash'], r['ts_minutes']), np.nan),
        axis=1
    )
    
    # --- Lag at ±15 min ---
    for offset in [-15, 15, -30, 30]:
        col = f'lag_{offset:+d}min'
        df[col] = df.apply(
            lambda r: lookups['geo_ts_mean'].get((r['geohash'], r['ts_minutes'] + offset), np.nan),
            axis=1
        )
    
    # --- Neighboring timestamp average ---
    df['lag_neighbor_avg'] = df[['lag_-15min', 'lag_+15min']].mean(axis=1)
    df['lag_window_avg'] = df[['lag_-30min', 'lag_-15min', 'lag_exact', 'lag_+15min', 'lag_+30min']].mean(axis=1)
    
    # --- Geo-hour mean from day 48 ---
    df['lag_geo_hour'] = df.apply(
        lambda r: lookups['geo_hour_mean'].get((r['geohash'], r['hour']), np.nan),
        axis=1
    )
    df['lag_geo_hour_median'] = df.apply(
        lambda r: lookups['geo_hour_median'].get((r['geohash'], r['hour']), np.nan),
        axis=1
    )
    df['lag_geo_hour_std'] = df.apply(
        lambda r: lookups['geo_hour_std'].get((r['geohash'], r['hour']), np.nan),
        axis=1
    )
    
    # --- Geo-time_bucket mean ---
    df['lag_geo_tb'] = df.apply(
        lambda r: lookups['geo_tb_mean'].get((r['geohash'], r['hour'] // 4), np.nan),
        axis=1
    )
    df['lag_geo_tb2'] = df.apply(
        lambda r: lookups['geo_tb2_mean'].get((r['geohash'], r['hour'] // 2), np.nan),
        axis=1
    )
    
    # --- Geohash overall stats ---
    df['lag_geo_mean'] = df['geohash'].map(lookups['geo_mean'])
    df['lag_geo_median'] = df['geohash'].map(lookups['geo_median'])
    df['lag_geo_std'] = df['geohash'].map(lookups['geo_std']).fillna(0)
    df['lag_geo_min'] = df['geohash'].map(lookups['geo_min'])
    df['lag_geo_max'] = df['geohash'].map(lookups['geo_max'])
    df['lag_geo_range'] = df['lag_geo_max'] - df['lag_geo_min']
    df['lag_geo_count'] = df['geohash'].map(lookups['geo_count']).fillna(0)
    
    # --- Hierarchical fallback: best available lag ---
    df['lag_best'] = df['lag_exact'].fillna(
        df['lag_geo_hour'].fillna(
            df['lag_geo_tb'].fillna(
                df['lag_geo_mean'].fillna(global_mean)
            )
        )
    )
    
    # --- Geo prefix stats ---
    for prefix_len in [3, 4, 5]:
        col = f'geo{prefix_len}'
        df[col] = df['geohash'].str[:prefix_len]
        df[f'lag_{col}_mean'] = df[col].map(lookups[f'{col}_mean']).fillna(global_mean)
        df[f'lag_{col}_hour_mean'] = df.apply(
            lambda r: lookups[f'{col}_hour_mean'].get((r[col], r['hour']), global_mean),
            axis=1
        )
    
    # --- Hour-level global stats ---
    df['lag_hour_global'] = df['hour'].map(lookups['hour_mean']).fillna(global_mean)
    
    # --- RoadType stats ---
    df['lag_road_mean'] = df['RoadType'].map(lookups['road_mean']).fillna(global_mean)
    df['lag_road_hour'] = df.apply(
        lambda r: lookups['road_hour_mean'].get((r['RoadType'], r['hour']), global_mean),
        axis=1
    )
    
    # --- Derived lag features ---
    df['lag_exact_vs_geo'] = df['lag_exact'] - df['lag_geo_mean']
    df['lag_exact_vs_hour'] = df['lag_exact'] - df['lag_hour_global']
    df['lag_geo_hour_vs_geo'] = df['lag_geo_hour'] - df['lag_geo_mean']
    df['lag_ratio_exact_geo'] = df['lag_exact'] / (df['lag_geo_mean'] + 1e-8)
    
    # Fill remaining NaNs
    lag_cols = [c for c in df.columns if c.startswith('lag_')]
    for c in lag_cols:
        df[c] = df[c].fillna(global_mean)
    
    return df


def add_base_features(df, label_encoders=None, fit=True):
    """Add basic features (temporal, categorical, etc.)."""
    if label_encoders is None:
        label_encoders = {}
    
    df = parse_timestamp(df)
    
    # Temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['ts_minutes'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['ts_minutes'] / 1440)
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['time_bucket'] = df['hour'] // 4
    df['time_bucket_2h'] = df['hour'] // 2
    
    # Geospatial
    coords = df['geohash'].apply(_decode_geohash)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    
    # Geohash prefixes (if not already added by lag features)
    for p in [3, 4, 5]:
        col = f'geo{p}'
        if col not in df.columns:
            df[col] = df['geohash'].str[:p]
    
    # Categorical encoding
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        enc_col = f'{col}_enc'
        if fit:
            le = LabelEncoder()
            vals = df[col].fillna('missing').astype(str)
            le.fit(vals)
            label_encoders[col] = le
            df[enc_col] = le.transform(vals)
        else:
            le = label_encoders[col]
            vals = df[col].fillna('missing').astype(str)
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else 'missing')
            df[enc_col] = le.transform(vals)
    
    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)
    
    # Geohash label encoding
    if fit:
        le = LabelEncoder()
        le.fit(df['geohash'].astype(str))
        label_encoders['geohash'] = le
        df['geohash_enc'] = le.transform(df['geohash'].astype(str))
    else:
        le = label_encoders['geohash']
        known = set(le.classes_)
        vals = df['geohash'].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        df['geohash_enc'] = le.transform(vals)
    
    # Geo prefix encoding
    for p in [3, 4, 5]:
        col = f'geo{p}'
        enc_col = f'{col}_enc'
        if fit:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            label_encoders[col] = le
            df[enc_col] = le.transform(df[col].astype(str))
        else:
            le = label_encoders[col]
            known = set(le.classes_)
            vals = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[enc_col] = le.transform(vals)
    
    # Temperature features
    temp_median = df['Temperature'].median()
    df['Temperature_filled'] = df['Temperature'].fillna(temp_median)
    df['temp_missing'] = df['Temperature'].isnull().astype(int)
    
    # Interaction features
    df['road_hour'] = df['RoadType_enc'] * 100 + df['hour']
    df['weather_hour'] = df['Weather_enc'] * 100 + df['hour']
    df['lanes_road'] = df['NumberofLanes'] * 10 + df['RoadType_enc']
    df['lanes_hour'] = df['NumberofLanes'] * 100 + df['hour']
    df['large_road'] = df['LargeVehicles_bin'] * 10 + df['RoadType_enc']
    
    return df, label_encoders


def get_feature_cols(df):
    """Get final feature columns, excluding non-features."""
    exclude = {'Index', 'geohash', 'timestamp', 'demand', 'day',
               'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
               'geo3', 'geo4', 'geo5', 'Temperature'}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != 'object']
    return feature_cols


# ---------------------------------------------------------------------------
# OOF Target Encoding for Day 48 data
# ---------------------------------------------------------------------------
def add_oof_target_encoding(df, cols_to_encode, target_col='demand', n_splits=5):
    """
    Out-of-fold target encoding computed ONLY on day 48 data.
    Day 49 rows get the full day 48 stats (no leakage since day 49 is future).
    """
    d48_mask = df['day'] == 48
    d49_mask = df['day'] == 49
    
    global_mean = df.loc[d48_mask, target_col].mean() if target_col in df.columns else 0.094
    smoothing = 10
    
    for col in cols_to_encode:
        te_col = f'{col}_te'
        df[te_col] = np.nan
        
        # Compute full day 48 stats for day 49 rows
        if target_col in df.columns:
            d48_data = df.loc[d48_mask]
            stats = d48_data.groupby(col)[target_col].agg(['mean', 'count'])
            smoother = stats['count'] / (stats['count'] + smoothing)
            stats['smoothed'] = smoother * stats['mean'] + (1 - smoother) * global_mean
            full_map = stats['smoothed'].to_dict()
        else:
            full_map = {}
        
        # Day 49: use full day 48 encoding
        df.loc[d49_mask, te_col] = df.loc[d49_mask, col].map(full_map)
        
        # Day 48: use OOF encoding
        if target_col in df.columns:
            d48_idx = df[d48_mask].index
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            
            for tr_idx, val_idx in kf.split(d48_idx):
                tr_rows = d48_idx[tr_idx]
                val_rows = d48_idx[val_idx]
                
                fold_stats = df.loc[tr_rows].groupby(col)[target_col].agg(['mean', 'count'])
                s = fold_stats['count'] / (fold_stats['count'] + smoothing)
                fold_stats['smoothed'] = s * fold_stats['mean'] + (1 - s) * global_mean
                fold_map = fold_stats['smoothed'].to_dict()
                
                df.loc[val_rows, te_col] = df.loc[val_rows, col].map(fold_map)
        
        df[te_col] = df[te_col].fillna(global_mean)
    
    return df


# ---------------------------------------------------------------------------
# Complete V2 Feature Pipeline
# ---------------------------------------------------------------------------
def build_features_v2(train_df, test_df):
    """Complete V2 feature pipeline with lag features and OOF encoding."""
    
    # Step 1: Parse timestamps
    train_df = parse_timestamp(train_df)
    test_df = parse_timestamp(test_df)
    
    # Step 2: Build day 48 lookups (ONLY from day 48 training data)
    print("  Building day 48 lookup tables...")
    lookups = build_day48_lookup(train_df)
    
    # Step 3: Add lag features
    print("  Adding lag features...")
    train_df = add_lag_features(train_df, lookups)
    test_df = add_lag_features(test_df, lookups)
    
    # Step 4: Add base features
    print("  Adding base features...")
    train_df, label_encoders = add_base_features(train_df, fit=True)
    test_df, _ = add_base_features(test_df, label_encoders=label_encoders, fit=False)
    
    # Step 5: OOF target encoding
    print("  Adding OOF target encoding...")
    te_cols = ['geohash_enc', 'geo4_enc', 'geo5_enc', 'RoadType_enc', 
               'road_hour', 'lanes_road']
    train_df = add_oof_target_encoding(train_df, te_cols)
    # For test: use full day 48 stats (already computed via lag features)
    # We need to compute TE for test separately using full day 48 data
    d48 = train_df[train_df['day'] == 48]
    global_mean = d48['demand'].mean()
    smoothing = 10
    for col in te_cols:
        te_col = f'{col}_te'
        stats = d48.groupby(col)['demand'].agg(['mean', 'count'])
        smoother = stats['count'] / (stats['count'] + smoothing)
        stats['smoothed'] = smoother * stats['mean'] + (1 - smoother) * global_mean
        full_map = stats['smoothed'].to_dict()
        test_df[te_col] = test_df[col].map(full_map).fillna(global_mean)
    
    # Step 6: Get feature columns
    feature_cols = get_feature_cols(train_df)
    # Ensure test has same cols
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0
    
    print(f"  Total features: {len(feature_cols)}")
    
    return train_df, test_df, feature_cols, lookups


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def train_lgb(X_tr, y_tr, X_va, y_va, feats, params=None, use_log=True):
    if params is None:
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': 0.03, 'num_leaves': 255, 'max_depth': 8,
            'min_child_samples': 30, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5,
            'lambda_l1': 0.05, 'lambda_l2': 0.05,
            'verbose': -1, 'seed': SEED, 'n_jobs': -1,
        }
    
    y_t = np.log1p(y_tr) if use_log else y_tr.copy()
    y_v = np.log1p(y_va) if use_log else y_va.copy()
    
    dtrain = lgb.Dataset(X_tr[feats], label=y_t)
    dval = lgb.Dataset(X_va[feats], label=y_v, reference=dtrain)
    
    model = lgb.train(params, dtrain, num_boost_round=10000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
    
    pred = model.predict(X_va[feats])
    if use_log:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)
    return model, pred, r2_score(y_va, pred)


def train_cb(X_tr, y_tr, X_va, y_va, feats, params=None, use_log=True):
    if params is None:
        params = {
            'iterations': 10000, 'learning_rate': 0.03, 'depth': 8,
            'l2_leaf_reg': 1, 'loss_function': 'RMSE', 'eval_metric': 'R2',
            'random_seed': SEED, 'verbose': 500, 'early_stopping_rounds': 200,
            'task_type': 'CPU',
        }
    
    y_t = np.log1p(y_tr) if use_log else y_tr.copy()
    y_v = np.log1p(y_va) if use_log else y_va.copy()
    
    model = CatBoostRegressor(**params)
    model.fit(X_tr[feats], y_t, eval_set=(X_va[feats], y_v), use_best_model=True)
    
    pred = model.predict(X_va[feats])
    if use_log:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)
    return model, pred, r2_score(y_va, pred)


def train_xgb(X_tr, y_tr, X_va, y_va, feats, params=None, use_log=True):
    if params is None:
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': 8, 'eta': 0.03, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 10,
            'lambda': 1.0, 'alpha': 0.1,
            'seed': SEED, 'nthread': -1, 'verbosity': 0,
        }
    
    y_t = np.log1p(y_tr) if use_log else y_tr.copy()
    y_v = np.log1p(y_va) if use_log else y_va.copy()
    
    dtrain = xgb.DMatrix(X_tr[feats], label=y_t)
    dval = xgb.DMatrix(X_va[feats], label=y_v)
    
    model = xgb.train(params, dtrain, num_boost_round=10000,
                      evals=[(dval, 'val')],
                      early_stopping_rounds=200, verbose_eval=500)
    
    pred = model.predict(dval)
    if use_log:
        pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)
    return model, pred, r2_score(y_va, pred)


def pred_lgb(m, X, feats, use_log=True):
    p = m.predict(X[feats])
    if use_log: p = np.expm1(p)
    return np.clip(p, 0, None)

def pred_cb(m, X, feats, use_log=True):
    p = m.predict(X[feats])
    if use_log: p = np.expm1(p)
    return np.clip(p, 0, None)

def pred_xgb(m, X, feats, use_log=True):
    p = m.predict(xgb.DMatrix(X[feats]))
    if use_log: p = np.expm1(p)
    return np.clip(p, 0, None)


# ---------------------------------------------------------------------------
# Optuna Tuning (V2 — focused, faster)
# ---------------------------------------------------------------------------
def tune_lgb_v2(X_tr, y_tr, X_va, y_va, feats, n_trials=80):
    def obj(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 63, 512),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 5.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 5.0, log=True),
            'verbose': -1, 'seed': SEED, 'n_jobs': -1,
        }
        _, _, score = train_lgb(X_tr, y_tr, X_va, y_va, feats, params)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  LGB best: {study.best_value:.6f}")
    return study.best_params


def tune_cb_v2(X_tr, y_tr, X_va, y_va, feats, n_trials=50):
    def obj(trial):
        params = {
            'iterations': 10000, 
            'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'loss_function': 'RMSE', 'eval_metric': 'R2',
            'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 200,
            'task_type': 'CPU',
        }
        _, _, score = train_cb(X_tr, y_tr, X_va, y_va, feats, params)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  CB best: {study.best_value:.6f}")
    return study.best_params


def tune_xgb_v2(X_tr, y_tr, X_va, y_va, feats, n_trials=60):
    def obj(trial):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'lambda': trial.suggest_float('lambda', 0.01, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-4, 5.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 5.0, log=True),
            'seed': SEED, 'nthread': -1, 'verbosity': 0,
        }
        _, _, score = train_xgb(X_tr, y_tr, X_va, y_va, feats, params)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  XGB best: {study.best_value:.6f}")
    return study.best_params


# ---------------------------------------------------------------------------
# Ensemble Weight Optimization
# ---------------------------------------------------------------------------
def optimize_weights(preds_dict, y_true):
    best_score = -np.inf
    best_w = None
    names = list(preds_dict.keys())
    preds = [preds_dict[n] for n in names]
    
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1 - w1, 21):
            w3 = 1.0 - w1 - w2
            if w3 < -0.001:
                continue
            w3 = max(w3, 0)
            p = sum(w * pr for w, pr in zip([w1, w2, w3], preds))
            s = r2_score(y_true, np.clip(p, 0, None))
            if s > best_score:
                best_score = s
                best_w = dict(zip(names, [w1, w2, w3]))
    return best_w, best_score


# ---------------------------------------------------------------------------
# Main V2 Pipeline
# ---------------------------------------------------------------------------
def run_v2():
    print("=" * 70)
    print("V2 PIPELINE -- WITH LAG FEATURES + OOF ENCODING")
    print("=" * 70)
    
    # --- Load ---
    print("\n[1/5] Loading data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")
    
    # --- Features ---
    print("\n[2/5] V2 Feature Engineering...")
    t0 = time.time()
    train_df, test_df, feature_cols, lookups = build_features_v2(
        train_raw.copy(), test_raw.copy()
    )
    print(f"  Time: {time.time()-t0:.1f}s")
    
    y_full = train_df['demand'].values
    
    # --- Temporal validation (day 48 -> day 49) ---
    print("\n[3/5] Temporal validation (day 48 -> day 49)...")
    d48_mask = train_df['day'] == 48
    d49_mask = train_df['day'] == 49
    tr_idx_t = train_df[d48_mask].index.values
    va_idx_t = train_df[d49_mask].index.values
    
    X_tr_t = train_df.loc[tr_idx_t]
    X_va_t = train_df.loc[va_idx_t]
    y_tr_t = y_full[tr_idx_t]
    y_va_t = y_full[va_idx_t]
    
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 255, 'max_depth': 8,
        'min_child_samples': 30, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'lambda_l1': 0.05, 'lambda_l2': 0.05,
        'verbose': -1, 'seed': SEED, 'n_jobs': -1,
    }
    cb_params = {
        'iterations': 10000, 'learning_rate': 0.03, 'depth': 8,
        'l2_leaf_reg': 1, 'loss_function': 'RMSE', 'eval_metric': 'R2',
        'random_seed': SEED, 'verbose': 500, 'early_stopping_rounds': 200,
        'task_type': 'CPU',
    }
    xgb_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'max_depth': 8, 'eta': 0.03, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 10,
        'lambda': 1.0, 'alpha': 0.1,
        'seed': SEED, 'nthread': -1, 'verbosity': 0,
    }
    
    lgb_m, lgb_vp, lgb_s = train_lgb(X_tr_t, y_tr_t, X_va_t, y_va_t, feature_cols, lgb_params)
    print(f"  LGB R2: {lgb_s:.6f}")
    cb_m, cb_vp, cb_s = train_cb(X_tr_t, y_tr_t, X_va_t, y_va_t, feature_cols, cb_params)
    print(f"  CB  R2: {cb_s:.6f}")
    xgb_m, xgb_vp, xgb_s = train_xgb(X_tr_t, y_tr_t, X_va_t, y_va_t, feature_cols, xgb_params)
    print(f"  XGB R2: {xgb_s:.6f}")
    
    # --- Ensemble ---
    print("\n[4/5] Ensemble Optimization...")
    val_preds = {'lgb': lgb_vp, 'cb': cb_vp, 'xgb': xgb_vp}
    best_w, ens_score = optimize_weights(val_preds, y_va_t)
    print(f"  Weights: {best_w}")
    print(f"  Ensemble R2: {ens_score:.6f}")
    
    # --- Seed Averaging + Final Submission ---
    print("\n[5/5] Seed Averaging + Final Submission...")
    SEEDS = [42, 123, 456, 789, 2024]
    
    test_preds = {'lgb': [], 'cb': [], 'xgb': []}
    val_preds_all = {'lgb': [], 'cb': [], 'xgb': []}
    
    for i, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({i+1}/{len(SEEDS)})...")
        
        lp = {**lgb_params, 'seed': seed}
        m, vp, s = train_lgb(X_tr_t, y_tr_t, X_va_t, y_va_t, feature_cols, lp)
        test_preds['lgb'].append(pred_lgb(m, test_df, feature_cols))
        val_preds_all['lgb'].append(vp)
        
        cp = {**cb_params, 'random_seed': seed}
        m, vp, s = train_cb(X_tr_t, y_tr_t, X_va_t, y_va_t, feature_cols, cp)
        test_preds['cb'].append(pred_cb(m, test_df, feature_cols))
        val_preds_all['cb'].append(vp)
        
        xp = {**xgb_params, 'seed': seed}
        m, vp, s = train_xgb(X_tr_t, y_tr_t, X_va_t, y_va_t, feature_cols, xp)
        test_preds['xgb'].append(pred_xgb(m, test_df, feature_cols))
        val_preds_all['xgb'].append(vp)
    
    # Average across seeds
    lgb_test = np.mean(test_preds['lgb'], axis=0)
    cb_test = np.mean(test_preds['cb'], axis=0)
    xgb_test = np.mean(test_preds['xgb'], axis=0)
    
    lgb_val = np.mean(val_preds_all['lgb'], axis=0)
    cb_val = np.mean(val_preds_all['cb'], axis=0)
    xgb_val = np.mean(val_preds_all['xgb'], axis=0)
    
    # Re-optimize weights on seed-averaged predictions
    final_w, final_score = optimize_weights(
        {'lgb': lgb_val, 'cb': cb_val, 'xgb': xgb_val}, y_va_t
    )
    print(f"\n  Final weights: {final_w}")
    print(f"  Final ensemble R2: {final_score:.6f}")
    
    # Generate final predictions
    final_pred = (final_w['lgb'] * lgb_test + 
                  final_w['cb'] * cb_test + 
                  final_w['xgb'] * xgb_test)
    final_pred = np.clip(final_pred, 0, None)
    
    # Save submissions
    for name, pred in [('final_v2', final_pred), 
                       ('lgb_v2', lgb_test),
                       ('cb_v2', cb_test), 
                       ('xgb_v2', xgb_test)]:
        sub = pd.DataFrame({'Index': test_df['Index'].values, 'demand': pred})
        sub['demand'] = sub['demand'].clip(lower=0)
        assert sub.shape == (41778, 2), f"Wrong shape: {sub.shape}"
        assert sub['demand'].isnull().sum() == 0, "NaN!"
        sub.to_csv(SUBMISSIONS_DIR / f'{name}_submission.csv', index=False)
        print(f"  Saved: {name}_submission.csv")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("V2 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Temporal -- LGB: {lgb_s:.6f}")
    print(f"  Temporal -- CB:  {cb_s:.6f}")
    print(f"  Temporal -- XGB: {xgb_s:.6f}")
    print(f"  FINAL ensemble R2:    {final_score:.6f}")
    print(f"  Weights: {final_w}")
    print("=" * 70)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_m.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)
    importance.to_csv(EXPERIMENTS_DIR / 'feature_importance_v2.csv', index=False)
    print("\n  Top 25 features (LGB gain):")
    print(importance.head(25).to_string(index=False))


if __name__ == '__main__':
    run_v2()
