"""
Traffic Demand Prediction - V3 Pipeline

FIX for V2's lag feature leakage:
  - Day 48 rows: lag_exact/lag_±15/30min set to NaN (no "same day" lag available)
  - Day 48 rows: aggregate lags (geo_hour, geo_mean) computed OOF
  - Day 49 rows: ALL lag features computed from full day 48 data (legitimate 24h lag)
  - Model learns: use base features + aggregate stats when exact lags unavailable,
    use everything when exact lags available
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
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
# Geohash decoder
# ---------------------------------------------------------------------------
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BITS = [16, 8, 4, 2, 1]

def _decode_geohash(gh):
    is_lon = True
    lat_r, lon_r = [-90.0, 90.0], [-180.0, 180.0]
    for c in gh:
        cd = _BASE32.index(c)
        for mask in _BITS:
            if is_lon:
                mid = (lon_r[0] + lon_r[1]) / 2
                if cd & mask: lon_r[0] = mid
                else: lon_r[1] = mid
            else:
                mid = (lat_r[0] + lat_r[1]) / 2
                if cd & mask: lat_r[0] = mid
                else: lat_r[1] = mid
            is_lon = not is_lon
    return (lat_r[0]+lat_r[1])/2, (lon_r[0]+lon_r[1])/2


def parse_timestamp(df):
    parts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = parts[0]
    df['minute'] = parts[1]
    df['ts_minutes'] = df['hour'] * 60 + df['minute']
    return df


# ---------------------------------------------------------------------------
# V3 Feature Engineering - Proper lag handling
# ---------------------------------------------------------------------------
def build_day48_lookups(train_df):
    """Build lookup tables from day 48 data only."""
    d48 = train_df[train_df['day'] == 48].copy()
    lookups = {}
    
    # Exact (geohash, ts_minutes) -> demand mean
    geo_ts = d48.groupby(['geohash', 'ts_minutes'])['demand'].agg(['mean','std','count']).reset_index()
    lookups['geo_ts_mean'] = dict(zip(zip(geo_ts['geohash'], geo_ts['ts_minutes']), geo_ts['mean']))
    
    # (geohash, hour) -> demand stats
    geo_hour = d48.groupby(['geohash', 'hour'])['demand'].agg(['mean','median','std','count']).reset_index()
    lookups['geo_hour_mean'] = dict(zip(zip(geo_hour['geohash'], geo_hour['hour']), geo_hour['mean']))
    lookups['geo_hour_median'] = dict(zip(zip(geo_hour['geohash'], geo_hour['hour']), geo_hour['median']))
    lookups['geo_hour_std'] = dict(zip(zip(geo_hour['geohash'], geo_hour['hour']), geo_hour['std']))
    
    # (geohash, time_bucket_2h) 
    d48['time_bucket_2h'] = d48['hour'] // 2
    geo_tb2 = d48.groupby(['geohash', 'time_bucket_2h'])['demand'].mean().to_dict()
    lookups['geo_tb2_mean'] = geo_tb2
    
    # (geohash, time_bucket_4h)
    d48['time_bucket_4h'] = d48['hour'] // 4
    geo_tb4 = d48.groupby(['geohash', 'time_bucket_4h'])['demand'].mean().to_dict()
    lookups['geo_tb4_mean'] = geo_tb4
    
    # geohash overall stats
    geo_stats = d48.groupby('geohash')['demand'].agg(['mean','median','std','min','max','count']).reset_index()
    lookups['geo_mean'] = dict(zip(geo_stats['geohash'], geo_stats['mean']))
    lookups['geo_median'] = dict(zip(geo_stats['geohash'], geo_stats['median']))
    lookups['geo_std'] = dict(zip(geo_stats['geohash'], geo_stats['std']))
    lookups['geo_min'] = dict(zip(geo_stats['geohash'], geo_stats['min']))
    lookups['geo_max'] = dict(zip(geo_stats['geohash'], geo_stats['max']))
    lookups['geo_count'] = dict(zip(geo_stats['geohash'], geo_stats['count']))
    
    # Geo prefix stats
    for p in [3, 4, 5]:
        col = f'geo{p}'
        d48[col] = d48['geohash'].str[:p]
        lookups[f'{col}_mean'] = d48.groupby(col)['demand'].mean().to_dict()
        lookups[f'{col}_hour_mean'] = d48.groupby([col, 'hour'])['demand'].mean().to_dict()
    
    # Hour-level global
    lookups['hour_mean'] = d48.groupby('hour')['demand'].mean().to_dict()
    
    # RoadType stats
    lookups['road_mean'] = d48.groupby('RoadType')['demand'].mean().to_dict()
    lookups['road_hour_mean'] = d48.groupby(['RoadType', 'hour'])['demand'].mean().to_dict()
    
    lookups['global_mean'] = d48['demand'].mean()
    
    return lookups


def add_lag_features_v3(df, lookups, is_day49_only=False):
    """Add lag features with proper handling:
    - Day 49 rows: full lag features from day 48 (24h lag - legitimate)
    - Day 48 rows: only aggregate stats (geo_hour, geo_mean) NOT exact timestamp match
    """
    gm = lookups['global_mean']
    
    # Mark which rows are day 49
    is_d49 = df['day'] == 49 if 'day' in df.columns else pd.Series(True, index=df.index)
    
    # --- EXACT LAG: only for day 49 rows ---
    # For day 48, these are NaN (no previous day to look up)
    df['lag_exact'] = np.nan
    df['lag_m15'] = np.nan
    df['lag_p15'] = np.nan
    df['lag_m30'] = np.nan
    df['lag_p30'] = np.nan
    
    if is_d49.any():
        d49_idx = df[is_d49].index
        df.loc[d49_idx, 'lag_exact'] = df.loc[d49_idx].apply(
            lambda r: lookups['geo_ts_mean'].get((r['geohash'], r['ts_minutes']), np.nan), axis=1)
        for offset, col in [(-15,'lag_m15'),(15,'lag_p15'),(-30,'lag_m30'),(30,'lag_p30')]:
            df.loc[d49_idx, col] = df.loc[d49_idx].apply(
                lambda r: lookups['geo_ts_mean'].get((r['geohash'], r['ts_minutes']+offset), np.nan), axis=1)
    
    df['lag_neighbor_avg'] = df[['lag_m15','lag_p15']].mean(axis=1)
    df['lag_window_avg'] = df[['lag_m30','lag_m15','lag_exact','lag_p15','lag_p30']].mean(axis=1)
    
    # --- AGGREGATE LAGS: for ALL rows (geo-hour, geo-mean, etc.) ---
    # These are legitimate for both day 48 and day 49
    df['lag_geo_hour'] = df.apply(
        lambda r: lookups['geo_hour_mean'].get((r['geohash'], r['hour']), np.nan), axis=1)
    df['lag_geo_hour_median'] = df.apply(
        lambda r: lookups['geo_hour_median'].get((r['geohash'], r['hour']), np.nan), axis=1)
    df['lag_geo_hour_std'] = df.apply(
        lambda r: lookups['geo_hour_std'].get((r['geohash'], r['hour']), np.nan), axis=1)
    
    df['lag_geo_tb2'] = df.apply(
        lambda r: lookups['geo_tb2_mean'].get((r['geohash'], r['hour']//2), np.nan), axis=1)
    df['lag_geo_tb4'] = df.apply(
        lambda r: lookups['geo_tb4_mean'].get((r['geohash'], r['hour']//4), np.nan), axis=1)
    
    df['lag_geo_mean'] = df['geohash'].map(lookups['geo_mean'])
    df['lag_geo_median'] = df['geohash'].map(lookups['geo_median'])
    df['lag_geo_std'] = df['geohash'].map(lookups['geo_std']).fillna(0)
    df['lag_geo_min'] = df['geohash'].map(lookups['geo_min'])
    df['lag_geo_max'] = df['geohash'].map(lookups['geo_max'])
    df['lag_geo_range'] = df['lag_geo_max'] - df['lag_geo_min']
    df['lag_geo_count'] = df['geohash'].map(lookups['geo_count']).fillna(0)
    
    # Hierarchical best lag
    df['lag_best'] = df['lag_exact'].fillna(
        df['lag_geo_hour'].fillna(
            df['lag_geo_tb2'].fillna(
                df['lag_geo_mean'].fillna(gm))))
    
    # Geo prefix stats
    for p in [3, 4, 5]:
        col = f'geo{p}'
        if col not in df.columns:
            df[col] = df['geohash'].str[:p]
        df[f'lag_{col}_mean'] = df[col].map(lookups[f'{col}_mean']).fillna(gm)
        df[f'lag_{col}_hour'] = df.apply(
            lambda r: lookups[f'{col}_hour_mean'].get((r[col], r['hour']), gm), axis=1)
    
    # Hour/road global
    df['lag_hour_global'] = df['hour'].map(lookups['hour_mean']).fillna(gm)
    df['lag_road_mean'] = df['RoadType'].map(lookups['road_mean']).fillna(gm)
    df['lag_road_hour'] = df.apply(
        lambda r: lookups['road_hour_mean'].get((r['RoadType'], r['hour']), gm), axis=1)
    
    # Derived features (only meaningful when lag_exact is not NaN)
    df['lag_exact_vs_geo'] = df['lag_exact'] - df['lag_geo_mean']
    df['lag_exact_vs_hour'] = df['lag_exact'] - df['lag_hour_global']
    df['lag_geo_hour_vs_geo'] = df['lag_geo_hour'] - df['lag_geo_mean']
    df['lag_ratio_geo_hour'] = df['lag_geo_hour'] / (df['lag_geo_mean'] + 1e-8)
    
    # Fill remaining NaN in AGGREGATE features only (NOT exact lags)
    agg_cols = [c for c in df.columns if c.startswith('lag_') and 
                c not in ('lag_exact','lag_m15','lag_p15','lag_m30','lag_p30',
                          'lag_neighbor_avg','lag_window_avg',
                          'lag_exact_vs_geo','lag_exact_vs_hour')]
    for c in agg_cols:
        df[c] = df[c].fillna(gm)
    
    return df


def add_base_features(df, label_encoders=None, fit=True):
    """Standard features: temporal, categorical, geospatial, interactions."""
    if label_encoders is None:
        label_encoders = {}
    
    # Temporal
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['ts_minutes'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['ts_minutes'] / 1440)
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['time_bucket_2h'] = df['hour'] // 2
    df['time_bucket_4h'] = df['hour'] // 4
    
    # Geospatial
    coords = df['geohash'].apply(_decode_geohash)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    
    # Geo prefixes
    for p in [3, 4, 5]:
        col = f'geo{p}'
        if col not in df.columns:
            df[col] = df['geohash'].str[:p]
    
    # Categorical encoding
    for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
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
    
    # Geohash encoding
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
    
    # Temperature
    temp_med = df['Temperature'].median()
    df['Temperature_filled'] = df['Temperature'].fillna(temp_med)
    df['temp_missing'] = df['Temperature'].isnull().astype(int)
    
    # Interactions
    df['road_hour'] = df['RoadType_enc'] * 100 + df['hour']
    df['weather_hour'] = df['Weather_enc'] * 100 + df['hour']
    df['lanes_road'] = df['NumberofLanes'] * 10 + df['RoadType_enc']
    df['lanes_hour'] = df['NumberofLanes'] * 100 + df['hour']
    df['large_road'] = df['LargeVehicles_bin'] * 10 + df['RoadType_enc']
    
    return df, label_encoders


def add_oof_target_encoding(df, cols, target='demand', n_splits=5):
    """OOF target encoding: day 48 uses KFold OOF, day 49 uses full day 48 stats."""
    d48_mask = df['day'] == 48
    d49_mask = df['day'] == 49
    gm = df.loc[d48_mask, target].mean() if target in df.columns else 0.094
    smooth = 10
    
    for col in cols:
        te_col = f'{col}_te'
        df[te_col] = np.nan
        
        if target in df.columns:
            d48 = df.loc[d48_mask]
            stats = d48.groupby(col)[target].agg(['mean','count'])
            s = stats['count'] / (stats['count'] + smooth)
            stats['sm'] = s * stats['mean'] + (1-s) * gm
            full_map = stats['sm'].to_dict()
            
            # Day 49: use full day 48 stats
            df.loc[d49_mask, te_col] = df.loc[d49_mask, col].map(full_map)
            
            # Day 48: OOF
            d48_idx = df[d48_mask].index
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            for tr, va in kf.split(d48_idx):
                tr_rows, va_rows = d48_idx[tr], d48_idx[va]
                fs = df.loc[tr_rows].groupby(col)[target].agg(['mean','count'])
                s2 = fs['count'] / (fs['count'] + smooth)
                fs['sm'] = s2 * fs['mean'] + (1-s2) * gm
                df.loc[va_rows, te_col] = df.loc[va_rows, col].map(fs['sm'].to_dict())
        
        df[te_col] = df[te_col].fillna(gm)
    
    return df


def get_feature_cols(df):
    exclude = {'Index','geohash','timestamp','demand','day',
               'RoadType','LargeVehicles','Landmarks','Weather',
               'geo3','geo4','geo5','Temperature'}
    return [c for c in df.columns if c not in exclude and df[c].dtype != 'object']


# ---------------------------------------------------------------------------
# V3 Complete Feature Pipeline
# ---------------------------------------------------------------------------
def build_features_v3(train_df, test_df):
    train_df = parse_timestamp(train_df)
    test_df = parse_timestamp(test_df)
    
    # Build lookups from day 48
    print("  Building day 48 lookups...")
    lookups = build_day48_lookups(train_df)
    
    # Add lag features with proper handling
    print("  Adding lag features (day 48: NaN exact, day 49: full)...")
    train_df = add_lag_features_v3(train_df, lookups)
    # Test is all day 49
    test_df['day'] = 49  # ensure day column exists for test
    test_df = add_lag_features_v3(test_df, lookups)
    
    # Add base features
    print("  Adding base features...")
    train_df, le = add_base_features(train_df, fit=True)
    test_df, _ = add_base_features(test_df, label_encoders=le, fit=False)
    
    # OOF target encoding
    print("  Adding OOF target encoding...")
    te_cols = ['geohash_enc', 'geo4_enc', 'geo5_enc', 'RoadType_enc', 'road_hour', 'lanes_road']
    train_df = add_oof_target_encoding(train_df, te_cols)
    
    d48 = train_df[train_df['day'] == 48]
    gm = d48['demand'].mean()
    smooth = 10
    for col in te_cols:
        te_col = f'{col}_te'
        stats = d48.groupby(col)['demand'].agg(['mean','count'])
        s = stats['count'] / (stats['count'] + smooth)
        stats['sm'] = s * stats['mean'] + (1-s) * gm
        test_df[te_col] = test_df[col].map(stats['sm'].to_dict()).fillna(gm)
    
    feature_cols = get_feature_cols(train_df)
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
    model = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
    pred = model.predict(X_va[feats])
    if use_log: pred = np.expm1(pred)
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
    if use_log: pred = np.expm1(pred)
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
                      evals=[(dval, 'val')], early_stopping_rounds=200, verbose_eval=500)
    pred = model.predict(dval)
    if use_log: pred = np.expm1(pred)
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
# Optuna Tuning
# ---------------------------------------------------------------------------
def tune_lgb(X_tr, y_tr, X_va, y_va, feats, n_trials=80):
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
        _, _, s = train_lgb(X_tr, y_tr, X_va, y_va, feats, params)
        return s
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  LGB best R2: {study.best_value:.6f}")
    return study.best_params


def tune_cb(X_tr, y_tr, X_va, y_va, feats, n_trials=50):
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
        _, _, s = train_cb(X_tr, y_tr, X_va, y_va, feats, params)
        return s
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  CB best R2: {study.best_value:.6f}")
    return study.best_params


def tune_xgb(X_tr, y_tr, X_va, y_va, feats, n_trials=60):
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
        _, _, s = train_xgb(X_tr, y_tr, X_va, y_va, feats, params)
        return s
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  XGB best R2: {study.best_value:.6f}")
    return study.best_params


def optimize_weights(preds_dict, y_true):
    best_score, best_w = -np.inf, None
    names = list(preds_dict.keys())
    preds = [preds_dict[n] for n in names]
    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1-w1, 21):
            w3 = max(1.0-w1-w2, 0)
            p = sum(w*pr for w, pr in zip([w1,w2,w3], preds))
            s = r2_score(y_true, np.clip(p, 0, None))
            if s > best_score:
                best_score = s
                best_w = dict(zip(names, [w1,w2,w3]))
    return best_w, best_score


# ---------------------------------------------------------------------------
# V3 Main Pipeline
# ---------------------------------------------------------------------------
def run_v3():
    print("=" * 70)
    print("V3 PIPELINE -- PROPER LAG HANDLING (no day48 leakage)")
    print("=" * 70)
    
    print("\n[1/6] Loading data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")
    
    print("\n[2/6] V3 Feature Engineering...")
    t0 = time.time()
    train_df, test_df, feature_cols, lookups = build_features_v3(train_raw.copy(), test_raw.copy())
    print(f"  Time: {time.time()-t0:.1f}s")
    
    y_full = train_df['demand'].values
    
    # Temporal validation
    print("\n[3/6] Temporal validation (day 48 -> day 49)...")
    d48_mask = train_df['day'] == 48
    d49_mask = train_df['day'] == 49
    X_tr = train_df[d48_mask]
    X_va = train_df[d49_mask]
    y_tr = y_full[d48_mask.values]
    y_va = y_full[d49_mask.values]
    
    _, lgb_vp, lgb_s = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  LGB baseline R2: {lgb_s:.6f}")
    _, cb_vp, cb_s = train_cb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  CB  baseline R2: {cb_s:.6f}")
    _, xgb_vp, xgb_s = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  XGB baseline R2: {xgb_s:.6f}")
    
    # Quick ensemble check
    best_w, ens_s = optimize_weights({'lgb': lgb_vp, 'cb': cb_vp, 'xgb': xgb_vp}, y_va)
    print(f"  Baseline ensemble R2: {ens_s:.6f} (weights: {best_w})")
    
    # Optuna tuning
    print("\n[4/6] Hyperparameter Tuning (Optuna)...")
    print("  Tuning LightGBM (80 trials)...")
    best_lgb = tune_lgb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=80)
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'verbose': -1, 'seed': SEED, 'n_jobs': -1,
        **{k: v for k, v in best_lgb.items()},
    }
    # Rename 'lr' to 'learning_rate'
    lgb_params['learning_rate'] = lgb_params.pop('lr')
    lgb_m, lgb_vp, lgb_s = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lgb_params)
    print(f"  LGB tuned R2: {lgb_s:.6f}")
    
    print("  Tuning CatBoost (50 trials)...")
    best_cb = tune_cb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=50)
    cb_params = {
        'iterations': 10000, 'loss_function': 'RMSE', 'eval_metric': 'R2',
        'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 200,
        'task_type': 'CPU',
        'learning_rate': best_cb['lr'],
        'depth': best_cb['depth'],
        'l2_leaf_reg': best_cb['l2_leaf_reg'],
        'min_data_in_leaf': best_cb['min_data_in_leaf'],
        'random_strength': best_cb['random_strength'],
        'bagging_temperature': best_cb['bagging_temperature'],
    }
    cb_m, cb_vp, cb_s = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cb_params)
    print(f"  CB  tuned R2: {cb_s:.6f}")
    
    print("  Tuning XGBoost (60 trials)...")
    best_xgb = tune_xgb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=60)
    xgb_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'seed': SEED, 'nthread': -1, 'verbosity': 0,
        'max_depth': best_xgb['max_depth'],
        'eta': best_xgb['eta'],
        'subsample': best_xgb['subsample'],
        'colsample_bytree': best_xgb['colsample_bytree'],
        'min_child_weight': best_xgb['min_child_weight'],
        'lambda': best_xgb['lambda'],
        'alpha': best_xgb['alpha'],
        'gamma': best_xgb['gamma'],
    }
    xgb_m, xgb_vp, xgb_s = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xgb_params)
    print(f"  XGB tuned R2: {xgb_s:.6f}")
    
    # Ensemble
    print("\n[5/6] Ensemble Optimization...")
    best_w, ens_s = optimize_weights({'lgb': lgb_vp, 'cb': cb_vp, 'xgb': xgb_vp}, y_va)
    print(f"  Weights: {best_w}")
    print(f"  Ensemble R2: {ens_s:.6f}")
    
    # Seed averaging
    print("\n[6/6] Seed Averaging + Final Submission...")
    SEEDS = [42, 123, 456, 789, 2024]
    test_preds = {'lgb': [], 'cb': [], 'xgb': []}
    val_preds_all = {'lgb': [], 'cb': [], 'xgb': []}
    
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)})...")
        lp = {**lgb_params, 'seed': seed}
        m, vp, s = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lp)
        test_preds['lgb'].append(pred_lgb(m, test_df, feature_cols))
        val_preds_all['lgb'].append(vp)
        
        cp = {**cb_params, 'random_seed': seed}
        m, vp, s = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cp)
        test_preds['cb'].append(pred_cb(m, test_df, feature_cols))
        val_preds_all['cb'].append(vp)
        
        xp = {**xgb_params, 'seed': seed}
        m, vp, s = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xp)
        test_preds['xgb'].append(pred_xgb(m, test_df, feature_cols))
        val_preds_all['xgb'].append(vp)
    
    lgb_test = np.mean(test_preds['lgb'], axis=0)
    cb_test = np.mean(test_preds['cb'], axis=0)
    xgb_test = np.mean(test_preds['xgb'], axis=0)
    lgb_val = np.mean(val_preds_all['lgb'], axis=0)
    cb_val = np.mean(val_preds_all['cb'], axis=0)
    xgb_val = np.mean(val_preds_all['xgb'], axis=0)
    
    final_w, final_s = optimize_weights({'lgb': lgb_val, 'cb': cb_val, 'xgb': xgb_val}, y_va)
    print(f"\n  Final weights: {final_w}")
    print(f"  Final ensemble R2: {final_s:.6f}")
    
    final_pred = final_w['lgb']*lgb_test + final_w['cb']*cb_test + final_w['xgb']*xgb_test
    final_pred = np.clip(final_pred, 0, None)
    
    for name, pred in [('final_v3', final_pred), ('lgb_v3', lgb_test),
                       ('cb_v3', cb_test), ('xgb_v3', xgb_test)]:
        sub = pd.DataFrame({'Index': test_df['Index'].values, 'demand': pred})
        sub['demand'] = sub['demand'].clip(lower=0)
        assert sub.shape == (41778, 2), f"Wrong shape: {sub.shape}"
        assert sub['demand'].isnull().sum() == 0, "NaN in predictions!"
        sub.to_csv(SUBMISSIONS_DIR / f'{name}_submission.csv', index=False)
        print(f"  Saved: {name}_submission.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("V3 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Features: {len(feature_cols)}")
    print(f"  LGB tuned R2:       {lgb_s:.6f}")
    print(f"  CB  tuned R2:       {cb_s:.6f}")
    print(f"  XGB tuned R2:       {xgb_s:.6f}")
    print(f"  FINAL ensemble R2:  {final_s:.6f}")
    print(f"  Weights: {final_w}")
    print("=" * 70)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_m.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)
    importance.to_csv(EXPERIMENTS_DIR / 'feature_importance_v3.csv', index=False)
    print("\n  Top 25 features (LGB gain):")
    print(importance.head(25).to_string(index=False))


if __name__ == '__main__':
    run_v3()
