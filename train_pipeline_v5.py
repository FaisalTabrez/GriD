"""
V5 Pipeline - Enhanced Feature Engineering + Multi-Target Ensemble

Building on V4's success (R2=0.831), adding:
  1. Structural geohash features (avg lanes, modal road type, diversity)
  2. Geospatial clustering (KMeans regions)
  3. Richer interaction target encodings
  4. Temperature/Weather interactions  
  5. Log + Raw target ensemble (captures different patterns)
  6. CatBoost native categorical mode
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
SEED = 42
DATA_DIR = PROJECT_ROOT / 'e88186124ec611f1' / 'dataset'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
np.random.seed(SEED)

_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BITS = [16, 8, 4, 2, 1]

def _decode_geohash(gh):
    is_lon = True
    lat_r, lon_r = [-90.0, 90.0], [-180.0, 180.0]
    for c in gh:
        cd = _BASE32.index(c)
        for mask in _BITS:
            if is_lon:
                mid = (lon_r[0]+lon_r[1])/2
                if cd & mask: lon_r[0] = mid
                else: lon_r[1] = mid
            else:
                mid = (lat_r[0]+lat_r[1])/2
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


def build_features_v5(train_df, test_df):
    """V5 features: V4 base + structural geohash + clustering + richer interactions."""
    
    train_df = parse_timestamp(train_df)
    test_df = parse_timestamp(test_df)
    
    # --- Temporal ---
    for df in [train_df, test_df]:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['ts_minutes'] / 1440)
        df['minute_cos'] = np.cos(2 * np.pi * df['ts_minutes'] / 1440)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_midday'] = ((df['hour'] >= 10) & (df['hour'] <= 15)).astype(int)
        df['time_bucket_2h'] = df['hour'] // 2
        df['time_bucket_4h'] = df['hour'] // 4
        df['time_bucket_3h'] = df['hour'] // 3
    
    # --- Geospatial ---
    for df in [train_df, test_df]:
        coords = df['geohash'].apply(_decode_geohash)
        df['latitude'] = coords.apply(lambda x: x[0])
        df['longitude'] = coords.apply(lambda x: x[1])
        for p in [3, 4, 5]:
            df[f'geo{p}'] = df['geohash'].str[:p]
    
    # --- Categorical encoding ---
    label_encoders = {}
    for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geohash', 'geo3', 'geo4', 'geo5']:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]]).fillna('missing').astype(str)
        le.fit(combined)
        label_encoders[col] = le
        train_df[f'{col}_enc'] = le.transform(train_df[col].fillna('missing').astype(str))
        test_df[f'{col}_enc'] = le.transform(test_df[col].fillna('missing').astype(str))
    
    for df in [train_df, test_df]:
        df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
        df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)
    
    # --- Temperature ---
    temp_med = train_df['Temperature'].median()
    for df in [train_df, test_df]:
        df['Temperature_filled'] = df['Temperature'].fillna(temp_med)
        df['temp_missing'] = df['Temperature'].isnull().astype(int)
        df['temp_high'] = (df['Temperature_filled'] > df['Temperature_filled'].quantile(0.75)).astype(int)
        df['temp_low'] = (df['Temperature_filled'] < df['Temperature_filled'].quantile(0.25)).astype(int)
    
    # --- Basic interactions ---
    for df in [train_df, test_df]:
        df['road_hour'] = df['RoadType_enc'] * 100 + df['hour']
        df['lanes_road'] = df['NumberofLanes'] * 10 + df['RoadType_enc']
        df['lanes_hour'] = df['NumberofLanes'] * 100 + df['hour']
        df['large_road'] = df['LargeVehicles_bin'] * 10 + df['RoadType_enc']
        df['weather_road'] = df['Weather_enc'] * 10 + df['RoadType_enc']
        df['weather_hour'] = df['Weather_enc'] * 100 + df['hour']
        df['landmark_road'] = df['Landmarks_bin'] * 10 + df['RoadType_enc']
        df['is_highway'] = (df['NumberofLanes'] >= 4).astype(int)
        df['road_large'] = df['RoadType_enc'] * 10 + df['LargeVehicles_bin']
        df['road_landmark'] = df['RoadType_enc'] * 10 + df['Landmarks_bin']
        df['road_weather_hour'] = df['RoadType_enc'] * 1000 + df['Weather_enc'] * 100 + df['hour']
        df['lanes_large'] = df['NumberofLanes'] * 10 + df['LargeVehicles_bin']
        # Temperature x RoadType interaction
        df['temp_x_road'] = df['Temperature_filled'] * (df['RoadType_enc'] + 1)
        df['temp_x_highway'] = df['Temperature_filled'] * df['is_highway']
    
    # --- Geospatial clustering ---
    print("  Computing geospatial clusters...")
    all_geos = pd.concat([train_df[['geohash','latitude','longitude']],
                          test_df[['geohash','latitude','longitude']]]).drop_duplicates('geohash')
    coords_arr = all_geos[['latitude','longitude']].values
    
    for n_clusters in [5, 10, 20, 50]:
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        all_geos[f'cluster_{n_clusters}'] = km.fit_predict(coords_arr)
        cluster_map = dict(zip(all_geos['geohash'], all_geos[f'cluster_{n_clusters}']))
        for df in [train_df, test_df]:
            df[f'cluster_{n_clusters}'] = df['geohash'].map(cluster_map).fillna(0).astype(int)
        # Distance to cluster center
        centers = km.cluster_centers_
        for df in [train_df, test_df]:
            c_ids = df[f'cluster_{n_clusters}'].values
            df[f'dist_center_{n_clusters}'] = np.sqrt(
                (df['latitude'].values - centers[c_ids, 0])**2 + 
                (df['longitude'].values - centers[c_ids, 1])**2
            )
    
    # --- Structural geohash features (non-demand-based) ---
    print("  Computing structural geohash features...")
    d48 = train_df[train_df['day'] == 48]
    
    # Average NumberofLanes per geohash
    geo_avg_lanes = d48.groupby('geohash')['NumberofLanes'].mean().to_dict()
    geo_max_lanes = d48.groupby('geohash')['NumberofLanes'].max().to_dict()
    
    # Modal RoadType per geohash  
    geo_modal_road = d48.groupby('geohash')['RoadType_enc'].agg(lambda x: x.mode().iloc[0]).to_dict()
    
    # RoadType diversity (number of unique road types)
    geo_road_diversity = d48.groupby('geohash')['RoadType_enc'].nunique().to_dict()
    
    # LargeVehicles percentage per geohash
    geo_large_pct = d48.groupby('geohash')['LargeVehicles_bin'].mean().to_dict()
    
    # Landmarks percentage per geohash
    geo_landmark_pct = d48.groupby('geohash')['Landmarks_bin'].mean().to_dict()
    
    # Temperature stats per geohash
    geo_temp_mean = d48.groupby('geohash')['Temperature_filled'].mean().to_dict()
    geo_temp_std = d48.groupby('geohash')['Temperature_filled'].std().fillna(0).to_dict()
    
    # Weather diversity per geohash
    geo_weather_div = d48.groupby('geohash')['Weather_enc'].nunique().to_dict()
    
    # Observation frequency
    geo_freq = d48.groupby('geohash').size().to_dict()
    
    # RoadType percentages per geohash
    for rt in ['Highway', 'Residential', 'Street']:
        key = f'geo_pct_{rt.lower()}'
        geo_rt = d48.groupby('geohash').apply(lambda x: (x['RoadType'] == rt).mean()).to_dict()
        for df in [train_df, test_df]:
            df[key] = df['geohash'].map(geo_rt).fillna(0)
    
    for df in [train_df, test_df]:
        df['geo_avg_lanes'] = df['geohash'].map(geo_avg_lanes).fillna(2)
        df['geo_max_lanes'] = df['geohash'].map(geo_max_lanes).fillna(2)
        df['geo_modal_road'] = df['geohash'].map(geo_modal_road).fillna(0)
        df['geo_road_diversity'] = df['geohash'].map(geo_road_diversity).fillna(1)
        df['geo_large_pct'] = df['geohash'].map(geo_large_pct).fillna(0.5)
        df['geo_landmark_pct'] = df['geohash'].map(geo_landmark_pct).fillna(0.5)
        df['geo_temp_mean'] = df['geohash'].map(geo_temp_mean).fillna(temp_med)
        df['geo_temp_std'] = df['geohash'].map(geo_temp_std).fillna(0)
        df['geo_weather_div'] = df['geohash'].map(geo_weather_div).fillna(1)
        df['geohash_freq'] = df['geohash'].map(geo_freq).fillna(0)
        # Interactions with structural features
        df['lanes_diff'] = df['NumberofLanes'] - df['geo_avg_lanes']
        df['is_max_lanes'] = (df['NumberofLanes'] == df['geo_max_lanes']).astype(int)
    
    # --- OOF Target Encoding ---
    print("  Computing OOF target encoding...")
    gm = d48['demand'].mean()
    smooth = 20
    
    te_cols = ['RoadType_enc', 'lanes_road', 'road_hour', 'NumberofLanes',
               'large_road', 'LargeVehicles_bin', 'road_large', 'weather_road',
               'landmark_road', 'road_landmark', 'lanes_large',
               'cluster_10', 'cluster_20']
    
    for col in te_cols:
        te_col = f'{col}_te'
        train_df[te_col] = np.nan
        
        d48_idx = train_df[train_df['day'] == 48].index
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr_i, va_i in kf.split(d48_idx):
            tr_rows, va_rows = d48_idx[tr_i], d48_idx[va_i]
            stats = train_df.loc[tr_rows].groupby(col)['demand'].agg(['mean','count'])
            s = stats['count'] / (stats['count'] + smooth)
            stats['sm'] = s * stats['mean'] + (1-s) * gm
            train_df.loc[va_rows, te_col] = train_df.loc[va_rows, col].map(stats['sm'].to_dict())
        
        d49_idx = train_df[train_df['day'] == 49].index
        stats = d48.groupby(col)['demand'].agg(['mean','count'])
        s = stats['count'] / (stats['count'] + smooth)
        stats['sm'] = s * stats['mean'] + (1-s) * gm
        full_map = stats['sm'].to_dict()
        train_df.loc[d49_idx, te_col] = train_df.loc[d49_idx, col].map(full_map)
        test_df[te_col] = test_df[col].map(full_map).fillna(gm)
        train_df[te_col] = train_df[te_col].fillna(gm)
    
    # --- Feature columns ---
    exclude = {'Index','geohash','timestamp','demand','day',
               'RoadType','LargeVehicles','Landmarks','Weather',
               'geo3','geo4','geo5','Temperature'}
    feature_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype != 'object']
    
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0
    
    print(f"  Total features: {len(feature_cols)}")
    return train_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# Model Training Functions
# ---------------------------------------------------------------------------
def train_lgb(X_tr, y_tr, X_va, y_va, feats, params=None, use_log=True):
    if params is None:
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': 0.03, 'num_leaves': 127, 'max_depth': 6,
            'min_child_samples': 50, 'feature_fraction': 0.7,
            'bagging_fraction': 0.7, 'bagging_freq': 5,
            'lambda_l1': 0.1, 'lambda_l2': 1.0,
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
            'iterations': 10000, 'learning_rate': 0.03, 'depth': 6,
            'l2_leaf_reg': 5, 'loss_function': 'RMSE', 'eval_metric': 'R2',
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
            'max_depth': 6, 'eta': 0.03, 'subsample': 0.7,
            'colsample_bytree': 0.7, 'min_child_weight': 20,
            'lambda': 5.0, 'alpha': 0.5, 'gamma': 1.0,
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


def tune_lgb(X_tr, y_tr, X_va, y_va, feats, n_trials=120):
    def obj(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 10.0, log=True),
            'verbose': -1, 'seed': SEED, 'n_jobs': -1,
        }
        _, _, s = train_lgb(X_tr, y_tr, X_va, y_va, feats, params)
        return s
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  LGB best R2: {study.best_value:.6f}")
    return study.best_params


def tune_cb(X_tr, y_tr, X_va, y_va, feats, n_trials=80):
    def obj(trial):
        params = {
            'iterations': 10000,
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 50.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
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


def tune_xgb(X_tr, y_tr, X_va, y_va, feats, n_trials=100):
    def obj(trial):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 100),
            'lambda': trial.suggest_float('lambda', 0.1, 50.0, log=True),
            'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 10.0, log=True),
            'seed': SEED, 'nthread': -1, 'verbosity': 0,
        }
        _, _, s = train_xgb(X_tr, y_tr, X_va, y_va, feats, params)
        return s
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    print(f"  XGB best R2: {study.best_value:.6f}")
    return study.best_params


def optimize_weights(preds_list, names, y_true):
    """Optimize weights for N models."""
    n = len(preds_list)
    if n == 2:
        best_score, best_w = -np.inf, None
        for w1 in np.linspace(0, 1, 51):
            w2 = 1 - w1
            p = w1*preds_list[0] + w2*preds_list[1]
            s = r2_score(y_true, np.clip(p, 0, None))
            if s > best_score:
                best_score, best_w = s, dict(zip(names, [w1, w2]))
        return best_w, best_score
    elif n == 3:
        best_score, best_w = -np.inf, None
        for w1 in np.linspace(0, 1, 21):
            for w2 in np.linspace(0, 1-w1, 21):
                w3 = max(1.0-w1-w2, 0)
                p = w1*preds_list[0] + w2*preds_list[1] + w3*preds_list[2]
                s = r2_score(y_true, np.clip(p, 0, None))
                if s > best_score:
                    best_score, best_w = s, dict(zip(names, [w1,w2,w3]))
        return best_w, best_score
    else:
        # For many models, do greedy forward selection
        best_score = -np.inf
        remaining = list(range(n))
        selected_w = np.zeros(n)
        for _ in range(min(n, 5)):
            best_add = -1
            for idx in remaining:
                for w in np.linspace(0.05, 0.5, 10):
                    test_w = selected_w.copy()
                    test_w[idx] += w
                    test_w /= test_w.sum()
                    p = sum(test_w[j]*preds_list[j] for j in range(n))
                    s = r2_score(y_true, np.clip(p, 0, None))
                    if s > best_score:
                        best_score = s
                        best_add = idx
                        best_new_w = test_w.copy()
            if best_add >= 0:
                selected_w = best_new_w
                remaining = [r for r in remaining if r != best_add]
        return dict(zip(names, selected_w)), best_score


def run_v5():
    print("=" * 70)
    print("V5 PIPELINE -- ENHANCED FEATURES + MULTI-TARGET ENSEMBLE")
    print("=" * 70)
    
    print("\n[1/7] Loading data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")
    
    print("\n[2/7] V5 Feature Engineering...")
    t0 = time.time()
    train_df, test_df, feature_cols = build_features_v5(train_raw.copy(), test_raw.copy())
    print(f"  Time: {time.time()-t0:.1f}s")
    
    y_full = train_df['demand'].values
    d48_mask = train_df['day'] == 48
    d49_mask = train_df['day'] == 49
    X_tr = train_df[d48_mask]
    X_va = train_df[d49_mask]
    y_tr = y_full[d48_mask.values]
    y_va = y_full[d49_mask.values]
    
    # --- Baseline ---
    print("\n[3/7] Baseline temporal validation...")
    _, lgb_vp, lgb_s = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  LGB R2: {lgb_s:.6f}")
    _, cb_vp, cb_s = train_cb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  CB  R2: {cb_s:.6f}")
    _, xgb_vp, xgb_s = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  XGB R2: {xgb_s:.6f}")
    
    # Raw target models (no log transform)
    _, lgb_vp_raw, lgb_s_raw = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, use_log=False)
    print(f"  LGB raw R2: {lgb_s_raw:.6f}")
    _, cb_vp_raw, cb_s_raw = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, use_log=False)
    print(f"  CB  raw R2: {cb_s_raw:.6f}")
    
    w0, ens0 = optimize_weights([lgb_vp, cb_vp, xgb_vp], ['lgb','cb','xgb'], y_va)
    print(f"  Baseline ensemble R2: {ens0:.6f}")
    
    # Log+Raw blend
    w0b, ens0b = optimize_weights([cb_vp, cb_vp_raw], ['cb_log','cb_raw'], y_va)
    print(f"  CB log+raw blend R2: {ens0b:.6f}")
    
    # --- Optuna Tuning ---
    print("\n[4/7] Hyperparameter Tuning (Optuna)...")
    
    print("  Tuning LightGBM (120 trials)...")
    best_lgb = tune_lgb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=120)
    lgb_params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
                  'verbose': -1, 'seed': SEED, 'n_jobs': -1}
    lgb_params['learning_rate'] = best_lgb.pop('lr')
    lgb_params.update(best_lgb)
    lgb_m, lgb_vp, lgb_s = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lgb_params)
    print(f"  LGB tuned R2: {lgb_s:.6f}")
    
    print("  Tuning CatBoost (80 trials)...")
    best_cb = tune_cb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=80)
    cb_params = {'iterations': 10000, 'loss_function': 'RMSE', 'eval_metric': 'R2',
                 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 200,
                 'task_type': 'CPU', 'learning_rate': best_cb['lr'],
                 'depth': best_cb['depth'], 'l2_leaf_reg': best_cb['l2_leaf_reg'],
                 'min_data_in_leaf': best_cb['min_data_in_leaf'],
                 'random_strength': best_cb['random_strength'],
                 'bagging_temperature': best_cb['bagging_temperature']}
    cb_m, cb_vp, cb_s = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cb_params)
    print(f"  CB  tuned R2: {cb_s:.6f}")
    
    print("  Tuning XGBoost (100 trials)...")
    best_xgb = tune_xgb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=100)
    xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                  'seed': SEED, 'nthread': -1, 'verbosity': 0}
    xgb_params.update(best_xgb)
    xgb_m, xgb_vp, xgb_s = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xgb_params)
    print(f"  XGB tuned R2: {xgb_s:.6f}")
    
    # Tuned raw models
    print("  Training tuned raw (no log) models...")
    lgb_params_raw = {**lgb_params}
    _, lgb_vp_raw, lgb_s_raw = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lgb_params_raw, use_log=False)
    cb_params_raw = {**cb_params}
    _, cb_vp_raw, cb_s_raw = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cb_params_raw, use_log=False)
    xgb_params_raw = {**xgb_params}
    _, xgb_vp_raw, xgb_s_raw = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xgb_params_raw, use_log=False)
    print(f"  LGB raw R2: {lgb_s_raw:.6f}, CB raw R2: {cb_s_raw:.6f}, XGB raw R2: {xgb_s_raw:.6f}")
    
    # --- Ensemble ---
    print("\n[5/7] Ensemble Optimization (6 models)...")
    all_names = ['lgb_log','cb_log','xgb_log','lgb_raw','cb_raw','xgb_raw']
    all_preds = [lgb_vp, cb_vp, xgb_vp, lgb_vp_raw, cb_vp_raw, xgb_vp_raw]
    
    # 3-model log ensemble
    w_log, s_log = optimize_weights([lgb_vp, cb_vp, xgb_vp], ['lgb','cb','xgb'], y_va)
    print(f"  Log ensemble R2: {s_log:.6f} (w={w_log})")
    
    # 3-model raw ensemble
    w_raw, s_raw = optimize_weights([lgb_vp_raw, cb_vp_raw, xgb_vp_raw], ['lgb','cb','xgb'], y_va)
    print(f"  Raw ensemble R2: {s_raw:.6f} (w={w_raw})")
    
    # Log+Raw blend
    log_ens = w_log['lgb']*lgb_vp + w_log['cb']*cb_vp + w_log['xgb']*xgb_vp
    raw_ens = w_raw['lgb']*lgb_vp_raw + w_raw['cb']*cb_vp_raw + w_raw['xgb']*xgb_vp_raw
    w_blend, s_blend = optimize_weights([log_ens, raw_ens], ['log','raw'], y_va)
    print(f"  Log+Raw blend R2: {s_blend:.6f} (w={w_blend})")
    
    # Full 6-model ensemble
    w_full, s_full = optimize_weights(all_preds, all_names, y_va)
    print(f"  Full 6-model R2: {s_full:.6f} (w={w_full})")
    
    # --- Seed Averaging ---
    print("\n[6/7] Seed Averaging + Submission...")
    SEEDS = [42, 123, 456, 789, 2024]
    
    test_log = {'lgb': [], 'cb': [], 'xgb': []}
    test_raw_d = {'lgb': [], 'cb': [], 'xgb': []}
    val_log = {'lgb': [], 'cb': [], 'xgb': []}
    val_raw_d = {'lgb': [], 'cb': [], 'xgb': []}
    
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)})...")
        
        # Log models
        lp = {**lgb_params, 'seed': seed}
        m, vp, _ = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lp)
        test_log['lgb'].append(pred_lgb(m, test_df, feature_cols))
        val_log['lgb'].append(vp)
        
        cp = {**cb_params, 'random_seed': seed}
        m, vp, _ = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cp)
        test_log['cb'].append(pred_cb(m, test_df, feature_cols))
        val_log['cb'].append(vp)
        
        xp = {**xgb_params, 'seed': seed}
        m, vp, _ = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xp)
        test_log['xgb'].append(pred_xgb(m, test_df, feature_cols))
        val_log['xgb'].append(vp)
        
        # Raw models
        m, vp, _ = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lp, use_log=False)
        test_raw_d['lgb'].append(pred_lgb(m, test_df, feature_cols, use_log=False))
        val_raw_d['lgb'].append(vp)
        
        m, vp, _ = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cp, use_log=False)
        test_raw_d['cb'].append(pred_cb(m, test_df, feature_cols, use_log=False))
        val_raw_d['cb'].append(vp)
        
        m, vp, _ = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xp, use_log=False)
        test_raw_d['xgb'].append(pred_xgb(m, test_df, feature_cols, use_log=False))
        val_raw_d['xgb'].append(vp)
    
    # Average across seeds
    lgb_test_log = np.mean(test_log['lgb'], axis=0)
    cb_test_log = np.mean(test_log['cb'], axis=0)
    xgb_test_log = np.mean(test_log['xgb'], axis=0)
    lgb_test_raw = np.mean(test_raw_d['lgb'], axis=0)
    cb_test_raw = np.mean(test_raw_d['cb'], axis=0)
    xgb_test_raw = np.mean(test_raw_d['xgb'], axis=0)
    
    lgb_val_log = np.mean(val_log['lgb'], axis=0)
    cb_val_log = np.mean(val_log['cb'], axis=0)
    xgb_val_log = np.mean(val_log['xgb'], axis=0)
    lgb_val_raw = np.mean(val_raw_d['lgb'], axis=0)
    cb_val_raw = np.mean(val_raw_d['cb'], axis=0)
    xgb_val_raw = np.mean(val_raw_d['xgb'], axis=0)
    
    # Final ensemble with seed-averaged predictions
    final_w, final_s = optimize_weights(
        [lgb_val_log, cb_val_log, xgb_val_log, lgb_val_raw, cb_val_raw, xgb_val_raw],
        ['lgb_log','cb_log','xgb_log','lgb_raw','cb_raw','xgb_raw'], y_va)
    print(f"\n  Final 6-model weights: {final_w}")
    print(f"  Final ensemble R2: {final_s:.6f}")
    
    # Build final predictions
    test_all = [lgb_test_log, cb_test_log, xgb_test_log, lgb_test_raw, cb_test_raw, xgb_test_raw]
    final_pred = sum(final_w[n]*p for n,p in zip(
        ['lgb_log','cb_log','xgb_log','lgb_raw','cb_raw','xgb_raw'], test_all))
    final_pred = np.clip(final_pred, 0, None)
    
    # Also build single-strategy submissions
    log_pred = w_log['lgb']*lgb_test_log + w_log['cb']*cb_test_log + w_log['xgb']*xgb_test_log
    log_pred = np.clip(log_pred, 0, None)
    
    # --- Save submissions ---
    print("\n[7/7] Saving submissions...")
    for name, pred in [('final_v5', final_pred), 
                       ('final_v5_log', log_pred),
                       ('cb_v5_log', cb_test_log),
                       ('cb_v5_raw', cb_test_raw)]:
        sub = pd.DataFrame({'Index': test_df['Index'].values, 'demand': pred})
        sub['demand'] = sub['demand'].clip(lower=0)
        assert sub.shape == (41778, 2)
        assert sub['demand'].isnull().sum() == 0
        sub.to_csv(SUBMISSIONS_DIR / f'{name}_submission.csv', index=False)
        print(f"  Saved: {name}_submission.csv")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("V5 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Features: {len(feature_cols)}")
    print(f"  LGB tuned R2 (log): {lgb_s:.6f}")
    print(f"  CB  tuned R2 (log): {cb_s:.6f}")
    print(f"  XGB tuned R2 (log): {xgb_s:.6f}")
    print(f"  LGB tuned R2 (raw): {lgb_s_raw:.6f}")
    print(f"  CB  tuned R2 (raw): {cb_s_raw:.6f}")
    print(f"  XGB tuned R2 (raw): {xgb_s_raw:.6f}")
    print(f"  Log ensemble R2:    {s_log:.6f}")
    print(f"  Raw ensemble R2:    {s_raw:.6f}")
    print(f"  Log+Raw blend R2:   {s_blend:.6f}")
    print(f"  FINAL ensemble R2:  {final_s:.6f}")
    print("=" * 70)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_m.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)
    importance.to_csv(EXPERIMENTS_DIR / 'feature_importance_v5.csv', index=False)
    print("\n  Top 25 features (LGB gain):")
    print(importance.head(25).to_string(index=False))


if __name__ == '__main__':
    run_v5()
