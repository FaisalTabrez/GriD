"""
V4 Pipeline - Focused on Generalization

KEY INSIGHTS from analysis:
  1. RoadType mean alone gives R2=0.756 on temporal validation
  2. geohash-level features OVERFIT (0.919 OOF vs 0.525 temporal)
  3. Our V3 ML model (0.735) performs WORSE than RoadType mean (0.756)
  4. Lag features hurt more than help (day48->day49 patterns don't repeat)

STRATEGY:
  - Use RoadType-based features as the core signal
  - Limit geohash-level features to avoid overfitting
  - Focus on features that GENERALIZE across days
  - Train on full data with careful regularization
  - Try training on day 49 ONLY (small but perfectly representative)
"""

import os, sys, time, warnings
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


def build_features_v4(train_df, test_df):
    """V4 features: focused on generalizable features, avoid geohash-level overfitting."""
    
    train_df = parse_timestamp(train_df)
    test_df = parse_timestamp(test_df)
    
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # --- Temporal features ---
    for df in [train_df, test_df]:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['ts_minutes'] / 1440)
        df['minute_cos'] = np.cos(2 * np.pi * df['ts_minutes'] / 1440)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['time_bucket_2h'] = df['hour'] // 2
        df['time_bucket_4h'] = df['hour'] // 4
    
    # --- Geospatial features ---
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
        enc_col = f'{col}_enc'
        train_df[enc_col] = le.transform(train_df[col].fillna('missing').astype(str))
        test_df[enc_col] = le.transform(test_df[col].fillna('missing').astype(str))
    
    for df in [train_df, test_df]:
        df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
        df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)
    
    # --- Temperature ---
    temp_med = train_df['Temperature'].median()
    for df in [train_df, test_df]:
        df['Temperature_filled'] = df['Temperature'].fillna(temp_med)
        df['temp_missing'] = df['Temperature'].isnull().astype(int)
    
    # --- Interaction features ---
    for df in [train_df, test_df]:
        df['road_hour'] = df['RoadType_enc'] * 100 + df['hour']
        df['lanes_road'] = df['NumberofLanes'] * 10 + df['RoadType_enc']
        df['lanes_hour'] = df['NumberofLanes'] * 100 + df['hour']
        df['large_road'] = df['LargeVehicles_bin'] * 10 + df['RoadType_enc']
        df['weather_hour'] = df['Weather_enc'] * 100 + df['hour']
        df['is_highway'] = (df['NumberofLanes'] >= 4).astype(int)
        df['road_large'] = df['RoadType_enc'] * 10 + df['LargeVehicles_bin']
    
    # --- OOF Target Encoding (ONLY on road-level features, NOT geohash) ---
    # Key: Only encode features that generalize across days
    d48 = train_df[train_df['day'] == 48]
    d49 = train_df[train_df['day'] == 49]
    gm = d48['demand'].mean()
    smooth = 20  # Higher smoothing to prevent overfitting
    
    te_cols = ['RoadType_enc', 'lanes_road', 'road_hour', 'NumberofLanes',
               'large_road', 'LargeVehicles_bin', 'road_large']
    
    for col in te_cols:
        te_col = f'{col}_te'
        train_df[te_col] = np.nan
        
        # Day 48: OOF
        d48_idx = train_df[train_df['day'] == 48].index
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr_i, va_i in kf.split(d48_idx):
            tr_rows, va_rows = d48_idx[tr_i], d48_idx[va_i]
            stats = train_df.loc[tr_rows].groupby(col)['demand'].agg(['mean','count'])
            s = stats['count'] / (stats['count'] + smooth)
            stats['sm'] = s * stats['mean'] + (1-s) * gm
            train_df.loc[va_rows, te_col] = train_df.loc[va_rows, col].map(stats['sm'].to_dict())
        
        # Day 49: full day 48 stats
        d49_idx = train_df[train_df['day'] == 49].index
        stats = d48.groupby(col)['demand'].agg(['mean','count'])
        s = stats['count'] / (stats['count'] + smooth)
        stats['sm'] = s * stats['mean'] + (1-s) * gm
        full_map = stats['sm'].to_dict()
        train_df.loc[d49_idx, te_col] = train_df.loc[d49_idx, col].map(full_map)
        test_df[te_col] = test_df[col].map(full_map).fillna(gm)
        train_df[te_col] = train_df[te_col].fillna(gm)
    
    # --- Geohash frequency (count-based, not target-based) ---
    geo_freq = train_df.groupby('geohash').size().to_dict()
    for df in [train_df, test_df]:
        df['geohash_freq'] = df['geohash'].map(geo_freq).fillna(0)
    
    # --- RoadType distribution per geohash (what fraction of time is this geo a Highway?) ---
    for rt in ['Highway', 'Residential', 'Street']:
        col = f'geo_pct_{rt.lower()}'
        geo_rt = d48.groupby('geohash').apply(lambda x: (x['RoadType'] == rt).mean()).to_dict()
        for df in [train_df, test_df]:
            df[col] = df['geohash'].map(geo_rt).fillna(0)
    
    # --- Get feature columns ---
    exclude = {'Index','geohash','timestamp','demand','day',
               'RoadType','LargeVehicles','Landmarks','Weather',
               'geo3','geo4','geo5','Temperature'}
    feature_cols = [c for c in train_df.columns if c not in exclude and train_df[c].dtype != 'object']
    
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0
    
    print(f"  Total features: {len(feature_cols)}")
    return train_df, test_df, feature_cols


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
    p = m.predict(X[feats]); 
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


def tune_lgb(X_tr, y_tr, X_va, y_va, feats, n_trials=100):
    def obj(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
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


def tune_cb(X_tr, y_tr, X_va, y_va, feats, n_trials=60):
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


def tune_xgb(X_tr, y_tr, X_va, y_va, feats, n_trials=80):
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
                best_score, best_w = s, dict(zip(names, [w1,w2,w3]))
    return best_w, best_score


def run_v4():
    print("=" * 70)
    print("V4 PIPELINE -- GENERALIZATION-FOCUSED (no geohash overfitting)")
    print("=" * 70)
    
    print("\n[1/6] Loading data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")
    
    print("\n[2/6] V4 Feature Engineering...")
    t0 = time.time()
    train_df, test_df, feature_cols = build_features_v4(train_raw.copy(), test_raw.copy())
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
    print(f"  LGB R2: {lgb_s:.6f}")
    _, cb_vp, cb_s = train_cb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  CB  R2: {cb_s:.6f}")
    _, xgb_vp, xgb_s = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols)
    print(f"  XGB R2: {xgb_s:.6f}")
    
    w0, ens0 = optimize_weights({'lgb': lgb_vp, 'cb': cb_vp, 'xgb': xgb_vp}, y_va)
    print(f"  Baseline ensemble R2: {ens0:.6f} (w={w0})")
    
    # Optuna tuning
    print("\n[4/6] Hyperparameter Tuning...")
    print("  Tuning LightGBM (100 trials)...")
    best_lgb = tune_lgb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=100)
    lgb_params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
                  'verbose': -1, 'seed': SEED, 'n_jobs': -1}
    lgb_params['learning_rate'] = best_lgb.pop('lr')
    lgb_params.update(best_lgb)
    lgb_m, lgb_vp, lgb_s = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lgb_params)
    print(f"  LGB tuned R2: {lgb_s:.6f}")
    
    print("  Tuning CatBoost (60 trials)...")
    best_cb = tune_cb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=60)
    cb_params = {'iterations': 10000, 'loss_function': 'RMSE', 'eval_metric': 'R2',
                 'random_seed': SEED, 'verbose': 0, 'early_stopping_rounds': 200,
                 'task_type': 'CPU', 'learning_rate': best_cb['lr'],
                 'depth': best_cb['depth'], 'l2_leaf_reg': best_cb['l2_leaf_reg'],
                 'min_data_in_leaf': best_cb['min_data_in_leaf'],
                 'random_strength': best_cb['random_strength'],
                 'bagging_temperature': best_cb['bagging_temperature']}
    cb_m, cb_vp, cb_s = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cb_params)
    print(f"  CB  tuned R2: {cb_s:.6f}")
    
    print("  Tuning XGBoost (80 trials)...")
    best_xgb = tune_xgb(X_tr, y_tr, X_va, y_va, feature_cols, n_trials=80)
    xgb_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                  'seed': SEED, 'nthread': -1, 'verbosity': 0}
    xgb_params.update(best_xgb)
    xgb_m, xgb_vp, xgb_s = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xgb_params)
    print(f"  XGB tuned R2: {xgb_s:.6f}")
    
    # Ensemble
    print("\n[5/6] Ensemble...")
    w1, ens1 = optimize_weights({'lgb': lgb_vp, 'cb': cb_vp, 'xgb': xgb_vp}, y_va)
    print(f"  Tuned ensemble R2: {ens1:.6f} (w={w1})")
    
    # Seed averaging
    print("\n[6/6] Seed Averaging + Submission...")
    SEEDS = [42, 123, 456, 789, 2024]
    test_preds = {'lgb': [], 'cb': [], 'xgb': []}
    val_preds_all = {'lgb': [], 'cb': [], 'xgb': []}
    
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)})...")
        lp = {**lgb_params, 'seed': seed}
        m, vp, _ = train_lgb(X_tr, y_tr, X_va, y_va, feature_cols, lp)
        test_preds['lgb'].append(pred_lgb(m, test_df, feature_cols))
        val_preds_all['lgb'].append(vp)
        
        cp = {**cb_params, 'random_seed': seed}
        m, vp, _ = train_cb(X_tr, y_tr, X_va, y_va, feature_cols, cp)
        test_preds['cb'].append(pred_cb(m, test_df, feature_cols))
        val_preds_all['cb'].append(vp)
        
        xp = {**xgb_params, 'seed': seed}
        m, vp, _ = train_xgb(X_tr, y_tr, X_va, y_va, feature_cols, xp)
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
    
    for name, pred in [('final_v4', final_pred), ('lgb_v4', lgb_test),
                       ('cb_v4', cb_test), ('xgb_v4', xgb_test)]:
        sub = pd.DataFrame({'Index': test_df['Index'].values, 'demand': pred})
        sub['demand'] = sub['demand'].clip(lower=0)
        assert sub.shape == (41778, 2)
        assert sub['demand'].isnull().sum() == 0
        sub.to_csv(SUBMISSIONS_DIR / f'{name}_submission.csv', index=False)
        print(f"  Saved: {name}_submission.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("V4 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Features: {len(feature_cols)}")
    print(f"  LGB tuned R2:      {lgb_s:.6f}")
    print(f"  CB  tuned R2:      {cb_s:.6f}")
    print(f"  XGB tuned R2:      {xgb_s:.6f}")
    print(f"  FINAL ensemble R2: {final_s:.6f}")
    print(f"  Weights: {final_w}")
    print("=" * 70)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_m.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)
    importance.to_csv(EXPERIMENTS_DIR / 'feature_importance_v4.csv', index=False)
    print("\n  Top 20 features (LGB gain):")
    print(importance.head(20).to_string(index=False))


if __name__ == '__main__':
    run_v4()
