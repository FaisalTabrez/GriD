"""
Traffic Demand Prediction — Full Training Pipeline

This script implements all phases:
  Phase 1: Baseline models (LightGBM, CatBoost, XGBoost)
  Phase 2: Advanced feature engineering
  Phase 3: Hyperparameter tuning with Optuna
  Phase 4: Ensemble (weighted average + seed averaging)
  Phase 5: Final submission

Usage:
    python train_pipeline.py
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
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import build_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
DATA_DIR = PROJECT_ROOT / 'e88186124ec611f1' / 'dataset'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    """Load train and test datasets."""
    train = pd.read_csv(DATA_DIR / 'train.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test, sample_sub


# ---------------------------------------------------------------------------
# Validation Strategy
# ---------------------------------------------------------------------------
def get_temporal_split(train_df):
    """
    Temporal split: train on day 48, validate on day 49.
    This matches the actual train/test split pattern.
    """
    train_mask = train_df['day'] == 48
    val_mask = train_df['day'] == 49
    
    train_idx = train_df[train_mask].index.values
    val_idx = train_df[val_mask].index.values
    
    print(f"Temporal split: train={len(train_idx)}, val={len(val_idx)}")
    return [(train_idx, val_idx)]


def get_kfold_splits(train_df, n_splits=5):
    """KFold cross-validation splits."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(kf.split(train_df))
    return splits


def get_group_kfold_splits(train_df, n_splits=5):
    """GroupKFold by geohash — prevents same location in both train/val."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df['geohash'].values
    splits = list(gkf.split(train_df, groups=groups))
    return splits


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
def train_lightgbm(X_train, y_train, X_val, y_val, feature_cols, params=None, 
                   use_log=False):
    """Train a LightGBM model."""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'max_depth': -1,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': SEED,
            'n_jobs': -1,
        }
    
    y_tr = np.log1p(y_train) if use_log else y_train.copy()
    y_va = np.log1p(y_val) if use_log else y_val.copy()
    
    dtrain = lgb.Dataset(X_train[feature_cols], label=y_tr)
    dval = lgb.Dataset(X_val[feature_cols], label=y_va, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(200),
        ],
    )
    
    pred = model.predict(X_val[feature_cols])
    if use_log:
        pred = np.expm1(pred)
    
    pred = np.clip(pred, 0, None)
    score = r2_score(y_val, pred)
    
    return model, pred, score


def predict_lightgbm(model, X, feature_cols, use_log=False):
    """Generate predictions from LightGBM model."""
    pred = model.predict(X[feature_cols])
    if use_log:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------
def train_catboost(X_train, y_train, X_val, y_val, feature_cols, params=None,
                   use_log=False):
    """Train a CatBoost model."""
    if params is None:
        params = {
            'iterations': 5000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'eval_metric': 'R2',
            'random_seed': SEED,
            'verbose': 200,
            'early_stopping_rounds': 100,
            'task_type': 'CPU',
        }
    
    y_tr = np.log1p(y_train) if use_log else y_train.copy()
    y_va = np.log1p(y_val) if use_log else y_val.copy()
    
    model = CatBoostRegressor(**params)
    model.fit(
        X_train[feature_cols], y_tr,
        eval_set=(X_val[feature_cols], y_va),
        use_best_model=True,
    )
    
    pred = model.predict(X_val[feature_cols])
    if use_log:
        pred = np.expm1(pred)
    
    pred = np.clip(pred, 0, None)
    score = r2_score(y_val, pred)
    
    return model, pred, score


def predict_catboost(model, X, feature_cols, use_log=False):
    """Generate predictions from CatBoost model."""
    pred = model.predict(X[feature_cols])
    if use_log:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train, X_val, y_val, feature_cols, params=None,
                  use_log=False):
    """Train an XGBoost model."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'lambda': 1.0,
            'alpha': 0.1,
            'seed': SEED,
            'nthread': -1,
            'verbosity': 0,
        }
    
    y_tr = np.log1p(y_train) if use_log else y_train.copy()
    y_va = np.log1p(y_val) if use_log else y_val.copy()
    
    dtrain = xgb.DMatrix(X_train[feature_cols], label=y_tr)
    dval = xgb.DMatrix(X_val[feature_cols], label=y_va)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=200,
    )
    
    pred = model.predict(dval)
    if use_log:
        pred = np.expm1(pred)
    
    pred = np.clip(pred, 0, None)
    score = r2_score(y_val, pred)
    
    return model, pred, score


def predict_xgboost(model, X, feature_cols, use_log=False):
    """Generate predictions from XGBoost model."""
    dtest = xgb.DMatrix(X[feature_cols])
    pred = model.predict(dtest)
    if use_log:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------
def tune_lightgbm(X_train, y_train, X_val, y_val, feature_cols, 
                  use_log=True, n_trials=100):
    """Tune LightGBM with Optuna."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 512),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
            'verbose': -1,
            'seed': SEED,
            'n_jobs': -1,
        }
        
        _, _, score = train_lightgbm(X_train, y_train, X_val, y_val, 
                                     feature_cols, params, use_log)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"  LightGBM best R²: {study.best_value:.6f}")
    return study.best_params


def tune_catboost(X_train, y_train, X_val, y_val, feature_cols,
                  use_log=True, n_trials=60):
    """Tune CatBoost with Optuna."""
    def objective(trial):
        params = {
            'iterations': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'loss_function': 'RMSE',
            'eval_metric': 'R2',
            'random_seed': SEED,
            'verbose': 0,
            'early_stopping_rounds': 100,
            'task_type': 'CPU',
        }
        
        _, _, score = train_catboost(X_train, y_train, X_val, y_val,
                                     feature_cols, params, use_log)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"  CatBoost best R²: {study.best_value:.6f}")
    return study.best_params


def tune_xgboost(X_train, y_train, X_val, y_val, feature_cols,
                 use_log=True, n_trials=80):
    """Tune XGBoost with Optuna."""
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'eta': trial.suggest_float('eta', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 5.0, log=True),
            'seed': SEED,
            'nthread': -1,
            'verbosity': 0,
        }
        
        _, _, score = train_xgboost(X_train, y_train, X_val, y_val,
                                    feature_cols, params, use_log)
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"  XGBoost best R²: {study.best_value:.6f}")
    return study.best_params


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def optimize_ensemble_weights(val_preds_dict, y_val):
    """Find optimal ensemble weights via grid search."""
    best_score = -np.inf
    best_weights = None
    model_names = list(val_preds_dict.keys())
    preds_list = [val_preds_dict[n] for n in model_names]
    
    # Grid search over weights
    steps = 21  # 0.0, 0.05, 0.10, ..., 1.0
    for w1 in np.linspace(0, 1, steps):
        for w2 in np.linspace(0, 1 - w1, steps):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            weights = [w1, w2, w3]
            pred = sum(w * p for w, p in zip(weights, preds_list))
            pred = np.clip(pred, 0, None)
            score = r2_score(y_val, pred)
            if score > best_score:
                best_score = score
                best_weights = dict(zip(model_names, weights))
    
    return best_weights, best_score


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def create_submission(test_df, predictions, filename='submission.csv'):
    """Create submission CSV."""
    sub = pd.DataFrame({
        'Index': test_df['Index'].values,
        'demand': predictions,
    })
    sub['demand'] = sub['demand'].clip(lower=0)
    
    # Validate
    assert sub.shape == (41778, 2), f"Wrong shape: {sub.shape}"
    assert sub['demand'].isnull().sum() == 0, "NaN in predictions!"
    
    filepath = SUBMISSIONS_DIR / filename
    sub.to_csv(filepath, index=False)
    print(f"Submission saved: {filepath} | Shape: {sub.shape}")
    return sub


# ---------------------------------------------------------------------------
# Experiment Logger
# ---------------------------------------------------------------------------
def log_experiment(name, cv_score, params=None, features=None, notes=''):
    """Log experiment to CSV."""
    log_file = EXPERIMENTS_DIR / 'experiment_log.csv'
    
    entry = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'name': name,
        'cv_r2': cv_score,
        'params': json.dumps(params) if params else '',
        'n_features': len(features) if features else 0,
        'notes': notes,
    }
    
    if log_file.exists():
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, pd.DataFrame([entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([entry])
    
    log_df.to_csv(log_file, index=False)
    print(f"  Logged: {name} -> R²={cv_score:.6f}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    """Execute the full training pipeline."""
    
    print("=" * 70)
    print("TRAFFIC DEMAND PREDICTION PIPELINE")
    print("=" * 70)
    
    # -----------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------
    print("\n[1/7] Loading data...")
    train_raw, test_raw, sample_sub = load_data()
    
    # -----------------------------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------------------------
    print("\n[2/7] Feature engineering...")
    t0 = time.time()
    
    train_df = train_raw.copy()
    test_df = test_raw.copy()
    
    train_df, test_df, feature_cols, artifacts = build_features(
        train_df, test_df,
        use_target_encoding=True,
        use_target_stats=True,
        use_clustering=True,
        n_clusters=25,
    )
    
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Time: {time.time() - t0:.1f}s")
    
    # Prepare target
    y_full = train_df['demand'].values
    test_indices = test_df['Index'].values
    
    # -----------------------------------------------------------------------
    # Temporal Validation Split
    # -----------------------------------------------------------------------
    print("\n[3/7] Temporal validation split...")
    splits = get_temporal_split(train_df)
    train_idx, val_idx = splits[0]
    
    X_train = train_df.iloc[train_idx]
    X_val = train_df.iloc[val_idx]
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Also do KFold on full data for more robust CV
    print("\n  Also setting up 5-fold KFold for robust evaluation...")
    kfold_splits = get_kfold_splits(train_df, n_splits=N_FOLDS)
    
    # -----------------------------------------------------------------------
    # Phase 1: Baseline Models
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[4/7] PHASE 1: Baseline Models (with log1p target)")
    print("=" * 70)
    
    USE_LOG = True  # Based on EDA, demand is right-skewed
    
    # --- LightGBM Baseline ---
    print("\n--- LightGBM Baseline ---")
    lgb_model, lgb_val_pred, lgb_score = train_lightgbm(
        X_train, y_train, X_val, y_val, feature_cols, use_log=USE_LOG
    )
    print(f"  LightGBM Val R²: {lgb_score:.6f}")
    log_experiment('lgb_baseline', lgb_score, notes='log1p target, temporal split')
    
    # --- CatBoost Baseline ---
    print("\n--- CatBoost Baseline ---")
    cb_model, cb_val_pred, cb_score = train_catboost(
        X_train, y_train, X_val, y_val, feature_cols, use_log=USE_LOG
    )
    print(f"  CatBoost Val R²: {cb_score:.6f}")
    log_experiment('cb_baseline', cb_score, notes='log1p target, temporal split')
    
    # --- XGBoost Baseline ---
    print("\n--- XGBoost Baseline ---")
    xgb_model, xgb_val_pred, xgb_score = train_xgboost(
        X_train, y_train, X_val, y_val, feature_cols, use_log=USE_LOG
    )
    print(f"  XGBoost Val R²: {xgb_score:.6f}")
    log_experiment('xgb_baseline', xgb_score, notes='log1p target, temporal split')
    
    # --- Quick ensemble check ---
    simple_avg = (lgb_val_pred + cb_val_pred + xgb_val_pred) / 3
    simple_avg_score = r2_score(y_val, np.clip(simple_avg, 0, None))
    print(f"\n  Simple Average Ensemble R²: {simple_avg_score:.6f}")
    
    # -----------------------------------------------------------------------
    # Phase 2: Hyperparameter Tuning with Optuna
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[5/7] PHASE 2: Hyperparameter Tuning (Optuna)")
    print("=" * 70)
    
    # Tune LightGBM
    print("\n--- Tuning LightGBM (100 trials) ---")
    best_lgb_params = tune_lightgbm(
        X_train, y_train, X_val, y_val, feature_cols, 
        use_log=USE_LOG, n_trials=100
    )
    print(f"  Best params: {best_lgb_params}")
    
    # Rebuild full params
    lgb_tuned_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
        **best_lgb_params,
    }
    
    lgb_tuned_model, lgb_tuned_pred, lgb_tuned_score = train_lightgbm(
        X_train, y_train, X_val, y_val, feature_cols,
        params=lgb_tuned_params, use_log=USE_LOG
    )
    print(f"  LightGBM Tuned R²: {lgb_tuned_score:.6f}")
    log_experiment('lgb_tuned', lgb_tuned_score, params=best_lgb_params,
                   features=feature_cols, notes='Optuna 100 trials')
    
    # Tune CatBoost
    print("\n--- Tuning CatBoost (60 trials) ---")
    best_cb_params = tune_catboost(
        X_train, y_train, X_val, y_val, feature_cols,
        use_log=USE_LOG, n_trials=60
    )
    print(f"  Best params: {best_cb_params}")
    
    cb_tuned_params = {
        'iterations': 5000,
        'loss_function': 'RMSE',
        'eval_metric': 'R2',
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 100,
        'task_type': 'CPU',
        **best_cb_params,
    }
    
    cb_tuned_model, cb_tuned_pred, cb_tuned_score = train_catboost(
        X_train, y_train, X_val, y_val, feature_cols,
        params=cb_tuned_params, use_log=USE_LOG
    )
    print(f"  CatBoost Tuned R²: {cb_tuned_score:.6f}")
    log_experiment('cb_tuned', cb_tuned_score, params=best_cb_params,
                   features=feature_cols, notes='Optuna 60 trials')
    
    # Tune XGBoost
    print("\n--- Tuning XGBoost (80 trials) ---")
    best_xgb_params = tune_xgboost(
        X_train, y_train, X_val, y_val, feature_cols,
        use_log=USE_LOG, n_trials=80
    )
    print(f"  Best params: {best_xgb_params}")
    
    xgb_tuned_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': SEED,
        'nthread': -1,
        'verbosity': 0,
        **best_xgb_params,
    }
    
    xgb_tuned_model, xgb_tuned_pred, xgb_tuned_score = train_xgboost(
        X_train, y_train, X_val, y_val, feature_cols,
        params=xgb_tuned_params, use_log=USE_LOG
    )
    print(f"  XGBoost Tuned R²: {xgb_tuned_score:.6f}")
    log_experiment('xgb_tuned', xgb_tuned_score, params=best_xgb_params,
                   features=feature_cols, notes='Optuna 80 trials')
    
    # -----------------------------------------------------------------------
    # Phase 3: Ensemble with Optimized Weights
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[6/7] PHASE 3: Ensemble Optimization")
    print("=" * 70)
    
    val_preds = {
        'lgb': lgb_tuned_pred,
        'cb': cb_tuned_pred,
        'xgb': xgb_tuned_pred,
    }
    
    best_weights, best_ens_score = optimize_ensemble_weights(val_preds, y_val)
    print(f"  Best weights: {best_weights}")
    print(f"  Ensemble R²: {best_ens_score:.6f}")
    log_experiment('ensemble_tuned', best_ens_score, params=best_weights,
                   features=feature_cols, notes='Weighted avg of tuned models')
    
    # -----------------------------------------------------------------------
    # Phase 4: Seed Averaging + Final Submission
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[7/7] PHASE 4: Final Submission (Seed Averaging)")
    print("=" * 70)
    
    SEEDS = [42, 123, 456, 789, 2024]
    
    # Train multiple models with different seeds and average predictions
    all_test_preds = {'lgb': [], 'cb': [], 'xgb': []}
    all_val_preds = {'lgb': [], 'cb': [], 'xgb': []}
    
    for i, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({i+1}/{len(SEEDS)})...")
        
        # LightGBM
        lgb_seed_params = {**lgb_tuned_params, 'seed': seed}
        lgb_m, lgb_vp, lgb_s = train_lightgbm(
            X_train, y_train, X_val, y_val, feature_cols,
            params=lgb_seed_params, use_log=USE_LOG
        )
        lgb_tp = predict_lightgbm(lgb_m, test_df, feature_cols, use_log=USE_LOG)
        all_test_preds['lgb'].append(lgb_tp)
        all_val_preds['lgb'].append(lgb_vp)
        print(f"    LGB R²: {lgb_s:.6f}")
        
        # CatBoost
        cb_seed_params = {**cb_tuned_params, 'random_seed': seed}
        cb_m, cb_vp, cb_s = train_catboost(
            X_train, y_train, X_val, y_val, feature_cols,
            params=cb_seed_params, use_log=USE_LOG
        )
        cb_tp = predict_catboost(cb_m, test_df, feature_cols, use_log=USE_LOG)
        all_test_preds['cb'].append(cb_tp)
        all_val_preds['cb'].append(cb_vp)
        print(f"    CB  R²: {cb_s:.6f}")
        
        # XGBoost
        xgb_seed_params = {**xgb_tuned_params, 'seed': seed}
        xgb_m, xgb_vp, xgb_s = train_xgboost(
            X_train, y_train, X_val, y_val, feature_cols,
            params=xgb_seed_params, use_log=USE_LOG
        )
        xgb_tp = predict_xgboost(xgb_m, test_df, feature_cols, use_log=USE_LOG)
        all_test_preds['xgb'].append(xgb_tp)
        all_val_preds['xgb'].append(xgb_vp)
        print(f"    XGB R²: {xgb_s:.6f}")
    
    # Average across seeds
    lgb_test_avg = np.mean(all_test_preds['lgb'], axis=0)
    cb_test_avg = np.mean(all_test_preds['cb'], axis=0)
    xgb_test_avg = np.mean(all_test_preds['xgb'], axis=0)
    
    lgb_val_avg = np.mean(all_val_preds['lgb'], axis=0)
    cb_val_avg = np.mean(all_val_preds['cb'], axis=0)
    xgb_val_avg = np.mean(all_val_preds['xgb'], axis=0)
    
    # Re-optimize weights on seed-averaged val preds
    val_preds_seed = {'lgb': lgb_val_avg, 'cb': cb_val_avg, 'xgb': xgb_val_avg}
    final_weights, final_val_score = optimize_ensemble_weights(val_preds_seed, y_val)
    print(f"\n  Seed-averaged ensemble weights: {final_weights}")
    print(f"  Seed-averaged ensemble Val R²: {final_val_score:.6f}")
    
    # Generate final test predictions
    w_lgb = final_weights['lgb']
    w_cb = final_weights['cb']
    w_xgb = final_weights['xgb']
    
    final_test_pred = w_lgb * lgb_test_avg + w_cb * cb_test_avg + w_xgb * xgb_test_avg
    final_test_pred = np.clip(final_test_pred, 0, None)
    
    # Save submission
    sub = create_submission(test_df, final_test_pred, 'final_submission.csv')
    
    log_experiment('final_ensemble', final_val_score, params=final_weights,
                   features=feature_cols, 
                   notes=f'Seed avg ({SEEDS}), weighted ensemble')
    
    # Also save individual model submissions
    create_submission(test_df, lgb_test_avg, 'lgb_submission.csv')
    create_submission(test_df, cb_test_avg, 'cb_submission.csv')
    create_submission(test_df, xgb_test_avg, 'xgb_submission.csv')
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  Features used: {len(feature_cols)}")
    print(f"  LightGBM baseline R²:   {lgb_score:.6f}")
    print(f"  CatBoost baseline R²:   {cb_score:.6f}")
    print(f"  XGBoost baseline R²:    {xgb_score:.6f}")
    print(f"  LightGBM tuned R²:      {lgb_tuned_score:.6f}")
    print(f"  CatBoost tuned R²:      {cb_tuned_score:.6f}")
    print(f"  XGBoost tuned R²:       {xgb_tuned_score:.6f}")
    print(f"  Ensemble tuned R²:      {best_ens_score:.6f}")
    print(f"  FINAL (seed avg) R²:    {final_val_score:.6f}")
    print(f"  Ensemble weights:       {final_weights}")
    print(f"  Submission: submissions/final_submission.csv")
    print("=" * 70)
    
    # Save feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_tuned_model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)
    importance.to_csv(OUTPUTS_DIR / 'feature_importance.csv', index=False)
    print("\n  Top 20 features (LightGBM gain):")
    print(importance.head(20).to_string(index=False))
    
    return {
        'final_val_score': final_val_score,
        'final_weights': final_weights,
        'lgb_params': lgb_tuned_params,
        'cb_params': cb_tuned_params,
        'xgb_params': xgb_tuned_params,
        'feature_cols': feature_cols,
    }


if __name__ == '__main__':
    results = run_pipeline()
