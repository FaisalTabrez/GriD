"""
V7 Pipeline - Leak-Free High-Performance Solution
Addresses the massive target leakage found in previous scripts by strictly aligning
the training data context with the test set context.

Key Architectural Changes:
1. Train set strictly filtered to Day 48 Afternoon (minutes > 120).
2. Morning profiles are matched to the SAME DAY (Day 48 for train, Day 49 for test).
3. Naive baseline (yesterday's exact time demand) is explicitly separated and blended
   at the very end, preventing tree overfitting to 24-hour exact lags.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')

# ─────────────── Configuration ───────────────
FAST_MODE = True  # Toggle to True for 1-minute single-seed temporal CV validation
SEED = 42
BLEND_ALPHA = 0.5  # Weight of ML Model (1-alpha will be the Naive Day 48 Baseline)
np.random.seed(SEED)
T0 = time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT
TR_PATH = DATA_DIR / "train.csv"
TE_PATH = DATA_DIR / "test.csv"
OUT_PATH = DATA_DIR / "submission_v7.csv"

# ─────────────── Geohash Decoding ───────────────
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

def parse_ts(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)

# ─────────────── Feature Engineering Pipeline ───────────────
def build_features_v7():
    log("Loading data...")
    tr_full = pd.read_csv(TR_PATH)
    te = pd.read_csv(TE_PATH)
    
    # Strictly filter the training data to Day 48 AFTERNOON
    tr_full["minutes"] = tr_full["timestamp"].map(parse_ts)
    tr = tr_full[(tr_full["day"] == 48) & (tr_full["minutes"] > 120)].copy()
    te["minutes"] = te["timestamp"].map(parse_ts)
    
    log(f"train (Day 48 Afternoon) {tr.shape}  test (Day 49 Afternoon) {te.shape}")

    # Decode lat/lon
    all_gh = pd.concat([tr_full["geohash"], te["geohash"]]).unique()
    gh_map = {g: _decode_geohash(g) for g in all_gh}
    
    # ── Time Features ──
    for df in [tr, te, tr_full]:
        df["hour"] = df["minutes"] // 60
        df["qhr"]  = df["minutes"] // 15
        
        a  = 2 * np.pi * df["minutes"] / 1440
        a2 = 4 * np.pi * df["minutes"] / 1440
        df["sin_t"], df["cos_t"]   = np.sin(a),  np.cos(a)
        df["sin_2t"], df["cos_2t"] = np.sin(a2), np.cos(a2)
        
        df["is_rush"]  = ((df["hour"].between(7, 9)) | (df["hour"].between(16, 19))).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
        df["is_midday"] = ((df["hour"] >= 10) & (df["hour"] <= 15)).astype(int)
        
        df["lat"] = df["geohash"].map(lambda g: gh_map[g][0])
        df["lon"] = df["geohash"].map(lambda g: gh_map[g][1])
        df["gh3"] = df["geohash"].str[:3]
        df["gh4"] = df["geohash"].str[:4]
        df["gh5"] = df["geohash"].str[:5]
        
    # ── Missing value handling ──
    for c in ["RoadType", "Weather"]:
        tr[c] = tr[c].fillna("Unknown")
        te[c] = te[c].fillna("Unknown")
        tr_full[c] = tr_full[c].fillna("Unknown")
        
    temp_med = tr_full["Temperature"].median()
    tr["Temperature"] = tr["Temperature"].fillna(temp_med)
    te["Temperature"] = te["Temperature"].fillna(temp_med)
    tr_full["Temperature"] = tr_full["Temperature"].fillna(temp_med)
    
    bin_map = {"Allowed": 1, "Not Allowed": 0, "Yes": 1, "No": 0}
    for c in ["LargeVehicles", "Landmarks"]:
        tr[c] = tr[c].map(bin_map).fillna(0).astype(int)
        te[c] = te[c].map(bin_map).fillna(0).astype(int)
        tr_full[c] = tr_full[c].map(bin_map).fillna(0).astype(int)
        
    # Encode categoricals
    label_encoders = {}
    for col in ['RoadType', 'Weather', 'geohash', 'gh3', 'gh4', 'gh5']:
        le = LabelEncoder()
        combined = pd.concat([tr_full[col], te[col]]).fillna('missing').astype(str)
        le.fit(combined)
        label_encoders[col] = le
        tr[f'{col}_enc'] = le.transform(tr[col].fillna('missing').astype(str))
        te[f'{col}_enc'] = le.transform(te[col].fillna('missing').astype(str))
        tr_full[f'{col}_enc'] = le.transform(tr_full[col].fillna('missing').astype(str))
        
    # ── Spatial neighbors ──
    log("computing neighbors...")
    g_uni = pd.DataFrame({"geohash": all_gh})
    g_uni["lat"] = g_uni["geohash"].map(lambda g: gh_map[g][0])
    g_uni["lon"] = g_uni["geohash"].map(lambda g: gh_map[g][1])
    tree = KDTree(g_uni[["lat", "lon"]].values)
    K_NBR = 8
    dists, idx = tree.query(g_uni[["lat", "lon"]].values, k=K_NBR + 1)
    neighbors_map, nbr_dist_map = {}, {}
    for i, gh in enumerate(g_uni["geohash"].values):
        nbrs = [g_uni["geohash"].iloc[j] for j in idx[i] if g_uni["geohash"].iloc[j] != gh][:K_NBR]
        neighbors_map[gh] = nbrs
        nbr_dist_map[gh]  = float(np.mean(dists[i][1:K_NBR+1]))

    tr["nbr_dist"] = tr["geohash"].map(nbr_dist_map)
    te["nbr_dist"] = te["geohash"].map(nbr_dist_map)

    # ── Same-Day Morning Profile Generation ──
    # We build the morning pivot for Day 48 (to merge with Train) and Day 49 (to merge with Test)
    log("building same-day morning profiles...")
    D49_SLOTS = [0, 15, 30, 45, 60, 75, 90, 105, 120]
    LAST_MORN = 120
    
    def build_morning_pivot(day_data):
        morn_data = day_data[day_data["minutes"].isin(D49_SLOTS)]
        pivot = morn_data.pivot(index="geohash", columns="minutes", values="demand")
        pivot.columns = [f"morn_m{c}" for c in pivot.columns]
        # Treat missing morning slots as NaNs, not 0.0 to prevent severe distortion!
        pivot = pivot.reindex(all_gh)
        
        morn_cols = [f"morn_m{s}" for s in D49_SLOTS]
        mm = pivot[morn_cols].values
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pivot["morn_mean"] = np.nanmean(mm, axis=1)
            pivot["morn_std"]  = np.nanstd(mm, axis=1)
            pivot["morn_max"]  = np.nanmax(mm, axis=1)
            pivot["morn_min"]  = np.nanmin(mm, axis=1)
            
        pivot["morn_last"] = pivot[f"morn_m{LAST_MORN}"]
        
        # Valid Slope Calculation
        xs = np.array(D49_SLOTS, dtype=float)
        xs_c = xs - xs.mean()
        denom = float((xs_c ** 2).sum())
        
        def slope_row(y):
            mask = ~np.isnan(y)
            if mask.sum() < 2: return np.nan
            yc = y[mask] - y[mask].mean()
            return float((xs_c[mask] * yc).sum() / max(denom, 1e-9))
            
        pivot["morn_slope"] = [slope_row(mm[i]) for i in range(len(mm))]
        
        # Neighbor morning mean
        nbr_morn_mean_dict = {}
        for gh in all_gh:
            nbrs = neighbors_map.get(gh, [])
            morn_vals = [pivot.loc[n, morn_cols].values for n in nbrs if n in pivot.index]
            # Flatten and remove NaNs
            morn_vals = [val for arr in morn_vals for val in arr if not np.isnan(val)]
            nbr_morn_mean_dict[gh] = float(np.mean(morn_vals)) if morn_vals else np.nan
            
        pivot["nbr_morn_mean"] = pivot.index.map(nbr_morn_mean_dict)
        return pivot

    d48_pivot = build_morning_pivot(tr_full[tr_full["day"] == 48])
    d49_pivot = build_morning_pivot(tr_full[tr_full["day"] == 49])
    
    tr = tr.merge(d48_pivot, on="geohash", how="left")
    te = te.merge(d49_pivot, on="geohash", how="left")
    
    # ── Day-48 Exact Same-Time Lookup (for blending baseline) ──
    # We will save this explicitly for blending, NOT as a feature (to prevent leak)
    d48_lookup = tr_full[tr_full["day"] == 48].set_index(["geohash", "minutes"])["demand"].to_dict()
    g_te, m_te = te["geohash"].values, te["minutes"].values
    te["day48_baseline"] = [d48_lookup.get((g_te[i], m_te[i]), 0.0) for i in range(len(te))]

    # ── V5 Structural Geohash Features (Computed purely on Day 48 to prevent any leak) ──
    log("building structural geohash features...")
    d48 = tr_full[tr_full["day"] == 48]
    geo_avg_lanes = d48.groupby('geohash')['NumberofLanes'].mean().to_dict()
    geo_max_lanes = d48.groupby('geohash')['NumberofLanes'].max().to_dict()
    geo_modal_road = d48.groupby('geohash')['RoadType_enc'].agg(lambda x: x.mode().iloc[0]).to_dict()
    geo_road_diversity = d48.groupby('geohash')['RoadType_enc'].nunique().to_dict()
    geo_large_pct = d48.groupby('geohash')['LargeVehicles'].mean().to_dict()
    geo_landmark_pct = d48.groupby('geohash')['Landmarks'].mean().to_dict()
    geo_temp_mean = d48.groupby('geohash')['Temperature'].mean().to_dict()
    geo_temp_std = d48.groupby('geohash')['Temperature'].std().fillna(0).to_dict()
    geo_weather_div = d48.groupby('geohash')['Weather_enc'].nunique().to_dict()
    geo_freq = d48.groupby('geohash').size().to_dict()
    
    for rt in ['Highway', 'Residential', 'Street']:
        key = f'geo_pct_{rt.lower()}'
        geo_rt = d48.groupby('geohash').apply(lambda x: (x['RoadType'] == rt).mean()).to_dict()
        for df in [tr, te]:
            df[key] = df['geohash'].map(geo_rt).fillna(0)

    for df in [tr, te]:
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
        
        # Interactions
        df['lanes_diff'] = df['NumberofLanes'] - df['geo_avg_lanes']
        df['is_max_lanes'] = (df['NumberofLanes'] == df['geo_max_lanes']).astype(int)
        df['road_hour'] = df['RoadType_enc'] * 100 + df['hour']
        df['lanes_road'] = df['NumberofLanes'] * 10 + df['RoadType_enc']
        df['large_road'] = df['LargeVehicles'] * 10 + df['RoadType_enc']
        df['weather_road'] = df['Weather_enc'] * 10 + df['RoadType_enc']
        df['landmark_road'] = df['Landmarks'] * 10 + df['RoadType_enc']
        df['road_large'] = df['RoadType_enc'] * 10 + df['LargeVehicles']
        df['road_landmark'] = df['RoadType_enc'] * 10 + df['Landmarks']
        df['lanes_large'] = df['NumberofLanes'] * 10 + df['LargeVehicles']
        df['is_highway'] = (df['NumberofLanes'] >= 4).astype(int)
        df['temp_x_road'] = df['Temperature'] * (df['RoadType_enc'] + 1)
        df['temp_x_highway'] = df['Temperature'] * df['is_highway']

    # ── Geospatial clustering ──
    log("Computing KMeans clusters...")
    coords_arr = [gh_map[g] for g in all_gh]
    coords_df = pd.DataFrame(coords_arr, columns=['lat','lon'], index=all_gh)
    for n_clusters in [5, 10, 20, 50]:
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        coords_df[f'cluster_{n_clusters}'] = km.fit_predict(coords_df[['lat','lon']].values)
        cluster_map = coords_df[f'cluster_{n_clusters}'].to_dict()
        tr[f'cluster_{n_clusters}'] = tr['geohash'].map(cluster_map).fillna(0).astype(int)
        te[f'cluster_{n_clusters}'] = te['geohash'].map(cluster_map).fillna(0).astype(int)
        
        centers = km.cluster_centers_
        for df in [tr, te]:
            c_ids = df[f'cluster_{n_clusters}'].values
            df[f'dist_center_{n_clusters}'] = np.sqrt(
                (df['lat'].values - centers[c_ids, 0])**2 + 
                (df['lon'].values - centers[c_ids, 1])**2
            )

    # ── OOF Target Encoding (KFold for Day 48 Train, map for Test) ──
    log("Computing OOF target encodings...")
    gm = tr['demand'].mean()
    smooth = 20
    te_cols = ['RoadType_enc', 'lanes_road', 'road_hour', 'NumberofLanes',
               'large_road', 'LargeVehicles', 'road_large', 'weather_road',
               'landmark_road', 'road_landmark', 'lanes_large',
               'cluster_10', 'cluster_20']
               
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx = tr.index
    
    for col in te_cols:
        te_col = f'{col}_te'
        tr[te_col] = np.nan
        
        # OOF KFold target encoding on day 48 train
        for tr_i, va_i in kf.split(tr_idx):
            tr_rows, va_rows = tr_idx[tr_i], tr_idx[va_i]
            stats = tr.loc[tr_rows].groupby(col)['demand'].agg(['mean','count'])
            s = stats['count'] / (stats['count'] + smooth)
            stats['sm'] = s * stats['mean'] + (1-s) * gm
            tr.loc[va_rows, te_col] = tr.loc[va_rows, col].map(stats['sm'].to_dict())
            
        # Direct group mapping for day 49 test
        stats = tr.groupby(col)['demand'].agg(['mean','count'])
        s = stats['count'] / (stats['count'] + smooth)
        stats['sm'] = s * stats['mean'] + (1-s) * gm
        full_map = stats['sm'].to_dict()
        
        te[te_col] = te[col].map(full_map).fillna(gm)
        tr[te_col] = tr[te_col].fillna(gm)

    # Clean cat columns
    cat_cols = ["geohash", "gh3", "gh4", "gh5", "RoadType", "Weather"]
    for c in cat_cols:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")

    exclude = {'Index','geohash','timestamp','demand','day', 'day48_baseline',
               'RoadType','LargeVehicles','Landmarks','Weather',
               'gh3','gh4','gh5','Temperature'}
    feature_cols = [c for c in tr.columns if c not in exclude and tr[c].dtype != 'object']
    feature_cols = list(dict.fromkeys(feature_cols))
    
    for c in feature_cols:
        if c not in te.columns:
            te[c] = 0.0

    log(f"Features created successfully: {len(feature_cols)} features total.")
    return tr, te, feature_cols, cat_cols

# ─────────────── Model Pipelines ───────────────
def run_v7():
    log("=" * 70)
    log(f"V7 PIPELINE RUN (FAST_MODE = {FAST_MODE})")
    log("=" * 70)
    
    tr_f, te_f, feats, cat_cols = build_features_v7()
    tr_f = tr_f.reset_index(drop=True)
    
    y = np.log1p(tr_f["demand"].values)
    TR_MAX = float(tr_f["demand"].max())

    N_TR, NT = len(tr_f), len(te_f)
    oof_l = np.zeros(N_TR); oof_x = np.zeros(N_TR); oof_c = np.zeros(N_TR)
    pr_l  = np.zeros(NT);   pr_x  = np.zeros(NT);   pr_c  = np.zeros(NT)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    lgb_params = dict(
        objective="regression", metric="rmse", verbosity=-1,
        learning_rate=0.035, num_leaves=160, min_data_in_leaf=25,
        feature_fraction=0.8, bagging_fraction=0.85, bagging_freq=5,
        lambda_l1=0.1, lambda_l2=0.2,
    )
    xgb_params = dict(
        objective="reg:squarederror", eval_metric="rmse",
        learning_rate=0.035, max_depth=8, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.8,
        tree_method="hist", enable_categorical=True,
    )
    cb_params = dict(
        loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
        learning_rate=0.04, depth=8, iterations=3000,
        l2_leaf_reg=3.0, verbose=False,
    )

    if FAST_MODE:
        SEEDS = [42]
        lgb_params["num_leaves"] = 127
        xgb_params["max_depth"] = 6
        cb_params["depth"] = 6
        cb_params["iterations"] = 1500
        log("Running in FAST MODE (1 seed, 5 folds, max_depth=6)...")
    else:
        SEEDS = [42, 1337, 2024]
        log("Running in FULL MODE (3 seeds, 5 folds, max_depth=8)...")

    for fold, (tri, vai) in enumerate(kf.split(tr_f), 1):
        log(f"--- Fold {fold}/5 ---")
        tr_fold_x = tr_f.iloc[tri]
        tr_fold_y = y[tri]
        va_x = tr_f.iloc[vai]
        va_y = y[vai]
        
        # LGBM
        for s in SEEDS:
            p = {**lgb_params, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
            dtr = lgb.Dataset(tr_fold_x[feats], tr_fold_y)
            dva = lgb.Dataset(va_x[feats], va_y, reference=dtr)
            m = lgb.train(p, dtr, 4000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            oof_l[vai] += m.predict(va_x[feats]) / len(SEEDS)
            pr_l       += m.predict(te_f[feats]) / (len(SEEDS) * 5)
            
        # XGBoost
        for s in SEEDS:
            m = xgb.XGBRegressor(**xgb_params, seed=s, n_estimators=4000, early_stopping_rounds=150)
            m.fit(tr_fold_x[feats], tr_fold_y, eval_set=[(va_x[feats], va_y)], verbose=False)
            oof_x[vai] += m.predict(va_x[feats]) / len(SEEDS)
            pr_x       += m.predict(te_f[feats]) / (len(SEEDS) * 5)
            
        # CatBoost
        tr_cb = tr_fold_x[feats].copy(); va_cb = va_x[feats].copy(); te_cb = te_f[feats].copy()
        cat_cols_clean = [c for c in cat_cols if c in feats]
        for c in cat_cols_clean:
            tr_cb[c] = tr_cb[c].astype(str).fillna("missing")
            va_cb[c] = va_cb[c].astype(str).fillna("missing")
            te_cb[c] = te_cb[c].astype(str).fillna("missing")
            
        cb_params_fold = {**cb_params, "cat_features": cat_cols_clean}
        m_c = CatBoostRegressor(**cb_params_fold)
        m_c.fit(tr_cb, tr_fold_y, eval_set=(va_cb, va_y), early_stopping_rounds=150, verbose=False)
        oof_c[vai] = m_c.predict(va_cb)
        pr_c      += m_c.predict(te_cb) / 5

    # expm1 back to original target space
    oof_l_e = np.expm1(oof_l); oof_x_e = np.expm1(oof_x); oof_c_e = np.expm1(oof_c)
    pr_l_e  = np.expm1(pr_l);  pr_x_e  = np.expm1(pr_x);  pr_c_e  = np.expm1(pr_c)
    for a in (oof_l_e, oof_x_e, oof_c_e, pr_l_e, pr_x_e, pr_c_e):
        np.clip(a, 0, TR_MAX, out=a)
        
    y_true = np.expm1(y)
    
    SX  = np.column_stack([oof_l_e, oof_x_e, oof_c_e])
    STX = np.column_stack([pr_l_e,  pr_x_e,  pr_c_e])
    
    # Non-negative ridge stacker
    ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(SX, y_true)
    oof_s = ridge.predict(SX)
    
    log(f"OOF R²  LGB={r2_score(y_true, oof_l_e):.4f}  "
        f"XGB={r2_score(y_true, oof_x_e):.4f}  "
        f"CB={r2_score(y_true, oof_c_e):.4f}  "
        f"STACK={r2_score(y_true, oof_s):.4f}")
    log(f"Stack weights={ridge.coef_}")
    
    # ML Prediction
    ml_pred = np.clip(ridge.predict(STX), 0, TR_MAX)
    baseline_pred = te_f["day48_baseline"].values
    
    # Blend with Baseline
    final = BLEND_ALPHA * ml_pred + (1 - BLEND_ALPHA) * baseline_pred
    
    pd.DataFrame({"Index": te_f["Index"].values, "demand": final}).to_csv(OUT_PATH, index=False)
    log(f"wrote {OUT_PATH}")

if __name__ == '__main__':
    run_v7()
