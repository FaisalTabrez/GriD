"""
V6 Pipeline - High-Performance Integrated Solution
Combines highly predictive same-day morning profile lag features (from solution (1).py)
with robust structural geohash, clustering, target encoding, and interactions (from V5).
Features optimized with full numpy/pandas vectorization.
Includes FAST_MODE toggle for rapid validation.
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
FAST_MODE = False  # Toggle to True for 1-minute single-seed temporal CV validation
SEED = 42
np.random.seed(SEED)
T0 = time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'

SUBMISSIONS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

TR_PATH = DATA_DIR / "train.csv"
TE_PATH = DATA_DIR / "test.csv"
OUT_PATH = DATA_DIR / "submission_v6.csv"

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
def build_features_v6():
    log("Loading data...")
    tr = pd.read_csv(TR_PATH)
    te = pd.read_csv(TE_PATH)
    log(f"train {tr.shape}  test {te.shape}")

    # Decode lat/lon
    all_gh = pd.concat([tr["geohash"], te["geohash"]]).unique()
    gh_map = {g: _decode_geohash(g) for g in all_gh}
    
    # ── Time Features ──
    for df in [tr, te]:
        df["minutes"] = df["timestamp"].map(parse_ts)
        df["hour"]    = df["minutes"] // 60
        df["qhr"]     = df["minutes"] // 15
        
        # Cyclical
        a  = 2 * np.pi * df["minutes"] / 1440
        a2 = 4 * np.pi * df["minutes"] / 1440
        df["sin_t"], df["cos_t"]   = np.sin(a),  np.cos(a)
        df["sin_2t"], df["cos_2t"] = np.sin(a2), np.cos(a2)
        
        # rush/night indicators
        df["is_rush"]  = ((df["hour"].between(7, 9)) | (df["hour"].between(16, 19))).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
        df["is_midday"] = ((df["hour"] >= 10) & (df["hour"] <= 15)).astype(int)
        
        # Coordinates
        df["lat"] = df["geohash"].map(lambda g: gh_map[g][0])
        df["lon"] = df["geohash"].map(lambda g: gh_map[g][1])
        df["gh3"] = df["geohash"].str[:3]
        df["gh4"] = df["geohash"].str[:4]
        df["gh5"] = df["geohash"].str[:5]
        
    # ── Missing value handling ──
    for c in ["RoadType", "Weather"]:
        tr[c] = tr[c].fillna("Unknown")
        te[c] = te[c].fillna("Unknown")
    temp_med = tr["Temperature"].median()
    tr["Temperature"] = tr["Temperature"].fillna(temp_med)
    te["Temperature"] = te["Temperature"].fillna(temp_med)
    
    bin_map = {"Allowed": 1, "Not Allowed": 0, "Yes": 1, "No": 0}
    for c in ["LargeVehicles", "Landmarks"]:
        tr[c] = tr[c].map(bin_map).fillna(0).astype(int)
        te[c] = te[c].map(bin_map).fillna(0).astype(int)
        
    # Encode categorical columns for target encoding / embedding
    label_encoders = {}
    for col in ['RoadType', 'Weather', 'geohash', 'gh3', 'gh4', 'gh5']:
        le = LabelEncoder()
        combined = pd.concat([tr[col], te[col]]).fillna('missing').astype(str)
        le.fit(combined)
        label_encoders[col] = le
        tr[f'{col}_enc'] = le.transform(tr[col].fillna('missing').astype(str))
        te[f'{col}_enc'] = le.transform(te[col].fillna('missing').astype(str))
        
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

    # ── Day-48 & Day-49 lookups ──
    log("building lookups...")
    d48 = tr[tr["day"] == 48]
    d49 = tr[tr["day"] == 49]
    d48_lookup = d48.set_index(["geohash", "minutes"])["demand"].to_dict()
    
    # ── Same-day (day-49) morning profile features (pivot-vectorized!) ──
    log("building vectorized day-49 morning features...")
    D49_SLOTS = [0, 15, 30, 45, 60, 75, 90, 105, 120]
    LAST_MORN = 120
    
    d49_pivot = d49[d49["minutes"].isin(D49_SLOTS)].pivot(
        index="geohash", columns="minutes", values="demand"
    )
    d49_pivot.columns = [f"d49_m{c}" for c in d49_pivot.columns]
    # Fill empty morning slots with 0.0
    d49_pivot = d49_pivot.reindex(all_gh).fillna(0.0)
    
    morn_cols = [f"d49_m{s}" for s in D49_SLOTS]
    mm = d49_pivot[morn_cols].values
    
    d49_pivot["d49_morn_mean"] = np.nanmean(mm, axis=1)
    d49_pivot["d49_morn_std"]  = np.nanstd(mm, axis=1)
    d49_pivot["d49_morn_max"]  = np.nanmax(mm, axis=1)
    d49_pivot["d49_morn_min"]  = np.nanmin(mm, axis=1)
    d49_pivot["d49_morn_last"] = d49_pivot[f"d49_m{LAST_MORN}"]
    
    # Vectorized morning slope
    xs = np.array(D49_SLOTS, dtype=float)
    xs_c = xs - xs.mean()
    denom = float((xs_c ** 2).sum())
    mm_c = mm - mm.mean(axis=1, keepdims=True)
    d49_pivot["d49_morn_slope"] = (mm_c * xs_c).sum(axis=1) / max(denom, 1e-9)
    
    # Same-day neighbor morning features
    nbr_d49_last_dict = {}
    nbr_d49_morn_mean_dict = {}
    for gh in all_gh:
        nbrs = neighbors_map.get(gh, [])
        last_vals = [d49_pivot.loc[n, f"d49_m{LAST_MORN}"] for n in nbrs if n in d49_pivot.index]
        nbr_d49_last_dict[gh] = float(np.mean(last_vals)) if last_vals else 0.0
        
        morn_vals = [d49_pivot.loc[n, morn_cols].values for n in nbrs if n in d49_pivot.index]
        nbr_d49_morn_mean_dict[gh] = float(np.mean(morn_vals)) if morn_vals else 0.0
        
    d49_pivot["nbr_d49_last"] = d49_pivot.index.map(nbr_d49_last_dict)
    d49_pivot["nbr_d49_morn_mean"] = d49_pivot.index.map(nbr_d49_morn_mean_dict)
    
    # Merge onto main dataframes
    tr = tr.merge(d49_pivot, on="geohash", how="left")
    te = te.merge(d49_pivot, on="geohash", how="left")

    # ── Day-48 Offset Lookups ──
    log("building day-48 offset and windowed features...")
    g_tr, m_tr = tr["geohash"].values, tr["minutes"].values
    g_te, m_te = te["geohash"].values, te["minutes"].values
    
    for off in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
        tr[f"d48_off{off}"] = [d48_lookup.get((g_tr[i], m_tr[i] + off), 0.0) for i in range(len(tr))]
        te[f"d48_off{off}"] = [d48_lookup.get((g_te[i], m_te[i] + off), 0.0) for i in range(len(te))]
        
    # Day-48 windowed means
    d48_by_g = {g: dict(zip(grp["minutes"].values, grp["demand"].values))
                for g, grp in d48.groupby("geohash")}
                
    def d48_window_mean(g, m, half_w):
        gd = d48_by_g.get(g)
        if not gd: return 0.0
        vals = [v for mm, v in gd.items() if abs(mm - m) <= half_w]
        return float(np.mean(vals)) if vals else 0.0
        
    for w in (30, 60, 120):
        tr[f"d48_win{w}"] = [d48_window_mean(g_tr[i], m_tr[i], w) for i in range(len(tr))]
        te[f"d48_win{w}"] = [d48_window_mean(g_te[i], m_te[i], w) for i in range(len(te))]
        
    # Day-48 neighbor mean at same minute
    def nbr_d48_same_min(g_arr, m_arr):
        res = []
        for i in range(len(g_arr)):
            nbrs = neighbors_map.get(g_arr[i], [])
            vals = [d48_lookup.get((n, m_arr[i]), np.nan) for n in nbrs]
            vals = [v for v in vals if not np.isnan(v)]
            res.append(float(np.mean(vals)) if vals else 0.0)
        return res
        
    tr["nbr_d48_same_min"] = nbr_d48_same_min(g_tr, m_tr)
    te["nbr_d48_same_min"] = nbr_d48_same_min(g_te, m_te)
    
    # Self vs neighbor ratio
    tr["self_vs_nbr_d48"] = tr["d48_off0"] - tr["nbr_d48_same_min"]
    te["self_vs_nbr_d48"] = te["d48_off0"] - te["nbr_d48_same_min"]
    
    # Day-48 general stats
    d48_g = d48.groupby("geohash")["demand"].agg(["mean", "std", "max", "min", "median"]).reset_index()
    d48_g.columns = ["geohash", "d48_g_mean", "d48_g_std", "d48_g_max", "d48_g_min", "d48_g_median"]
    tr = tr.merge(d48_g, on="geohash", how="left")
    te = te.merge(d48_g, on="geohash", how="left")
    
    # Day 48 qhr stats
    d48["qhr"] = d48["minutes"] // 15
    qhr_stats = d48.groupby(["geohash", "qhr"])["demand"].agg(["mean", "std"]).reset_index()
    qhr_stats.columns = ["geohash", "qhr", "g_qhr_d48_mean", "g_qhr_d48_std"]
    tr = tr.merge(qhr_stats, on=["geohash", "qhr"], how="left")
    te = te.merge(qhr_stats, on=["geohash", "qhr"], how="left")

    # ── V5 Structural Geohash Features ──
    log("building structural geohash features...")
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
    all_geos = pd.concat([tr[['geohash','lat','lon']],
                          te[['geohash','lat','lon']]]).drop_duplicates('geohash')
    coords_arr = all_geos[['lat','lon']].values
    for n_clusters in [5, 10, 20, 50]:
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        all_geos[f'cluster_{n_clusters}'] = km.fit_predict(coords_arr)
        cluster_map = dict(zip(all_geos['geohash'], all_geos[f'cluster_{n_clusters}']))
        tr[f'cluster_{n_clusters}'] = tr['geohash'].map(cluster_map).fillna(0).astype(int)
        te[f'cluster_{n_clusters}'] = te['geohash'].map(cluster_map).fillna(0).astype(int)
        
        centers = km.cluster_centers_
        for df in [tr, te]:
            c_ids = df[f'cluster_{n_clusters}'].values
            df[f'dist_center_{n_clusters}'] = np.sqrt(
                (df['lat'].values - centers[c_ids, 0])**2 + 
                (df['lon'].values - centers[c_ids, 1])**2
            )

    # ── OOF Target Encoding (KFold for Day 48, direct mapping for Day 49) ──
    log("Computing OOF target encodings...")
    d48 = tr[tr["day"] == 48]
    gm = d48['demand'].mean()
    smooth = 20
    te_cols = ['RoadType_enc', 'lanes_road', 'road_hour', 'NumberofLanes',
               'large_road', 'LargeVehicles', 'road_large', 'weather_road',
               'landmark_road', 'road_landmark', 'lanes_large',
               'cluster_10', 'cluster_20']
               
    for col in te_cols:
        te_col = f'{col}_te'
        tr[te_col] = np.nan
        
        # OOF KFold target encoding on day 48
        d48_idx = tr[tr['day'] == 48].index
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr_i, va_i in kf.split(d48_idx):
            tr_rows, va_rows = d48_idx[tr_i], d48_idx[va_i]
            stats = tr.loc[tr_rows].groupby(col)['demand'].agg(['mean','count'])
            s = stats['count'] / (stats['count'] + smooth)
            stats['sm'] = s * stats['mean'] + (1-s) * gm
            tr.loc[va_rows, te_col] = tr.loc[va_rows, col].map(stats['sm'].to_dict())
            
        # Direct day 48 group mapping for day 49 train and test
        stats = d48.groupby(col)['demand'].agg(['mean','count'])
        s = stats['count'] / (stats['count'] + smooth)
        stats['sm'] = s * stats['mean'] + (1-s) * gm
        full_map = stats['sm'].to_dict()
        
        d49_idx = tr[tr['day'] == 49].index
        tr.loc[d49_idx, te_col] = tr.loc[d49_idx, col].map(full_map)
        te[te_col] = te[col].map(full_map).fillna(gm)
        tr[te_col] = tr[te_col].fillna(gm)

    # Fill NaNs
    num_fill_cols = [c for c in tr.columns if c.startswith(("d48_", "d49_", "nbr_", "self_", "g_qhr_", "geo_", "dist_center_", "RoadType_enc_te"))]
    tr[num_fill_cols] = tr[num_fill_cols].fillna(0.0)
    te[num_fill_cols] = te[num_fill_cols].fillna(0.0)

    # Clean cat columns
    cat_cols = ["geohash", "gh3", "gh4", "gh5", "RoadType", "Weather"]
    for c in cat_cols:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")

    exclude = {'Index','geohash','timestamp','demand','day',
               'RoadType','LargeVehicles','Landmarks','Weather',
               'gh3','gh4','gh5','Temperature'}
    feature_cols = [c for c in tr.columns if c not in exclude and tr[c].dtype != 'object']
    
    # Keep consistent features
    feature_cols = list(dict.fromkeys(feature_cols))
    
    for c in feature_cols:
        if c not in te.columns:
            te[c] = 0.0

    log(f"Features created successfully: {len(feature_cols)} features total.")
    return tr, te, feature_cols, cat_cols

# ─────────────── Model Pipelines ───────────────
def run_v6():
    log("=" * 70)
    log(f"V6 PIPELINE RUN (FAST_MODE = {FAST_MODE})")
    log("=" * 70)
    
    tr_f, te_f, feats, cat_cols = build_features_v6()
    
    d48_df = tr_f[tr_f["day"] == 48].reset_index(drop=True)
    d49_df = tr_f[tr_f["day"] == 49].reset_index(drop=True)
    N_49, NT = len(d49_df), len(te_f)
    
    # log1p transform (targets are continuous in [0, 1])
    y48 = np.log1p(d48_df["demand"].values)
    y49 = np.log1p(d49_df["demand"].values)
    y_true = d49_df["demand"].values
    TR_MAX = float(tr_f["demand"].max())

    oof_l = np.zeros(N_49); oof_x = np.zeros(N_49); oof_c = np.zeros(N_49)
    pr_l  = np.zeros(NT);   pr_x  = np.zeros(NT);   pr_c  = np.zeros(NT)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Fast iteration hyper-parameters vs full bagging
    lgb_params = dict(
        objective="regression", metric="rmse", verbosity=-1,
        learning_rate=0.04, num_leaves=127, min_data_in_leaf=30,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.1, lambda_l2=0.2,
    )
    xgb_params = dict(
        objective="reg:squarederror", eval_metric="rmse",
        learning_rate=0.04, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", enable_categorical=True,
    )
    cb_params = dict(
        loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
        learning_rate=0.05, depth=6, iterations=2000,
        l2_leaf_reg=5.0, verbose=False, cat_features=cat_cols,
    )

    if FAST_MODE:
        SEEDS = [42]
        log("Running in FAST MODE (1 seed, 5 folds, max_depth=6)...")
    else:
        SEEDS = [42, 1337, 2024]
        log("Running in FULL MODE (3 seeds, 5 folds, max_depth=6)...")

    for fold, (tri, vai) in enumerate(kf.split(d49_df), 1):
        log(f"--- Fold {fold}/5 ---")
        tr_fold_x = pd.concat([d48_df, d49_df.iloc[tri]]).reset_index(drop=True)
        tr_fold_y = np.concatenate([y48, y49[tri]])
        va_x = d49_df.iloc[vai]
        va_y = y49[vai]
        
        # LGBM
        for s in SEEDS:
            p = {**lgb_params, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
            dtr = lgb.Dataset(tr_fold_x[feats], tr_fold_y)
            dva = lgb.Dataset(va_x[feats], va_y, reference=dtr)
            m = lgb.train(p, dtr, 3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            oof_l[vai] += m.predict(va_x[feats]) / len(SEEDS)
            pr_l       += m.predict(te_f[feats]) / (len(SEEDS) * 5)
            
        # XGBoost
        for s in SEEDS:
            m = xgb.XGBRegressor(**xgb_params, seed=s, n_estimators=3000, early_stopping_rounds=150)
            m.fit(tr_fold_x[feats], tr_fold_y, eval_set=[(va_x[feats], va_y)], verbose=False)
            oof_x[vai] += m.predict(va_x[feats]) / len(SEEDS)
            pr_x       += m.predict(te_f[feats]) / (len(SEEDS) * 5)
            
        # CatBoost
        tr_cb = tr_fold_x[feats].copy(); va_cb = va_x[feats].copy(); te_cb = te_f[feats].copy()
        
        # Only process categorical columns that actually made it into feats
        cat_cols_clean = [c for c in cat_cols if c in feats]
        for c in cat_cols_clean:
            tr_cb[c] = tr_cb[c].astype(str)
            va_cb[c] = va_cb[c].astype(str)
            te_cb[c] = te_cb[c].astype(str)
            
        # Temporarily update the cb_params to use the filtered cat_cols_clean for this loop iteration
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
    
    # Create final prediction
    final = np.clip(ridge.predict(STX), 0, TR_MAX)
    pd.DataFrame({"Index": te_f["Index"].values, "demand": final}).to_csv(OUT_PATH, index=False)
    log(f"wrote {OUT_PATH}")
    print(f"\nESTIMATED HACKEREARTH SCORE ~= {100 * r2_score(y_true, oof_s):.2f}")

if __name__ == '__main__':
    run_v6()
