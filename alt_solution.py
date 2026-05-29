# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "lightgbm",
#   "xgboost",
#   "catboost",
#   "scipy",
# ]
# ///

import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.spatial import KDTree

SEED = 42
np.random.seed(SEED)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

# ───────────────────────── Load ─────────────────────────
TR_PATH = "train.csv"
TE_PATH = "test.csv"
OUT_PATH = "submission.csv"

tr = pd.read_csv(TR_PATH)
te = pd.read_csv(TE_PATH)
log(f"train {tr.shape}  test {te.shape}")

# ─────────────────── Geohash → lat / lon ─────────────────
_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DEC = {c: i for i, c in enumerate(_B32)}
def gh_decode(gh):
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for c in gh:
        v = _DEC[c]
        for mask in (16, 8, 4, 2, 1):
            bit = 1 if (v & mask) else 0
            if even:
                mid = (lon_lo + lon_hi) / 2
                if bit: lon_lo = mid
                else:   lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if bit: lat_lo = mid
                else:   lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

all_gh = pd.concat([tr["geohash"], te["geohash"]]).unique()
gh_map = {g: gh_decode(g) for g in all_gh}

def add_geo(df):
    df["lat"] = df["geohash"].map(lambda g: gh_map[g][0])
    df["lon"] = df["geohash"].map(lambda g: gh_map[g][1])
    df["gh3"] = df["geohash"].str[:3]
    df["gh4"] = df["geohash"].str[:4]
    df["gh5"] = df["geohash"].str[:5]
    return df

# ─────────────────────── Time features ──────────────────
def parse_ts(s):
    h, m = s.split(":"); return int(h) * 60 + int(m)
def add_time(df):
    df["minutes"] = df["timestamp"].map(parse_ts)
    df["hour"]    = df["minutes"] // 60
    df["qhr"]     = df["minutes"] // 15
    a  = 2 * np.pi * df["minutes"] / 1440
    a2 = 4 * np.pi * df["minutes"] / 1440
    df["sin_t"], df["cos_t"]   = np.sin(a),  np.cos(a)
    df["sin_2t"], df["cos_2t"] = np.sin(a2), np.cos(a2)
    df["is_rush"]  = ((df["hour"].between(7, 9)) | (df["hour"].between(16, 19))).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    return df

tr = add_geo(add_time(tr))
te = add_geo(add_time(te))

# ──────────────── Missing-value handling ────────────────
for c in ["RoadType", "Weather"]:
    tr[c] = tr[c].fillna("Unknown")
    te[c] = te[c].fillna("Unknown")
temp_med = tr["Temperature"].median()
tr["Temperature"] = tr["Temperature"].fillna(temp_med)
te["Temperature"] = te["Temperature"].fillna(temp_med)

bin_map = {"Allowed": 1, "Not Allowed": 0, "Yes": 1, "No": 0}
for c in ["LargeVehicles", "Landmarks"]:
    tr[c] = tr[c].map(bin_map).astype(int)
    te[c] = te[c].map(bin_map).astype(int)

# ────────────────── Spatial Neighborhoods ────────────────
log("calculating neighborhoods...")
g48 = tr[tr["day"] == 48][["geohash", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
tree = KDTree(g48[["lat", "lon"]].values)
all_g_df = pd.DataFrame({"geohash": all_gh})
all_g_df["lat"] = all_g_df["geohash"].map(lambda g: gh_map[g][0])
all_g_df["lon"] = all_g_df["geohash"].map(lambda g: gh_map[g][1])
dists, indices = tree.query(all_g_df[["lat", "lon"]].values, k=6)

neighbors_map = {}
nbr_dist_map = {}
for i, row in all_g_df.iterrows():
    nbrs = g48.iloc[indices[i]]["geohash"].values
    nbr_dists = dists[i]
    valid_idx = [j for j in range(len(nbrs)) if nbrs[j] != row["geohash"]][:5]
    neighbors_map[row["geohash"]] = [nbrs[j] for j in valid_idx]
    nbr_dist_map[row["geohash"]] = np.mean([nbr_dists[j] for j in valid_idx])

# ──────────────── Previous Day Lookup ────────────────────
log("calculating historical features...")
d48_lookup = tr[tr["day"] == 48].set_index(["geohash", "minutes"])["demand"].to_dict()

def get_prev_day_feat(row):
    if row["day"] == 48: return np.nan
    return d48_lookup.get((row["geohash"], row["minutes"]), 0.0)

def get_prev_day_lag1(row):
    if row["day"] == 48: return np.nan
    return d48_lookup.get((row["geohash"], row["minutes"] - 15), 0.0)

def get_prev_day_lead1(row):
    if row["day"] == 48: return np.nan
    return d48_lookup.get((row["geohash"], row["minutes"] + 15), 0.0)

def get_prev_day_lag2(row):
    if row["day"] == 48: return np.nan
    return d48_lookup.get((row["geohash"], row["minutes"] - 30), 0.0)

def get_prev_day_lead2(row):
    if row["day"] == 48: return np.nan
    return d48_lookup.get((row["geohash"], row["minutes"] + 30), 0.0)

def get_nbr_demand(row):
    if row["day"] == 48: return np.nan
    nbrs = neighbors_map[row["geohash"]]
    vals = [d48_lookup.get((n, row["minutes"]), 0.0) for n in nbrs]
    return np.mean(vals) if vals else 0.0

tr["demand_prev_day"] = tr.apply(get_prev_day_feat, axis=1)
tr["demand_prev_day_lag1"] = tr.apply(get_prev_day_lag1, axis=1)
tr["demand_prev_day_lead1"] = tr.apply(get_prev_day_lead1, axis=1)
tr["demand_prev_day_lag2"] = tr.apply(get_prev_day_lag2, axis=1)
tr["demand_prev_day_lead2"] = tr.apply(get_prev_day_lead2, axis=1)
tr["nbr_demand_prev_day"] = tr.apply(get_nbr_demand, axis=1)

te["demand_prev_day"] = te.apply(get_prev_day_feat, axis=1)
te["demand_prev_day_lag1"] = te.apply(get_prev_day_lag1, axis=1)
te["demand_prev_day_lead1"] = te.apply(get_prev_day_lead1, axis=1)
te["demand_prev_day_lag2"] = te.apply(get_prev_day_lag2, axis=1)
te["demand_prev_day_lead2"] = te.apply(get_prev_day_lead2, axis=1)
te["nbr_demand_prev_day"] = te.apply(get_nbr_demand, axis=1)

tr["nbr_dist"] = tr["geohash"].map(nbr_dist_map)
te["nbr_dist"] = te["geohash"].map(nbr_dist_map)

# ─────────────────── Morning Profiles ────────────────────
log("calculating morning profiles...")
# Combine train and test to search for morning profile demands on Day 49
all_df = pd.concat([tr, te]).reset_index(drop=True)
m_0am = all_df[(all_df["minutes"] == 0) & (all_df["demand"].notna())][["geohash", "day", "demand"]].rename(columns={"demand": "demand_0am"})
m_1am = all_df[(all_df["minutes"] == 60) & (all_df["demand"].notna())][["geohash", "day", "demand"]].rename(columns={"demand": "demand_1am"})
m_2am = all_df[(all_df["minutes"] == 120) & (all_df["demand"].notna())][["geohash", "day", "demand"]].rename(columns={"demand": "demand_2am"})

tr = tr.merge(m_0am, on=["geohash", "day"], how="left")
tr = tr.merge(m_1am, on=["geohash", "day"], how="left")
tr = tr.merge(m_2am, on=["geohash", "day"], how="left")

te = te.merge(m_0am, on=["geohash", "day"], how="left")
te = te.merge(m_1am, on=["geohash", "day"], how="left")
te = te.merge(m_2am, on=["geohash", "day"], how="left")

# Day 48 general stats
d48_stats = tr[tr["day"] == 48].groupby("geohash")["demand"].agg(["mean", "std", "max", "min"]).reset_index()
d48_stats.columns = ["geohash", "d48_mean", "d48_std", "d48_max", "d48_min"]
tr = tr.merge(d48_stats, on="geohash", how="left")
te = te.merge(d48_stats, on="geohash", how="left")

demand_feats = ["demand_prev_day", "demand_prev_day_lag1", "demand_prev_day_lead1",
                "demand_prev_day_lag2", "demand_prev_day_lead2", "nbr_demand_prev_day",
                "demand_0am", "demand_1am", "demand_2am", "d48_mean", "d48_std", "d48_max", "d48_min"]

# For test and day 49, fill NaN morning profiles with 0
tr[demand_feats] = tr[demand_feats].fillna(0.0)
te[demand_feats] = te[demand_feats].fillna(0.0)

# ─────────────────── Categoricals ────────────────────────
cat_cols = ["geohash", "gh3", "gh4", "gh5", "RoadType", "Weather"]
for c in cat_cols:
    tr[c] = tr[c].astype("category")
    te[c] = te[c].astype("category")

feats = ["lat", "lon", "minutes", "hour", "qhr", "sin_t", "cos_t", "sin_2t", "cos_2t",
         "NumberofLanes", "LargeVehicles", "Landmarks", "Temperature", "is_rush", "is_night", "nbr_dist"] + \
        cat_cols + demand_feats

log(f"#features = {len(feats)}")

# ────────────────── Stacking Pipeline ─────────────────────
d48_df = tr[tr["day"] == 48].reset_index(drop=True)
d49_df = tr[tr["day"] == 49].reset_index(drop=True)

N_49 = len(d49_df)
oof_l = np.zeros(N_49)
oof_x = np.zeros(N_49)
oof_c = np.zeros(N_49)

NT = len(te)
pr_l = np.zeros(NT)
pr_x = np.zeros(NT)
pr_c = np.zeros(NT)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": SEED,
    "learning_rate": 0.04,
    "num_leaves": 128,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": SEED,
    "learning_rate": 0.04,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "learning_rate": 0.05,
    "depth": 7,
    "iterations": 2000,
    "verbose": False,
    "cat_features": cat_cols,
}

for fold, (tri, vai) in enumerate(kf.split(d49_df), 1):
    log(f"--- Fold {fold}/5 ---")
    train_fold = pd.concat([d48_df, d49_df.iloc[tri]]).reset_index(drop=True)
    val_fold = d49_df.iloc[vai]
    
    # LGBM
    dtr = lgb.Dataset(train_fold[feats], train_fold["demand"])
    dva = lgb.Dataset(val_fold[feats], val_fold["demand"], reference=dtr)
    m_l = lgb.train(lgb_params, dtr, 3000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_l[vai] = m_l.predict(val_fold[feats])
    pr_l += m_l.predict(te[feats]) / 5
    
    # XGBoost
    m_x = xgb.XGBRegressor(**xgb_params, n_estimators=3000)
    m_x.fit(train_fold[feats], train_fold["demand"],
            eval_set=[(val_fold[feats], val_fold["demand"])],
            verbose=False)
    # Get predictions
    oof_x[vai] = m_x.predict(val_fold[feats])
    pr_x += m_x.predict(te[feats]) / 5
    
    # CatBoost
    # Convert category back to string for CatBoost
    train_cb = train_fold[feats].copy()
    val_cb = val_fold[feats].copy()
    te_cb = te[feats].copy()
    for c in cat_cols:
        train_cb[c] = train_cb[c].astype(str)
        val_cb[c] = val_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)
        
    m_c = CatBoostRegressor(**cb_params)
    m_c.fit(train_cb, train_fold["demand"],
            eval_set=(val_cb, val_fold["demand"]),
            early_stopping_rounds=100, verbose=False)
    oof_c[vai] = m_c.predict(val_cb)
    pr_c += m_c.predict(te_cb) / 5

# ─────────────── Stack with non-negative Ridge ──────────
y_true = d49_df["demand"].values
SX  = np.column_stack([oof_l, oof_x, oof_c])
STX = np.column_stack([pr_l,  pr_x,  pr_c])

ridge = Ridge(alpha=1.0, positive=True).fit(SX, y_true)
oof_s = ridge.predict(SX)

log(f"OOF R²  LGB={r2_score(y_true, oof_l):.4f}  "
    f"XGB={r2_score(y_true, oof_x):.4f}  "
    f"CB={r2_score(y_true, oof_c):.4f}  "
    f"STACK={r2_score(y_true, oof_s):.4f}")
log(f"stack weights={ridge.coef_}  intercept={ridge.intercept_:.4f}")

# Final submission
final = np.clip(ridge.predict(STX), 0, tr["demand"].max() * 1.05)
pd.DataFrame({"Index": te["Index"].values, "demand": final}).to_csv(OUT_PATH, index=False)
log(f"wrote {OUT_PATH}")
print(f"\nESTIMATED HACKEREARTH SCORE ≈ {100 * r2_score(y_true, oof_s):.2f}")