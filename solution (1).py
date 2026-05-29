# /// script
# dependencies = [
#   "numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "scipy",
# ]
# ///
"""
Refined solution — targets ≥94 score.

Key upgrades vs. the 91.3 version (in order of impact):
  1.  Same-day (day-49) morning context features per geohash:
        - demand at every available morning slot (0..120 min) as flat features
        - rolling mean / std / max / last-value / linear trend across morning
        - neighbor-mean of demand at the most recent morning slot (t=120)
      => the strongest signal we were leaving on the table.
  2.  Day-48 windowed lookups (±15/30/45/60 min) and qhr-of-day mean/std
      => smooths the noisy single-minute prev-day lookup.
  3.  log1p target transform (demand is heavily right-skewed) — fit on
      log space, expm1 back. Improves R² for skewed targets.
  4.  Multi-seed bagging (3 seeds × 5 folds) for LGB & XGB; CatBoost 1 seed.
  5.  Non-negative ridge stacker with intercept disabled (cleaner blend).
  6.  Tighter clipping at observed train max (no 1.05x inflation).
"""

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
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

TR_PATH = "train.csv"
TE_PATH = "test.csv"
OUT_PATH = "submission.csv"

tr = pd.read_csv(TR_PATH)
te = pd.read_csv(TE_PATH)
log(f"train {tr.shape}  test {te.shape}")

# ─────────────── Geohash → lat/lon ───────────────
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

# ─────────────── Missing handling ───────────────
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

# ─────────────── Spatial neighbors ───────────────
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

# ─────────────── Day-48 & Day-49 lookups ───────────────
log("building lookups...")
d48 = tr[tr["day"] == 48]
d49 = tr[tr["day"] == 49]
d48_lookup = d48.set_index(["geohash", "minutes"])["demand"].to_dict()
d49_lookup = d49.set_index(["geohash", "minutes"])["demand"].to_dict()

# day-48 per-geohash time-of-day arrays for windowed means
d48_by_g = {g: dict(zip(grp["minutes"].values, grp["demand"].values))
            for g, grp in d48.groupby("geohash")}

D49_SLOTS = sorted(d49["minutes"].unique().tolist())  # [0,15,30,...,120]
LAST_MORN = max(D49_SLOTS)                            # 120

def d48_val(g, m): return d48_lookup.get((g, m), np.nan)
def d49_val(g, m): return d49_lookup.get((g, m), np.nan)

# ─────────────── Day-48 windowed features ───────────────
def d48_window_mean(g, m, half_w):
    """Mean of day-48 demand within ±half_w minutes of m at same geohash."""
    gd = d48_by_g.get(g)
    if not gd: return np.nan
    vals = [v for mm, v in gd.items() if abs(mm - m) <= half_w]
    return float(np.mean(vals)) if vals else np.nan

# Per-geohash day-48 same-qhr-of-day stats
d48["qhr"] = d48["minutes"] // 15
qhr_stats = d48.groupby(["geohash", "qhr"])["demand"].agg(["mean", "std"]).reset_index()
qhr_stats.columns = ["geohash", "qhr", "g_qhr_d48_mean", "g_qhr_d48_std"]

# ─────────────── Build features ───────────────
log("building per-row features...")

def build_features(df):
    df = df.copy()
    g = df["geohash"].values
    m = df["minutes"].values

    # Day-48 lookups at offsets
    for off in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
        df[f"d48_off{off}"] = [d48_lookup.get((g[i], m[i] + off), np.nan) for i in range(len(df))]

    # Day-48 windowed means
    df["d48_win30"] = [d48_window_mean(g[i], m[i], 30) for i in range(len(df))]
    df["d48_win60"] = [d48_window_mean(g[i], m[i], 60) for i in range(len(df))]
    df["d48_win120"] = [d48_window_mean(g[i], m[i], 120) for i in range(len(df))]

    # Day-48 neighbor mean at same minute
    def nbr_d48(i):
        nbrs = neighbors_map.get(g[i], [])
        vals = [d48_lookup.get((n, m[i]), np.nan) for n in nbrs]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan
    df["nbr_d48_same_min"] = [nbr_d48(i) for i in range(len(df))]

    # ── SAME-DAY (day-49) morning features ──
    # Per-geohash demand at each morning slot 0..120 (flat features)
    for s in D49_SLOTS:
        df[f"d49_m{s}"] = [d49_lookup.get((g[i], s), np.nan) for i in range(len(df))]

    # Stats over morning window per geohash
    morn_cols = [f"d49_m{s}" for s in D49_SLOTS]
    mm = df[morn_cols].values
    df["d49_morn_mean"] = np.nanmean(mm, axis=1)
    df["d49_morn_std"]  = np.nanstd(mm, axis=1)
    df["d49_morn_max"]  = np.nanmax(mm, axis=1)
    df["d49_morn_min"]  = np.nanmin(mm, axis=1)
    df["d49_morn_last"] = df[f"d49_m{LAST_MORN}"]

    # Linear trend across morning (slope)
    xs = np.array(D49_SLOTS, dtype=float)
    xs_c = xs - xs.mean()
    denom = float((xs_c ** 2).sum())
    def slope_row(r):
        y = r.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2: return 0.0
        yc = y[mask] - y[mask].mean()
        return float((xs_c[mask] * yc).sum() / max(denom, 1e-9))
    df["d49_morn_slope"] = [slope_row(mm[i]) for i in range(len(df))]

    # Same-day neighbor mean at LAST morning slot
    def nbr_d49_last(i):
        nbrs = neighbors_map.get(g[i], [])
        vals = [d49_lookup.get((n, LAST_MORN), np.nan) for n in nbrs]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan
    df["nbr_d49_last"] = [nbr_d49_last(i) for i in range(len(df))]

    # Same-day morning mean across neighbors
    def nbr_d49_morn(i):
        nbrs = neighbors_map.get(g[i], [])
        vals = []
        for n in nbrs:
            for s in D49_SLOTS:
                v = d49_lookup.get((n, s), np.nan)
                if not np.isnan(v): vals.append(v)
        return float(np.mean(vals)) if vals else np.nan
    df["nbr_d49_morn_mean"] = [nbr_d49_morn(i) for i in range(len(df))]

    # Ratio: how does this geohash compare to its neighbors on day 48?
    df["self_vs_nbr_d48"] = df["d48_off0"] - df["nbr_d48_same_min"]

    df["nbr_dist"] = df["geohash"].map(nbr_dist_map)
    return df

tr_f = build_features(tr)
te_f = build_features(te)

# Merge per-(geohash, qhr) day-48 stats
tr_f = tr_f.merge(qhr_stats, on=["geohash", "qhr"], how="left")
te_f = te_f.merge(qhr_stats, on=["geohash", "qhr"], how="left")

# Per-geohash day-48 overall stats
d48_g = d48.groupby("geohash")["demand"].agg(["mean", "std", "max", "min", "median"]).reset_index()
d48_g.columns = ["geohash", "d48_g_mean", "d48_g_std", "d48_g_max", "d48_g_min", "d48_g_median"]
tr_f = tr_f.merge(d48_g, on="geohash", how="left")
te_f = te_f.merge(d48_g, on="geohash", how="left")

# Fill numeric NaNs with 0 (tree models handle NaN fine, but keep consistent)
num_fill_cols = [c for c in tr_f.columns if c.startswith(("d48_", "d49_", "nbr_", "self_", "g_qhr_"))]
tr_f[num_fill_cols] = tr_f[num_fill_cols].fillna(0.0)
te_f[num_fill_cols] = te_f[num_fill_cols].fillna(0.0)

# ─────────────── Categoricals ───────────────
cat_cols = ["geohash", "gh3", "gh4", "gh5", "RoadType", "Weather"]
for c in cat_cols:
    tr_f[c] = tr_f[c].astype("category")
    te_f[c] = te_f[c].astype("category")

feats = (["lat", "lon", "minutes", "hour", "qhr",
          "sin_t", "cos_t", "sin_2t", "cos_2t",
          "NumberofLanes", "LargeVehicles", "Landmarks", "Temperature",
          "is_rush", "is_night", "nbr_dist"]
         + cat_cols
         + num_fill_cols)
feats = list(dict.fromkeys(feats))
log(f"#features = {len(feats)}")

# ─────────────── Stacking ───────────────
d48_df = tr_f[tr_f["day"] == 48].reset_index(drop=True)
d49_df = tr_f[tr_f["day"] == 49].reset_index(drop=True)
N_49, NT = len(d49_df), len(te_f)

# log1p target — refit on log space, invert with expm1
y48 = np.log1p(d48_df["demand"].values)
y49 = np.log1p(d49_df["demand"].values)
y_true = d49_df["demand"].values
TR_MAX = float(tr["demand"].max())

oof_l = np.zeros(N_49); oof_x = np.zeros(N_49); oof_c = np.zeros(N_49)
pr_l  = np.zeros(NT);   pr_x  = np.zeros(NT);   pr_c  = np.zeros(NT)

SEEDS = [42, 1337, 2024]
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

lgb_params_base = dict(
    objective="regression", metric="rmse", verbosity=-1,
    learning_rate=0.035, num_leaves=160, min_data_in_leaf=25,
    feature_fraction=0.8, bagging_fraction=0.85, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=0.2,
)
xgb_params_base = dict(
    objective="reg:squarederror", eval_metric="rmse",
    learning_rate=0.035, max_depth=8, min_child_weight=4,
    subsample=0.85, colsample_bytree=0.8,
    tree_method="hist", enable_categorical=True,
)
cb_params = dict(
    loss_function="RMSE", eval_metric="RMSE", random_seed=SEED,
    learning_rate=0.04, depth=8, iterations=3000,
    l2_leaf_reg=3.0, verbose=False, cat_features=cat_cols,
)

for fold, (tri, vai) in enumerate(kf.split(d49_df), 1):
    log(f"--- Fold {fold}/5 ---")
    tr_fold_x = pd.concat([d48_df, d49_df.iloc[tri]]).reset_index(drop=True)
    tr_fold_y = np.concatenate([y48, y49[tri]])
    va_x = d49_df.iloc[vai]
    va_y = y49[vai]

    # LGB multi-seed bag
    for s in SEEDS:
        p = {**lgb_params_base, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        dtr = lgb.Dataset(tr_fold_x[feats], tr_fold_y)
        dva = lgb.Dataset(va_x[feats], va_y, reference=dtr)
        m = lgb.train(p, dtr, 4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_l[vai] += m.predict(va_x[feats]) / len(SEEDS)
        pr_l       += m.predict(te_f[feats]) / (len(SEEDS) * 5)

    # XGB multi-seed bag
    for s in SEEDS:
        m = xgb.XGBRegressor(**xgb_params_base, seed=s, n_estimators=4000,
                             early_stopping_rounds=150)
        m.fit(tr_fold_x[feats], tr_fold_y,
              eval_set=[(va_x[feats], va_y)], verbose=False)
        oof_x[vai] += m.predict(va_x[feats]) / len(SEEDS)
        pr_x       += m.predict(te_f[feats]) / (len(SEEDS) * 5)

    # CatBoost (string cats)
    tr_cb = tr_fold_x[feats].copy(); va_cb = va_x[feats].copy(); te_cb = te_f[feats].copy()
    for c in cat_cols:
        tr_cb[c] = tr_cb[c].astype(str)
        va_cb[c] = va_cb[c].astype(str)
        te_cb[c] = te_cb[c].astype(str)
    m_c = CatBoostRegressor(**cb_params)
    m_c.fit(tr_cb, tr_fold_y, eval_set=(va_cb, va_y),
            early_stopping_rounds=150, verbose=False)
    oof_c[vai] = m_c.predict(va_cb)
    pr_c      += m_c.predict(te_cb) / 5

# back to original scale
oof_l_e = np.expm1(oof_l); oof_x_e = np.expm1(oof_x); oof_c_e = np.expm1(oof_c)
pr_l_e  = np.expm1(pr_l);  pr_x_e  = np.expm1(pr_x);  pr_c_e  = np.expm1(pr_c)
for a in (oof_l_e, oof_x_e, oof_c_e, pr_l_e, pr_x_e, pr_c_e):
    np.clip(a, 0, TR_MAX, out=a)

SX  = np.column_stack([oof_l_e, oof_x_e, oof_c_e])
STX = np.column_stack([pr_l_e,  pr_x_e,  pr_c_e])

# Non-negative ridge w/o intercept — cleaner blend, can't bias
ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(SX, y_true)
oof_s = ridge.predict(SX)

log(f"OOF R²  LGB={r2_score(y_true, oof_l_e):.4f}  "
    f"XGB={r2_score(y_true, oof_x_e):.4f}  "
    f"CB={r2_score(y_true, oof_c_e):.4f}  "
    f"STACK={r2_score(y_true, oof_s):.4f}")
log(f"weights={ridge.coef_}")

final = np.clip(ridge.predict(STX), 0, TR_MAX)
pd.DataFrame({"Index": te_f["Index"].values, "demand": final}).to_csv(OUT_PATH, index=False)
log(f"wrote {OUT_PATH}")
print(f"\nESTIMATED HACKEREARTH SCORE ≈ {100 * r2_score(y_true, oof_s):.2f}")
