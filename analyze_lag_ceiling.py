"""Quick analysis: How well does day 48 demand predict day 49 demand?
This tells us the theoretical ceiling for lag-based approaches."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

DATA_DIR = Path(r'c:\Volume D\GriD\e88186124ec611f1\dataset')

train = pd.read_csv(DATA_DIR / 'train.csv')
parts = train['timestamp'].str.split(':', expand=True).astype(int)
train['hour'] = parts[0]
train['minute'] = parts[1]
train['ts_minutes'] = train['hour'] * 60 + train['minute']

d48 = train[train['day'] == 48]
d49 = train[train['day'] == 49]

print(f"Day 48: {len(d48)} rows, Day 49: {len(d49)} rows")
print(f"Day 49 timestamps: {sorted(d49['timestamp'].unique())}")
print(f"Day 49 hours: {sorted(d49['hour'].unique())}")

# Build day 48 lookup
d48_lookup = d48.groupby(['geohash', 'ts_minutes'])['demand'].mean().to_dict()
d48_hour_lookup = d48.groupby(['geohash', 'hour'])['demand'].mean().to_dict()
d48_geo_lookup = d48.groupby('geohash')['demand'].mean().to_dict()
global_mean = d48['demand'].mean()

# For each day 49 row, find the matching day 48 demand
d49 = d49.copy()
d49['d48_exact'] = d49.apply(lambda r: d48_lookup.get((r['geohash'], r['ts_minutes']), np.nan), axis=1)
d49['d48_hour'] = d49.apply(lambda r: d48_hour_lookup.get((r['geohash'], r['hour']), np.nan), axis=1)
d49['d48_geo'] = d49['geohash'].map(d48_geo_lookup)
d49['d48_best'] = d49['d48_exact'].fillna(d49['d48_hour'].fillna(d49['d48_geo'].fillna(global_mean)))

# Correlation analysis
exact_mask = d49['d48_exact'].notna()
print(f"\n--- Day 48 -> Day 49 Correlation Analysis ---")
print(f"Day 49 rows with exact match from day 48: {exact_mask.sum()}/{len(d49)} ({exact_mask.mean()*100:.1f}%)")

if exact_mask.sum() > 0:
    corr = np.corrcoef(d49.loc[exact_mask, 'demand'], d49.loc[exact_mask, 'd48_exact'])[0,1]
    r2 = r2_score(d49.loc[exact_mask, 'demand'], d49.loc[exact_mask, 'd48_exact'])
    print(f"Exact match correlation: {corr:.6f}")
    print(f"Exact match R2 (lag = prediction): {r2:.6f}")

r2_hour = r2_score(d49['demand'], d49['d48_hour'].fillna(global_mean))
r2_geo = r2_score(d49['demand'], d49['d48_geo'].fillna(global_mean))
r2_best = r2_score(d49['demand'], d49['d48_best'])

print(f"Geo-hour mean R2: {r2_hour:.6f}")
print(f"Geo mean R2: {r2_geo:.6f}")
print(f"Hierarchical best R2: {r2_best:.6f}")

# By hour analysis
print(f"\n--- R2 by Hour (Day 49 validation hours only) ---")
for h in sorted(d49['hour'].unique()):
    mask = d49['hour'] == h
    sub = d49[mask]
    r2_h = r2_score(sub['demand'], sub['d48_best'])
    exact_h = sub['d48_exact'].notna().mean()
    print(f"Hour {h:2d}: R2={r2_h:.4f}, n={len(sub)}, exact_match={exact_h*100:.0f}%")

# Demand statistics comparison
print(f"\n--- Demand Statistics ---")
print(f"Day 48: mean={d48['demand'].mean():.6f}, std={d48['demand'].std():.6f}")
print(f"Day 49: mean={d49['demand'].mean():.6f}, std={d49['demand'].std():.6f}")

# By RoadType
print(f"\n--- R2 by RoadType ---")
for rt in d49['RoadType'].unique():
    if pd.isna(rt):
        mask = d49['RoadType'].isna()
        label = 'NaN'
    else:
        mask = d49['RoadType'] == rt
        label = rt
    sub = d49[mask]
    if len(sub) > 10:
        r2_rt = r2_score(sub['demand'], sub['d48_best'])
        print(f"{label:15s}: R2={r2_rt:.4f}, n={len(sub)}")
