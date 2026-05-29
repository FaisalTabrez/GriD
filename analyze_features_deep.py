"""Deep feature analysis to find what actually predicts demand well."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

DATA_DIR = Path(r'c:\Volume D\GriD\e88186124ec611f1\dataset')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

# Parse
parts = train['timestamp'].str.split(':', expand=True).astype(int)
train['hour'] = parts[0]
train['minute'] = parts[1]
train['ts_minutes'] = train['hour'] * 60 + train['minute']

# Check how many unique geohashes appear in BOTH train and test
train_geos = set(train['geohash'].unique())
test_geos = set(test['geohash'].unique())
print(f"Train unique geohashes: {len(train_geos)}")
print(f"Test unique geohashes: {len(test_geos)}")
print(f"Overlap: {len(train_geos & test_geos)}")
print(f"Test-only: {len(test_geos - train_geos)}")

# Check how many unique (geohash, RoadType) combos there are
d48 = train[train['day'] == 48]
d49 = train[train['day'] == 49]

# Check if features are truly dynamic or mostly static per geohash
print("\n--- Feature variability per geohash ---")
for col in ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Weather', 'Temperature']:
    nunique = d48.groupby('geohash')[col].nunique()
    pct_multi = (nunique > 1).mean() * 100
    print(f"{col:20s}: {pct_multi:.1f}% of geohashes have >1 unique value")

# Analysis: what combination of features best predicts demand?
print("\n--- Feature combination R2 on day 48 (OOF) ---")

# For each feature combo, compute OOF R2 using simple group means
combos = [
    ['RoadType'],
    ['NumberofLanes'],
    ['RoadType', 'NumberofLanes'],
    ['RoadType', 'hour'],
    ['geohash'],
    ['RoadType', 'NumberofLanes', 'hour'],
    ['geohash', 'hour'],
    ['RoadType', 'NumberofLanes', 'LargeVehicles'],
    ['RoadType', 'NumberofLanes', 'LargeVehicles', 'hour'],
]

global_mean = d48['demand'].mean()

for combo in combos:
    # Skip combos with NaN issues
    cols = combo.copy()
    sub = d48[cols + ['demand']].dropna(subset=cols)
    
    # OOF prediction using group means
    preds = np.full(len(sub), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for tr_idx, va_idx in kf.split(sub):
        tr_data = sub.iloc[tr_idx]
        va_data = sub.iloc[va_idx]
        
        group_means = tr_data.groupby(cols)['demand'].mean()
        
        # Map to validation
        keys = va_data[cols].apply(tuple, axis=1) if len(cols) > 1 else va_data[cols[0]]
        if len(cols) > 1:
            preds[va_idx] = keys.map(group_means.to_dict()).fillna(global_mean).values
        else:
            preds[va_idx] = va_data[cols[0]].map(group_means.to_dict()).fillna(global_mean).values
    
    r2 = r2_score(sub['demand'], preds)
    print(f"  {' x '.join(cols):50s} R2={r2:.6f}")

# Check day 49 prediction using day 48 group means
print("\n--- Temporal validation: day 48 group means -> day 49 ---")
for combo in combos:
    cols = combo.copy()
    
    # Compute group means from day 48
    d48_sub = d48[cols + ['demand']].dropna(subset=cols)
    group_means = d48_sub.groupby(cols)['demand'].mean()
    
    # Apply to day 49
    d49_sub = d49[cols + ['demand']].dropna(subset=cols)
    if len(cols) > 1:
        keys = d49_sub[cols].apply(tuple, axis=1)
        d49_preds = keys.map(group_means.to_dict()).fillna(global_mean)
    else:
        d49_preds = d49_sub[cols[0]].map(group_means.to_dict()).fillna(global_mean)
    
    r2 = r2_score(d49_sub['demand'], d49_preds)
    print(f"  {' x '.join(cols):50s} R2={r2:.6f}")

# Check demand distribution stability
print("\n--- Demand by RoadType x Day ---")
for rt in ['Residential', 'Street', 'Highway']:
    for day in [48, 49]:
        sub = train[(train['RoadType'] == rt) & (train['day'] == day)]
        print(f"  {rt:12s} Day {day}: mean={sub['demand'].mean():.6f}, std={sub['demand'].std():.6f}, n={len(sub)}")

# Check if NumberofLanes is highly predictive
print("\n--- Demand by NumberofLanes ---")
for lanes in sorted(train['NumberofLanes'].unique()):
    sub = train[train['NumberofLanes'] == lanes]
    print(f"  Lanes={lanes}: mean={sub['demand'].mean():.6f}, std={sub['demand'].std():.6f}, n={len(sub)}")
