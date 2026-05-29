import pandas as pd
import numpy as np

train = pd.read_csv('e88186124ec611f1/dataset/train.csv')
test = pd.read_csv('e88186124ec611f1/dataset/test.csv')

# Parse timestamp properly
def parse_ts(ts_str):
    parts = ts_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

train['ts_minutes'] = train['timestamp'].apply(parse_ts)
test['ts_minutes'] = test['timestamp'].apply(parse_ts)

print("=" * 60)
print("DETAILED TEMPORAL ANALYSIS")
print("=" * 60)

# Train day 48 timestamps
train48 = train[train['day'] == 48]
train49 = train[train['day'] == 49]
print(f"Train day 48: {len(train48)} rows, timestamps: {sorted(train48['ts_minutes'].unique())[:5]}...{sorted(train48['ts_minutes'].unique())[-5:]}")
print(f"Train day 49: {len(train49)} rows, timestamps: {sorted(train49['ts_minutes'].unique())[:5]}...{sorted(train49['ts_minutes'].unique())[-5:]}")
print(f"Test day 49: {len(test)} rows, timestamps: {sorted(test['ts_minutes'].unique())[:5]}...{sorted(test['ts_minutes'].unique())[-5:]}")
print()

# More detail on timestamps
train48_ts = sorted(train48['timestamp'].unique())
train49_ts = sorted(train49['timestamp'].unique(), key=lambda x: parse_ts(x))
test_ts = sorted(test['timestamp'].unique(), key=lambda x: parse_ts(x))

print(f"Train day 48 timestamps ({len(train48_ts)}): all 96 15-min intervals")
print(f"Train day 49 timestamps ({len(train49_ts)}):")
for t in train49_ts:
    print(f"  {t} ({parse_ts(t)} min)")
print()
print(f"Test day 49 timestamps ({len(test_ts)}):")
for t in test_ts:
    print(f"  {t} ({parse_ts(t)} min)")
print()

# Check overlap
train49_ts_set = set(train49_ts)
test_ts_set = set(test_ts)
print(f"Train49 & Test overlap: {train49_ts_set & test_ts_set}")
print(f"Train49-only: {train49_ts_set - test_ts_set}")
print(f"Test-only: {test_ts_set - train49_ts_set}")
print()

# Check geohash consistency per location
print("=" * 60)
print("GEOHASH CONSISTENCY CHECK")
print("=" * 60)
# Do same geohashes keep same features?
sample_geo = train['geohash'].value_counts().index[0]
print(f"Sample geohash: {sample_geo}")
sample = train[train['geohash'] == sample_geo]
print(f"Rows: {len(sample)}")
print(f"RoadType: {sample['RoadType'].unique()}")
print(f"NumberofLanes: {sample['NumberofLanes'].unique()}")
print(f"LargeVehicles: {sample['LargeVehicles'].unique()}")
print(f"Landmarks: {sample['Landmarks'].unique()}")
print()

# Check multiple geohashes
print("Checking if location features are static per geohash:")
for col in ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks']:
    nunique_per_geo = train.groupby('geohash')[col].nunique()
    print(f"  {col}: max unique per geohash = {nunique_per_geo.max()}, geohashes with >1 = {(nunique_per_geo > 1).sum()}")
print()

# Geohash frequency analysis
print("=" * 60)
print("GEOHASH FREQUENCY")
print("=" * 60)
geo_counts = train['geohash'].value_counts()
print(f"Top 10:\n{geo_counts.head(10)}")
print(f"Bottom 10:\n{geo_counts.tail(10)}")
print(f"Mean rows per geohash: {geo_counts.mean():.1f}")
print()

# Demand by features
print("=" * 60)
print("DEMAND BY FEATURES")
print("=" * 60)
for col in ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'NumberofLanes']:
    grp = train.groupby(col)['demand'].agg(['mean', 'median', 'std', 'count'])
    print(f"\n{col}:")
    print(grp)
print()

# Demand by hour
train['hour'] = train['ts_minutes'] // 60
hourly = train.groupby('hour')['demand'].agg(['mean', 'median', 'count'])
print("\nDemand by hour:")
print(hourly)
