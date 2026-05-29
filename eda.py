import pandas as pd
import numpy as np

train = pd.read_csv('e88186124ec611f1/dataset/train.csv')
test = pd.read_csv('e88186124ec611f1/dataset/test.csv')

print("=" * 60)
print("TRAIN DATASET")
print("=" * 60)
print(f"Shape: {train.shape}")
print(f"Columns: {list(train.columns)}")
print()
print("Dtypes:")
print(train.dtypes)
print()
print("Missing values:")
print(train.isnull().sum())
print()
print("Describe:")
print(train.describe())
print()

print("=" * 60)
print("TEST DATASET")
print("=" * 60)
print(f"Shape: {test.shape}")
print(f"Columns: {list(test.columns)}")
print()
print("Missing values:")
print(test.isnull().sum())
print()

print("=" * 60)
print("UNIQUE VALUES")
print("=" * 60)
for c in train.columns:
    print(f"  {c}: {train[c].nunique()} unique")
print()

print("=" * 60)
print("CATEGORICAL DISTRIBUTIONS")
print("=" * 60)
for c in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
    print(f"\n{c}:")
    print(train[c].value_counts(dropna=False))
print()

print("=" * 60)
print("DAY RANGES")
print("=" * 60)
print(f"Train day: {train['day'].min()} - {train['day'].max()}")
print(f"Test day: {test['day'].min()} - {test['day'].max()}")
print(f"Train day unique values: {sorted(train['day'].unique())}")
print(f"Test day unique values: {sorted(test['day'].unique())}")
print()

print("=" * 60)
print("TIMESTAMP ANALYSIS")
print("=" * 60)
train_ts = sorted(train['timestamp'].unique())
test_ts = sorted(test['timestamp'].unique())
print(f"Train timestamp count: {len(train_ts)}")
print(f"Test timestamp count: {len(test_ts)}")
print(f"Train timestamps (first 10): {train_ts[:10]}")
print(f"Test timestamps (first 10): {test_ts[:10]}")
ts_overlap = set(train_ts) & set(test_ts)
print(f"Timestamp overlap: {len(ts_overlap)} common timestamps")
print()

print("=" * 60)
print("GEOHASH ANALYSIS")
print("=" * 60)
train_geo = set(train['geohash'].unique())
test_geo = set(test['geohash'].unique())
print(f"Train geohashes: {len(train_geo)}")
print(f"Test geohashes: {len(test_geo)}")
print(f"Overlap: {len(train_geo & test_geo)}")
print(f"Test-only geohashes: {len(test_geo - train_geo)}")
print(f"Sample geohashes: {list(train_geo)[:10]}")
print()

print("=" * 60)
print("DEMAND DISTRIBUTION")
print("=" * 60)
d = train['demand']
print(f"min: {d.min()}")
print(f"max: {d.max()}")
print(f"mean: {d.mean():.6f}")
print(f"median: {d.median():.6f}")
print(f"std: {d.std():.6f}")
print(f"skewness: {d.skew():.4f}")
print(f"kurtosis: {d.kurtosis():.4f}")
print(f"quantiles:")
for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  {q*100:.0f}%: {d.quantile(q):.6f}")
print()

# Log transform analysis
log_d = np.log1p(d)
print(f"After log1p: skew={log_d.skew():.4f}, kurtosis={log_d.kurtosis():.4f}")
print()

print("=" * 60)
print("TEMPORAL SPLIT CHECK")
print("=" * 60)
train_days = set(train['day'].unique())
test_days = set(test['day'].unique())
print(f"Day overlap: {train_days & test_days}")
print(f"Train-only days: {train_days - test_days}")
print(f"Test-only days: {test_days - train_days}")
print()

# Check if it's temporal split
if max(train['day'].unique()) < min(test['day'].unique()):
    print(">>> TEMPORAL SPLIT: Test days are AFTER train days")
elif len(train_days & test_days) > 0:
    print(">>> OVERLAP: Train and test share some days")
else:
    print(">>> RANDOM or OTHER split")
print()

print("=" * 60)
print("GEOHASH + DAY COMBINATIONS")
print("=" * 60)
print(f"Train: {len(train.groupby(['geohash', 'day']))} geo-day combos")
print(f"Test: {len(test.groupby(['geohash', 'day']))} geo-day combos")
print()

# NumberofLanes
print("=" * 60)
print("NUMBEROF LANES")
print("=" * 60)
print(f"Train: {sorted(train['NumberofLanes'].unique())}")
print(f"Test: {sorted(test['NumberofLanes'].unique())}")
print()

# Temperature
print("=" * 60)
print("TEMPERATURE")
print("=" * 60)
print(f"Train: min={train['Temperature'].min():.2f}, max={train['Temperature'].max():.2f}")
print(f"Test: min={test['Temperature'].min():.2f}, max={test['Temperature'].max():.2f}")
print(f"Missing train: {train['Temperature'].isnull().sum()}")
print(f"Missing test: {test['Temperature'].isnull().sum()}")
