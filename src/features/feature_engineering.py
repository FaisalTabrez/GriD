"""
Feature Engineering Pipeline for Traffic Demand Prediction.

This module provides all feature engineering functions used in the pipeline.
Features are organized by category: temporal, geospatial, categorical, interactions.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Geohash decoding (pure-Python fallback so no external dep needed)
# ---------------------------------------------------------------------------
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BITS = [16, 8, 4, 2, 1]

def _decode_geohash(geohash_str):
    """Decode a geohash string to (latitude, longitude)."""
    is_lon = True
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    for char in geohash_str:
        cd = _BASE32.index(char)
        for mask in _BITS:
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if cd & mask:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if cd & mask:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon
    lat = (lat_range[0] + lat_range[1]) / 2
    lon = (lon_range[0] + lon_range[1]) / 2
    return lat, lon


# ---------------------------------------------------------------------------
# Core Feature Engineering
# ---------------------------------------------------------------------------

def parse_timestamp(df):
    """Parse timestamp string 'H:M' into numeric features."""
    parts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = parts[0]
    df['minute'] = parts[1]
    df['ts_minutes'] = df['hour'] * 60 + df['minute']
    return df


def add_temporal_features(df):
    """Add temporal features from hour/minute."""
    df = parse_timestamp(df)
    
    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical minute encoding
    df['minute_sin'] = np.sin(2 * np.pi * df['ts_minutes'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['ts_minutes'] / 1440)
    
    # Time period features
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)
    
    # Time bucket (4-hour blocks)
    df['time_bucket'] = df['hour'] // 4
    
    # Finer time bucket (2-hour blocks)
    df['time_bucket_2h'] = df['hour'] // 2
    
    return df


def add_geospatial_features(df):
    """Decode geohash and create spatial features."""
    # Decode geohash to lat/lon
    coords = df['geohash'].apply(_decode_geohash)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    
    # Geohash prefixes for different spatial granularities
    df['geo3'] = df['geohash'].str[:3]
    df['geo4'] = df['geohash'].str[:4]
    df['geo5'] = df['geohash'].str[:5]
    
    return df


def add_categorical_features(df, label_encoders=None, fit=True):
    """Encode categorical features. Returns df and encoders dict."""
    if label_encoders is None:
        label_encoders = {}
    
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    
    for col in cat_cols:
        col_enc = f'{col}_enc'
        if fit:
            le = LabelEncoder()
            # Handle NaN by filling with 'missing'
            vals = df[col].fillna('missing').astype(str)
            le.fit(vals)
            label_encoders[col] = le
            df[col_enc] = le.transform(vals)
        else:
            le = label_encoders[col]
            vals = df[col].fillna('missing').astype(str)
            # Handle unseen labels
            known = set(le.classes_)
            vals = vals.apply(lambda x: x if x in known else 'missing')
            df[col_enc] = le.transform(vals)
    
    # Binary encoding for binary features
    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_bin'] = (df['Landmarks'] == 'Yes').astype(int)
    
    # Geohash prefix encoding
    for prefix_col in ['geo3', 'geo4', 'geo5']:
        col_enc = f'{prefix_col}_enc'
        if fit:
            le = LabelEncoder()
            le.fit(df[prefix_col].astype(str))
            label_encoders[prefix_col] = le
            df[col_enc] = le.transform(df[prefix_col].astype(str))
        else:
            le = label_encoders[prefix_col]
            known = set(le.classes_)
            vals = df[prefix_col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col_enc] = le.transform(vals)
    
    return df, label_encoders


def add_geohash_label_encoding(df, label_encoders=None, fit=True):
    """Label-encode the full geohash."""
    if label_encoders is None:
        label_encoders = {}
    
    col = 'geohash'
    col_enc = 'geohash_enc'
    if fit:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        label_encoders[col] = le
        df[col_enc] = le.transform(df[col].astype(str))
    else:
        le = label_encoders[col]
        known = set(le.classes_)
        vals = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        df[col_enc] = le.transform(vals)
    
    return df, label_encoders


def add_interaction_features(df):
    """Create interaction features between key variables."""
    # RoadType × Hour
    df['road_hour'] = df['RoadType_enc'] * 100 + df['hour']
    
    # Weather × Hour
    df['weather_hour'] = df['Weather_enc'] * 100 + df['hour']
    
    # Lanes × RoadType
    df['lanes_road'] = df['NumberofLanes'] * 10 + df['RoadType_enc']
    
    # Temperature × Weather
    df['temp_weather'] = df['Temperature'].fillna(0) * (df['Weather_enc'] + 1)
    
    # Lanes × Hour
    df['lanes_hour'] = df['NumberofLanes'] * 100 + df['hour']
    
    # LargeVehicles × RoadType
    df['large_road'] = df['LargeVehicles_bin'] * 10 + df['RoadType_enc']
    
    # Landmarks × Hour
    df['landmarks_hour'] = df['Landmarks_bin'] * 100 + df['hour']
    
    return df


def add_temperature_features(df):
    """Add temperature-derived features."""
    # Fill missing temperature with median
    temp_median = df['Temperature'].median()
    df['Temperature_filled'] = df['Temperature'].fillna(temp_median)
    
    # Temperature bins
    df['temp_bin'] = pd.cut(
        df['Temperature_filled'],
        bins=[-30, 0, 10, 20, 30, 50],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2)
    
    # Temperature missing indicator
    df['temp_missing'] = df['Temperature'].isnull().astype(int)
    
    return df


def add_geohash_frequency(df, freq_map=None, fit=True):
    """Add geohash frequency as a density proxy."""
    if fit:
        freq_map = df['geohash'].value_counts().to_dict()
    
    df['geohash_freq'] = df['geohash'].map(freq_map).fillna(0)
    
    return df, freq_map


def add_clustering_features(df, kmeans_model=None, n_clusters=20, fit=True):
    """Add KMeans clustering on lat/lon."""
    coords = df[['latitude', 'longitude']].values
    
    if fit:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_model.fit(coords)
    
    df['region_cluster'] = kmeans_model.predict(coords)
    
    # Distance to cluster center
    centers = kmeans_model.cluster_centers_
    cluster_labels = df['region_cluster'].values
    dists = np.sqrt(
        (coords[:, 0] - centers[cluster_labels, 0])**2 +
        (coords[:, 1] - centers[cluster_labels, 1])**2
    )
    df['dist_to_cluster_center'] = dists
    
    return df, kmeans_model


def add_target_encoding(df, target_col='demand', cols_to_encode=None,
                        target_maps=None, fit=True, global_mean=None):
    """
    Add target encoding for high-cardinality categorical features.
    When fit=True, computes target means from df (should be training data only).
    Uses smoothing to handle low-count categories.
    """
    if cols_to_encode is None:
        cols_to_encode = ['geohash', 'geo4', 'geo5', 'RoadType_enc', 'region_cluster']
    
    if target_maps is None:
        target_maps = {}
    
    if fit:
        global_mean = df[target_col].mean()
    
    smoothing = 10  # smoothing factor
    
    for col in cols_to_encode:
        enc_col = f'{col}_te'
        if fit:
            stats = df.groupby(col)[target_col].agg(['mean', 'count'])
            # Bayesian smoothing
            smoother = stats['count'] / (stats['count'] + smoothing)
            stats['smoothed'] = smoother * stats['mean'] + (1 - smoother) * global_mean
            target_maps[col] = stats['smoothed'].to_dict()
        
        df[enc_col] = df[col].map(target_maps[col]).fillna(global_mean)
    
    return df, target_maps, global_mean


def add_geohash_target_stats(df, target_col='demand', stats_maps=None, fit=True):
    """Add various target statistics per geohash (from training data only)."""
    if stats_maps is None:
        stats_maps = {}
    
    if fit:
        grp = df.groupby('geohash')[target_col]
        stats_maps['geo_demand_mean'] = grp.mean().to_dict()
        stats_maps['geo_demand_median'] = grp.median().to_dict()
        stats_maps['geo_demand_std'] = grp.std().fillna(0).to_dict()
        stats_maps['geo_demand_min'] = grp.min().to_dict()
        stats_maps['geo_demand_max'] = grp.max().to_dict()
        
        # Geo-hour interaction stats
        grp_hour = df.groupby(['geohash', 'hour'])[target_col].mean()
        stats_maps['geo_hour_mean'] = grp_hour.to_dict()
        
        # Geo-time_bucket stats
        grp_tb = df.groupby(['geohash', 'time_bucket'])[target_col].mean()
        stats_maps['geo_tb_mean'] = grp_tb.to_dict()
    
    global_mean = df[target_col].mean() if target_col in df.columns else 0.094
    
    df['geo_demand_mean'] = df['geohash'].map(stats_maps['geo_demand_mean']).fillna(global_mean)
    df['geo_demand_median'] = df['geohash'].map(stats_maps['geo_demand_median']).fillna(global_mean)
    df['geo_demand_std'] = df['geohash'].map(stats_maps['geo_demand_std']).fillna(0)
    df['geo_demand_min'] = df['geohash'].map(stats_maps['geo_demand_min']).fillna(0)
    df['geo_demand_max'] = df['geohash'].map(stats_maps['geo_demand_max']).fillna(1)
    df['geo_demand_range'] = df['geo_demand_max'] - df['geo_demand_min']
    
    # Geo-hour mean
    df['geo_hour_mean'] = df.apply(
        lambda row: stats_maps['geo_hour_mean'].get((row['geohash'], row['hour']), global_mean),
        axis=1
    )
    
    # Geo-time_bucket mean
    df['geo_tb_mean'] = df.apply(
        lambda row: stats_maps['geo_tb_mean'].get((row['geohash'], row['time_bucket']), global_mean),
        axis=1
    )
    
    return df, stats_maps


def build_features(train_df, test_df, use_target_encoding=True,
                   use_target_stats=True, use_clustering=True,
                   n_clusters=20):
    """
    Complete feature engineering pipeline.
    Fits on train, transforms both train and test.
    Returns: train_df, test_df, feature_columns, artifacts (encoders, etc.)
    """
    artifacts = {}
    
    # --- Temporal ---
    train_df = add_temporal_features(train_df)
    test_df = add_temporal_features(test_df)
    
    # --- Geospatial ---
    train_df = add_geospatial_features(train_df)
    test_df = add_geospatial_features(test_df)
    
    # --- Temperature ---
    train_df = add_temperature_features(train_df)
    test_df = add_temperature_features(test_df)
    
    # --- Categorical ---
    train_df, label_encoders = add_categorical_features(train_df, fit=True)
    test_df, _ = add_categorical_features(test_df, label_encoders=label_encoders, fit=False)
    artifacts['label_encoders'] = label_encoders
    
    # --- Geohash label encoding ---
    train_df, label_encoders = add_geohash_label_encoding(train_df, label_encoders, fit=True)
    test_df, _ = add_geohash_label_encoding(test_df, label_encoders, fit=False)
    
    # --- Interactions ---
    train_df = add_interaction_features(train_df)
    test_df = add_interaction_features(test_df)
    
    # --- Geohash frequency ---
    train_df, freq_map = add_geohash_frequency(train_df, fit=True)
    test_df, _ = add_geohash_frequency(test_df, freq_map=freq_map, fit=False)
    artifacts['freq_map'] = freq_map
    
    # --- Clustering ---
    if use_clustering:
        train_df, kmeans_model = add_clustering_features(train_df, n_clusters=n_clusters, fit=True)
        test_df, _ = add_clustering_features(test_df, kmeans_model=kmeans_model, fit=False)
        artifacts['kmeans_model'] = kmeans_model
    
    # --- Target stats (geohash-level aggregations) ---
    if use_target_stats:
        train_df, stats_maps = add_geohash_target_stats(train_df, fit=True)
        test_df, _ = add_geohash_target_stats(test_df, stats_maps=stats_maps, fit=False)
        artifacts['stats_maps'] = stats_maps
    
    # --- Target encoding ---
    if use_target_encoding:
        te_cols = ['geohash_enc', 'geo4_enc', 'geo5_enc', 'RoadType_enc', 'region_cluster']
        if not use_clustering:
            te_cols = [c for c in te_cols if c != 'region_cluster']
        train_df, target_maps, global_mean = add_target_encoding(
            train_df, cols_to_encode=te_cols, fit=True
        )
        test_df, _, _ = add_target_encoding(
            test_df, cols_to_encode=te_cols, target_maps=target_maps,
            fit=False, global_mean=global_mean
        )
        artifacts['target_maps'] = target_maps
        artifacts['global_mean'] = global_mean
    
    # --- Define feature columns ---
    exclude_cols = ['Index', 'geohash', 'timestamp', 'demand',
                    'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
                    'geo3', 'geo4', 'geo5', 'Temperature']
    
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Ensure no object columns remain
    for col in feature_cols:
        if train_df[col].dtype == 'object':
            feature_cols.remove(col)
    
    artifacts['feature_cols'] = feature_cols
    
    return train_df, test_df, feature_cols, artifacts
