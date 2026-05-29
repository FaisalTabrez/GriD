# Traffic Demand Prediction Hackathon — Development Instructions

## Objective

Build a high-performance machine learning pipeline for the Hackerearth "Traffic demand prediction" competition.

Primary metric:

* R² Score

Target variable:

* `demand`

Goal:

* maximize leaderboard score
* avoid data leakage
* maintain reproducibility
* produce modular, competition-grade code

---

# Project Philosophy

This is a tabular ML competition.

Prioritize:

1. Feature engineering
2. Validation correctness
3. Leakage prevention
4. Gradient boosting models
5. Ensemble methods

Do NOT prioritize deep learning unless strong evidence suggests otherwise.

Expected strongest models:

* LightGBM
* CatBoost
* XGBoost

---

# Initial Tasks (MANDATORY)

Before building models, perform detailed exploratory analysis.

## 1. Dataset Inspection

For both train and test datasets:

* print shape
* inspect dtypes
* identify missing values
* inspect unique values
* inspect categorical distributions
* inspect timestamp range
* inspect day distribution

Generate:

* summary statistics
* missing value heatmap/table
* feature cardinality report

---

## 2. Determine Split Strategy

This is CRITICAL.

Check whether:

* test set contains future timestamps
  OR
* train/test are randomly sampled

Tasks:

* compare timestamp ranges
* compare day distributions
* check overlap patterns

If future-based split:

* use TimeSeriesSplit or custom temporal validation

If random:

* use KFold cross-validation

Never use random KFold on future forecasting data.

---

## 3. Analyze Target Distribution

Inspect `demand` distribution carefully.

Generate:

* histogram
* KDE plot
* boxplot
* skewness statistics

Determine whether log transformation is beneficial.

If highly right-skewed:

Use:

```python
y_train = np.log1p(y_train)
pred = np.expm1(pred)
```

Compare validation performance with and without transformation.

---

# Feature Engineering Requirements

Feature engineering is the highest priority.

Create modular feature pipelines.

---

# Temporal Features

From `timestamp`, extract:

* hour
* minute
* day_of_week
* day_of_month
* month
* is_weekend
* is_business_hour
* is_morning_rush
* is_evening_rush
* time_bucket

Create cyclical encodings:

```python
hour_sin
hour_cos
weekday_sin
weekday_cos
```

using sine/cosine transformations.

---

# Geospatial Features

`geohash` is extremely important.

## Mandatory:

Decode geohash into:

* latitude
* longitude

Use appropriate geohash libraries.

## Also create:

* geohash prefixes:

  * geo3
  * geo4
  * geo5

These represent different spatial granularities.

---

# Spatial Intelligence Features

Experiment with:

## Clustering

Use KMeans or similar clustering on:

* latitude
* longitude

Generate:

* region_cluster

## Density Features

Create:

* frequency/count of each geohash
* traffic density proxies

---

# Categorical Features

Handle:

* RoadType
* Weather
* geohash prefixes
* cluster labels

Experiment with:

* label encoding
* native categorical handling
* target encoding

---

# Target Encoding Rules

If using target encoding:

MUST avoid leakage.

Use:

* out-of-fold encoding
* fold-wise mean computation

Never compute target encoding on full training data before validation.

---

# Interaction Features

Create interaction features such as:

* Weather × Hour
* RoadType × Hour
* Weekend × Geohash
* Temperature × Weather
* Lanes × RoadType

Test feature importance after creation.

---

# Missing Value Strategy

Explicitly handle missing values.

Compare:

* median imputation
* mode imputation
* model-native handling

Track performance impact.

---

# Validation Strategy

Validation correctness is extremely important.

Requirements:

* fixed random seed
* reproducible splits
* separate train/validation logic
* no leakage

Track:

* fold scores
* mean CV score
* standard deviation

Store validation predictions for analysis.

---

# Modeling Strategy

## Phase 1 — Baseline

Build quick baselines using:

* CatBoostRegressor
* LightGBMRegressor

Goal:

* establish benchmark score rapidly

---

## Phase 2 — Advanced Features

Integrate:

* spatial features
* cyclical features
* target encoding
* interaction features

---

## Phase 3 — Hyperparameter Tuning

Use Optuna.

Tune:

### LightGBM

* num_leaves
* learning_rate
* max_depth
* min_child_samples
* feature_fraction
* bagging_fraction
* lambda_l1
* lambda_l2

### CatBoost

* depth
* learning_rate
* l2_leaf_reg
* iterations

### XGBoost

* max_depth
* eta
* subsample
* colsample_bytree
* min_child_weight

---

# Ensemble Strategy

Train multiple models.

Experiment with:

* weighted averaging
* seed averaging
* stacking (optional)

Primary ensemble candidates:

* LightGBM
* CatBoost
* XGBoost

Track:

* single model performance
* ensemble uplift

---

# Feature Importance & Diagnostics

Generate:

* SHAP plots
* gain importance
* permutation importance

Identify:

* most influential features
* unstable features
* leakage indicators

---

# Experiment Tracking

Maintain structured experiments.

For every run, log:

* feature set
* validation scheme
* parameters
* CV score
* public leaderboard score

Store results in:

* CSV
  OR
* lightweight experiment tracker

---

# Codebase Requirements

Structure project cleanly.

Recommended layout:

```text
project/
│
├── data/
├── notebooks/
├── src/
│   ├── features/
│   ├── models/
│   ├── validation/
│   ├── utils/
│   └── inference/
├── outputs/
├── submissions/
└── experiments/
```

---

# Notebook Requirements

Create notebooks in stages:

1. EDA
2. Feature Engineering
3. Baseline Models
4. Hyperparameter Tuning
5. Ensemble
6. Final Submission

Keep notebooks clean and reproducible.

---

# Submission Requirements

Submission format:

```text
Index,demand
```

Ensure:

* exact ordering
* correct index values
* no NaNs
* valid numeric predictions

---

# Important Constraints

DO NOT:

* leak future information
* use full-data target encoding
* overfit public leaderboard
* hardcode assumptions without validation

DO:

* validate every engineering choice
* compare transformations scientifically
* maintain reproducibility

---

# Expected Workflow

1. EDA
2. Validation design
3. Baseline CatBoost
4. Temporal features
5. Geospatial features
6. Interaction features
7. Target encoding
8. Hyperparameter tuning
9. Ensemble
10. Final submission optimization

---

# Deliverables

Produce:

* clean notebooks
* reusable training scripts
* modular feature engineering pipeline
* tuned ensemble models
* final submission CSV
* experiment documentation

---

# Performance Goal

Aim for:

* robust CV stability
* strong private leaderboard generalization
* minimal leakage risk
* reproducible top-tier solution
