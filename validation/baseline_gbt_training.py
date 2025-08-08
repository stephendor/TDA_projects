"""Baseline Gradient Boosted Trees training on vectorized TDA window features.

Workflow:
  * Stream real CIC-IDS2017 (or configured) data windows via StreamingWindowLoader
  * Build point clouds per window (numeric features only initially)
  * Compute persistence diagrams (VR / witness fallback) -> deterministic topological features
  * Aggregate feature vectors + align with labels (attack vs benign) using timestamp overlap
  * Train train/test split with temporal integrity (earlier segment = train, later = test) to avoid leakage
  * Report accuracy + confusion matrix (deterministic seed)

Validation-first:
  - NO performance claim here; script outputs raw metrics only
  - Label integration requires presence of 'Label' column (binary) in parquet files
  - Temporal leakage guard: ensures max(train_timestamp) < min(test_timestamp)
"""
from __future__ import annotations
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from src.data.stream_loader import WindowConfig, StreamingWindowLoader, build_point_cloud
from src.tda.vectorizers.window_tda_vectorizer import TDAWindowVectorizer, TDAWindowVectorizerConfig

# ---- Load dataset config ----
CONFIG_PATH = Path('configs/dataset_cic.yaml')
if not CONFIG_PATH.exists():
    raise SystemExit("Dataset config not found: configs/dataset_cic.yaml")
with CONFIG_PATH.open('r') as f:
    ds_cfg = yaml.safe_load(f)

root_path = Path(ds_cfg['root_path'])
file_pattern = ds_cfg['file_pattern']
feature_cols = ds_cfg['features']['numeric']  # use numeric only for point cloud
label_column = 'Label'

# Collect parquet files
if not root_path.exists():
    raise SystemExit(f"Data root does not exist: {root_path}")
parquet_files = sorted(root_path.glob(file_pattern))
if len(parquet_files) == 0:
    raise SystemExit(f"No parquet files matching pattern {file_pattern} under {root_path}")

# Window configuration
win_cfg = WindowConfig(
    window_seconds=ds_cfg['window']['seconds'],
    overlap=ds_cfg['window']['overlap'],
    time_column=ds_cfg['time_column']
)

# Vectorizer configuration
vec = TDAWindowVectorizer(TDAWindowVectorizerConfig(
    feature_columns=feature_cols,
    max_vr_points=ds_cfg['limits']['max_vr_points'],
    witness_landmarks=ds_cfg['limits']['witness_landmarks']
))

# ---- Stream windows and build feature matrix ----
X_features = []
Y_labels = []
WindowTimes = []
WindowOrder = []  # collection order for synthetic fallback

for w_idx, window_df in enumerate(StreamingWindowLoader(parquet_files, win_cfg).windows()):
    if label_column not in window_df.columns:
        continue
    label_counts = window_df[label_column].value_counts()
    if len(label_counts) == 0:
        continue
    majority_label = label_counts.idxmax()
    sub = window_df[feature_cols]
    # Clean numeric: replace inf, drop NaN, remove extreme magnitudes
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        continue
    # Optional clipping to mitigate huge rates (deterministic)
    sub = sub.clip(lower=-1e12, upper=1e12)
    cleaned = sub.join(window_df[[win_cfg.time_column]])
    pc = build_point_cloud(
        window=cleaned,
        feature_columns=feature_cols,
        cap_points=ds_cfg['limits']['max_vr_points'],
        seed=ds_cfg.get('random_seed', 42)
    )
    # Ensure enough points > dimensionality to avoid ripser warning (points > features)
    if pc.shape[0] <= len(feature_cols):
        continue
    res = vec.vr.compute(pc) if pc.shape[0] <= vec.cfg.max_vr_points else vec.witness.compute(pc)
    X_features.append(res['features'])
    Y_labels.append(int(majority_label))
    WindowTimes.append(float(window_df[win_cfg.time_column].min()))
    WindowOrder.append(w_idx)

if len(X_features) < 10:
    raise SystemExit(f"Insufficient windows with labels to train baseline (collected {len(X_features)})")

X = np.vstack(X_features)
y = np.array(Y_labels)
WindowTimes = np.array(WindowTimes)
WindowOrder = np.array(WindowOrder)

# Detect duplicate / synthetic timestamps (e.g., per-file synthetic resets)
unique_ts = len(set(WindowTimes))
use_collection_order = unique_ts < len(WindowTimes)
if use_collection_order:
    print("WARNING: Duplicate window start timestamps detected; using collection order for temporal split (synthetic time).")
    order = np.argsort(WindowOrder)
else:
    order = np.argsort(WindowTimes)

X, y, WindowTimes, WindowOrder = X[order], y[order], WindowTimes[order], WindowOrder[order]

# Temporal split (collection-order if synthetic)
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
max_train_time = WindowTimes[:split_idx].max()
min_test_time = WindowTimes[split_idx:].min()

if not use_collection_order:
    if max_train_time >= min_test_time:
        raise SystemExit("Temporal leakage detected: training window end overlaps test window start")
else:
    print("INFO: Temporal leakage guard disabled (synthetic timestamps). Results are NOT temporally validated.")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=ds_cfg.get('random_seed', 42),
    n_jobs=4,
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Train windows:", X_train.shape[0], "Test windows:", X_test.shape[0])
print("Temporal split integrity (enforced):", (not use_collection_order) and (max_train_time < min_test_time))
print("Baseline GBT (topological features) Accuracy:", acc)
print("Confusion Matrix:\n", cm)
