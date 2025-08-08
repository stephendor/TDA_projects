"""Baseline Gradient Boosted Trees training on vectorized TDA window features.

Workflow:
  * Stream synthetic (placeholder) data windows (real dataset integration later)
  * For each window build point cloud and compute persistence diagrams via VR
  * Aggregate feature vectors
  * Train train/test split with XGBoost classifier (binary labels synthetic)
  * Report accuracy + confusion matrix (deterministic seed)

Real integration step will replace synthetic generation with actual StreamingWindowLoader
yielding windows aligned temporally with attack labels ensuring no leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from src.tda.vectorizers.window_tda_vectorizer import TDAWindowVectorizer, TDAWindowVectorizerConfig

# Synthetic placeholder data generation (FIXME: replace with real streaming windows)
RNG = np.random.default_rng(42)
N_WINDOWS = 60
POINTS_PER_WIN = 300
FEATURES = ['f1','f2','f3']

windows = []
labels = []
for i in range(N_WINDOWS):
    base = RNG.normal(size=(POINTS_PER_WIN, len(FEATURES)))
    label = 1 if i % 5 == 0 else 0  # sparse positives
    if label == 1:
        # Inject structural shift (larger variance) to create topological signal
        base += RNG.normal(0, 1.5, size=base.shape)
    df = pd.DataFrame(base, columns=FEATURES)
    df['timestamp'] = i * 10 + RNG.random(POINTS_PER_WIN)
    windows.append(df)
    labels.append(label)

vec = TDAWindowVectorizer(TDAWindowVectorizerConfig(feature_columns=FEATURES, max_vr_points=500))
X_feats = []
for w in windows:
    r = vec.vectorize(w)
    X_feats.append(r['features'])
X = np.vstack(X_feats)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=4,
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Baseline GBT Accuracy:", acc)
print("Confusion Matrix:\n", cm)
