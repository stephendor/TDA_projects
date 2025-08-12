"""Witness complex based persistence builder.

Applies landmark selection (k-medoids / random fallback) to large point clouds then computes persistence.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from src.core.persistent_homology import PersistentHomologyAnalyzer

def _farthest_point_sampling(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    if k >= X.shape[0]:
        return X
    rng = np.random.default_rng(seed)
    first = rng.integers(0, X.shape[0])
    selected = [first]
    dists = np.linalg.norm(X - X[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dists))
        selected.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(X - X[nxt], axis=1))
    return X[selected]

class WitnessPersistenceBuilder:
    def __init__(self, landmarks: int = 128, maxdim: int = 2, backend: str = "ripser", metric: str = "euclidean", seed: int = 42):
        self.landmarks = landmarks
        self.maxdim = maxdim
        self.backend = backend
        self.metric = metric
        self.seed = seed

    def compute(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        if point_cloud.ndim != 2:
            raise ValueError("point_cloud must be 2D array [n_points, n_features]")
        lm_points = _farthest_point_sampling(point_cloud, self.landmarks, seed=self.seed)
        analyzer = PersistentHomologyAnalyzer(backend=self.backend, maxdim=self.maxdim, metric=self.metric)
        analyzer.fit(lm_points)
        return {
            "diagrams": analyzer.persistence_diagrams_,
            "features": analyzer.extract_features(),
            "landmark_count": lm_points.shape[0]
        }

__all__ = ["WitnessPersistenceBuilder"]
