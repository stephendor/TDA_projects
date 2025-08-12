"""Vietoris-Rips (VR) complex builder wrapper.

Uses PersistentHomologyAnalyzer to compute persistence diagrams from point clouds.
Validation scope: compute diagrams up to H2 with point capping handled upstream.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from src.core.persistent_homology import PersistentHomologyAnalyzer

class VRPersistenceBuilder:
    def __init__(self, maxdim: int = 2, backend: str = "ripser", thresh: float = float("inf"), metric: str = "euclidean"):
        self.maxdim = maxdim
        self.backend = backend
        self.thresh = thresh
        self.metric = metric

    def compute(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        if point_cloud.ndim != 2:
            raise ValueError("point_cloud must be 2D array [n_points, n_features]")
        analyzer = PersistentHomologyAnalyzer(backend=self.backend, maxdim=self.maxdim, thresh=self.thresh, metric=self.metric)
        analyzer.fit(point_cloud)
        return {
            "diagrams": analyzer.persistence_diagrams_,
            "features": analyzer.extract_features()
        }

__all__ = ["VRPersistenceBuilder"]
