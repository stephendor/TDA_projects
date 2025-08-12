"""Window-level TDA vectorizer.

Takes a dataframe window + feature columns -> point cloud -> persistence diagrams -> feature vector.
Uses PersistentHomologyAnalyzer (VR) first, optionally witness fallback for large windows.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from src.data.stream_loader import build_point_cloud
from src.tda.builders.vr import VRPersistenceBuilder
from src.tda.builders.witness import WitnessPersistenceBuilder

@dataclass
class TDAWindowVectorizerConfig:
    feature_columns: List[str]
    max_vr_points: int = 2000
    witness_landmarks: int = 128
    maxdim: int = 2

class TDAWindowVectorizer:
    def __init__(self, cfg: TDAWindowVectorizerConfig):
        self.cfg = cfg
        self.vr = VRPersistenceBuilder(maxdim=cfg.maxdim)
        self.witness = WitnessPersistenceBuilder(landmarks=cfg.witness_landmarks, maxdim=cfg.maxdim)

    def vectorize(self, window_df: pd.DataFrame) -> Dict[str, Any]:
        pc = build_point_cloud(window_df, self.cfg.feature_columns, cap_points=self.cfg.max_vr_points)
        builder = self.vr if pc.shape[0] <= self.cfg.max_vr_points else self.witness
        res = builder.compute(pc)
        return {
            'features': res['features'],
            'diagrams': res['diagrams'],
            'n_points': pc.shape[0]
        }

__all__ = ["TDAWindowVectorizer", "TDAWindowVectorizerConfig"]
