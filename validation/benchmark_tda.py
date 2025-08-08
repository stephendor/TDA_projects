"""TDA scaling & memory benchmark.

Measures time and memory for persistence computations across point counts
and vectorizers (raw stats, persistence image, landscape).

Protocol:
  * Generate synthetic Gaussian point clouds with controlled size
  * Compute VR diagrams (ripser) up to H2
  * Record: n_points, wall_time, peak_rss_delta, feature_dim for each vectorizer

This does NOT validate model performance; purely computational characteristics.
"""
from __future__ import annotations
import time
import psutil
import os
import json
from pathlib import Path
import numpy as np
from src.core.persistent_homology import PersistentHomologyAnalyzer
from src.tda.vectorizers.diagram_vectorizers import (
    PersistenceImageConfig, PersistenceImageVectorizer,
    PersistenceLandscapeConfig, PersistenceLandscapeVectorizer,
)

OUT = Path('results/tda_benchmark.json')
OUT.parent.mkdir(parents=True, exist_ok=True)

POINT_SCALES = [200, 400, 800]
REPEATS = 2

pi_vec = PersistenceImageVectorizer(PersistenceImageConfig(resolution=(15,15), sigma=0.08))
pl_vec = PersistenceLandscapeVectorizer(PersistenceLandscapeConfig(resolution=80, k_layers=3))

results = []
proc = psutil.Process(os.getpid())
base_rss = proc.memory_info().rss

for n in POINT_SCALES:
    for r in range(REPEATS):
        pc = np.random.randn(n, 3)
        analyzer = PersistentHomologyAnalyzer(backend='ripser', maxdim=2)
        t0 = time.time()
        analyzer.fit(pc)
        diagrams = analyzer.persistence_diagrams_
        if diagrams is None:
            raise RuntimeError("Persistence diagrams not computed")
        raw_features = analyzer.extract_features()
        t1 = time.time()
        rss_mid = proc.memory_info().rss
        pi = [pi_vec.transform_diagram(d) for d in diagrams[:2]]  # only H0,H1 for images
        pl = [pl_vec.transform_diagram(d) for d in diagrams[:2]]
        t2 = time.time()
        rss_end = proc.memory_info().rss
        results.append({
            'n_points': n,
            'repeat': r,
            'fit_time': t1 - t0,
            'vectorize_time': t2 - t1,
            'total_time': t2 - t0,
            'rss_fit_delta': rss_mid - base_rss,
            'rss_total_delta': rss_end - base_rss,
            'raw_feat_dim': int(raw_features.shape[0]),
            'pi_dim': int(sum(v.shape[0] for v in pi)),
            'pl_dim': int(sum(v.shape[0] for v in pl)),
        })

with OUT.open('w') as f:
    json.dump(results, f, indent=2)

print(f"Wrote benchmark results to {OUT} ({len(results)} rows)")
