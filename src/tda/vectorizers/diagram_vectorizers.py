"""Diagram-level TDA vectorizers (persistence image / landscape).

Uses existing PersistentHomologyAnalyzer outputs (persistence diagrams) and
constructs stable, vectorizable representations:
  * Persistence Image (Adams et al. 2017) via simple Gaussian weighting
  * Persistence Landscape (Bubenik 2015) first k layers

No statistical proxy features; all derived from birth/death topology.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Tuple
import numpy as np

# ---- Persistence Image ----

def _diagram_to_birth_persistence(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return diagram
    b = diagram[:, 0]
    d = diagram[:, 1]
    p = d - b
    finite = np.isfinite(d)
    return np.column_stack([b[finite], p[finite]])

@dataclass
class PersistenceImageConfig:
    resolution: Tuple[int, int] = (20, 20)
    sigma: float = 0.05
    birth_range: Tuple[float, float] | None = None
    pers_range: Tuple[float, float] | None = None

class PersistenceImageVectorizer:
    def __init__(self, cfg: PersistenceImageConfig):
        self.cfg = cfg

    def transform_diagram(self, diagram: np.ndarray) -> np.ndarray:
        bp = _diagram_to_birth_persistence(diagram)
        res_b, res_p = self.cfg.resolution
        if bp.size == 0:
            return np.zeros(res_b * res_p, dtype=float)
        bmin = self.cfg.birth_range[0] if self.cfg.birth_range else float(bp[:,0].min())
        bmax = self.cfg.birth_range[1] if self.cfg.birth_range else float(bp[:,0].max())
        pmin = self.cfg.pers_range[0] if self.cfg.pers_range else float(bp[:,1].min())
        pmax = self.cfg.pers_range[1] if self.cfg.pers_range else float(bp[:,1].max())
        if bmax == bmin:
            bmax += 1e-6
        if pmax == pmin:
            pmax += 1e-6
        b_lin = np.linspace(bmin, bmax, res_b)
        p_lin = np.linspace(pmin, pmax, res_p)
        bb, pp = np.meshgrid(b_lin, p_lin, indexing='xy')
        img = np.zeros_like(bb)
        inv_2s2 = 1.0 / (2 * self.cfg.sigma * self.cfg.sigma)
        for birth, pers in bp:
            # Weight persistence stronger (common heuristic)
            w = pers
            img += w * np.exp(-((bb - birth)**2 + (pp - pers)**2) * inv_2s2)
        # Flatten row-major
        return img.astype(float).ravel()

# ---- Persistence Landscape ----
@dataclass
class PersistenceLandscapeConfig:
    resolution: int = 100
    k_layers: int = 5

class PersistenceLandscapeVectorizer:
    def __init__(self, cfg: PersistenceLandscapeConfig):
        self.cfg = cfg

    @staticmethod
    def _landscape_layer(values: List[float], k: int) -> float:
        if len(values) < k:
            return 0.0
        return sorted(values, reverse=True)[k-1]

    def transform_diagram(self, diagram: np.ndarray) -> np.ndarray:
        if diagram.size == 0:
            return np.zeros(self.cfg.resolution * self.cfg.k_layers)
        # remove infinite deaths
        finite = np.isfinite(diagram[:,1])
        diag = diagram[finite]
        if diag.size == 0:
            return np.zeros(self.cfg.resolution * self.cfg.k_layers)
        bmin = float(diag[:,0].min())
        dmax = float(diag[:,1].max())
        t = np.linspace(bmin, dmax, self.cfg.resolution)
        layers = []
        for ti in t:
            vals = []
            for birth, death in diag:
                if birth <= ti <= death:
                    vals.append(min(ti - birth, death - ti))
            # collect first k layers
            for k in range(1, self.cfg.k_layers+1):
                layers.append(self._landscape_layer(vals, k))
        return np.array(layers, dtype=float)

__all__ = [
    'PersistenceImageConfig', 'PersistenceImageVectorizer',
    'PersistenceLandscapeConfig', 'PersistenceLandscapeVectorizer'
]
