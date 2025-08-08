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

# ---- Betti Curves & Lifetime Stats (deterministic) ----
@dataclass
class BettiCurveConfig:
    radii_resolution: int = 50  # number of radius samples
    pad_epsilon: float = 1e-6   # ensure non-zero span
    maxdim: int = 2             # up to which homology dimension to compute curves

class BettiCurveVectorizer:
    """Generate Betti curves for each homology dimension 0..maxdim across a radius grid.
    Uses birth/death intervals directly from diagrams (no statistical proxies).
    Returns concatenated curves (dim-major order).
    """
    def __init__(self, cfg: BettiCurveConfig):
        self.cfg = cfg

    def transform(self, diagrams: Sequence[np.ndarray]) -> np.ndarray:
        # diagrams: list-like index = homology dimension
        maxdim = min(self.cfg.maxdim, len(diagrams)-1)
        # Determine global min birth and max death across used diagrams
        births = []
        deaths = []
        for d in diagrams[:maxdim+1]:
            if d is None or d.size == 0:
                continue
            finite = np.isfinite(d[:,1])
            if finite.any():
                births.append(d[finite,0])
                deaths.append(d[finite,1])
        if not births or not deaths:
            return np.zeros((maxdim+1) * self.cfg.radii_resolution, dtype=float)
        bmin = float(min(map(np.min, births)))
        dmax = float(max(map(np.max, deaths)))
        if dmax <= bmin:
            dmax = bmin + self.cfg.pad_epsilon
        radii = np.linspace(bmin, dmax, self.cfg.radii_resolution)
        curves = []
        for dim in range(maxdim+1):
            dgm = diagrams[dim] if dim < len(diagrams) else None
            if dgm is None or dgm.size == 0:
                curves.append(np.zeros(self.cfg.radii_resolution, dtype=float))
                continue
            finite = np.isfinite(dgm[:,1])
            dgm = dgm[finite]
            if dgm.size == 0:
                curves.append(np.zeros(self.cfg.radii_resolution, dtype=float))
                continue
            births_d = dgm[:,0]
            deaths_d = dgm[:,1]
            # Betti number at radius r = count of intervals where birth <= r < death
            betti_vals = [( (births_d <= r) & (deaths_d > r) ).sum() for r in radii]
            curves.append(np.array(betti_vals, dtype=float))
        return np.concatenate(curves, axis=0)

@dataclass
class LifetimeStatsConfig:
    top_k: int = 5        # capture top-K lifetimes per dimension
    maxdim: int = 2

class LifetimeStatsVectorizer:
    """Extract lifetime-based deterministic statistics strictly from diagrams:
    - Total persistence per dimension
    - Top-K lifetimes per dimension (zero-padded)
    - Persistence entropy per dimension (Shannon over normalized lifetimes)
    No raw data statistics used; all derived from birth/death.
    """
    def __init__(self, cfg: LifetimeStatsConfig):
        self.cfg = cfg

    @staticmethod
    def _lifetimes(diagram):  # accept None or ndarray
        if diagram is None or getattr(diagram, 'size', 0) == 0:
            return np.array([], dtype=float)
        finite = np.isfinite(diagram[:,1])
        if not finite.any():
            return np.array([], dtype=float)
        d = diagram[finite]
        return d[:,1] - d[:,0]

    def transform(self, diagrams: Sequence[np.ndarray]) -> np.ndarray:
        parts = []
        maxdim = min(self.cfg.maxdim, len(diagrams)-1)
        for dim in range(maxdim+1):
            dgm = diagrams[dim] if dim < len(diagrams) else None
            lifetimes = self._lifetimes(dgm)
            if lifetimes.size == 0:
                total_persistence = 0.0
                entropy = 0.0
                topk = np.zeros(self.cfg.top_k, dtype=float)
            else:
                total_persistence = float(lifetimes.sum())
                # Top-K lifetimes sorted descending
                sorted_lt = np.sort(lifetimes)[::-1]
                topk = np.zeros(self.cfg.top_k, dtype=float)
                take = min(self.cfg.top_k, sorted_lt.size)
                topk[:take] = sorted_lt[:take]
                # Persistence entropy
                norm = lifetimes / lifetimes.sum() if lifetimes.sum() > 0 else np.zeros_like(lifetimes)
                # Avoid log(0)
                nz = norm[norm > 0]
                entropy = float(-(nz * np.log(nz)).sum()) if nz.size > 0 else 0.0
            parts.extend([total_persistence, entropy])
            parts.extend(topk.tolist())
        return np.array(parts, dtype=float)

__all__ = [
    'PersistenceImageConfig', 'PersistenceImageVectorizer',
    'PersistenceLandscapeConfig', 'PersistenceLandscapeVectorizer',
    'BettiCurveConfig','BettiCurveVectorizer',
    'LifetimeStatsConfig','LifetimeStatsVectorizer'
]
