"""Deterministic vector stack feature construction for persistence diagrams.

Implements multiple stable topological liftings combined into a single
feature vector per sample (window). Only genuine topological transforms
are used. No statistical proxy features (means, std, etc.).

Blocks included:
 1. Persistence Landscapes (multi-resolution, levels L)
 2. Persistence Images (multi-scale grids, adaptive sigma)
 3. Betti Curves (sampled counts over filtration)
 4. Sliced Wasserstein Projections (fixed angle set)
 5. Static Kernel Dictionary Responses (multi-scale Gaussian dictionaries)

Silhouettes intentionally deferred for initial evaluation.

All computations are dimension-wise (H0, H1) and concatenated in a fixed
block order recorded in the manifest for reproducibility & ablation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Sequence, Any, Optional
import numpy as np
import math
import json
import hashlib
from pathlib import Path

# Optional: SciPy for KD-tree nearest neighbor for sigma estimation (fallback pure numpy if absent)
try:  # optional dependency
    from scipy import spatial as _scipy_spatial  # type: ignore
    _HAS_SCIPY = hasattr(_scipy_spatial, 'cKDTree')
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


@dataclass
class VectorStackConfig:
    homology_dims: Sequence[int] = (0, 1)
    landscape_levels: int = 3
    landscape_resolutions: Sequence[int] = (100, 300)
    image_grids: Sequence[Tuple[int, int]] = ((16, 16), (32, 32))
    betti_resolution: int = 300
    sw_num_angles: int = 32
    sw_resolution: int = 300
    kernel_small_k: int = 32
    kernel_large_k: int = 32
    kernel_sample_small: int = 10000
    kernel_sample_large: int = 20000
    random_seed: int = 1337
    max_points_per_block: int = 5000  # downsample cap (except Betti curves)
    intensity_normalize_images: bool = True
    clamp_sigma_min: float = 1e-3
    clamp_sigma_max_factor: float = 0.2  # fraction of (birth_range + life_range)
    use_log_lifetime_images: bool = False
    # New enable flags for ablations
    enable_landscapes: bool = True
    enable_images: bool = True
    enable_betti: bool = True
    enable_sw: bool = True
    enable_kernels: bool = True
    # New optional future block (silhouettes scaffold) - not yet implemented
    enable_silhouettes: bool = False  # placeholder (Task 20 automation)
    # Verbosity / progress tracking
    verbose: bool = False


# ----------------------- Utility & Determinism ----------------------- #

def _set_rng(seed: int):
    np.random.seed(seed)


def _log(msg: str, cfg: Optional[VectorStackConfig]):  # lightweight logger
    if cfg is not None and getattr(cfg, 'verbose', False):
        print(f"[VectorStack] {msg}")


def _hash_manifest(items: Sequence[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode('utf-8'))
    return h.hexdigest()


# ----------------------- Diagram Preprocessing ----------------------- #

def _sanitize_diagram(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return dgm.reshape(0, 2)
    if dgm.shape[1] != 2:
        raise ValueError("Diagram must have shape (n,2)")
    mask = np.isfinite(dgm).all(axis=1)
    dgm = dgm[mask]
    # Remove death < birth anomalies
    mask2 = dgm[:, 1] >= dgm[:, 0]
    return dgm[mask2]


def _diagram_to_birth_lifetime(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return dgm.reshape(0, 2)
    life = np.maximum(0.0, dgm[:, 1] - dgm[:, 0])
    return np.column_stack([dgm[:, 0], life])


# ----------------------- Block: Persistence Landscapes ----------------------- #

def _compute_landscapes(dgm: np.ndarray, levels: int, resolutions: Sequence[int]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if dgm.size == 0:
        for r in resolutions:
            out[f"landscape_r{r}"] = np.zeros((levels, r), dtype=float)
        return out
    finite = dgm[np.isfinite(dgm[:, 1])]
    if finite.size == 0:
        for r in resolutions:
            out[f"landscape_r{r}"] = np.zeros((levels, r), dtype=float)
        return out
    births = finite[:, 0]
    deaths = finite[:, 1]
    min_b = births.min()
    max_d = deaths.max()
    if max_d <= min_b:
        max_d = min_b + 1e-6
    for r in resolutions:
        grid = np.linspace(min_b, max_d, r)
        # For each grid point compute triangle heights, then take kth largest per level
        triangles = []  # list per point of values along grid
        # More efficient vectorization: compute all contributions
        # For each point define lambda(t)=max(0, min(t-b, d-t)) for t in [b,d]
        values = np.zeros((finite.shape[0], r), dtype=float)
        for i, (b, d) in enumerate(finite):
            mask = (grid >= b) & (grid <= d)
            if not mask.any():
                continue
            t_segment = grid[mask]
            vals = np.minimum(t_segment - b, d - t_segment)
            values[i, mask] = vals
        # Now for each level k take kth order statistic across points at each grid location
        lvl_stack = np.zeros((levels, r), dtype=float)
        # Sort descending along axis 0 per column (points) - use partial sort for efficiency if needed
        # Simpler: we gather non-zero contributions per column
        for j in range(r):
            col = values[:, j]
            nz = col[col > 0]
            if nz.size == 0:
                continue
            nz_sorted = np.sort(nz)[::-1]
            upto = min(levels, nz_sorted.size)
            lvl_stack[:upto, j] = nz_sorted[:upto]
        out[f"landscape_r{r}"] = lvl_stack
    return out


# ----------------------- Block: Persistence Images ----------------------- #

def _adaptive_sigma(birth_life: np.ndarray, clamp_min: float, clamp_max: float) -> float:
    if birth_life.size == 0:
        return clamp_min
    life = birth_life[:, 1]
    finite = life[np.isfinite(life)]
    if finite.size == 0:
        return clamp_min
    med = np.median(finite)
    sigma = 0.5 * med
    sigma = max(clamp_min, min(sigma, clamp_max))
    return float(sigma)


def _compute_persistence_images(dgm: np.ndarray, grids: Sequence[Tuple[int, int]], cfg: VectorStackConfig,
                                birth_range: Tuple[float, float], life_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if dgm.size == 0:
        for (gx, gy) in grids:
            out[f"pimg_{gx}x{gy}"] = np.zeros((gx, gy), dtype=float)
        return out
    bl = _diagram_to_birth_lifetime(dgm)
    if cfg.use_log_lifetime_images:
        bl[:, 1] = np.log1p(bl[:, 1])
    bmin, bmax = birth_range
    lmin, lmax = life_range
    if bmax <= bmin:
        bmax = bmin + 1e-6
    if lmax <= lmin:
        lmax = lmin + 1e-6
    clamp_max_sigma = cfg.clamp_sigma_max_factor * ((bmax - bmin) + (lmax - lmin))
    sigma = _adaptive_sigma(bl, cfg.clamp_sigma_min, clamp_max_sigma)
    for (gx, gy) in grids:
        img = np.zeros((gx, gy), dtype=float)
        # Grid edges
        xs = np.linspace(bmin, bmax, gx)
        ys = np.linspace(lmin, lmax, gy)
        # Precompute gaussian normalization 1/(2*pi*sigma^2)
        inv = 1.0 / (2.0 * sigma * sigma)
        for (b, l) in bl:
            # Evaluate contribution on grid (vectorized per point)
            dx2 = (xs - b) ** 2
            dy2 = (ys - l) ** 2
            # outer add via broadcasting
            contrib = np.exp(-(dx2[:, None] + dy2[None, :]) * inv)
            img += contrib
        if cfg.intensity_normalize_images and img.sum() > 0:
            img = img / img.sum()
        out[f"pimg_{gx}x{gy}"] = img
    return out


# ----------------------- Block: Betti Curves ----------------------- #

def _compute_betti_curve(dgm: np.ndarray, resolution: int) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros(resolution, dtype=float)
    finite = dgm[np.isfinite(dgm[:, 1])]
    if finite.size == 0:
        return np.zeros(resolution, dtype=float)
    births = finite[:, 0]
    deaths = finite[:, 1]
    min_b = births.min()
    max_d = deaths.max()
    if max_d <= min_b:
        max_d = min_b + 1e-6
    grid = np.linspace(min_b, max_d, resolution)
    counts = np.zeros(resolution, dtype=int)
    # Sweep-line style: for each point add interval
    for (b, d) in finite:
        mask = (grid >= b) & (grid <= d)
        counts[mask] += 1
    return counts.astype(float)


# ----------------------- Block: Sliced Wasserstein Projections ----------------------- #

def _generate_angles(m: int) -> np.ndarray:
    # Golden angle sequence for deterministic coverage
    angles = np.zeros(m, dtype=float)
    golden = math.pi * (3 - math.sqrt(5))
    for i in range(m):
        angles[i] = (i * golden) % (2 * math.pi)
    return angles


def _compute_sliced_wasserstein(dgm: np.ndarray, num_angles: int, resolution: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if dgm.size == 0:
        out['sw'] = np.zeros((num_angles, resolution), dtype=float)
        return out
    finite = dgm[np.isfinite(dgm[:, 1])]
    if finite.size == 0:
        out['sw'] = np.zeros((num_angles, resolution), dtype=float)
        return out
    births = finite[:, 0]
    deaths = finite[:, 1]
    min_b = births.min()
    max_d = deaths.max()
    if max_d <= min_b:
        max_d = min_b + 1e-6
    grid = np.linspace(min_b, max_d, resolution)
    angles = _generate_angles(num_angles)
    proj_stack = np.zeros((num_angles, resolution), dtype=float)
    pts = finite
    for ai, ang in enumerate(angles):
        c = math.cos(ang)
        s = math.sin(ang)
        proj = pts @ np.array([[c, 0.0], [0.0, s]])  # birth*c, death*s
        # collapse to 1D signature: project births and deaths separately then combine by sorting
        v = np.sort(np.concatenate([proj[:, 0], proj[:, 1]]))
        if v.size == 0:
            continue
        # Interpolate histogram-like curve over grid
        # Map v range to grid range
        vmin, vmax = v[0], v[-1]
        if vmax <= vmin:
            vmax = vmin + 1e-6
        # compute CDF style curve
        counts = np.searchsorted(v, grid, side='right') / v.size
        proj_stack[ai] = counts
    out['sw'] = proj_stack
    return out


# ----------------------- Block: Kernel Dictionary Responses ----------------------- #

def _sample_points(diagrams: List[np.ndarray], total: int) -> np.ndarray:
    collected = []
    for dgm in diagrams:
        if dgm.size == 0:
            continue
        pts = _diagram_to_birth_lifetime(_sanitize_diagram(dgm))
        collected.append(pts)
    if not collected:
        return np.zeros((0, 2), dtype=float)
    all_pts = np.vstack(collected)
    if all_pts.shape[0] <= total:
        return all_pts
    idx = np.random.choice(all_pts.shape[0], size=total, replace=False)
    return all_pts[idx]


def _estimate_sigma_from_points(pts: np.ndarray, k: int = 8) -> float:
    if pts.shape[0] < 2:
        return 0.1
    if _HAS_SCIPY and pts.shape[0] >= k:
        tree = _scipy_spatial.cKDTree(pts)  # type: ignore[attr-defined]
        dists, _ = tree.query(pts, k=k)
        # dists shape (n,k); ignore self-distance at col0
        vals = dists[:, 1:].reshape(-1)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return 0.1
        return float(np.median(vals))
    # Fallback: pairwise subset
    n = pts.shape[0]
    take = min(n, 500)
    idx = np.random.choice(n, take, replace=False)
    sub = pts[idx]
    d2 = ((sub[None, :, :] - sub[:, None, :]) ** 2).sum(axis=2)
    d2 = d2[np.triu_indices(sub.shape[0], k=1)]
    d = np.sqrt(np.maximum(d2, 0))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 0.1
    return float(np.median(d))


def prepare_kernel_dictionaries(train_diagrams_by_dim: Dict[int, List[np.ndarray]], cfg: VectorStackConfig,
                                 out_path: Optional[Path] = None) -> Dict[str, Any]:
    _set_rng(cfg.random_seed)
    dicts: Dict[str, Any] = {}
    _log("Preparing kernel dictionaries", cfg)
    for scale_name, k, sample_n in [
        ("small", cfg.kernel_small_k, cfg.kernel_sample_small),
        ("large", cfg.kernel_large_k, cfg.kernel_sample_large),
    ]:
        diagrams = []
        for dim in cfg.homology_dims:
            diagrams.extend(train_diagrams_by_dim.get(dim, []))
        _log(f"Sampling up to {sample_n} points for '{scale_name}' scale", cfg)
        pts = _sample_points(diagrams, sample_n)
        if pts.shape[0] == 0:
            centers = np.zeros((k, 2), dtype=float)
            sigma_base = 0.1
            _log(f"No points available for '{scale_name}', creating zero centers", cfg)
        else:
            centers = np.zeros((k, 2), dtype=float)
            # k-means++ deterministic
            # pick first center
            idx0 = np.random.randint(0, pts.shape[0])
            centers[0] = pts[idx0]
            # distances
            d2 = ((pts - centers[0]) ** 2).sum(axis=1)
            for i in range(1, k):
                probs = d2 / d2.sum() if d2.sum() > 0 else np.ones_like(d2) / d2.size
                c_idx = np.random.choice(pts.shape[0], p=probs)
                centers[i] = pts[c_idx]
                new_d2 = ((pts - centers[i]) ** 2).sum(axis=1)
                d2 = np.minimum(d2, new_d2)
            sigma_base = _estimate_sigma_from_points(pts)
            _log(f"Built {k} centers for '{scale_name}' (sigma_base≈{sigma_base:.4f})", cfg)
        dicts[f"kernel_{scale_name}"] = {
            'centers': centers,
            'sigma_base': sigma_base,
        }
    if out_path is not None:
        _log(f"Persisting kernel dictionaries to {out_path}", cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **{k: v['centers'] for k, v in dicts.items()})
        with open(out_path.with_suffix('.json'), 'w') as f:
            json.dump({k: {'sigma_base': v['sigma_base']} for k, v in dicts.items()}, f, indent=2)
    _log("Kernel dictionaries ready", cfg)
    return dicts


def _kernel_responses(dgm: np.ndarray, centers: np.ndarray, sigma: float, max_points: int) -> np.ndarray:
    if dgm.size == 0 or centers.size == 0:
        return np.zeros(centers.shape[0], dtype=float)
    pts = _diagram_to_birth_lifetime(_sanitize_diagram(dgm))
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    # Compute gaussian responses
    # (n,2) vs (k,2)
    diff = pts[:, None, :] - centers[None, :, :]
    d2 = (diff ** 2).sum(axis=2)
    sigma2 = (sigma ** 2)
    vals = np.exp(-d2 / (2 * sigma2))
    # Average pooling for scale invariance w.r.t number of points
    resp = vals.mean(axis=0)
    return resp


# ----------------------- Complete Block Feature Assembly ----------------------- #

def compute_block_features(diagrams_by_dim: Dict[int, np.ndarray], cfg: VectorStackConfig,
                           kernel_dicts: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
    blocks: Dict[str, np.ndarray] = {}
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    _log("Beginning block feature assembly", cfg)
    for dim in cfg.homology_dims:
        dgm = _sanitize_diagram(diagrams_by_dim.get(dim, np.zeros((0, 2), dtype=float)))
        _log(f"Dim H{dim}: {dgm.shape[0]} points", cfg)
        # Landscapes
        if cfg.enable_landscapes:
            lans = _compute_landscapes(dgm, cfg.landscape_levels, cfg.landscape_resolutions)
            for name, arr in lans.items():
                flat = arr.reshape(-1)
                key = f"H{dim}_{name}"
                blocks[key] = flat
                spans[key] = (cursor, cursor + flat.size)
                cursor += flat.size
                _log(f"Added {key} size={flat.size} (cursor={cursor})", cfg)
        # Persistence Images
        if cfg.enable_images:
            finite = dgm[np.isfinite(dgm[:, 1])] if dgm.size > 0 else dgm
            if finite.size > 0:
                births = finite[:, 0]
                deaths = finite[:, 1]
                birth_range = (births.min(), births.max())
                life_vals = np.maximum(0.0, deaths - births)
                life_range = (life_vals.min(), life_vals.max())
            else:
                birth_range = (0.0, 1.0)
                life_range = (0.0, 1.0)
            pimgs = _compute_persistence_images(dgm, cfg.image_grids, cfg, birth_range, life_range)
            for name, arr in pimgs.items():
                flat = arr.reshape(-1)
                key = f"H{dim}_{name}"
                blocks[key] = flat
                spans[key] = (cursor, cursor + flat.size)
                cursor += flat.size
                _log(f"Added {key} size={flat.size} (cursor={cursor})", cfg)
        # Betti curve
        if cfg.enable_betti:
            betti = _compute_betti_curve(dgm, cfg.betti_resolution)
            key_betti = f"H{dim}_betti"
            blocks[key_betti] = betti
            spans[key_betti] = (cursor, cursor + betti.size)
            cursor += betti.size
            _log(f"Added {key_betti} size={betti.size} (cursor={cursor})", cfg)
        # SW projections
        if cfg.enable_sw:
            sw = _compute_sliced_wasserstein(dgm, cfg.sw_num_angles, cfg.sw_resolution)['sw']
            key_sw = f"H{dim}_sw"
            blocks[key_sw] = sw.reshape(-1)
            spans[key_sw] = (cursor, cursor + sw.size)
            cursor += sw.size
            _log(f"Added {key_sw} size={sw.size} (cursor={cursor})", cfg)
        # Optional silhouettes (not implemented yet)
        if getattr(cfg, 'enable_silhouettes', False):
            _log("Silhouette block requested but not yet implemented; skipping (placeholder)", cfg)
        # Kernel dictionaries
        if cfg.enable_kernels:
            for scale in ["kernel_small", "kernel_large"]:
                centers = kernel_dicts[scale]['centers']
                sigma_base = kernel_dicts[scale]['sigma_base']
                resp = _kernel_responses(dgm, centers, sigma_base, cfg.max_points_per_block)
                key_k = f"H{dim}_{scale}"
                blocks[key_k] = resp
                spans[key_k] = (cursor, cursor + resp.size)
                cursor += resp.size
                _log(f"Added {key_k} size={resp.size} (cursor={cursor})", cfg)
    _log(f"Completed assembly. Total feature length={cursor}", cfg)
    return blocks, spans


def stack_and_normalize(blocks: Dict[str, np.ndarray], spans: Dict[str, Tuple[int, int]],
                        norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
                        fit: bool = False, cfg: Optional[VectorStackConfig] = None) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    # Compute / apply per-block z-score using train-only stats
    updated_stats: Dict[str, Dict[str, float]] = {} if fit else (norm_stats or {})
    vector_parts = []
    for key, arr in blocks.items():
        if fit:
            mean = float(arr.mean()) if arr.size > 0 else 0.0
            std = float(arr.std()) if arr.std() > 0 else 1.0
            updated_stats[key] = {'mean': mean, 'std': std}
            _log(f"Fit norm stats for {key}: mean={mean:.4e} std={std:.4e}", cfg)
        stats = updated_stats[key]
        std = stats['std'] if stats['std'] > 0 else 1.0
        normed = (arr - stats['mean']) / std
        vector_parts.append(normed)
    flat = np.concatenate(vector_parts) if vector_parts else np.zeros(0, dtype=float)
    _log(f"Stacked vector length={flat.size}", cfg)
    return flat, updated_stats


def build_vector_stack(diagrams_by_dim: Dict[int, np.ndarray], cfg: VectorStackConfig,
                       kernel_dicts: Dict[str, Any], norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
                       fit_norm: bool = False) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], Dict[str, Tuple[int, int]]]:
    blocks, spans = compute_block_features(diagrams_by_dim, cfg, kernel_dicts)
    vec, stats = stack_and_normalize(blocks, spans, norm_stats, fit=fit_norm, cfg=cfg)
    return vec, stats, spans


__all__ = [
    'VectorStackConfig',
    'prepare_kernel_dictionaries',
    'compute_block_features',
    'stack_and_normalize',
    'build_vector_stack',
]
