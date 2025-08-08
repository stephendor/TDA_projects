"""Topological visualization module.

Generates visual artifacts for a small sample of streamed windows:
  * Persistence diagrams (H0, H1, ...)
  * Betti curves (per homology dimension)
  * Persistence images (H0..H1) reshaped grids
  * Persistence landscapes (H0..H1, k layers)

STRICT TOPOLOGY ONLY:
Uses existing infrastructure (NO statistical proxy features):
  - TDAWindowVectorizer (builds VR / witness diagrams)
  - PersistenceImageVectorizer
  - PersistenceLandscapeVectorizer
  - BettiCurveVectorizer
  - LifetimeStatsVectorizer (not visualized directly, but available)

The script re-streams a limited number of windows to avoid heavy runtime.
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Sequence

from src.data.stream_loader import WindowConfig, StreamingWindowLoader, build_point_cloud
from src.tda.vectorizers.window_tda_vectorizer import TDAWindowVectorizer, TDAWindowVectorizerConfig
from src.tda.vectorizers.diagram_vectorizers import (
    PersistenceImageConfig, PersistenceImageVectorizer,
    PersistenceLandscapeConfig, PersistenceLandscapeVectorizer,
    BettiCurveConfig, BettiCurveVectorizer,
    LifetimeStatsConfig, LifetimeStatsVectorizer
)

try:
    import persim  # for diagram plotting
except ImportError:  # pragma: no cover
    persim = None

# ---------------- Configuration ----------------
CONFIG_PATH = Path('configs/dataset_cic.yaml')
if not CONFIG_PATH.exists():
    raise SystemExit("Dataset config not found: configs/dataset_cic.yaml")
with CONFIG_PATH.open('r') as f:
    ds_cfg = yaml.safe_load(f)

OUTPUT_DIR = Path(os.getenv('TOPO_VIZ_DIR', 'validation/visualizations'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WINDOWS = int(os.getenv('VIZ_MAX_WINDOWS', '5'))  # number of windows to visualize
INCLUDE_LANDSCAPE = os.getenv('VIZ_INCLUDE_LANDSCAPE', '1') == '1'
INCLUDE_IMAGES = os.getenv('VIZ_INCLUDE_IMAGES', '1') == '1'
BETTI_RES = int(os.getenv('VIZ_BETTI_RES', '40'))
LAND_RES = int(os.getenv('VIZ_LAND_RES', '60'))
LAND_K = int(os.getenv('VIZ_LAND_K', '3'))
PIMG_RES = int(os.getenv('VIZ_PIMG_RES', '16'))
PIMG_SIGMA = float(os.getenv('VIZ_PIMG_SIGMA', '0.05'))
MAX_VR_POINTS = min(500, ds_cfg['limits']['max_vr_points'])

# ---------------- Initialize Vectorizers ----------------
vec = TDAWindowVectorizer(TDAWindowVectorizerConfig(
    feature_columns=ds_cfg['features']['numeric'],
    max_vr_points=MAX_VR_POINTS,
    witness_landmarks=min(64, ds_cfg['limits']['witness_landmarks'])
))
_pimg = PersistenceImageVectorizer(PersistenceImageConfig(resolution=(PIMG_RES, PIMG_RES), sigma=PIMG_SIGMA))
_pland = PersistenceLandscapeVectorizer(PersistenceLandscapeConfig(resolution=LAND_RES, k_layers=LAND_K))
_betti = BettiCurveVectorizer(BettiCurveConfig(radii_resolution=BETTI_RES, maxdim=2))
_life = LifetimeStatsVectorizer(LifetimeStatsConfig(top_k=5, maxdim=2))

# ---------------- Helpers ----------------

def _collect_sample_windows() -> list[dict]:
    root_path = Path(ds_cfg['root_path'])
    file_pattern = ds_cfg['file_pattern']
    if not root_path.exists():
        raise SystemExit(f"Data root does not exist: {root_path}")
    parquet_files = sorted(root_path.glob(file_pattern))
    if len(parquet_files) == 0:
        raise SystemExit(f"No parquet files matching pattern {file_pattern} under {root_path}")

    win_cfg = WindowConfig(
        window_seconds=ds_cfg['window']['seconds'],
        overlap=ds_cfg['window']['overlap'],
        time_column=ds_cfg['time_column']
    )

    samples = []
    for w_idx, window_df in enumerate(StreamingWindowLoader(parquet_files, win_cfg).windows()):
        if 'Label' not in window_df.columns:
            continue
        # Clean features
        feats = window_df[ds_cfg['features']['numeric']].replace([np.inf, -np.inf], np.nan).dropna()
        if feats.shape[0] < 3:
            continue
        feats = feats.clip(lower=-1e12, upper=1e12)
        joined = feats.join(window_df[[win_cfg.time_column]])
        pc = build_point_cloud(
            window=joined,
            feature_columns=ds_cfg['features']['numeric'],
            cap_points=ds_cfg['limits']['max_vr_points'],
            seed=ds_cfg.get('random_seed', 42)
        )
        if pc.shape[0] < 3:
            continue
        # Compute diagrams using VR (fallback witness if cap exceeded handled externally if needed)
        if pc.shape[0] <= vec.cfg.max_vr_points:
            res = vec.vr.compute(pc)
        else:
            res = vec.witness.compute(pc)
        diagrams = res['diagrams']
        samples.append({
            'index': w_idx,
            'diagrams': diagrams,
            'raw_points': pc,
            'label': int(window_df['Label'].value_counts().idxmax()),
            'start_time': float(window_df[win_cfg.time_column].min())
        })
        if len(samples) >= MAX_WINDOWS:
            break
    return samples


def _betti_axis_and_curves(diagrams: Sequence[np.ndarray], radii_resolution: int) -> tuple[np.ndarray, list[np.ndarray]]:
    # Reproduce internal BettiCurveVectorizer axis for plotting (pure topology: birth/death only)
    maxdim = min(2, len(diagrams)-1)
    births, deaths = [], []
    for d in diagrams[:maxdim+1]:
        if d is None or getattr(d, 'size', 0) == 0:
            continue
        finite = np.isfinite(d[:,1])
        if finite.any():
            births.append(d[finite,0])
            deaths.append(d[finite,1])
    if not births or not deaths:
        radii = np.linspace(0.0, 1.0, radii_resolution)
        return radii, [np.zeros(radii_resolution) for _ in range(maxdim+1)]
    bmin = float(min(map(np.min, births)))
    dmax = float(max(map(np.max, deaths)))
    if dmax <= bmin:
        dmax = bmin + 1e-6
    radii = np.linspace(bmin, dmax, radii_resolution)
    curves = []
    for dim in range(maxdim+1):
        dgm = diagrams[dim] if dim < len(diagrams) else None
        if dgm is None or getattr(dgm, 'size', 0) == 0:
            curves.append(np.zeros(radii_resolution))
            continue
        finite = np.isfinite(dgm[:,1])
        dgm = dgm[finite]
        if dgm.size == 0:
            curves.append(np.zeros(radii_resolution))
            continue
        births_d = dgm[:,0]
        deaths_d = dgm[:,1]
        betti_vals = [(((births_d <= r) & (deaths_d > r)).sum()) for r in radii]
        curves.append(np.array(betti_vals, dtype=float))
    return radii, curves


def _plot_persistence_diagrams(diagrams: Sequence[np.ndarray], out_path: Path):
    if persim is None:
        print("WARNING: persim not installed; skipping diagram plot.")
        return
    # Collect up to first 3 dims
    diag_list = []
    labels = []
    for dim, dgm in enumerate(diagrams[:3]):
        if dgm is None:
            dgm = np.empty((0,2))
        diag_list.append(dgm)
        labels.append(f"H{dim}")
    plt.figure(figsize=(4,4))
    try:
        persim.plot_diagrams(diag_list, show=False, xy_range=None, labels=labels)
        plt.title('Persistence Diagrams')
        plt.tight_layout()
        plt.savefig(out_path)
    finally:
        plt.close()


def _plot_betti_curves(diagrams: Sequence[np.ndarray], out_path: Path):
    radii, curves = _betti_axis_and_curves(diagrams, BETTI_RES)
    plt.figure(figsize=(6,4))
    for dim, c in enumerate(curves):
        plt.plot(radii, c, label=f"H{dim}")
    plt.xlabel('Radius / Filtration')
    plt.ylabel('Betti Number')
    plt.title('Betti Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_persistence_images(diagrams: Sequence[np.ndarray], base_path: Path):
    if not INCLUDE_IMAGES:
        return
    for dim in range(min(2, len(diagrams))):
        dgm = diagrams[dim]
        if dgm is None or getattr(dgm, 'size', 0) == 0:
            img = np.zeros((PIMG_RES, PIMG_RES))
        else:
            vec_img = _pimg.transform_diagram(dgm)
            img = vec_img.reshape(PIMG_RES, PIMG_RES)
        plt.figure(figsize=(3,3))
        plt.imshow(img, origin='lower', cmap='magma')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f'Persistence Image H{dim}')
        plt.tight_layout()
        plt.savefig(base_path / f"pimage_H{dim}.png")
        plt.close()


def _plot_persistence_landscape(diagrams: Sequence[np.ndarray], out_path: Path):
    if not INCLUDE_LANDSCAPE:
        return
    # Combine H0 and H1 landscapes side-by-side layers overlay
    plt.figure(figsize=(6,4))
    for dim in range(min(2, len(diagrams))):
        dgm = diagrams[dim]
        if dgm is None or getattr(dgm, 'size', 0) == 0:
            land = np.zeros(LAND_RES * LAND_K)
        else:
            land = _pland.transform_diagram(dgm)
        # land vector is resolution * k_layers sequentially
        for k in range(LAND_K):
            segment = land[k*LAND_RES:(k+1)*LAND_RES]
            if segment.size == 0:
                continue
            plt.plot(range(LAND_RES), segment, label=f"H{dim}-L{k+1}")
    plt.xlabel('Discretized t')
    plt.ylabel('Landscape value')
    plt.title('Persistence Landscapes (layers)')
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def visualize():
    samples = _collect_sample_windows()
    if not samples:
        print('No windows collected for visualization.')
        return
    meta = []
    for i, s in enumerate(samples):
        stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        win_dir = OUTPUT_DIR / f"window_{i:03d}_{stamp}"
        win_dir.mkdir(parents=True, exist_ok=True)
        diagrams = s['diagrams']
        # Core diagram plot
        _plot_persistence_diagrams(diagrams, win_dir / 'diagrams.png')
        # Betti curves
        _plot_betti_curves(diagrams, win_dir / 'betti_curves.png')
        # Persistence images
        _plot_persistence_images(diagrams, win_dir)
        # Persistence landscapes
        _plot_persistence_landscape(diagrams, win_dir / 'landscapes.png')
        # Lifetime stats (numeric summary only)
        life_vec = _life.transform(diagrams)
        meta.append({
            'window_index': s['index'],
            'label': s['label'],
            'start_time': s['start_time'],
            'lifetime_stats_vector': life_vec.tolist(),
            'output_dir': str(win_dir)
        })
    # Persist metadata
    import json
    meta_path = OUTPUT_DIR / f"viz_metadata_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with meta_path.open('w') as mf:
        json.dump({'samples': meta}, mf, indent=2)
    print(f"Visualization written for {len(samples)} windows -> {OUTPUT_DIR}")


if __name__ == '__main__':
    visualize()
