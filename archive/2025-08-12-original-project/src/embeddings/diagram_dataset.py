"""Dataset utilities for loading persisted persistence diagrams for learnable embeddings.

Uses diagram manifest produced by `validation/baseline_gbt_training.py` when DIAGRAM_DUMP_DIR is enabled.
Implements:
  * DiagramWindowRecord dataclass metadata
  * DiagramWindowDataset for PyTorch style iteration (lazy load npz)
  * Collation helper to pad / truncate diagrams per homology dimension

Topological Integrity:
  - Uses raw birth/death pairs (no statistical proxies)
  - Provides optional lifetime sort & truncation (same as original dump cap)
  - Does NOT compute new statistics; only prepares tensors for embedding model

Reproducibility:
  - Stores config snapshot in dataset instance
  - Deterministic ordering = manifest order
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Tuple
import numpy as np

@dataclass
class DiagramWindowRecord:
    order: int
    start_time: float
    label: int
    file: str
    dims: List[str]
    counts: Dict[str, int]
    used_witness: bool

class DiagramWindowDataset:
    """Lightweight iterable over dumped diagram .npz files.

    Parameters
    ----------
    manifest_path : Path
        Path to diagram_manifest_*.json
    root_dir : Path
        Directory containing the .npz files (defaults to manifest parent)
    max_points_per_dim : int | None
        Optional cap (re-applied) per homology dimension. If None uses manifest config
    dims : Sequence[int] | None
        Restrict to subset of homology dimensions (e.g., [0,1])
    sort_by_lifetime : bool
        Sort points descending by lifetime before truncation
    as_float32 : bool
        Cast arrays to float32 for torch efficiency
    """
    def __init__(self,
                 manifest_path: Path,
                 root_dir: Optional[Path] = None,
                 max_points_per_dim: Optional[int] = None,
                 dims: Optional[Sequence[int]] = None,
                 sort_by_lifetime: bool = True,
                 as_float32: bool = True):
        self.manifest_path = Path(manifest_path)
        if root_dir is None:
            root_dir = self.manifest_path.parent
        self.root_dir = Path(root_dir)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with self.manifest_path.open('r') as f:
            man = json.load(f)
        self.raw_manifest = man
        self.records: List[DiagramWindowRecord] = [
            DiagramWindowRecord(**entry) for entry in man.get('entries', [])
        ]
        self.config = man.get('config', {})
        self._base_cap = int(self.config.get('DIAGRAM_MAX_POINTS_PER_DIM', 0)) or None
        self.cap = max_points_per_dim if max_points_per_dim is not None else self._base_cap
        self.dims = list(dims) if dims is not None else None
        self.sort_by_lifetime = sort_by_lifetime
        self.as_float32 = as_float32

    def __len__(self) -> int:
        return len(self.records)

    def _load_npz(self, rec: DiagramWindowRecord) -> Dict[str, np.ndarray]:
        fpath = self.root_dir / rec.file
        if not fpath.exists():
            raise FileNotFoundError(f"Diagram npz missing: {fpath}")
        try:
            data = np.load(fpath)
        except Exception as e:
            raise RuntimeError(f"Failed to load {fpath}: {e}")
        return {k: data[k] for k in data.files}

    @staticmethod
    def _truncate(points: np.ndarray, cap: Optional[int], sort_by_lifetime: bool) -> np.ndarray:
        if points is None or points.size == 0:
            return points.reshape(0,2)
        if cap is None or points.shape[0] <= cap:
            return points
        if sort_by_lifetime:
            life = points[:,1] - points[:,0]
            idx = np.argsort(life)[::-1][:cap]
            return points[idx]
        return points[:cap]

    def get(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        arrs = self._load_npz(rec)
        sample: Dict[str, Any] = {
            'order': rec.order,
            'start_time': rec.start_time,
            'label': rec.label,
            'used_witness': rec.used_witness,
            'diagrams': {}
        }
        for k, v in arrs.items():
            if not k.startswith('dgm_H'):
                continue
            dim = int(k.split('H')[1])
            if self.dims is not None and dim not in self.dims:
                continue
            pts = v.astype(np.float32) if self.as_float32 else v
            pts = self._truncate(pts, self.cap, self.sort_by_lifetime)
            sample['diagrams'][dim] = pts
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get(idx)

    def iter_batches(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            batch = [self.get(i) for i in range(start, min(start+batch_size, len(self)))]
            yield self.collate(batch)

    @staticmethod
    def collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Determine dims present
        dims = sorted({d for s in samples for d in s['diagrams'].keys()})
        batch = {
            'orders': np.array([s['order'] for s in samples]),
            'start_times': np.array([s['start_time'] for s in samples]),
            'labels': np.array([s['label'] for s in samples], dtype=int),
            'used_witness': np.array([s['used_witness'] for s in samples], dtype=bool),
            'diagrams': {},
            'counts': {}
        }
        for d in dims:
            max_pts = max(s['diagrams'][d].shape[0] if d in s['diagrams'] else 0 for s in samples)
            # Build padded array shape (B, max_pts, 2)
            arr = np.zeros((len(samples), max_pts, 2), dtype=np.float32)
            counts = []
            for i, s in enumerate(samples):
                if d in s['diagrams']:
                    pts = s['diagrams'][d]
                    arr[i, :pts.shape[0], :] = pts
                    counts.append(pts.shape[0])
                else:
                    counts.append(0)
            batch['diagrams'][d] = arr
            batch['counts'][d] = np.array(counts, dtype=int)
        return batch

__all__ = [
    'DiagramWindowRecord',
    'DiagramWindowDataset'
]
