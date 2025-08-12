"""Manifest splitting utility for temporal train/test separation.

Converts unified diagram manifests into temporally-separated train/test manifests
while preserving metadata integrity and ensuring zero temporal leakage.

Key Features:
- Temporal boundary enforcement (train_max_time < test_min_time)
- Metadata preservation across splits
- Path resolution compatibility
- Label distribution reporting
- Configurable split ratios with timestamp-based fallback
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import numpy as np


@dataclass
class SplitConfig:
    """Configuration for manifest splitting."""
    train_ratio: float = 0.7
    min_test_samples: int = 50
    min_train_samples: int = 100
    temporal_gap_seconds: float = 0.0  # Enforce gap between train/test
    preserve_label_distribution: bool = True
    random_seed: int = 1337


class ManifestSplitter:
    """Handles temporal splitting of unified diagram manifests."""
    
    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()
        np.random.seed(self.config.random_seed)
    
    def split_manifest(self, 
                      manifest_path: Path, 
                      output_dir: Path, 
                      train_name: str = "train_manifest.json",
                      test_name: str = "test_manifest.json") -> Dict[str, Any]:
        """Split unified manifest into temporal train/test manifests.
        
        Parameters
        ----------
        manifest_path : Path
            Path to unified diagram manifest JSON
        output_dir : Path  
            Directory to write train/test manifests
        train_name : str
            Filename for training manifest
        test_name : str
            Filename for test manifest
            
        Returns
        -------
        Dict[str, Any]
            Split statistics and metadata
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        # Load unified manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        entries = manifest['entries']
        if len(entries) < self.config.min_train_samples + self.config.min_test_samples:
            raise ValueError(f"Insufficient entries ({len(entries)}) for minimum split requirements")
        
        # Sort by timestamp to ensure temporal ordering
        entries.sort(key=lambda x: x['start_time'])
        
        # Calculate split point
        split_idx = self._calculate_split_index(entries)
        
        # Apply temporal gap if configured
        if self.config.temporal_gap_seconds > 0:
            split_idx = self._apply_temporal_gap(entries, split_idx)
        
        # Split entries
        train_entries = entries[:split_idx]
        test_entries = entries[split_idx:]
        
        # Validate temporal integrity
        self._validate_temporal_split(train_entries, test_entries)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate split manifests
        train_manifest = self._create_split_manifest(
            manifest, train_entries, "train", manifest_path
        )
        test_manifest = self._create_split_manifest(
            manifest, test_entries, "test", manifest_path
        )
        
        # Write manifests
        train_path = output_dir / train_name
        test_path = output_dir / test_name
        
        with open(train_path, 'w') as f:
            json.dump(train_manifest, f, indent=2)
        
        with open(test_path, 'w') as f:
            json.dump(test_manifest, f, indent=2)
        
        # Generate split statistics
        stats = self._generate_split_stats(
            train_entries, test_entries, train_path, test_path, manifest_path
        )
        
        return stats
    
    def _calculate_split_index(self, entries: List[Dict[str, Any]]) -> int:
        """Calculate optimal split index based on configuration."""
        n_total = len(entries)
        
        # Simple ratio-based split
        ratio_idx = int(n_total * self.config.train_ratio)
        
        # Ensure minimum requirements
        min_train_idx = self.config.min_train_samples
        max_train_idx = n_total - self.config.min_test_samples
        
        # Clamp to valid range
        split_idx = max(min_train_idx, min(ratio_idx, max_train_idx))
        
        return split_idx
    
    def _apply_temporal_gap(self, entries: List[Dict[str, Any]], split_idx: int) -> int:
        """Apply temporal gap requirement by adjusting split index."""
        if split_idx >= len(entries) - 1:
            return split_idx
            
        train_max_time = entries[split_idx - 1]['start_time']
        
        # Find first test entry that satisfies gap requirement
        for i in range(split_idx, len(entries)):
            test_time = entries[i]['start_time']
            if test_time >= train_max_time + self.config.temporal_gap_seconds:
                return i
        
        # If no suitable gap found, use original split
        return split_idx
    
    def _validate_temporal_split(self, 
                                train_entries: List[Dict[str, Any]], 
                                test_entries: List[Dict[str, Any]]) -> None:
        """Validate temporal integrity of split."""
        if not train_entries or not test_entries:
            raise ValueError("Empty train or test split after temporal separation")
        
        train_max = max(entry['start_time'] for entry in train_entries)
        test_min = min(entry['start_time'] for entry in test_entries)
        
        if train_max >= test_min:
            raise ValueError(
                f"Temporal leakage detected: train_max_time ({train_max}) >= "
                f"test_min_time ({test_min})"
            )
    
    def _create_split_manifest(self, 
                              original_manifest: Dict[str, Any],
                              entries: List[Dict[str, Any]], 
                              split_name: str,
                              original_path: Path) -> Dict[str, Any]:
        """Create manifest for a specific split."""
        # Preserve original metadata with split-specific updates
        split_manifest = {
            "schema_version": original_manifest.get("schema_version", 1),
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "total_windows": len(entries),
            "config": original_manifest.get("config", {}),
            "original_manifest": str(original_path),
            "split_type": split_name,
            "entries": []
        }
        
        # Reindex entries and update diagram paths to be relative to manifest location
        for new_idx, entry in enumerate(entries):
            new_entry = entry.copy()
            new_entry['order'] = new_idx
            # Convert file path to be relative to validation/diagrams/
            if not new_entry['file'].startswith('validation/'):
                new_entry['diagram_path'] = f"validation/diagrams/{new_entry['file']}"
            else:
                new_entry['diagram_path'] = new_entry['file']
            split_manifest["entries"].append(new_entry)
        
        return split_manifest
    
    def _generate_split_stats(self, 
                             train_entries: List[Dict[str, Any]],
                             test_entries: List[Dict[str, Any]], 
                             train_path: Path,
                             test_path: Path,
                             original_path: Path) -> Dict[str, Any]:
        """Generate comprehensive split statistics."""
        
        # Basic counts
        n_train = len(train_entries)
        n_test = len(test_entries)
        n_total = n_train + n_test
        
        # Temporal boundaries
        train_times = [e['start_time'] for e in train_entries]
        test_times = [e['start_time'] for e in test_entries]
        
        # Label distribution
        train_labels = [e['label'] for e in train_entries]
        test_labels = [e['label'] for e in test_entries]
        
        train_label_counts = {}
        test_label_counts = {}
        
        for label in set(train_labels + test_labels):
            train_label_counts[str(label)] = train_labels.count(label)
            test_label_counts[str(label)] = test_labels.count(label)
        
        # Homology dimension statistics
        train_dims = {}
        test_dims = {}
        
        for entry in train_entries:
            for dim in entry.get('dims', []):
                train_dims[dim] = train_dims.get(dim, 0) + 1
                
        for entry in test_entries:
            for dim in entry.get('dims', []):
                test_dims[dim] = test_dims.get(dim, 0) + 1
        
        stats = {
            "split_timestamp": datetime.utcnow().isoformat() + "Z",
            "original_manifest": str(original_path),
            "train_manifest": str(train_path),
            "test_manifest": str(test_path),
            "configuration": {
                "train_ratio": self.config.train_ratio,
                "temporal_gap_seconds": self.config.temporal_gap_seconds,
                "random_seed": self.config.random_seed
            },
            "counts": {
                "total": n_total,
                "train": n_train,
                "test": n_test,
                "train_ratio_actual": n_train / n_total
            },
            "temporal_boundaries": {
                "train_time_range": [min(train_times), max(train_times)],
                "test_time_range": [min(test_times), max(test_times)],
                "temporal_gap": min(test_times) - max(train_times)
            },
            "label_distribution": {
                "train": train_label_counts,
                "test": test_label_counts
            },
            "homology_dimensions": {
                "train": train_dims,
                "test": test_dims
            },
            "integrity_checks": {
                "temporal_leakage": max(train_times) >= min(test_times),
                "min_samples_satisfied": n_train >= self.config.min_train_samples and n_test >= self.config.min_test_samples
            }
        }
        
        return stats


def split_existing_manifest(manifest_path: str | Path, 
                           output_dir: str | Path, 
                           config: Optional[SplitConfig] = None) -> Dict[str, Any]:
    """Convenience function to split an existing manifest.
    
    Parameters
    ----------
    manifest_path : str | Path
        Path to unified diagram manifest 
    output_dir : str | Path
        Directory for output train/test manifests
    config : SplitConfig, optional
        Split configuration (uses defaults if None)
        
    Returns
    -------
    Dict[str, Any]
        Split statistics and metadata
    """
    splitter = ManifestSplitter(config)
    return splitter.split_manifest(Path(manifest_path), Path(output_dir))


if __name__ == "__main__":
    # Example usage for existing diagram manifest
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python manifest_splitter.py <manifest_path> <output_dir> [train_ratio]")
        sys.exit(1)
    
    manifest_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    train_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    
    config = SplitConfig(train_ratio=train_ratio)
    stats = split_existing_manifest(manifest_path, output_dir, config)
    
    print("Split completed successfully!")
    print(f"Train samples: {stats['counts']['train']}")
    print(f"Test samples: {stats['counts']['test']}")
    print(f"Temporal gap: {stats['temporal_boundaries']['temporal_gap']:.2f} seconds")
    print(f"Output directory: {output_dir}")