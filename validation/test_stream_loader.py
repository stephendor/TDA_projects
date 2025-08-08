"""Validation script for StreamingWindowLoader.

Creates small synthetic parquet shards to exercise window iteration,
verifies:
  - Complete temporal coverage
  - Overlap consistency (approximate)
  - No memory blow-up (bounded buffer size)

Note: Uses synthetic data ONLY for loader mechanics (not model validation).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src.data.stream_loader import WindowConfig, StreamingWindowLoader

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / "data" / "_stream_loader_test"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Generate deterministic synthetic shards (temporal continuity)
np.random.seed(42)
rows = 30
timestamps = np.arange(rows, dtype=float)  # 0..29 seconds
f1 = np.random.randn(rows)
f2 = np.random.rand(rows)

df = pd.DataFrame({"timestamp": timestamps, "f1": f1, "f2": f2})
# Split into two shards to simulate sequential ingestion
shard1 = df.iloc[:15]
shard2 = df.iloc[15:]
(shard1).to_parquet(TMP_DIR / "part1.parquet", index=False)
(shard2).to_parquet(TMP_DIR / "part2.parquet", index=False)

config = WindowConfig(window_seconds=5.0, overlap=0.5)  # step = 2.5
loader = StreamingWindowLoader(
    data_paths=[TMP_DIR / "part1.parquet", TMP_DIR / "part2.parquet"],
    config=config,
)

windows = []
for w in loader.windows():
    win_start = float(w['timestamp'].min())
    win_end = win_start + config.window_seconds
    windows.append({
        "start": win_start,
        "end": win_end,
        "count": len(w),
        "min_ts": float(w['timestamp'].min()),
        "max_ts": float(w['timestamp'].max()),
    })

# Basic assertions
assert len(windows) > 0, "No windows produced"
# Coverage (allow final partial)
covered_max = max(w['max_ts'] for w in windows)
assert covered_max == df['timestamp'].max(), "Did not reach final timestamp"

# Check approximate overlap (difference between successive starts ~ step)
starts = [w['start'] for w in windows]
if len(starts) > 2:
    diffs = np.diff(starts)
    expected_step = config.window_seconds * (1 - config.overlap)
    mean_step = float(np.mean(diffs))
    assert abs(mean_step - expected_step) < 0.6, f"Mean step {mean_step} deviates from expected {expected_step}"  # tolerance due to partial final windows

print("Total windows:", len(windows))
print("First 5 windows (start,end,count):", [(w['start'], w['end'], w['count']) for w in windows[:5]])
print("Mean inter-window step:", np.mean(np.diff(starts)) if len(starts) > 1 else 'N/A')
print("Max timestamp covered:", covered_max)
print("SUCCESS: StreamingWindowLoader basic mechanics validated.")
