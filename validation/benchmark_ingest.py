"""Benchmark ingestion & window generation for CIC config.
Produces CSV with per-window timing and cumulative memory (RSS) stats.
Validation-first: no modeling or TDA, only ingestion metrics.
"""
from __future__ import annotations
import time, csv, os, psutil
from pathlib import Path
import yaml
from src.data.stream_loader import StreamingWindowLoader, WindowConfig

def load_config(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def benchmark(dataset_cfg: Path, limit_windows: int = 200):
    cfg = load_config(dataset_cfg)
    root = Path(cfg['root_path'])
    files = sorted(root.glob(cfg['file_pattern']))
    wc = WindowConfig(window_seconds=cfg['window']['seconds'], overlap=cfg['window']['overlap'], time_column=cfg['time_column'])
    loader = StreamingWindowLoader(files, wc)
    proc = psutil.Process(os.getpid())
    out_rows = []
    start_global = time.time()
    for i, w in enumerate(loader.windows()):
        if i >= limit_windows:
            break
        t0 = time.time()
        rows = len(w)
        t1 = time.time()
        rss = proc.memory_info().rss
        out_rows.append({
            'window_index': i,
            'rows': rows,
            'elapsed_window_s': t1 - t0,
            'rss_mb': rss / (1024*1024),
            'since_start_s': t1 - start_global
        })
    out = Path('validation') / 'ingest_benchmark.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader(); writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out}")

if __name__ == '__main__':
    benchmark(Path('configs/dataset_cic.yaml'))
