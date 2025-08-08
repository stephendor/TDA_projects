"""Variance assessment across multiple baseline deterministic re-runs.

Purpose:
  Execute the existing baseline script multiple times (unmodified) under
  identical environment settings to empirically measure run-to-run variance
  in key metrics (Accuracy, ROC-AUC, PR-AUC, confusion matrix components).

Topology Integrity:
  The baseline script already performs: streaming window construction, VR / witness
  persistence diagram computation, and deterministic diagram-level feature stacking:
    - analyzer.extract_features() (internal TDAWindowVectorizer output)
    - BettiCurveVectorizer
    - LifetimeStatsVectorizer
    - PersistenceImageVectorizer (H0..H1)
    - PersistenceLandscapeVectorizer (H0..H1)
  This runner ONLY orchestrates repeated executions; it does NOT add any statistical proxies.

Determinism Notes:
  The underlying pipeline uses a fixed random seed from dataset config for point cloud capping & model seed.
  Residual variance can still arise from:
    - File system / streaming window boundary alignment (if timing columns produce different accumulation grouping)
    - XGBoost parallelism / non-deterministic reductions
  We keep parameters constant and (optionally) disable tuning to isolate pipeline variance.

Usage:
  python validation/variance_assessment.py \
      --runs 5 \
      --enable-tuning 0 \
      --max-feature-windows 6000 \
      --min-minority-windows 80 \
      --target-min-minority 80

Outputs:
  JSON summary written to validation/metrics/variance_summary_*.json
  Prints mean / std for metrics and relative std (% of mean) for quick inspection.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
import time
import json
from typing import List, Dict, Any
import numpy as np

METRICS_DIR = Path('validation/metrics')
BASELINE_SCRIPT = 'validation/baseline_gbt_training.py'
PREFIX = 'baseline_gbt_metrics_'


def _latest_metric_files() -> List[Path]:
    if not METRICS_DIR.exists():
        return []
    return sorted(METRICS_DIR.glob(f'{PREFIX}*.json'))


def _extract_metrics(path: Path) -> Dict[str, Any]:
    try:
        with path.open('r') as f:
            data = json.load(f)
        # ensure numeric extraction with fallback None
        return {
            'file': str(path),
            'accuracy': data.get('accuracy'),
            'roc_auc': data.get('roc_auc'),
            'pr_auc': data.get('pr_auc'),
            'confusion': data.get('confusion_matrix') or None,  # baseline stores only cm in stdout, not JSON? keep placeholder
            'train_counts': data.get('train_counts'),
            'test_counts': data.get('test_counts')
        }
    except Exception as e:
        return {'file': str(path), 'error': str(e)}


def _run_baseline(env_overrides: Dict[str, str]) -> Path | None:
    before = set(_latest_metric_files())
    # Compose environment
    env = os.environ.copy()
    env.update(env_overrides)
    # Run baseline
    cmd = [sys.executable, BASELINE_SCRIPT]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Capture output for debugging in summary if needed
    run_log = proc.stdout
    after = set(_latest_metric_files())
    new_files = sorted(list(after - before))
    metrics_path = new_files[-1] if new_files else None
    if metrics_path is None:
        # Persist log for forensic analysis
        fail_log = METRICS_DIR / f"variance_failure_log_{int(time.time())}.txt"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text(run_log)
    return metrics_path


def assess_variance(runs: int, env_overrides: Dict[str, str]) -> Dict[str, Any]:
    records = []
    for i in range(runs):
        print(f"[Variance] Starting run {i+1}/{runs}")
        path = _run_baseline(env_overrides)
        if path is None:
            print(f"[Variance] WARNING: Run {i+1} produced no metrics file; aborting further runs.")
            break
        rec = _extract_metrics(path)
        rec['run_index'] = i
        records.append(rec)
        print(f"[Variance] Completed run {i+1}: metrics file={path.name}")
    # Aggregate numeric metrics
    agg = {}
    for key in ('accuracy', 'roc_auc', 'pr_auc'):
        vals = [r[key] for r in records if isinstance(r.get(key), (int, float))]
        if vals:
            arr = np.array(vals, dtype=float)
            agg[key] = {
                'mean': float(arr.mean()),
                'std': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                'relative_std_pct': float((arr.std(ddof=1)/arr.mean()*100.0)) if arr.size > 1 and arr.mean() != 0 else 0.0,
                'min': float(arr.min()),
                'max': float(arr.max()),
                'n': int(arr.size)
            }
    return {'runs': records, 'aggregate': agg, 'env_overrides': env_overrides}


def main():
    parser = argparse.ArgumentParser(description='Variance assessment for baseline GBT TDA pipeline.')
    parser.add_argument('--runs', type=int, default=5, help='Number of repeated executions')
    parser.add_argument('--enable-tuning', type=int, default=0, help='Enable tuning inside baseline (adds extra variance)')
    parser.add_argument('--max-feature-windows', type=int, default=6000)
    parser.add_argument('--min-minority-windows', type=int, default=80)
    parser.add_argument('--target-min-minority', type=int, default=80)
    parser.add_argument('--post-target-extra-windows', type=int, default=800)
    parser.add_argument('--max-class-ratio', type=float, default=60.0)
    parser.add_argument('--min-train-per-class', type=int, default=20)
    parser.add_argument('--min-test-per-class', type=int, default=5)
    args = parser.parse_args()

    env_overrides = {
        'ENABLE_TUNING': '1' if args.enable_tuning else '0',
        'MAX_FEATURE_WINDOWS': str(args.max_feature_windows),
        'MIN_MINORITY_WINDOWS': str(args.min_minority_windows),
        'TARGET_MIN_MINORITY': str(args.target_min_minority),
        'POST_TARGET_EXTRA_WINDOWS': str(args.post_target_extra_windows),
        'MAX_CLASS_RATIO': str(args.max_class_ratio),
        'MIN_TRAIN_PER_CLASS': str(args.min_train_per_class),
        'MIN_TEST_PER_CLASS': str(args.min_test_per_class)
    }

    summary = assess_variance(args.runs, env_overrides)

    # Persist summary
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"variance_summary_{int(time.time())}.json"
    with out_path.open('w') as f:
        json.dump(summary, f, indent=2)
    print('\n[Variance] Summary saved to', out_path)
    if 'aggregate' in summary:
        print('\n[Variance] Aggregates:')
        for k, stats in summary['aggregate'].items():
            print(f"  {k}: mean={stats['mean']:.6f} std={stats['std']:.6f} rel%={stats['relative_std_pct']:.3f} n={stats['n']} range=({stats['min']:.6f},{stats['max']:.6f})")


if __name__ == '__main__':
    main()
