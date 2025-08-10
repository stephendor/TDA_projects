"""Compare baseline vs enhanced runs producing delta JSON.

Usage:
  python scripts/enhancement_delta_eval.py --baseline path/to/baseline/results --enhanced path/to/enhanced/results --out delta.json

It loads metrics.json and diagnostics.json (if present) from each run directory
and computes absolute/relative deltas for key metrics plus feature dimension changes.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
from typing import Dict, Any

def load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    with open(p,'r') as f:
        return json.load(f)

def extract_primary_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics = load_json(run_dir / 'results' / 'metrics.json')
    diagnostics = load_json(run_dir / 'results' / 'diagnostics.json')
    out: Dict[str, Any] = {}
    # Pull logistic + gbt if present
    for model in ['logistic_regression','gradient_boosted_trees']:
        if model in metrics:
            out[model] = {k: metrics[model][k] for k in ['pr_auc','roc_auc','best_f1'] if k in metrics[model]}
    if 'train_shape' in diagnostics:
        out['train_dim'] = diagnostics['train_shape'][1]
    if 'post_prune_feature_dim' in diagnostics:
        out['post_prune_dim'] = diagnostics['post_prune_feature_dim']
    if 'sw_angle_strategy' in diagnostics:
        out['sw_angle_strategy'] = diagnostics['sw_angle_strategy']
    if 'filtration_recompute' in diagnostics:
        out['filtration_recompute'] = diagnostics['filtration_recompute']
    return out

def compute_delta(base: Dict[str,Any], enh: Dict[str,Any]) -> Dict[str,Any]:
    delta: Dict[str,Any] = {'baseline': base, 'enhanced': enh, 'deltas': {}}
    for model in ['logistic_regression','gradient_boosted_trees']:
        if model in base and model in enh:
            dsub = {}
            for k in base[model]:
                if k in enh[model]:
                    b = base[model][k]; e = enh[model][k]
                    if isinstance(b,(int,float)) and isinstance(e,(int,float)):
                        dsub[k] = {'abs': e-b, 'rel': (e-b)/b if b not in (0,None) else None}
            delta['deltas'][model] = dsub
    for key in ['train_dim','post_prune_dim']:
        if key in base and key in enh:
            b = base[key]; e = enh[key]
            if isinstance(b,(int,float)) and isinstance(e,(int,float)):
                delta['deltas'][key] = {'abs': e-b, 'rel': (e-b)/b if b else None}
    return delta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, help='Baseline run directory (contains results/)')
    ap.add_argument('--enhanced', required=True, help='Enhanced run directory (contains results/)')
    ap.add_argument('--out', default='delta_enhancement.json', help='Output JSON path')
    args = ap.parse_args()
    base_dir = Path(args.baseline)
    enh_dir = Path(args.enhanced)
    base = extract_primary_metrics(base_dir)
    enh = extract_primary_metrics(enh_dir)
    delta = compute_delta(base, enh)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,'w') as f:
        json.dump(delta, f, indent=2)
    print(f"Wrote delta metrics to {out_path}")

if __name__ == '__main__':
    main()
