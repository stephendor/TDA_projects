#!/usr/bin/env python3
"""
Analyze JSONL/CSV artifacts produced by the streaming Čech harness.

Gates on:
- serial threshold runs (soft-cap active implies parallel-threshold==0)
- overshoot_sum==0 and overshoot_max==0
- presence of adjacency histogram CSVs (optional flag)
- presence of peak memory fields

Usage:
  ./scripts/analyze_performance_logs.py --artifacts /tmp/tda-artifacts --gate --require-csv
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import glob
from typing import Any, Dict, List, Tuple

FIELDS = [
    "mode",
    "dm_blocks",
    "dm_edges",
    "simplices",
    "softcap_overshoot_sum",
    "softcap_overshoot_max",
    "dm_peakMB",
    "rss_peakMB",
    "parallel_threshold",
    # Pool telemetry (aggregate)
    "pool_total_blocks",
    "pool_free_blocks",
    "pool_fragmentation",
    # Optional compact per-bucket serialization
    "pool_bucket_stats_compact",
    # Optional page info (if exported later)
    "pool_pages",
    "pool_blocks_per_page",
    # Early-stop diagnostics
    "dm_total_blocks",
    "dm_total_pairs",
    "dm_stopped_by_time",
    "dm_stopped_by_pairs",
    "dm_stopped_by_blocks",
    "dm_last_block_i0",
    "dm_last_block_j0",
    # Manifest & recompute fields
    "manifest_hash",
    "attempted_count",
    "attempted_kind",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # Skip non-JSON lines
                continue
    return out


def summarize_jsonl(path: str) -> Dict[str, Any]:
    rows = load_jsonl(path)
    # Take the last JSON object as the summary if present
    last = rows[-1] if rows else {}
    return {k: last.get(k) for k in FIELDS if k in last}


def find_accuracy_reports(artifacts_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(artifacts_dir, "accuracy_report_*.json")))


def load_accuracy_report(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def analyze_accuracy_reports(artifacts_dir: str, thr_edges: float | None, thr_h1: float | None, gate: bool) -> Tuple[bool, int]:
    """
    Parse accuracy_report_*.json files and optionally enforce delta thresholds.

    Returns (ok, count_reports).
    """
    reports = find_accuracy_reports(artifacts_dir)
    if not reports:
        print("[analyzer] accuracy: no accuracy_report_*.json found")
        return True, 0

    ok = True
    for rp in reports:
        data = load_accuracy_report(rp)
        base = data.get("baseline", {})
        be, bh = base.get("edges", 0), base.get("h1", 0)
        print(f"[analyzer] accuracy: {os.path.basename(rp)} baseline edges={be}, h1={bh}")
        worst_edges_pct = 0.0
        worst_h1_pct = 0.0
        for v in data.get("variants", []):
            k = v.get("K")
            de_pct = float(v.get("delta_edges_pct", 0.0) or 0.0)
            dh_pct = float(v.get("delta_h1_pct", 0.0) or 0.0)
            worst_edges_pct = max(worst_edges_pct, abs(de_pct))
            worst_h1_pct = max(worst_h1_pct, abs(dh_pct))
            print(f"[analyzer]  - K={k}: Δedges%={de_pct:.3f}, Δh1%={dh_pct:.3f}")
        print(f"[analyzer] accuracy: worst Δedges%={worst_edges_pct:.3f}, worst Δh1%={worst_h1_pct:.3f}")
        if gate:
            if thr_edges is not None and worst_edges_pct > thr_edges:
                print(f"[analyzer] WARNING: Δedges% {worst_edges_pct:.3f} exceeds {thr_edges:.3f}")
                ok = False
            if thr_h1 is not None and worst_h1_pct > thr_h1:
                print(f"[analyzer] WARNING: Δh1% {worst_h1_pct:.3f} exceeds {thr_h1:.3f}")
                ok = False
    return ok, len(reports)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="Directory with jsonl/csv outputs")
    ap.add_argument("--gate", action="store_true", help="Exit non-zero on violations (perf + optional accuracy)")
    ap.add_argument("--require-csv", action="store_true", help="Require adjacency histogram CSVs")
    ap.add_argument("--recompute-gate", action="store_true", help="Fail if manifest hash changed but attempted_count==0 (no recompute)")
    # Accuracy thresholds (non-blocking unless --accuracy-gate is provided)
    ap.add_argument("--accuracy-edges-pct-threshold", type=float, default=None, help="Max allowed |Δedges%%| across variants")
    ap.add_argument("--accuracy-h1-pct-threshold", type=float, default=None, help="Max allowed |Δh1%%| across variants")
    ap.add_argument("--accuracy-gate", action="store_true", help="Apply gating to accuracy thresholds (otherwise just print)")
    # Allow env overrides for CI without editing workflows
    args = ap.parse_args()
    # Environment toggles (optional): ACCURACY_GATE=1, REQUIRE_CSV=1
    if not args.require_csv and os.environ.get("REQUIRE_CSV", "0") in ("1", "true", "TRUE"):  # pragma: no cover
        args.require_csv = True
    if not args.accuracy_gate and os.environ.get("ACCURACY_GATE", "0") in ("1", "true", "TRUE"):  # pragma: no cover
        args.accuracy_gate = True
    if not args.recompute_gate and os.environ.get("RECOMPUTE_GATE", "0") in ("1", "true", "TRUE"):  # pragma: no cover
        args.recompute_gate = True
    # Thresholds from env if not specified
    def _env_float(name: str) -> float | None:
        v = os.environ.get(name)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None
    if args.accuracy_edges_pct_threshold is None:  # pragma: no cover
        args.accuracy_edges_pct_threshold = _env_float("ACCURACY_EDGES_PCT_THRESHOLD")
    if args.accuracy_h1_pct_threshold is None:  # pragma: no cover
        args.accuracy_h1_pct_threshold = _env_float("ACCURACY_H1_PCT_THRESHOLD")

    art = os.path.abspath(args.artifacts)
    jsonls = sorted(glob.glob(os.path.join(art, "*.jsonl")))
    parents = [p for p in jsonls if os.path.basename(p).startswith("run_")]
    bases = [p for p in jsonls if os.path.basename(p).startswith("baseline_")]

    print(f"[analyzer] artifacts: {art}")
    print(f"[analyzer] parent JSONLs: {len(parents)}, baseline JSONLs: {len(bases)}")

    ok = True
    if not parents:
        print("[analyzer] ERROR: no parent run_*.jsonl found")
        ok = False
    if not bases:
        print("[analyzer] WARNING: no baseline_*.jsonl found (skipping baseline checks)")

    # Print summaries and enforce gates
    last_parent_summary = None
    for p in parents:
        s = summarize_jsonl(p)
        print(f"[analyzer] parent summary {os.path.basename(p)} -> {s}")
        last_parent_summary = s or last_parent_summary
        if all(k in s for k in ("pool_total_blocks","pool_free_blocks","pool_fragmentation")):
            preview = ""
            compact = s.get("pool_bucket_stats_compact")
            if isinstance(compact, str) and compact:
                preview = compact[:60]
                if len(compact) > 60:
                    preview += "…"
                preview = f", buckets={preview}"
            extra = ""
            if "pool_pages" in s and "pool_blocks_per_page" in s:
                extra = f", pages={s.get('pool_pages')} x {s.get('pool_blocks_per_page')}"
            print(f"[analyzer] pool: total={s['pool_total_blocks']} free={s['pool_free_blocks']} frag={s['pool_fragmentation']}{extra}{preview}")
            # Warn-only heuristic for extreme fragmentation on sizable pools
            try:
                if float(s.get("pool_total_blocks", 0) or 0) >= 100 and float(s.get("pool_fragmentation", 0.0) or 0.0) >= 0.95:
                    print("[analyzer] WARNING: High SimplexPool fragmentation detected (frag>=0.95 with total_blocks>=100). Investigate reuse patterns.")
            except Exception:
                pass
        # Gate: overshoot must be zero in serial threshold mode
        if args.gate:
            pt = s.get("parallel_threshold")
            oversum = s.get("softcap_overshoot_sum")
            overmax = s.get("softcap_overshoot_max")
            if pt == 0:
                if oversum not in (0, None) and oversum != 0:
                    print("[analyzer] ERROR: overshoot_sum != 0 with serial threshold")
                    ok = False
                if overmax not in (0, None) and overmax != 0:
                    print("[analyzer] ERROR: overshoot_max != 0 with serial threshold")
                    ok = False
            # Presence of memory fields
            for memk in ("dm_peakMB", "rss_peakMB"):
                if memk not in s or s[memk] is None:
                    print(f"[analyzer] ERROR: missing memory field {memk}")
                    ok = False
        # Print early-stop diagnostics if present
        if any(k in s for k in ("dm_stopped_by_time","dm_stopped_by_pairs","dm_stopped_by_blocks")):
            print(f"[analyzer] dm-stop: blocks={s.get('dm_total_blocks','-')} pairs={s.get('dm_total_pairs','-')} time={s.get('dm_stopped_by_time','-')} pairsStop={s.get('dm_stopped_by_pairs','-')} blocksStop={s.get('dm_stopped_by_blocks','-')} last=({s.get('dm_last_block_i0','-')},{s.get('dm_last_block_j0','-')})")
        # Print manifest if present
        if 'manifest_hash' in s:
            print(f"[analyzer] manifest: hash={s.get('manifest_hash','-')} attempted={s.get('attempted_count','-')} kind={s.get('attempted_kind','-')}")

    if args.require_csv:
        csvs = sorted(glob.glob(os.path.join(art, "*.csv")))
        # Accept several naming patterns for flexibility:
        #  - *_parent.csv (legacy)
        #  - adj_parent_*.csv (current)
        #  - parent_*.csv (fallback)
        # Baseline CSVs are optional for gating, but we detect them for info.
        basenames = [os.path.basename(c) for c in csvs]
        have_parent = any(
            name.endswith("_parent.csv")
            or name.startswith("adj_parent_")
            or name.startswith("parent_")
            for name in basenames
        )
        have_baseline = any(
            name.endswith("_baseline.csv")
            or name.startswith("adj_baseline_")
            or name.startswith("baseline_")
            for name in basenames
        )
        if not csvs or not have_parent:
            print("[analyzer] ERROR: adjacency histogram CSVs missing (no parent CSV found)")
            if csvs:
                print("[analyzer]   found CSVs:")
                for name in basenames:
                    print(f"[analyzer]    - {name}")
            ok = False
        else:
            print(f"[analyzer] CSVs present: {len(csvs)} (parent={'YES' if have_parent else 'NO'}, baseline={'YES' if have_baseline else 'NO'})")

    # Accuracy analysis (non-blocking by default unless --accuracy-gate)
    acc_ok, acc_count = analyze_accuracy_reports(
        art,
        args.accuracy_edges_pct_threshold,
        args.accuracy_h1_pct_threshold,
        args.accuracy_gate,
    )
    if acc_count > 0:
        print(f"[analyzer] accuracy gate={'ON' if args.accuracy_gate else 'OFF'} result={'PASS' if acc_ok else 'FAIL'}")
    # Recompute safeguard: if manifest changed but attempted_count==0, warn/fail
    if last_parent_summary and 'manifest_hash' in last_parent_summary:
        store_path = os.path.join(art, ".manifest_hash")
        prev_hash = None
        try:
            with open(store_path, 'r') as fh:
                prev_hash = fh.read().strip() or None
        except Exception:
            prev_hash = None
        curr_hash = last_parent_summary.get('manifest_hash')
        attempted = int(last_parent_summary.get('attempted_count') or 0)
        if prev_hash is not None and curr_hash and curr_hash != prev_hash:
            if attempted == 0:
                msg = "[analyzer] {}: manifest changed ({} -> {}), attempted_count==0".format(
                    'ERROR' if args.recompute_gate and args.gate else 'WARNING', prev_hash, curr_hash)
                print(msg)
                if args.recompute_gate and args.gate:
                    ok = False
        # Always update manifest store to current
        try:
            with open(store_path, 'w') as fh:
                fh.write(curr_hash or "")
        except Exception:
            pass
    # Final status considers perf gating and optional accuracy gating
    final_ok = ok and (acc_ok or not args.accuracy_gate)
    print(f"[analyzer] gate={'ON' if args.gate else 'OFF'} result={'PASS' if final_ok else 'FAIL'}")
    return 0 if (final_ok or not args.gate) else 2


if __name__ == "__main__":
    sys.exit(main())
