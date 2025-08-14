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
from typing import Any, Dict, List

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="Directory with jsonl/csv outputs")
    ap.add_argument("--gate", action="store_true", help="Exit non-zero on violations")
    ap.add_argument("--require-csv", action="store_true", help="Require adjacency histogram CSVs")
    args = ap.parse_args()

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
    for p in parents:
        s = summarize_jsonl(p)
        print(f"[analyzer] parent summary {os.path.basename(p)} -> {s}")
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

    if args.require_csv:
        csvs = sorted(glob.glob(os.path.join(art, "*.csv")))
        have_parent = any("_parent.csv" in c for c in csvs)
        if not csvs or not have_parent:
            print("[analyzer] ERROR: adjacency histogram CSVs missing")
            ok = False
        else:
            print(f"[analyzer] CSVs present: {len(csvs)} (sample: {os.path.basename(csvs[0])})")

    print(f"[analyzer] gate={'ON' if args.gate else 'OFF'} result={'PASS' if ok else 'FAIL'}")
    return 0 if (ok or not args.gate) else 2


if __name__ == "__main__":
    sys.exit(main())
