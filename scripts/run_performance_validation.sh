#!/usr/bin/env bash
set -euo pipefail

# run_performance_validation.sh
# Orchestrates small/medium probes using the streaming Čech harness and writes artifacts.
#
# Usage:
#   ./scripts/run_performance_validation.sh \
#     --harness build/release/bin/test_streaming_cech_perf \
#     --artifacts /tmp/tda-artifacts
#
# Notes:
# - Expects the harness to be already built. This script does NOT build.
# - Produces JSONL + adjacency histogram CSVs and invokes the analyzer.

HARNESS=""
ARTIFACTS_DIR="${TDA_ARTIFACT_DIR:-/tmp/tda-artifacts}"
N=8000
D=8
RADIUS=0.5
MAXDIM=1
K_SOFT=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness)
      HARNESS="$2"; shift 2;;
    --artifacts)
      ARTIFACTS_DIR="$2"; shift 2;;
    --n)
      N="$2"; shift 2;;
    --d)
      D="$2"; shift 2;;
    --radius)
      RADIUS="$2"; shift 2;;
    --maxDim)
      MAXDIM="$2"; shift 2;;
    --K|--soft-knn-cap)
      K_SOFT="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$HARNESS" ]]; then
  echo "--harness path is required (e.g., build/release/bin/test_streaming_cech_perf)" >&2
  exit 2
fi

mkdir -p "$ARTIFACTS_DIR"
PARENT_JSON="$ARTIFACTS_DIR/run_${N}.jsonl"
BASE_JSON="$ARTIFACTS_DIR/baseline_${N}.jsonl"
ADJ_PARENT="$ARTIFACTS_DIR/adj_${N}_parent.csv"
ADJ_BASE="$ARTIFACTS_DIR/adj_${N}_baseline.csv"

echo "[run_performance_validation] Running probe: n=$N d=$D K=$K_SOFT radius=$RADIUS maxDim=$MAXDIM"
"$HARNESS" \
  --n "$N" --d "$D" --radius "$RADIUS" --maxDim "$MAXDIM" \
  --soft-knn-cap "$K_SOFT" --parallel-threshold 0 \
  --baseline-separate-process --baseline-maxDim "$MAXDIM" \
  --baseline-json-out "$BASE_JSON" \
  --adj-hist-csv "$ADJ_PARENT" \
  --adj-hist-csv-baseline "$ADJ_BASE" \
  --json "$PARENT_JSON"

echo "[run_performance_validation] Analyzing artifacts in $ARTIFACTS_DIR"
"$(dirname "$0")/analyze_performance_logs.py" --artifacts "$ARTIFACTS_DIR" --gate --require-csv

echo "[run_performance_validation] Done"
