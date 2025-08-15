#!/usr/bin/env bash
set -euo pipefail

# run_performance_validation.sh
# Orchestrates small/medium probes using the streaming Čech/VR harness and writes artifacts.
#
# Usage:
#   ./scripts/run_performance_validation.sh \
#     --harness build/release/tests/cpp/test_streaming_cech_perf \
#     --artifacts /tmp/tda-artifacts
#
# Notes:
# - Expects the harness to be already built. This script does NOT build.
# - Produces JSONL + adjacency histogram CSVs and invokes the analyzer.

HARNESS=""
ARTIFACTS_DIR="${TDA_ARTIFACT_DIR:-/tmp/tda-artifacts}"
# Defaults aligned with CI baseline
N=10000
D=8
MODE="vr"          # vr | cech
# Threshold params
RADIUS=0.5          # used when MODE=cech
EPSILON=0.5         # used when MODE=vr
MAXDIM=1
K_SOFT=16
PAR_THRESH=0        # serial threshold for deterministic gating
CUDA=0              # 1 to request CUDA path (if built)
STAIRCASE=""

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
    --mode)
      MODE="$2"; shift 2;;
    --radius)
      RADIUS="$2"; shift 2;;
    --epsilon)
      EPSILON="$2"; shift 2;;
    --maxDim)
      MAXDIM="$2"; shift 2;;
    --K|--soft-knn-cap)
      K_SOFT="$2"; shift 2;;
    --parallel-threshold)
      PAR_THRESH="$2"; shift 2;;
    --cuda)
      CUDA="$2"; shift 2;;
    --staircase)
      STAIRCASE="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$HARNESS" ]]; then
  echo "--harness path is required (e.g., build/release/bin/test_streaming_cech_perf)" >&2
  exit 2
fi

mkdir -p "$ARTIFACTS_DIR"

run_one() {
  local nval="$1"
  local pjson="$ARTIFACTS_DIR/run_${nval}.jsonl"
  local bjson="$ARTIFACTS_DIR/baseline_${nval}.jsonl"
  local adjp="$ARTIFACTS_DIR/adj_${nval}_parent.csv"
  local adjb="$ARTIFACTS_DIR/adj_${nval}_baseline.csv"

  # Compose common args
  local args=(
    --n "$nval" --d "$D" --maxDim "$MAXDIM"
    --soft-knn-cap "$K_SOFT" --parallel-threshold "$PAR_THRESH"
    --baseline-compare 1 --baseline-separate-process 1 --baseline-maxDim "$MAXDIM"
    --baseline-json-out "$bjson"
    --adj-hist-csv "$adjp" --adj-hist-csv-baseline "$adjb"    --json "$pjson"
  )
  # Mode-specific threshold
  if [[ "$MODE" == "vr" ]]; then
    echo "[run_performance_validation] Running VR probe: n=$nval d=$D K=$K_SOFT epsilon=$EPSILON maxDim=$MAXDIM parThresh=$PAR_THRESH cuda=$CUDA"
    args=( --mode vr --epsilon "$EPSILON" "${args[@]}" )
  else
    echo "[run_performance_validation] Running Čech probe: n=$nval d=$D K=$K_SOFT radius=$RADIUS maxDim=$MAXDIM parThresh=$PAR_THRESH cuda=$CUDA"
    args=( --radius "$RADIUS" "${args[@]}" )
  fi
  # Optional CUDA
  if [[ "$CUDA" != "0" ]]; then
    args+=( --cuda 1 )
    export TDA_USE_CUDA=1
  fi
  "$HARNESS" "${args[@]}"
}

if [[ -n "$STAIRCASE" ]]; then
  IFS=',' read -r -a sizes <<< "$STAIRCASE"
  for s in "${sizes[@]}"; do
    run_one "$s"
  done
else
  run_one "$N"
fi

echo "[run_performance_validation] Analyzing artifacts in $ARTIFACTS_DIR"
python3 "$(dirname "$0")/analyze_performance_logs.py" --artifacts "$ARTIFACTS_DIR" --gate --require-csv

echo "[run_performance_validation] Done"
