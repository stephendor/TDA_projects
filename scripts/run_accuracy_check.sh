#!/usr/bin/env bash
set -euo pipefail

# run_accuracy_check.sh
# Purpose: Small-n exact vs soft-cap variants accuracy checks. Emits JSON report in artifacts dir.

HARNESS=""
ARTIFACTS_DIR="${TDA_ARTIFACT_DIR:-/tmp/tda-artifacts}"
N=4000
D=8
RADIUS=0.5
MAXDIM=1
K_LIST="8,16,32"

usage() {
  echo "Usage: $0 --harness <path> [--artifacts <dir>] [--n <int>] [--d <int>] [--radius <float>] [--maxDim <int>] [--klist <comma>]" >&2
}

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
    --klist)
      K_LIST="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$HARNESS" ]]; then
  echo "--harness is required" >&2; exit 2
fi

mkdir -p "$ARTIFACTS_DIR"

# Exact baseline (dense, serial threshold, no soft cap)
BASE_JSON="$ARTIFACTS_DIR/accuracy_baseline_${N}.jsonl"
BASE_CHILD_JSON="$ARTIFACTS_DIR/accuracy_baseline_child_${N}.jsonl"

echo "[run_accuracy_check] Running exact baseline n=$N d=$D"
"$HARNESS" \
  --n "$N" --d "$D" --radius "$RADIUS" --maxDim "$MAXDIM" \
  --soft-knn-cap 0 --parallel-threshold 0 \
  --baseline-separate-process --baseline-maxDim "$MAXDIM" \
  --baseline-json-out "$BASE_CHILD_JSON" \
  --json "$BASE_JSON"

# Parse helper: extract numeric field from last JSONL line
extract_field() {
  local file="$1"; local field="$2"
  python3 - "$file" "$field" <<'PY'
import json,sys
path,field = sys.argv[1:3]
last=None
with open(path) as f:
    for line in f:
        try:
            last=json.loads(line)
        except Exception:
            pass
if isinstance(last, dict):
    val = last.get(field, 0)
    try:
        print(int(val))
    except Exception:
        try:
            print(float(val))
        except Exception:
            print(0)
else:
    print(0)
PY
}

BASE_EDGES=$(extract_field "$BASE_JSON" dm_edges)
BASE_H1=$(extract_field "$BASE_JSON" h1_intervals)

REPORT="$ARTIFACTS_DIR/accuracy_report_${N}.json"
python3 - "$REPORT" "$N" "$D" "$RADIUS" "$MAXDIM" "$BASE_EDGES" "$BASE_H1" <<'PY'
import json,sys
report,n,d,r,maxdim,be,bh = sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7])
with open(report,'w') as f:
    json.dump({
        "n": n,
        "d": d,
        "radius": r,
        "maxDim": maxdim,
        "baseline": {"edges": be, "h1": bh},
        "variants": []
    }, f)
PY

append_variant() {
  local k
  local file
  local e
  local h
  k="$1"
  file="$2"
  e=$(extract_field "$file" dm_edges)
  h=$(extract_field "$file" h1_intervals)
  python3 - "$REPORT" "$k" "$e" "$h" "$BASE_EDGES" "$BASE_H1" <<'PY'
import json,sys
path,k,e,h,be,bh = sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])
with open(path) as f: data=json.load(f)

def pct(delta,base):
    return (100.0*delta/base) if base>0 else (0.0 if delta==0 else float('inf'))

de=e-be; dh=h-bh
entry={"K":k,"edges":e,"h1":h,"delta_edges":de,"delta_edges_pct":pct(de,be),"delta_h1":dh,"delta_h1_pct":pct(dh,bh)}
data["variants"].append(entry)
with open(path,'w') as f: json.dump(data,f)
PY
}

IFS=',' read -r -a KARR <<< "$K_LIST"
for k in "${KARR[@]}"; do
  VAR_JSON="$ARTIFACTS_DIR/accuracy_K${k}_${N}.jsonl"
  echo "[run_accuracy_check] Running soft-cap variant K=$k"
  "$HARNESS" \
    --n "$N" --d "$D" --radius "$RADIUS" --maxDim "$MAXDIM" \
    --soft-knn-cap "$k" --parallel-threshold 0 \
    --json "$VAR_JSON"
  append_variant "$k" "$VAR_JSON"
done

echo "[run_accuracy_check] Wrote report: $REPORT"
