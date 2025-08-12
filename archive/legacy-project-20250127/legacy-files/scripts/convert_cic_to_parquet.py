"""CIC-IDS2017 CSV -> Parquet conversion utility (validation-first).

Reads raw CSV files from an input directory, extracts required numeric features + timestamp + Label,
performs minimal cleaning, and writes parquet shards compatible with dataset_cic.yaml config.

Updates (synthetic time support):
  * If no real timestamp column present, create synthetic monotonically increasing 'timestamp' from row index
  * Also add 'RowIndex' (per-file) and optional 'PseudoTime' (cumulative FlowDuration if available)
  * Explicitly marks that synthetic time MUST NOT be used for temporal leakage claims

Usage (example):
    python scripts/convert_cic_to_parquet.py \
        --input data/apt_datasets/CIC-IDS-2017/raw_csv \
        --output data/apt_datasets/CIC-IDS-2017 \
        --glob "*.csv" \
        --chunksize 200000

Notes:
  * Only deterministic transformations; no statistical feature engineering
  * Timestamp column in raw may be one of TIMESTAMP_CANDIDATES; else synthetic
  * Label mapped: BENIGN->0 else 1 (keep binary focus)
  * Never abort file solely due to missing timestamp anymore (synthetic fallback)
  * Writes one parquet per source CSV base name (no compression by default)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

# Mapping from verbose raw CIC-IDS2017 column names (trimmed) to simplified internal feature names
VERBOSE_TO_INTERNAL = {
    'Flow Duration': 'FlowDuration',
    'Total Fwd Packets': 'TotFwdPkts',
    'Total Backward Packets': 'TotBwdPkts',
    'Total Length of Fwd Packets': 'TotLenFwdPkts',
    'Total Length of Bwd Packets': 'TotLenBwdPkts',
    'Fwd Packet Length Mean': 'FwdPktLenMean',
    'Bwd Packet Length Mean': 'BwdPktLenMean',
    'Flow Bytes/s': 'FlowByts/s',
    'Flow Packets/s': 'FlowPkts/s',
}

REQUIRED_NUMERIC = [
    'FlowDuration', 'TotFwdPkts', 'TotBwdPkts', 'TotLenFwdPkts', 'TotLenBwdPkts',
    'FwdPktLenMean', 'BwdPktLenMean', 'FlowByts/s', 'FlowPkts/s'
]
TIMESTAMP_CANDIDATES = ['timestamp', 'Timestamp', 'Time', 'StartTime', 'ts']
LABEL_CANDIDATES = ['Label', 'label', 'Attack', 'Category']


def detect_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    original = list(df.columns)
    # Trim whitespace first (maintain original tokens for mapping)
    trimmed = [c.strip() for c in original]
    # Build rename dict for verbose -> internal
    rename_map = {}
    for c in trimmed:
        if c in VERBOSE_TO_INTERNAL:
            rename_map[c] = VERBOSE_TO_INTERNAL[c]
    # Apply verbose mapping first (need to address duplicates separately if any)
    df.columns = trimmed
    df = df.rename(columns=rename_map)
    # Secondary normalization: collapse spaces to single underscore for remaining columns (non-required kept raw-ish)
    new_cols = []
    for c in df.columns:
        c2 = re.sub(r'\s+', ' ', c).strip()
        # Replace spaces with nothing for internal required numeric (to keep expected names) unless already mapped
        if c2 not in VERBOSE_TO_INTERNAL.values():
            c2 = c2.replace(' ', '_').replace('/', '_').replace('-', '_')
        new_cols.append(c2)
    df.columns = new_cols
    return df


def load_chunk_process(chunk: pd.DataFrame, ts_col: str | None, label_col: str, row_offset: int) -> tuple[pd.DataFrame, int]:
    from pandas.api.types import is_numeric_dtype
    # Label mapping (binary) - robust before dropping rows
    lbl_raw = chunk[label_col].astype(str).str.upper()
    labels = (~lbl_raw.str.contains('BENIGN')).astype(int)
    chunk = chunk.assign(Label=labels)

    # Timestamp handling
    synthetic = False
    if ts_col is not None:
        ts_series = chunk[ts_col]
        if is_numeric_dtype(ts_series):
            ts = ts_series.astype(float)
            # Heuristic: if looks like ns epoch
            if ts.mean() > 1e12:
                ts = ts / 1e9
        else:
            ts = pd.to_datetime(ts_series, errors='coerce').astype('int64') / 1e9
        chunk = chunk.assign(timestamp=ts)
        chunk = chunk[np.isfinite(chunk['timestamp'])]
    else:
        # Synthetic monotonic timestamp from row index order (per-file)
        synthetic = True
        local_idx = np.arange(len(chunk), dtype=float) + float(row_offset)
        chunk = chunk.assign(timestamp=local_idx, RowIndex=local_idx.astype(int))

    # PseudoTime (cumulative FlowDuration) if available
    if 'FlowDuration' in chunk.columns:
        flow_dur = pd.to_numeric(chunk['FlowDuration'], errors='coerce').fillna(0).astype(float)
        pseudo_time = flow_dur.cumsum() + (0.0 if row_offset == 0 else row_offset)
        chunk = chunk.assign(PseudoTime=pseudo_time)
    elif synthetic:
        # Guarantee a PseudoTime column even if FlowDuration absent
        chunk = chunk.assign(PseudoTime=chunk['timestamp'])

    keep_cols = ['timestamp', 'Label']
    if 'RowIndex' in chunk.columns:
        keep_cols.append('RowIndex')
    if 'PseudoTime' in chunk.columns:
        keep_cols.append('PseudoTime')
    # Append available required numeric
    for c in REQUIRED_NUMERIC:
        if c in chunk.columns:
            keep_cols.append(c)
    processed = chunk[keep_cols]
    return processed, row_offset + len(chunk)


def process_file(path: Path, out_dir: Path, chunksize: int) -> dict:
    total_rows_in = 0
    total_rows_out = 0
    detected = False
    ts_col: str | None = None
    label_col: str | None = None
    row_offset = 0  # for synthetic timestamp continuity within file
    out_path = out_dir / (path.stem + '.parquet')
    writer_frames = []

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total_rows_in += len(chunk)
        chunk = normalize_and_map_columns(chunk)
        if not detected:
            ts_col = detect_column(chunk.columns, TIMESTAMP_CANDIDATES)
            label_col = detect_column(chunk.columns, LABEL_CANDIDATES)
            if label_col is None:
                return {'file': str(path), 'skipped': True, 'reason': 'missing label'}
            detected = True
        assert label_col is not None  # typing guard
        processed, row_offset = load_chunk_process(chunk, ts_col, label_col, row_offset)
        total_rows_out += len(processed)
        writer_frames.append(processed)

    if total_rows_out == 0:
        return {'file': str(path), 'skipped': True, 'reason': 'no valid rows'}

    df_out = pd.concat(writer_frames, axis=0)
    # Ensure deterministic ordering by synthetic/real timestamp
    df_out = df_out.sort_values('timestamp').reset_index(drop=True)
    df_out.to_parquet(out_path, index=False)
    synthetic_flag = ts_col is None
    return {
        'file': str(path),
        'out': str(out_path),
        'rows_in': total_rows_in,
        'rows_out': total_rows_out,
        'synthetic_timestamp': synthetic_flag,
        'skipped': False
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Input directory containing raw CSV files')
    ap.add_argument('--output', required=True, help='Output directory for parquet files')
    ap.add_argument('--glob', default='*.csv', help='Glob for CSV selection')
    ap.add_argument('--chunksize', type=int, default=200000)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.exists():
        raise SystemExit(f'Input directory not found: {in_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit('No CSV files found with pattern')

    summary = []
    for f in files:
        res = process_file(f, out_dir, args.chunksize)
        summary.append(res)
        status = 'SKIP' if res.get('skipped') else 'OK'
        synth = ' synthetic-ts' if (not res.get('skipped') and res.get('synthetic_timestamp')) else ''
        print(f"[{status}] {f.name}: in={res.get('rows_in',0)} out={res.get('rows_out',0)}{synth} reason={res.get('reason','-')}")
    ok = [r for r in summary if not r.get('skipped')]
    synth_count = sum(1 for r in ok if r.get('synthetic_timestamp'))
    print(f"Converted {len(ok)}/{len(summary)} files to parquet -> {out_dir} (synthetic timestamps in {synth_count})")
    if synth_count > 0:
        print('WARNING: Synthetic timestamps were generated; temporal leakage safeguards cannot assert real capture ordering.')


if __name__ == '__main__':
    main()
