"""Baseline Gradient Boosted Trees training on vectorized TDA window features.

Workflow:
  * Stream real CIC-IDS2017 (or configured) data windows via StreamingWindowLoader
  * Build point clouds per window (numeric features only initially)
  * Compute persistence diagrams (VR / witness fallback) -> deterministic topological features
  * Aggregate feature vectors + align with labels (attack vs benign) using timestamp overlap
  * Train train/test split with temporal integrity (earlier segment = train, later = test) to avoid leakage
  * Report accuracy + confusion matrix (deterministic seed)

Validation-first:
  - NO performance claim here; script outputs raw metrics only
  - Label integration requires presence of 'Label' column (binary) in parquet files
  - Temporal leakage guard: ensures max(train_timestamp) < min(test_timestamp)
"""
from __future__ import annotations
import os
import json
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from xgboost import XGBClassifier
from src.data.stream_loader import WindowConfig, StreamingWindowLoader, build_point_cloud
from src.tda.vectorizers.window_tda_vectorizer import TDAWindowVectorizer, TDAWindowVectorizerConfig
from src.tda.vectorizers.diagram_vectorizers import (
    PersistenceImageConfig, PersistenceImageVectorizer,
    PersistenceLandscapeConfig, PersistenceLandscapeVectorizer,
    BettiCurveConfig, BettiCurveVectorizer,
    LifetimeStatsConfig, LifetimeStatsVectorizer
)
from pathlib import Path

# ---- Load dataset config ----
CONFIG_PATH = Path('configs/dataset_cic.yaml')
if not CONFIG_PATH.exists():
    raise SystemExit("Dataset config not found: configs/dataset_cic.yaml")
with CONFIG_PATH.open('r') as f:
    ds_cfg = yaml.safe_load(f)

root_path = Path(ds_cfg['root_path'])
file_pattern = ds_cfg['file_pattern']
feature_cols = ds_cfg['features']['numeric']  # use numeric only for point cloud
label_column = 'Label'

# Collect parquet files
if not root_path.exists():
    raise SystemExit(f"Data root does not exist: {root_path}")
parquet_files = sorted(root_path.glob(file_pattern))
if len(parquet_files) == 0:
    raise SystemExit(f"No parquet files matching pattern {file_pattern} under {root_path}")
# Removed single-file debug limit – processing all files
print(f"INFO: Using {len(parquet_files)} parquet files for sampling (no single-file limit).")

# Window configuration
win_cfg = WindowConfig(
    window_seconds=ds_cfg['window']['seconds'],
    overlap=ds_cfg['window']['overlap'],
    time_column=ds_cfg['time_column']
)

# Vectorizer configuration
vec = TDAWindowVectorizer(TDAWindowVectorizerConfig(
    feature_columns=feature_cols,
    max_vr_points=min(500, ds_cfg['limits']['max_vr_points']),  # quick patch: lower cap for speed
    witness_landmarks=min(64, ds_cfg['limits']['witness_landmarks'])
))
# Additional diagram-level vectorizers (pure topology)
_pimg = PersistenceImageVectorizer(PersistenceImageConfig(resolution=(16,16), sigma=0.05))
_pland = PersistenceLandscapeVectorizer(PersistenceLandscapeConfig(resolution=60, k_layers=3))
_betti = BettiCurveVectorizer(BettiCurveConfig(radii_resolution=40, maxdim=2))
_life = LifetimeStatsVectorizer(LifetimeStatsConfig(top_k=5, maxdim=2))
# Track how many persistence image dimensions we include (H0..Hk)
_PI_DIMS = 2  # persistence image homology dims included (starting from H0)
_LAND_DIMS = 2  # landscape homology dims included (H0,H1)

# ---- Thresholds (env-overridable) ----
MIN_POINTS_ACCUM = int(os.getenv('MIN_POINTS_ACCUM', '20'))  # minimum raw rows to accumulate before topology
MAX_FEATURE_WINDOWS = int(os.getenv('MAX_FEATURE_WINDOWS', '30'))  # limit number of feature windows to build (can raise/remove)
MIN_MINORITY_WINDOWS = int(os.getenv('MIN_MINORITY_WINDOWS', '5'))  # require at least this many minority windows before stopping
MAX_TOTAL_STREAM_WINDOWS = int(os.getenv('MAX_TOTAL_STREAM_WINDOWS', '5000'))  # hard safety cap
TARGET_MIN_MINORITY = int(os.getenv('TARGET_MIN_MINORITY', str(MIN_MINORITY_WINDOWS)))  # optional higher target after initial collection
FEATURE_DUMP_DIR = os.getenv('FEATURE_DUMP_DIR', '').strip()
DENSITY_SCAN_DISABLE = os.getenv('DISABLE_DENSITY_SCAN', '0') == '1'
AUDIT_DIR = Path(os.getenv('AUDIT_DIR', 'validation/audits'))
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
POST_TARGET_EXTRA_WINDOWS = int(os.getenv('POST_TARGET_EXTRA_WINDOWS', '0'))  # extra windows to collect after reaching target minority
MIN_TRAIN_PER_CLASS = int(os.getenv('MIN_TRAIN_PER_CLASS', '30'))
MIN_TEST_PER_CLASS = int(os.getenv('MIN_TEST_PER_CLASS', '10'))
EVAL_REQUIRE_BOTH_CLASSES = os.getenv('EVAL_REQUIRE_BOTH_CLASSES', '1') == '1'
MAX_CLASS_RATIO = float(os.getenv('MAX_CLASS_RATIO', '2.0'))  # max allowed (majority/minority) after target met before stopping
REPORT_CLASSIFICATION = os.getenv('REPORT_CLASSIFICATION', '1') == '1'

# Optional: create feature dump directory
if FEATURE_DUMP_DIR:
    fdd = Path(FEATURE_DUMP_DIR)
    fdd.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Feature dump enabled -> per-window .npz files under {fdd}")

# ---- Minority density scan (pre-stream) ----
minority_scan_report = {}
extra_post_target_remaining = POST_TARGET_EXTRA_WINDOWS  # initialize here
if not DENSITY_SCAN_DISABLE:
    def scan_minority_density(files, label_col: str):
        file_stats = []
        global_counts = {}
        for i, pf in enumerate(files):
            try:
                # Only load label column for speed
                df_lbl = pd.read_parquet(pf, columns=[label_col])
            except Exception:
                continue
            vc = df_lbl[label_col].value_counts()
            if len(vc) == 0:
                continue
            counts_dict = {(
                int(k) if isinstance(k, (int, np.integer)) or (isinstance(k, str) and k.isdigit()) else k
            ): int(v) for k, v in vc.items()}
            for k, v in counts_dict.items():
                global_counts[k] = global_counts.get(k, 0) + v
            file_stats.append({
                'file': str(pf),
                'counts': counts_dict,
                'total': int(vc.sum())
            })
        if len(global_counts) < 2:
            return {
                'file_stats': file_stats,
                'global_counts': global_counts,
                'minority_label': None,
                'note': 'Only one label observed globally; density scan limited.'
            }
        minority_label = min(global_counts, key=lambda k: global_counts[k])
        for fs in file_stats:
            minority_ct = fs['counts'].get(minority_label, 0)
            fs['minority_density'] = minority_ct / fs['total'] if fs['total'] > 0 else 0.0
        return {
            'file_stats': file_stats,
            'global_counts': global_counts,
            'minority_label': minority_label,
            'note': 'OK'
        }
    minority_scan_report = scan_minority_density(parquet_files, label_column)
    scan_path = AUDIT_DIR / f"minority_density_scan_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with scan_path.open('w') as fjs:
        json.dump(minority_scan_report, fjs, indent=2)
    print(f"INFO: Minority density scan written to {scan_path}")
    if minority_scan_report.get('minority_label') is not None:
        print(f"INFO: Global counts {minority_scan_report['global_counts']} minority label={minority_scan_report['minority_label']}")
else:
    print('INFO: Minority density scan disabled via DISABLE_DENSITY_SCAN=1')

# ---- Stream windows and build feature matrix ----
X_features = []
Y_labels = []
WindowTimes = []
WindowOrder = []  # collection order for synthetic fallback

_accum_frames = []
_accum_rows = 0
_accum_counter = 0

# DEBUG BYPASS: Direct sample without streaming if env var set
if os.environ.get('BYPASS_STREAM', '0') == '1':
    print('DEBUG: Bypass streaming window loader; direct point-cloud sample from first parquet file.')
    first_file = parquet_files[0]
    df_sample = pd.read_parquet(first_file).head(5000)
    if label_column not in df_sample.columns:
        raise SystemExit('Label column missing in sample parquet for bypass test')
    label_counts = df_sample[label_column].value_counts()
    majority_label = label_counts.idxmax() if len(label_counts) else 0
    sub = df_sample[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub.clip(lower=-1e12, upper=1e12)
    if sub.empty:
        raise SystemExit('No valid numeric rows after cleaning in bypass sample')
    # Build point cloud directly
    pc = sub.to_numpy(dtype=float)
    # Cap points deterministically
    if pc.shape[0] > vec.cfg.max_vr_points:
        pc = pc[:vec.cfg.max_vr_points]
    # Compute VR topology
    res = vec.vr.compute(pc)
    feat = res['features']
    print('DEBUG: Feature vector shape:', feat.shape)
    print('DEBUG: Majority label:', majority_label)
    print('DEBUG: SUCCESS bypass path; exiting early.')
    raise SystemExit(0)

for w_idx, window_df in enumerate(StreamingWindowLoader(parquet_files, win_cfg).windows()):
    # Safety cap
    if _accum_counter >= MAX_TOTAL_STREAM_WINDOWS:
        print(f"WARNING: Reached MAX_TOTAL_STREAM_WINDOWS={MAX_TOTAL_STREAM_WINDOWS}; stopping stream loop.")
        break
    if label_column not in window_df.columns:
        continue
    _accum_frames.append(window_df[feature_cols + [label_column, win_cfg.time_column]])
    _accum_rows += len(window_df)
    if _accum_rows < MIN_POINTS_ACCUM:
        continue
    acc_df = pd.concat(_accum_frames, axis=0, ignore_index=True)
    label_counts = acc_df[label_column].value_counts()
    if len(label_counts) == 0:
        _accum_frames.clear(); _accum_rows = 0
        continue
    majority_label = label_counts.idxmax()
    sub = acc_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty or sub.shape[0] < 3:
        _accum_frames.clear(); _accum_rows = 0
        continue
    sub = sub.clip(lower=-1e12, upper=1e12)
    start_time_val = float(acc_df[win_cfg.time_column].min())
    cleaned = sub.join(acc_df[[win_cfg.time_column]])
    pc = build_point_cloud(
        window=cleaned,
        feature_columns=feature_cols,
        cap_points=ds_cfg['limits']['max_vr_points'],
        seed=ds_cfg.get('random_seed', 42)
    )
    if pc.shape[0] < 3:
        _accum_frames.clear(); _accum_rows = 0
        continue
    res = vec.vr.compute(pc) if pc.shape[0] <= vec.cfg.max_vr_points else vec.witness.compute(pc)
    # Augment features with diagram-derived Betti curves + lifetime stats + persistence images
    diagrams = res['diagrams']
    betti_vec = _betti.transform(diagrams)
    life_vec = _life.transform(diagrams)
    # Persistence images per dimension
    pimg_vecs = []
    for dim_i in range(min(_PI_DIMS, len(diagrams))):
        dgm = diagrams[dim_i]
        if dgm is None or getattr(dgm, 'size', 0) == 0:
            pimg_vecs.append(np.zeros(_pimg.cfg.resolution[0]*_pimg.cfg.resolution[1], dtype=float))
        else:
            pimg_vecs.append(_pimg.transform_diagram(dgm))
    pimg_vec = np.concatenate(pimg_vecs, axis=0) if pimg_vecs else np.array([], dtype=float)
    # Persistence landscapes per dimension
    land_vecs = []
    for dim_i in range(min(_LAND_DIMS, len(diagrams))):
        dgm = diagrams[dim_i]
        if dgm is None or getattr(dgm, 'size', 0) == 0:
            land_vecs.append(np.zeros(_pland.cfg.resolution * _pland.cfg.k_layers, dtype=float))
        else:
            land_vecs.append(_pland.transform_diagram(dgm))
    land_vec = np.concatenate(land_vecs, axis=0) if land_vecs else np.array([], dtype=float)
    augmented_features = np.concatenate([res['features'], betti_vec, life_vec, pimg_vec, land_vec])
    X_features.append(augmented_features)
    Y_labels.append(int(majority_label))
    WindowTimes.append(start_time_val)
    WindowOrder.append(_accum_counter)
    # Optional per-window dump
    if FEATURE_DUMP_DIR:
        try:
            np.savez_compressed(
                Path(FEATURE_DUMP_DIR) / f"win_{_accum_counter:06d}.npz",
                features=res['features'], label=int(majority_label), start_time=start_time_val, order=_accum_counter
            )
        except Exception as e:
            print(f"WARNING: Feature dump failed for window {_accum_counter}: {e}")
    _accum_counter += 1
    _accum_frames.clear(); _accum_rows = 0
    # Dynamic class ratio early stop (after target reached) to prevent inversion / overshoot
    if len(set(Y_labels)) == 2:
        counts_now = pd.Series(Y_labels).value_counts()
        minority_ct_live = counts_now.min()
        majority_ct_live = counts_now.max()
        ratio_live = majority_ct_live / max(1, minority_ct_live)
        target_min_live = max(MIN_MINORITY_WINDOWS, TARGET_MIN_MINORITY)
        if minority_ct_live >= target_min_live and ratio_live > MAX_CLASS_RATIO:
            print(f"DEBUG: Early stop due to class ratio {ratio_live:.2f} > MAX_CLASS_RATIO={MAX_CLASS_RATIO} (minority_ct={minority_ct_live} >= target_min={target_min_live}).")
            break
    if (_accum_counter % 50) == 0:
        print(f"Accumulated windows processed: {_accum_counter} (classes so far: {set(Y_labels)})")
    # Early stop only if both classes present AND minority count threshold satisfied
    if _accum_counter >= MAX_FEATURE_WINDOWS and len(set(Y_labels)) >= 2:
        # Respect higher target if provided
        target_min = max(MIN_MINORITY_WINDOWS, TARGET_MIN_MINORITY)
        counts_tmp = pd.Series(Y_labels).value_counts()
        minority_ct = counts_tmp.min()
        majority_ct = counts_tmp.max()
        ratio_tmp = majority_ct / max(1, minority_ct)
        if minority_ct >= target_min:
            if ratio_tmp > MAX_CLASS_RATIO:
                print(f"DEBUG: Early stop at {_accum_counter} windows (ratio {ratio_tmp:.2f} > MAX_CLASS_RATIO={MAX_CLASS_RATIO}).")
                break
            if extra_post_target_remaining > 0:
                extra_post_target_remaining -= 1
                if extra_post_target_remaining == 0:
                    print(f"DEBUG: Early stop after collecting post-target extras (target_min={target_min}, extras done, ratio={ratio_tmp:.2f}).")
                    break
                else:
                    print(f"INFO: Target minority reached (minority_ct={minority_ct}); collecting {extra_post_target_remaining} more windows (ratio={ratio_tmp:.2f}).")
            else:
                print(f"DEBUG: Early stop at {_accum_counter} windows (minority_ct={minority_ct} >= target_min={target_min}, ratio={ratio_tmp:.2f}).")
                break

# Fallback quick patch: if still no features, sample directly
if len(X_features) == 0:
    print("WARNING: Streaming produced zero feature windows; activating fallback direct sampling across all files.")
    X_features = []
    Y_labels = []
    WindowTimes = []
    WindowOrder = []
    order_counter = 0
    for file_idx, pf in enumerate(parquet_files):
        if len(set(Y_labels)) >= 2:
            break
        df_all = pd.read_parquet(pf)
        sample_head = df_all.head(10000)
        sample_tail = df_all.tail(10000)
        sample_block = pd.concat([sample_head, sample_tail], axis=0)
        if label_column not in sample_block.columns:
            continue
        block_size = max(500, MIN_POINTS_ACCUM)
        blocks = [sample_block.iloc[i:i+block_size] for i in range(0, len(sample_block), block_size)]
        for bi, blk in enumerate(blocks):
            if order_counter >= MAX_FEATURE_WINDOWS and len(set(Y_labels)) >= 2:
                break
            lbl_counts = blk[label_column].value_counts()
            if len(lbl_counts) == 0:
                continue
            maj = lbl_counts.idxmax()
            sub = blk[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty:
                continue
            sub = sub.clip(lower=-1e12, upper=1e12)
            pc = sub.to_numpy(dtype=float)
            if pc.shape[0] > vec.cfg.max_vr_points:
                pc = pc[:vec.cfg.max_vr_points]
            try:
                res = vec.vr.compute(pc)
            except Exception:
                continue
            diagrams = res['diagrams']
            betti_vec = _betti.transform(diagrams)
            life_vec = _life.transform(diagrams)
            pimg_vecs = []
            for dim_i in range(min(_PI_DIMS, len(diagrams))):
                dgm = diagrams[dim_i]
                if dgm is None or getattr(dgm, 'size', 0) == 0:
                    pimg_vecs.append(np.zeros(_pimg.cfg.resolution[0]*_pimg.cfg.resolution[1], dtype=float))
                else:
                    pimg_vecs.append(_pimg.transform_diagram(dgm))
            pimg_vec = np.concatenate(pimg_vecs, axis=0) if pimg_vecs else np.array([], dtype=float)
            land_vecs = []
            for dim_i in range(min(_LAND_DIMS, len(diagrams))):
                dgm = diagrams[dim_i]
                if dgm is None or getattr(dgm, 'size', 0) == 0:
                    land_vecs.append(np.zeros(_pland.cfg.resolution * _pland.cfg.k_layers, dtype=float))
                else:
                    land_vecs.append(_pland.transform_diagram(dgm))
            land_vec = np.concatenate(land_vecs, axis=0) if land_vecs else np.array([], dtype=float)
            augmented_features = np.concatenate([res['features'], betti_vec, life_vec, pimg_vec, land_vec])
            X_features.append(augmented_features)
            Y_labels.append(int(maj))
            WindowTimes.append(float(blk[win_cfg.time_column].min()) if win_cfg.time_column in blk.columns else order_counter)
            WindowOrder.append(order_counter)
            order_counter += 1
        print(f"Fallback file {file_idx+1}/{len(parquet_files)} processed; classes: {set(Y_labels)}")
    if len(X_features) == 0:
        raise SystemExit('Fallback sampling also produced zero features; deeper investigation needed.')
    print(f"Fallback produced {len(X_features)} feature vectors (classes: {set(Y_labels)}).")

# --- Targeted minority class acquisition (trigger if single class OR minority below threshold) ---
label_set = set(Y_labels)
if len(label_set) == 0:
    raise SystemExit("No windows collected.")
if len(label_set) == 1 or (len(label_set) == 2 and min(pd.Series(Y_labels).value_counts()) < MIN_MINORITY_WINDOWS):
    counts_now = pd.Series(Y_labels).value_counts()
    minority_label = counts_now.idxmin() if len(label_set) == 2 else (1 if list(label_set)[0] == 0 else 0)
    # target additional minority windows until threshold satisfied
    print(f"INFO: Initiating targeted minority search for label={minority_label}; current minority count={counts_now.get(minority_label,0)} target>={MIN_MINORITY_WINDOWS}.")
    added_minority_count = 0
    local_radius = 250
    max_targets_per_file = 20
    for pf_idx, pf in enumerate(parquet_files):
        if pd.Series(Y_labels).value_counts().get(minority_label,0) >= MIN_MINORITY_WINDOWS:
            break
        try:
            df_lbl = pd.read_parquet(pf, columns=[label_column] + feature_cols + [win_cfg.time_column])
        except Exception:
            try:
                df_lbl = pd.read_parquet(pf)
            except Exception:
                continue
        if label_column not in df_lbl.columns:
            continue
        mask = (df_lbl[label_column] == minority_label)
        if not mask.any():
            continue
        pos_indices = np.where(mask.to_numpy())[0]
        for pi in pos_indices[:max_targets_per_file]:
            if pd.Series(Y_labels).value_counts().get(minority_label,0) >= MIN_MINORITY_WINDOWS:
                break
            start = max(0, pi - local_radius)
            end = min(len(df_lbl), pi + local_radius)
            blk = df_lbl.iloc[start:end]
            sub = blk[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty or sub.shape[0] < 3:
                continue
            sub = sub.clip(lower=-1e12, upper=1e12)
            pc = sub.to_numpy(dtype=float)
            if pc.shape[0] > vec.cfg.max_vr_points:
                pc = pc[:vec.cfg.max_vr_points]
            try:
                res = vec.vr.compute(pc)
            except Exception:
                continue
            diagrams = res['diagrams']
            betti_vec = _betti.transform(diagrams)
            life_vec = _life.transform(diagrams)
            augmented_features = np.concatenate([res['features'], betti_vec, life_vec])
            X_features.append(augmented_features)
            Y_labels.append(int(minority_label))
            WindowTimes.append(float(blk[win_cfg.time_column].min()) if win_cfg.time_column in blk.columns else len(WindowTimes))
            WindowOrder.append(len(WindowOrder))
            if FEATURE_DUMP_DIR:
                try:
                    np.savez_compressed(
                        Path(FEATURE_DUMP_DIR) / f"targeted_{len(WindowOrder):06d}.npz",
                        features=res['features'], label=int(minority_label), start_time=float(blk[win_cfg.time_column].min()) if win_cfg.time_column in blk.columns else len(WindowTimes), order=len(WindowOrder)
                    )
                except Exception as e:
                    print(f"WARNING: Targeted feature dump failed: {e}")
            added_minority_count += 1
        print(f"Targeted scan file {pf_idx+1}/{len(parquet_files)} added so far minority={pd.Series(Y_labels).value_counts().get(minority_label,0)}")
    final_minority_ct = pd.Series(Y_labels).value_counts().get(minority_label,0)
    if final_minority_ct < MIN_MINORITY_WINDOWS:
        print(f"WARNING: Minority label {minority_label} count={final_minority_ct} < required {MIN_MINORITY_WINDOWS}.")
    else:
        print(f"SUCCESS: Achieved minority label {minority_label} count={final_minority_ct} (>= {MIN_MINORITY_WINDOWS}).")

# Final class diversity guard
if len(set(Y_labels)) < 2:
    raise SystemExit(f"Still single class after targeted search (labels={set(Y_labels)}). Cannot train.")

# Build arrays and compute global counts before split
X = np.vstack(X_features)
# Record feature schema manifest (topological components only)
feature_manifest = {
    'base_feature_length': int(len(X_features[0]) if X_features else 0),
    'components': [
        'analyzer.extract_features()',
        'BettiCurveVectorizer(radii_resolution=40,maxdim=2)',
        'LifetimeStatsVectorizer(top_k=5,maxdim=2)',
        'PersistenceImageVectorizer(resolution=(16,16),dims=H0..H1)',
        'PersistenceLandscapeVectorizer(resolution=60,k_layers=3,dims=H0..H1)'
    ],
    'ordering_note': 'Concatenated in listed order',
}
y = np.array(Y_labels)
WindowTimes = np.array(WindowTimes)
WindowOrder = np.array(WindowOrder)
class_counts = np.bincount(y)
class_ratio = float(class_counts.max() / class_counts.min()) if class_counts.min() > 0 else None
# After global class counts print (audit logging)
print(f"INFO: Global class counts: {dict(enumerate(class_counts))}")
# Persist audit metadata
# Helper to make JSON-safe
def _json_safe(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return [_json_safe(x) for x in o.tolist()]
    if isinstance(o, (dict,)):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_json_safe(v) for v in o]
    return o
final_audit = {
    'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
    'config': {
        'MIN_POINTS_ACCUM': MIN_POINTS_ACCUM,
        'MAX_FEATURE_WINDOWS': MAX_FEATURE_WINDOWS,
        'MIN_MINORITY_WINDOWS': MIN_MINORITY_WINDOWS,
        'TARGET_MIN_MINORITY': TARGET_MIN_MINORITY,
        'MAX_TOTAL_STREAM_WINDOWS': MAX_TOTAL_STREAM_WINDOWS,
        'POST_TARGET_EXTRA_WINDOWS': POST_TARGET_EXTRA_WINDOWS,
        'MIN_TRAIN_PER_CLASS': MIN_TRAIN_PER_CLASS,
        'MIN_TEST_PER_CLASS': MIN_TEST_PER_CLASS,
        'MAX_CLASS_RATIO': MAX_CLASS_RATIO
    },
    'class_counts': dict(enumerate(class_counts)),
    'class_ratio': class_ratio,
    'minority_density_scan': minority_scan_report if minority_scan_report else None
}
final_audit = _json_safe(final_audit)
final_audit_path = AUDIT_DIR / f"collection_audit_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
try:
    with final_audit_path.open('w') as fa:
        json.dump(final_audit, fa, indent=2)
    print(f"INFO: Collection audit written to {final_audit_path}")
except Exception as e:
    print(f"WARNING: Failed to write collection audit: {e}")

if class_counts.min() == 0:
    raise SystemExit("One class has zero count globally; adjust sampling or data source before training.")

# Detect duplicate / synthetic timestamps (e.g., per-file synthetic resets)
unique_ts = len(set(WindowTimes))
use_collection_order = False  # placeholder initialization; real value set after time uniqueness check
use_collection_order = unique_ts < len(WindowTimes)
if use_collection_order:
    print("WARNING: Duplicate window start timestamps detected; using collection order for temporal split (synthetic time).")
    order = np.argsort(WindowOrder)
else:
    order = np.argsort(WindowTimes)

X, y, WindowTimes, WindowOrder = X[order], y[order], WindowTimes[order], WindowOrder[order]

# Temporal split (collection-order if synthetic)
split_idx = int(0.7 * len(X))
# Initial provisional split
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Adjust split to guarantee both classes in train and test (stratified temporal adjustment)
if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
    minority_label = 1 if (y == 1).sum() < (y == 0).sum() else 0
    minority_indices = np.where(y == minority_label)[0]
    total_minority = len(minority_indices)
    if total_minority == 0:
        raise SystemExit("No minority class present globally after collection; aborting.")
    # Target fraction of minority in train, leave at least one for test
    if total_minority == 1:
        # Force that single minority into train; test will lack minority (report clearly)
        split_idx = max(split_idx, minority_indices[0] + 1)
    else:
        train_minority_target = min(max(1, int(0.7 * total_minority)), total_minority - 1)
        split_idx = max(split_idx, minority_indices[train_minority_target - 1] + 1)
    # Ensure test not empty and retains at least one minority
    if split_idx >= len(X) - 1:
        split_idx = len(X) - 2
    # If test lost all minority, move boundary left before last minority
    if (y[split_idx:] == minority_label).sum() == 0 and total_minority > 1:
        # place boundary before last minority to keep it in test
        last_minority = minority_indices[-1]
        if last_minority > 0:
            split_idx = max(min(split_idx, last_minority), 1)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"INFO: Adjusted split_idx={split_idx} for stratified temporal split. Train minority={(y_train==minority_label).sum()} / {total_minority}; Test minority={(y_test==minority_label).sum()}.")

# After stratified temporal adjustment, enforce bidirectional class presence if required
if EVAL_REQUIRE_BOTH_CLASSES:
    train_unique = set(np.unique(y_train))
    test_unique = set(np.unique(y_test))
    all_classes = set(np.unique(y))
    missing_in_test = all_classes - test_unique
    missing_in_train = all_classes - train_unique
    # If test missing a class but dataset tail is pure other class and no future windows remain, flag insufficiency.
    if missing_in_test:
        # Determine last index per class
        last_indices = {c: np.where(y == c)[0][-1] for c in all_classes}
        # If for any missing class its last index < split_idx we can shift boundary earlier only if it preserves temporal ordering (reduces train, enlarges test with earlier windows) leading to leakage risk.
        # We forbid moving earlier windows into test (would leak) -> require additional collection instead.
        print(f"WARNING: Test split missing classes {missing_in_test}; cannot inject earlier windows without temporal leakage. Marking evaluation as NOT READY.")
    if missing_in_train:
        print(f"WARNING: Train split missing classes {missing_in_train}; evaluation NOT READY.")
    # Minimum per-class counts check
    train_counts = {int(c): int((y_train==c).sum()) for c in all_classes}
    test_counts = {int(c): int((y_test==c).sum()) for c in all_classes}
    for c in all_classes:
        if train_counts[c] < MIN_TRAIN_PER_CLASS or test_counts[c] < MIN_TEST_PER_CLASS:
            print(f"WARNING: Per-class count threshold unmet: class {c}: train={train_counts[c]} (min {MIN_TRAIN_PER_CLASS}), test={test_counts[c]} (min {MIN_TEST_PER_CLASS}). Evaluation NOT READY.")
    # Abort training if evaluation not ready
    if (missing_in_test or missing_in_train or any(train_counts[c] < MIN_TRAIN_PER_CLASS or test_counts[c] < MIN_TEST_PER_CLASS for c in all_classes)):
        raise SystemExit("EVALUATION_NOT_READY: Insufficient balanced temporal split. Increase POST_TARGET_EXTRA_WINDOWS or collection thresholds and re-run.")

# Recompute times for integrity check
max_train_time = WindowTimes[:split_idx].max()
min_test_time = WindowTimes[split_idx:].min()

if not use_collection_order:
    if max_train_time >= min_test_time:
        raise SystemExit("Temporal leakage detected: training window end overlaps test window start")
else:
    print("INFO: Temporal leakage guard disabled (synthetic timestamps). Results are NOT temporally validated.")

# After computing train/test split but before model instantiation, compute class weights
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
if pos == 0 or neg == 0:
    raise SystemExit(f"Train or test split still single class after stratified adjustment (train neg={neg}, pos={pos}).")
base_score = pos / (pos + neg)
scale_pos_weight = neg / pos
print(f"INFO: Class distribution train: neg={neg}, pos={pos}, base_score={base_score:.4f}, scale_pos_weight={scale_pos_weight:.2f}")

# ---- Optional simple temporal tuning (single pass, no CV) ----
ENABLE_TUNING = os.getenv('ENABLE_TUNING', '0') == '1'
TUNING_RESULTS = []
final_params = None
if ENABLE_TUNING:
    print("INFO: ENABLE_TUNING=1 -> starting simple temporal holdout tuning.")
    inner_split = int(0.8 * len(X_train)) if len(X_train) > 50 else max(10, int(0.7 * len(X_train)))
    X_inner_tr, X_inner_val = X_train[:inner_split], X_train[inner_split:]
    y_inner_tr, y_inner_val = y_train[:inner_split], y_train[inner_split:]
    if np.unique(y_inner_val).size < 2:
        print("WARNING: Inner validation split lacks class diversity; skipping tuning.")
    else:
        param_grid = [
            {'max_depth': d, 'learning_rate': lr, 'n_estimators': n}
            for d in (3, 5)
            for lr in (0.05, 0.1)
            for n in (100, 200)
        ]
        best_score = -1.0
        for pg in param_grid:
            m = XGBClassifier(
                n_estimators=pg['n_estimators'],
                max_depth=pg['max_depth'],
                learning_rate=pg['learning_rate'],
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=ds_cfg.get('random_seed', 42),
                n_jobs=4,
                verbosity=0,
                base_score=base_score,
                scale_pos_weight=scale_pos_weight
            )
            try:
                m.fit(X_inner_tr, y_inner_tr)
                val_proba = m.predict_proba(X_inner_val)[:, 1]
                pr_auc_inner = average_precision_score(y_inner_val, val_proba)
                roc_auc_inner = roc_auc_score(y_inner_val, val_proba)
                score = pr_auc_inner  # primary criterion
                TUNING_RESULTS.append({
                    'params': pg,
                    'pr_auc': float(pr_auc_inner),
                    'roc_auc': float(roc_auc_inner)
                })
                if score > best_score:
                    best_score = score
                    final_params = pg
            except Exception as e:
                TUNING_RESULTS.append({'params': pg, 'error': str(e)})
        if final_params:
            print(f"INFO: Tuning selected params: {final_params} (best PR-AUC={best_score:.4f})")
        else:
            print("WARNING: No valid tuning configuration succeeded; falling back to defaults.")

# Base / fallback parameters
if final_params is None:
    final_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05
    }

model = XGBClassifier(
    n_estimators=final_params['n_estimators'],
    max_depth=final_params['max_depth'],
    learning_rate=final_params['learning_rate'],
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=ds_cfg.get('random_seed', 42),
    n_jobs=4,
    verbosity=0,
    base_score=base_score,
    scale_pos_weight=scale_pos_weight
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# --- Added probabilistic predictions for ROC/PR metrics ---
try:
    y_proba = model.predict_proba(X_test)[:, 1]
except Exception:
    y_proba = None
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Train windows:", X_train.shape[0], "Test windows:", X_test.shape[0])
print("Temporal split integrity (enforced):", (not use_collection_order) and (max_train_time < min_test_time))
print("Baseline GBT (topological features) Accuracy:", acc)
print("Confusion Matrix:\n", cm)
# --- Compute ROC-AUC & PR-AUC (Average Precision) ---
roc_auc = None
pr_auc = None
roc_curve_pts = None
pr_curve_pts = None
if y_proba is not None and len(np.unique(y_test)) == 2:
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, roc_thr = roc_curve(y_test, y_proba)
        roc_curve_pts = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': roc_thr.tolist()}
    except Exception as e:
        print(f"WARNING: roc_auc_score failed: {e}")
    try:
        pr_auc = average_precision_score(y_test, y_proba)
        precision, recall, pr_thr = precision_recall_curve(y_test, y_proba)
        pr_curve_pts = {'precision': precision.tolist(), 'recall': recall.tolist(), 'thresholds': pr_thr.tolist()}
    except Exception as e:
        print(f"WARNING: average_precision_score failed: {e}")
print(f"ROC-AUC: {roc_auc if roc_auc is not None else 'NA'}  PR-AUC (AvgPrecision): {pr_auc if pr_auc is not None else 'NA'}")
if REPORT_CLASSIFICATION:
    try:
        print("Classification Report (test):")
        print(classification_report(y_test, y_pred, digits=4))
        test_counts = pd.Series(y_test).value_counts().to_dict()
        # Determine minority label robustly
        minority_label_test = sorted(test_counts.items(), key=lambda kv: kv[1])[0][0]
        minority_precision = ((y_pred==minority_label_test) & (y_test==minority_label_test)).sum() / max(1, (y_pred==minority_label_test).sum())
        minority_recall = ((y_pred==minority_label_test) & (y_test==minority_label_test)).sum() / max(1, (y_test==minority_label_test).sum())
        print(f"Minority label (test)={minority_label_test} precision={minority_precision:.4f} recall={minority_recall:.4f}")
    except Exception as e:
        print(f"WARNING: Failed to generate classification report: {e}")

# --- Persist metrics to JSON (validation output) ---
try:
    metrics_dir = Path('validation/metrics'); metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_record = {
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'accuracy': float(acc),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'pr_auc': float(pr_auc) if pr_auc is not None else None,
        'train_counts': {'neg': int(neg), 'pos': int(pos)},
        'test_counts': { (int(k) if isinstance(k, (int, np.integer)) else str(k)): int(v) for k, v in pd.Series(y_test).value_counts().items() },
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
        'tuning_enabled': ENABLE_TUNING,
        'tuning_final_params': final_params,
        'tuning_results': TUNING_RESULTS,
        'roc_curve': roc_curve_pts,
        'pr_curve': pr_curve_pts,
        'feature_manifest': feature_manifest
    }
    out_path = metrics_dir / f"baseline_gbt_metrics_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with out_path.open('w') as mf:
        json.dump(metrics_record, mf, indent=2)
    print(f"INFO: Metrics written to {out_path}")
except Exception as e:
    print(f"WARNING: Failed to persist metrics JSON: {e}")

# Persist model + vector stack manifest for reproducibility
try:
    model_dir = Path('validation/models'); model_dir.mkdir(parents=True, exist_ok=True)
    # Save XGBoost model
    model_out = model_dir / f"baseline_gbt_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    model.save_model(model_out)
    manifest_out = model_dir / f"feature_manifest_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with manifest_out.open('w') as mfjs:
        json.dump(feature_manifest, mfjs, indent=2)
    print(f"INFO: Model saved to {model_out}; feature manifest saved to {manifest_out}")
except Exception as e:
    print(f"WARNING: Failed to persist model/manifest: {e}")

# Pre-train class diversity guard for train split
if np.unique(y_train).size < 2:
    raise SystemExit(f"Training split has single class (labels={np.unique(y_train)}). Adjust sampling/windowing.")
