#!/usr/bin/env python3
"""
Batch Dataset Analysis for TDA Suitability
=========================================

Automates comprehensive analysis of all datasets in /data/apt_datasets
using the logic from analyze_unsw_nb15_dataset.py.

Author: TDA Project Team
Date: August 7, 2025
"""


import pandas as pd
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Dask for large file support
try:
    import dask.dataframe as dd
except ImportError:
    dd = None

PROJECT_ROOT = Path("/home/stephen-dorman/dev/TDA_projects")
DATA_ROOT = PROJECT_ROOT / "data" / "apt_datasets"
OUTPUT_ROOT = PROJECT_ROOT / "notebooks" / "dataset_analysis_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Supported file extensions
FILE_EXTS = ['.parquet', '.csv']

# Helper: Find all dataset files
def find_dataset_files():
    dataset_files = {}
    for dataset_dir in DATA_ROOT.iterdir():
        if dataset_dir.is_dir():
            files = [f for f in dataset_dir.glob("**/*") if f.suffix in FILE_EXTS]
            if files:
                dataset_files[dataset_dir.name] = files
    # Also include top-level files
    top_files = [f for f in DATA_ROOT.glob("*.parquet") if f.suffix in FILE_EXTS]
    if top_files:
        dataset_files['top_level'] = top_files
    return dataset_files

# Helper: Load a dataset file

def load_dataset(file_path, use_dask=False):
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"[MEM] Before loading {file_path}: {psutil.virtual_memory().percent}% used")
    adaptive_sample_size = 100000
    if file_size_mb > 1000:
        adaptive_sample_size = 10000
    elif file_size_mb > 500:
        adaptive_sample_size = 25000
    try:
        if use_dask and dd is not None:
            if file_path.suffix == '.parquet':
                df = dd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = dd.read_csv(file_path, blocksize="16MB", assume_missing=True)
            else:
                print(f"Unsupported file type for Dask: {file_path}")
                return None
            # Stratified sampling
            label_column = None
            possible_labels = ['attack_cat', 'label', 'Label', 'Attack_Type', 'attack_type', 'Class', 'class']
            for col in df.columns:
                if col in possible_labels:
                    label_column = col
                    break
            if label_column:
                df_sample = stratified_sample_dask(df, label_column, adaptive_sample_size)
                df_sample = df_sample.compute()
            else:
                df_sample = df.sample(n=adaptive_sample_size, random_state=42).compute()
        else:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                print(f"Unsupported file type for pandas: {file_path}")
                return None
            label_column = None
            possible_labels = ['attack_cat', 'label', 'Label', 'Attack_Type', 'attack_type', 'Class', 'class']
            for col in df.columns:
                if col in possible_labels:
                    label_column = col
                    break
            if label_column:
                df_sample = df.groupby(label_column, group_keys=False).apply(
                    lambda x: x.sample(n=max(1, int(len(x) / len(df) * adaptive_sample_size)), random_state=42)
                )
            else:
                df_sample = df.sample(n=min(adaptive_sample_size, len(df)), random_state=42)
        print(f"[MEM] After loading {file_path}: {psutil.virtual_memory().percent}% used")
        gc.collect()
        print(f"[MEM] After gc.collect() for {file_path}: {psutil.virtual_memory().percent}% used")
        return df_sample
    except MemoryError as me:
        print(f"[ERROR] MemoryError for {file_path}: {me}")
        return None
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# Main analysis logic (adapted from analyze_unsw_nb15_dataset.py)

def stratified_sample_dask(df, label_column, sample_size=100000):
    class_counts = df[label_column].value_counts().compute()
    total = class_counts.sum()
    samples = []
        for cls, count in class_counts.items():
            frac = count / total
            n_cls = max(1, int(frac * sample_size))
            cls_df = df[df[label_column] == cls].sample(n=n_cls, random_state=42)
            samples.append(cls_df)
        sampled_df = pd.concat(samples, ignore_index=True)
    return sampled_df

def analyze_dataset(dataset_name, file_paths):
    print(f"\n{'='*80}\nAnalyzing dataset: {dataset_name}\n{'='*80}")
    dfs = []
    use_dask = any(fp.stat().st_size > 500 * 1024 * 1024 for fp in file_paths)
    for fp in file_paths:
        print(f"Loading: {fp}")
        df = load_dataset(fp, use_dask=use_dask)
        if df is not None:
            dfs.append(df)
        gc.collect()
        print(f"[MEM] After loading and gc.collect() for {fp}: {psutil.virtual_memory().percent}% used")
    if not dfs:
        print(f"❌ No valid files loaded for {dataset_name}")
        return
    if use_dask:
        full_df = pd.concat(dfs, ignore_index=True)
        columns = full_df.columns
        label_column = None
        possible_labels = ['attack_cat', 'label', 'Label', 'Attack_Type', 'attack_type', 'Class', 'class']
        for col in possible_labels:
            if col in columns:
                label_column = col
                break
        if label_column:
            sample_df = full_df.groupby(label_column, group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(len(x) / len(full_df) * 100000)), random_state=42)
            )
        else:
            sample_df = full_df.sample(n=min(100000, len(full_df)), random_state=42)
        total_samples = len(full_df)
    else:
        full_df = pd.concat(dfs, ignore_index=True)
        columns = full_df.columns
        label_column = None
        possible_labels = ['attack_cat', 'label', 'Label', 'Attack_Type', 'attack_type', 'Class', 'class']
        for col in possible_labels:
            if col in columns:
                label_column = col
                break
        if label_column:
            sample_df = full_df.groupby(label_column, group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(len(x) / len(full_df) * 100000)), random_state=42)
            )
        else:
            sample_df = full_df.sample(n=min(100000, len(full_df)), random_state=42)
        total_samples = len(full_df)
    gc.collect()
    print(f"[MEM] After analysis and gc.collect() for {dataset_name}: {psutil.virtual_memory().percent}% used")

    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # --- Structural Analysis ---
    feature_info = {}
    for col in columns:
        dtype = str(sample_df[col].dtype)
        feature_info[col] = {
            'dtype': dtype,
            'unique_values': sample_df[col].nunique(),
            'missing_percentage': (sample_df[col].isna().sum() / len(sample_df)) * 100
        }
    numerical_features = [col for col in columns if sample_df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in columns if sample_df[col].dtype == 'object']
    network_flow_features = [col for col in columns if any(k in col.lower() for k in ['src', 'dst', 'port', 'proto', 'dur', 'byte', 'packet'])]
    temporal_features = [col for col in columns if any(k in col.lower() for k in ['time', 'dur', 'start', 'end'])]
    structural_analysis = {
        "name": dataset_name,
        "file_count": len(file_paths),
        "total_samples": int(total_samples),
        "feature_count": len(columns),
        "feature_types": {
            "numerical": len(numerical_features),
            "categorical": len(categorical_features),
            "temporal": len(temporal_features),
            "network_flow": len(network_flow_features)
        },
        "missing_values": {
            "columns_with_missing": [col for col, info in feature_info.items() if info['missing_percentage'] > 0],
            "total_missing_percentage": float(np.mean([info['missing_percentage'] for info in feature_info.values()]))
        },
        "feature_details": feature_info
    }
    with open(output_dir / "structural_analysis.json", 'w') as f:
        json.dump(structural_analysis, f, indent=2)
    print(f"Total samples: {structural_analysis['total_samples']:,}")
    print(f"Total features: {structural_analysis['feature_count']}")

    # --- Class Distribution Analysis ---
    if label_column:
        class_distribution = sample_df[label_column].value_counts()
        class_percentages = (class_distribution / len(sample_df)) * 100
        class_analysis = {
            "label_column": label_column,
            "attack_types": class_distribution.to_dict(),
            "class_percentages": class_percentages.to_dict(),
            "normal_vs_attack_ratio": None,
            "imbalance_severity": None,
            "smallest_class_percentage": float(class_percentages.min()),
            "largest_class_percentage": float(class_percentages.max())
        }
        normal_classes = [cls for cls in class_distribution.index if any(k in str(cls).lower() for k in ['normal', 'benign'])]
        if normal_classes:
            normal_count = sum(class_distribution[cls] for cls in normal_classes)
            attack_count = len(sample_df) - normal_count
            class_analysis["normal_vs_attack_ratio"] = normal_count / attack_count if attack_count > 0 else float('inf')
        if class_analysis["smallest_class_percentage"] < 1:
            class_analysis["imbalance_severity"] = "severe"
        elif class_analysis["smallest_class_percentage"] < 5:
            class_analysis["imbalance_severity"] = "moderate"
        else:
            class_analysis["imbalance_severity"] = "balanced"
        with open(output_dir / "class_analysis.json", 'w') as f:
            json.dump(class_analysis, f, indent=2)
        # Plot
        plt.figure(figsize=(12, 8))
        class_distribution.plot(kind='bar')
        plt.title(f'{dataset_name} Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class imbalance severity: {class_analysis['imbalance_severity']}")
    else:
        print(f"No label column found for {dataset_name}")

    # --- Temporal Structure Analysis ---
    timestamp_columns = [col for col in columns if any(k in col.lower() for k in ['time', 'timestamp', 'date'])]
    temporal_analysis = {
        "has_timestamps": len(timestamp_columns) > 0,
        "timestamp_columns": timestamp_columns,
        "co_occurrence_verified": False
    }
    if timestamp_columns and label_column:
        ts_col = timestamp_columns[0]
        try:
            if sample_df[ts_col].dtype == 'object':
                timestamps = pd.to_datetime(sample_df[ts_col], errors='coerce')
            else:
                timestamps = pd.to_datetime(sample_df[ts_col], unit='s', errors='coerce')
            valid_timestamps = timestamps.dropna()
            if len(valid_timestamps) > 0:
                df_with_time = sample_df.copy()
                df_with_time['parsed_timestamp'] = timestamps
                df_with_time = df_with_time.dropna(subset=['parsed_timestamp'])
                normal_classes = [cls for cls in sample_df[label_column].unique() if any(k in str(cls).lower() for k in ['normal', 'benign'])]
                if len(df_with_time) > 0 and normal_classes:
                    df_with_time['hour'] = df_with_time['parsed_timestamp'].dt.hour
                    hours_with_normal = set()
                    hours_with_attacks = set()
                    for hour in df_with_time['hour'].unique():
                        hour_data = df_with_time[df_with_time['hour'] == hour]
                        normal_in_hour = any(hour_data[label_column].isin(normal_classes))
                        attack_in_hour = any(~hour_data[label_column].isin(normal_classes))
                        if normal_in_hour:
                            hours_with_normal.add(hour)
                        if attack_in_hour:
                            hours_with_attacks.add(hour)
                    overlapping_hours = hours_with_normal.intersection(hours_with_attacks)
                    temporal_analysis["co_occurrence_verified"] = len(overlapping_hours) > 0
        except Exception as e:
            print(f"Timestamp parsing failed: {e}")
    with open(output_dir / "temporal_analysis.json", 'w') as f:
        json.dump(temporal_analysis, f, indent=2)

    # --- TDA Suitability Assessment ---
    tda_suitable_features = [col for col in numerical_features if col != label_column and sample_df[col].nunique() > 10]
    graph_features = [col for col in columns if any(k in col.lower() for k in ['src', 'dst', 'sport', 'dport', 'proto'])]
    tda_suitability = {
        "tda_suitable_features": tda_suitable_features[:20],
        "total_tda_features": len(tda_suitable_features),
        "graph_constructible": len(graph_features) >= 2,
        "graph_features": graph_features,
        "time_series_constructible": len(temporal_features) > 0,
        "tda_readiness_score": 0
    }
    score = 0
    if len(tda_suitable_features) >= 10: score += 1
    if tda_suitability["graph_constructible"]: score += 1
    if tda_suitability["time_series_constructible"]: score += 1
    if temporal_analysis["co_occurrence_verified"]: score += 1
    tda_suitability["tda_readiness_score"] = score
    with open(output_dir / "tda_suitability.json", 'w') as f:
        json.dump(tda_suitability, f, indent=2)
    print(f"TDA readiness score: {score}/4")

    # --- Data Leakage Risk Assessment ---
    ids_features = [col for col in columns if any(k in col.lower() for k in ['alert', 'anomaly', 'score', 'detection', 'ids', 'classify'])]
    future_info_features = [col for col in columns if any(k in col.lower() for k in ['response', 'result', 'outcome', 'verdict'])]
    leakage_risks = {
        "ids_detection_features": ids_features,
        "future_information": future_info_features,
        "temporal_ordering_issues": not temporal_analysis["co_occurrence_verified"],
        "high_risk_features_to_remove": ids_features + future_info_features
    }
    with open(output_dir / "leakage_risks.json", 'w') as f:
        json.dump(leakage_risks, f, indent=2)
    print(f"High-risk features to remove: {len(leakage_risks['high_risk_features_to_remove'])}")

    # --- Summary ---
    summary = {
        "dataset_name": dataset_name,
        "total_samples": int(total_samples),
        "tda_readiness_score": score,
        "key_findings": [
            f"Class imbalance severity: {class_analysis['imbalance_severity']}" if label_column else "No label column found",
            f"TDA suitable features: {len(tda_suitable_features)}",
            f"Temporal co-occurrence: {temporal_analysis['co_occurrence_verified']}",
            f"Data leakage risks: {len(leakage_risks['high_risk_features_to_remove'])} risky features"
        ]
    }
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Analysis completed for {dataset_name}. Results saved to {output_dir}")

if __name__ == "__main__":
    dataset_files = find_dataset_files()
    for dataset_name, file_paths in dataset_files.items():
        analyze_dataset(dataset_name, file_paths)
