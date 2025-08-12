#!/usr/bin/env python3
"""
Dataset Analysis: UNSW-NB15
==========================

Comprehensive analysis of UNSW-NB15 dataset for TDA suitability.
Following UNIFIED_AGENT_INSTRUCTIONS.md requirements.

Author: TDA Project Team
Date: August 7, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Project structure setup
PROJECT_ROOT = Path("/home/stephen-dorman/dev/TDA_projects")
DATA_DIR = PROJECT_ROOT / "data" / "apt_datasets" / "UNSW-NB15"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "dataset_analysis_results" / "unsw_nb15"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_unsw_nb15():
    """
    Comprehensive analysis of UNSW-NB15 dataset following project requirements.
    
    Critical Checkpoints:
    1. Data Quality Gate
    2. Temporal Integrity Gate  
    3. TDA Viability Gate
    4. Baseline Performance Gate
    """
    
    print("=" * 80)
    print("UNSW-NB15 Dataset Analysis for TDA Suitability")
    print("=" * 80)
    
    # Phase 1: Load and Basic Structure
    print("\n🔍 Phase 1: Loading and Basic Structure Analysis")
    print("-" * 60)
    
    try:
        # Load training and testing datasets
        train_path = DATA_DIR / "UNSW_NB15_training-set.parquet"
        test_path = DATA_DIR / "UNSW_NB15_testing-set.parquet"
        
        print(f"Loading training data from: {train_path}")
        train_df = pd.read_parquet(train_path)
        
        print(f"Loading testing data from: {test_path}")
        test_df = pd.read_parquet(test_path)
        
        print(f"✅ Training set loaded: {train_df.shape}")
        print(f"✅ Testing set loaded: {test_df.shape}")
        
    except Exception as e:
        print(f"❌ CHECKPOINT 1 FAILED: Data loading error: {e}")
        return False
    
    # Combine for full analysis
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"✅ Combined dataset: {full_df.shape}")
    
    # Phase 2: Structural Analysis
    print("\n📊 Phase 2: Structural Analysis")
    print("-" * 60)
    
    # Feature types analysis
    feature_info = {}
    for col in full_df.columns:
        dtype = str(full_df[col].dtype)
        feature_info[col] = {
            'dtype': dtype,
            'unique_values': full_df[col].nunique(),
            'missing_percentage': (full_df[col].isna().sum() / len(full_df)) * 100
        }
    
    # Categorize features
    numerical_features = [col for col in full_df.columns 
                         if full_df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in full_df.columns 
                           if full_df[col].dtype == 'object']
    
    # Network flow features (typical for UNSW-NB15)
    network_flow_features = [col for col in full_df.columns 
                            if any(keyword in col.lower() for keyword in 
                                 ['src', 'dst', 'port', 'proto', 'dur', 'byte', 'packet'])]
    
    temporal_features = [col for col in full_df.columns 
                        if any(keyword in col.lower() for keyword in 
                             ['time', 'dur', 'start', 'end'])]
    
    structural_analysis = {
        "name": "UNSW-NB15",
        "file_format": "Parquet",
        "total_samples": len(full_df),
        "feature_count": len(full_df.columns),
        "feature_types": {
            "numerical": len(numerical_features),
            "categorical": len(categorical_features),
            "temporal": len(temporal_features),
            "network_flow": len(network_flow_features)
        },
        "missing_values": {
            "columns_with_missing": [col for col, info in feature_info.items() 
                                   if info['missing_percentage'] > 0],
            "total_missing_percentage": (full_df.isna().sum().sum() / 
                                       (len(full_df) * len(full_df.columns))) * 100
        },
        "feature_details": feature_info
    }
    
    print(f"Total samples: {structural_analysis['total_samples']:,}")
    print(f"Total features: {structural_analysis['feature_count']}")
    print(f"Numerical features: {structural_analysis['feature_types']['numerical']}")
    print(f"Categorical features: {structural_analysis['feature_types']['categorical']}")
    print(f"Network flow features: {structural_analysis['feature_types']['network_flow']}")
    print(f"Temporal features: {structural_analysis['feature_types']['temporal']}")
    print(f"Missing data percentage: {structural_analysis['missing_values']['total_missing_percentage']:.2f}%")
    
    # Save structural analysis
    with open(OUTPUT_DIR / "structural_analysis.json", 'w') as f:
        json.dump(structural_analysis, f, indent=2)
    
    # Phase 3: Class Distribution Analysis
    print("\n🎯 Phase 3: Class Distribution Analysis")
    print("-" * 60)
    
    # Find label column (typically 'attack_cat' or 'label' in UNSW-NB15)
    label_column = None
    possible_labels = ['attack_cat', 'label', 'Label', 'Attack_Type', 'attack_type']
    for col in possible_labels:
        if col in full_df.columns:
            label_column = col
            break
    
    if label_column is None:
        print("❌ CHECKPOINT 1 FAILED: No label column found!")
        print(f"Available columns: {list(full_df.columns)}")
        return False
    
    print(f"✅ Found label column: {label_column}")
    
    # Class distribution
    class_distribution = full_df[label_column].value_counts()
    class_percentages = (class_distribution / len(full_df)) * 100
    
    class_analysis = {
        "label_column": label_column,
        "attack_types": class_distribution.to_dict(),
        "class_percentages": class_percentages.to_dict(),
        "normal_vs_attack_ratio": None,
        "imbalance_severity": None,
        "smallest_class_percentage": class_percentages.min(),
        "largest_class_percentage": class_percentages.max()
    }
    
    # Determine normal vs attack
    normal_classes = [cls for cls in class_distribution.index 
                     if any(keyword in cls.lower() for keyword in ['normal', 'benign'])]
    
    if normal_classes:
        normal_count = sum(class_distribution[cls] for cls in normal_classes)
        attack_count = len(full_df) - normal_count
        class_analysis["normal_vs_attack_ratio"] = normal_count / attack_count if attack_count > 0 else float('inf')
        
        print(f"Normal samples: {normal_count:,} ({(normal_count/len(full_df)*100):.1f}%)")
        print(f"Attack samples: {attack_count:,} ({(attack_count/len(full_df)*100):.1f}%)")
    else:
        print("⚠️  Could not identify normal vs attack classes")
    
    # Imbalance severity
    if class_analysis["smallest_class_percentage"] < 1:
        class_analysis["imbalance_severity"] = "severe"
    elif class_analysis["smallest_class_percentage"] < 5:
        class_analysis["imbalance_severity"] = "moderate"
    else:
        class_analysis["imbalance_severity"] = "balanced"
    
    print(f"Class imbalance severity: {class_analysis['imbalance_severity']}")
    print(f"Smallest class: {class_analysis['smallest_class_percentage']:.2f}%")
    print(f"Largest class: {class_analysis['largest_class_percentage']:.2f}%")
    
    # Create class distribution plot
    plt.figure(figsize=(12, 8))
    ax = class_distribution.plot(kind='bar')
    plt.title('UNSW-NB15 Class Distribution')
    plt.xlabel('Attack Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save class analysis
    with open(OUTPUT_DIR / "class_analysis.json", 'w') as f:
        json.dump(class_analysis, f, indent=2)
    
    # Phase 4: Temporal Structure Analysis  
    print("\n⏰ Phase 4: Temporal Structure Analysis")
    print("-" * 60)
    
    # Look for timestamp columns
    timestamp_columns = []
    for col in full_df.columns:
        if any(keyword in col.lower() for keyword in ['time', 'timestamp', 'date']):
            timestamp_columns.append(col)
        elif full_df[col].dtype == 'object':
            # Check if string column could be timestamps
            sample_values = full_df[col].dropna().head(10)
            if sample_values.empty:
                continue
            # Simple heuristic for timestamp detection
            sample_str = str(sample_values.iloc[0])
            if any(char in sample_str for char in ['-', ':', '/', ' ']) and len(sample_str) > 8:
                timestamp_columns.append(col)
    
    temporal_analysis = {
        "has_timestamps": len(timestamp_columns) > 0,
        "timestamp_columns": timestamp_columns,
        "time_span": {},
        "temporal_resolution": None,
        "temporal_gaps": [],
        "attack_temporal_distribution": {},
        "co_occurrence_verified": False
    }
    
    if timestamp_columns:
        print(f"✅ Found timestamp columns: {timestamp_columns}")
        
        # Analyze first timestamp column
        ts_col = timestamp_columns[0]
        try:
            # Try to parse timestamps
            if full_df[ts_col].dtype == 'object':
                timestamps = pd.to_datetime(full_df[ts_col], errors='coerce')
            else:
                timestamps = pd.to_datetime(full_df[ts_col], unit='s', errors='coerce')
            
            valid_timestamps = timestamps.dropna()
            if len(valid_timestamps) > 0:
                temporal_analysis["time_span"] = {
                    "start": str(valid_timestamps.min()),
                    "end": str(valid_timestamps.max()),
                    "duration_days": (valid_timestamps.max() - valid_timestamps.min()).days
                }
                
                print(f"Time span: {temporal_analysis['time_span']['start']} to {temporal_analysis['time_span']['end']}")
                print(f"Duration: {temporal_analysis['time_span']['duration_days']} days")
                
                # Check co-occurrence of normal and attack samples
                df_with_time = full_df.copy()
                df_with_time['parsed_timestamp'] = timestamps
                df_with_time = df_with_time.dropna(subset=['parsed_timestamp'])
                
                if len(df_with_time) > 0 and normal_classes:
                    df_with_time['hour'] = df_with_time['parsed_timestamp'].dt.hour
                    
                    # Simple co-occurrence check
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
                    
                    if temporal_analysis["co_occurrence_verified"]:
                        print(f"✅ CHECKPOINT 2 PASSED: Normal and attack samples co-occur in {len(overlapping_hours)} hours")
                    else:
                        print("❌ CHECKPOINT 2 FAILED: No temporal co-occurrence of normal and attack samples")
                    
        except Exception as e:
            print(f"⚠️  Timestamp parsing failed: {e}")
    else:
        print("❌ CHECKPOINT 2 FAILED: No timestamp columns found")
        print("Available columns:", list(full_df.columns)[:10], "...")
    
    # Save temporal analysis
    with open(OUTPUT_DIR / "temporal_analysis.json", 'w') as f:
        json.dump(temporal_analysis, f, indent=2)
    
    # Phase 5: TDA Suitability Assessment
    print("\n🔮 Phase 5: TDA Suitability Assessment")
    print("-" * 60)
    
    # Features suitable for TDA
    tda_suitable_features = []
    for col in numerical_features:
        if col != label_column and full_df[col].nunique() > 10:  # Avoid binary/categorical
            tda_suitable_features.append(col)
    
    # Graph construction potential
    graph_features = [col for col in full_df.columns 
                     if any(keyword in col.lower() for keyword in 
                          ['src', 'dst', 'sport', 'dport', 'proto'])]
    
    tda_suitability = {
        "tda_suitable_features": tda_suitable_features[:20],  # Top 20 for analysis
        "total_tda_features": len(tda_suitable_features),
        "graph_constructible": len(graph_features) >= 2,
        "graph_features": graph_features,
        "time_series_constructible": len(temporal_features) > 0,
        "expected_homology_dimensions": ["H0", "H1"] if len(tda_suitable_features) >= 3 else ["H0"],
        "computational_feasibility": "high" if len(full_df) < 1000000 else "medium",
        "tda_readiness_score": 0
    }
    
    # Calculate TDA readiness score (1-5)
    score = 0
    if len(tda_suitable_features) >= 10: score += 1
    if tda_suitability["graph_constructible"]: score += 1 
    if tda_suitability["time_series_constructible"]: score += 1
    if temporal_analysis["co_occurrence_verified"]: score += 1
    if class_analysis["imbalance_severity"] != "severe": score += 1
    
    tda_suitability["tda_readiness_score"] = score
    
    print(f"TDA suitable features: {len(tda_suitable_features)} (showing first 10)")
    print(f"Top TDA features: {tda_suitable_features[:10]}")
    print(f"Graph constructible: {tda_suitability['graph_constructible']}")
    print(f"Time series constructible: {tda_suitability['time_series_constructible']}")
    print(f"TDA readiness score: {score}/5")
    
    if score >= 3:
        print("✅ CHECKPOINT 3 PASSED: Dataset suitable for TDA")
    else:
        print("❌ CHECKPOINT 3 FAILED: Dataset may not be suitable for TDA")
    
    # Save TDA suitability
    with open(OUTPUT_DIR / "tda_suitability.json", 'w') as f:
        json.dump(tda_suitability, f, indent=2)
    
    # Phase 6: Data Leakage Risk Assessment
    print("\n🔒 Phase 6: Data Leakage Risk Assessment")
    print("-" * 60)
    
    # Identify risky features
    ids_features = [col for col in full_df.columns 
                   if any(keyword in col.lower() for keyword in 
                        ['alert', 'anomaly', 'score', 'detection', 'ids', 'classify'])]
    
    future_info_features = [col for col in full_df.columns 
                           if any(keyword in col.lower() for keyword in 
                                ['response', 'result', 'outcome', 'verdict'])]
    
    leakage_risks = {
        "ids_detection_features": ids_features,
        "future_information": future_info_features,
        "temporal_ordering_issues": not temporal_analysis["co_occurrence_verified"],
        "train_test_temporal_overlap": True,  # UNSW-NB15 specific
        "recommended_split_strategy": "stratified_random",
        "high_risk_features_to_remove": ids_features + future_info_features
    }
    
    print(f"IDS detection features found: {len(ids_features)}")
    print(f"Future information features: {len(future_info_features)}")
    print(f"High-risk features to remove: {len(leakage_risks['high_risk_features_to_remove'])}")
    
    if len(leakage_risks['high_risk_features_to_remove']) == 0:
        print("✅ No obvious data leakage risks detected")
    else:
        print("⚠️  Data leakage risks detected - remove high-risk features")
        print(f"Risky features: {leakage_risks['high_risk_features_to_remove']}")
    
    # Save leakage analysis
    with open(OUTPUT_DIR / "leakage_risks.json", 'w') as f:
        json.dump(leakage_risks, f, indent=2)
    
    # Phase 7: Summary and Recommendations
    print("\n📋 Phase 7: Summary and Recommendations")
    print("-" * 60)
    
    # Overall assessment
    checkpoints_passed = 0
    if structural_analysis['missing_values']['total_missing_percentage'] < 10:
        checkpoints_passed += 1
        print("✅ Data Quality Gate: PASSED")
    else:
        print("❌ Data Quality Gate: FAILED")
    
    if temporal_analysis["co_occurrence_verified"]:
        checkpoints_passed += 1
        print("✅ Temporal Integrity Gate: PASSED")
    else:
        print("❌ Temporal Integrity Gate: FAILED")
    
    if tda_suitability["tda_readiness_score"] >= 3:
        checkpoints_passed += 1
        print("✅ TDA Viability Gate: PASSED")
    else:
        print("❌ TDA Viability Gate: FAILED")
    
    print(f"\nOverall checkpoints passed: {checkpoints_passed}/3")
    
    if checkpoints_passed >= 2:
        recommendation = "PROCEED with TDA implementation"
        print(f"🎯 RECOMMENDATION: {recommendation}")
    else:
        recommendation = "CAUTION - Address failed checkpoints before proceeding"
        print(f"⚠️  RECOMMENDATION: {recommendation}")
    
    # Final summary
    summary = {
        "dataset_name": "UNSW-NB15",
        "analysis_date": "2025-08-07",
        "checkpoints_passed": checkpoints_passed,
        "recommendation": recommendation,
        "total_samples": len(full_df),
        "tda_readiness_score": tda_suitability["tda_readiness_score"],
        "key_findings": [
            f"Dataset has {len(full_df):,} samples with {len(full_df.columns)} features",
            f"Class imbalance severity: {class_analysis['imbalance_severity']}",
            f"TDA suitable features: {len(tda_suitable_features)}",
            f"Temporal co-occurrence: {temporal_analysis['co_occurrence_verified']}",
            f"Data leakage risks: {len(leakage_risks['high_risk_features_to_remove'])} risky features"
        ]
    }
    
    # Save summary
    with open(OUTPUT_DIR / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n💾 Analysis completed. Results saved to:")
    print(f"   {OUTPUT_DIR}")
    
    return checkpoints_passed >= 2

if __name__ == "__main__":
    # Create output directories
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)
    
    # Run analysis
    success = analyze_unsw_nb15()
    
    if success:
        print("\n🎉 UNSW-NB15 analysis completed successfully!")
        print("✅ Dataset approved for TDA implementation")
    else:
        print("\n⚠️  UNSW-NB15 analysis revealed issues")
        print("❌ Address critical checkpoints before proceeding")
