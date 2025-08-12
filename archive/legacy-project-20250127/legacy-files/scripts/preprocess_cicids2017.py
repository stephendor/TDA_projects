#!/usr/bin/env python3
"""
Preprocess CIC-IDS2017 dataset for TDA analysis
This script prepares the data for temporal and topological analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def check_data_availability():
    """Check if CIC-IDS2017 data files are available."""
    data_dir = Path("data/apt_datasets/cicids2017")
    
    expected_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv", 
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ]
    
    available_files = []
    missing_files = []
    
    # Search in subdirectories for CSV files
    for csv_file in data_dir.rglob("*.csv"):
        if csv_file.name in expected_files:
            available_files.append(str(csv_file))
    
    # Also check direct paths
    for filename in expected_files:
        file_path = data_dir / filename
        if file_path.exists() and str(file_path) not in available_files:
            available_files.append(str(file_path))
    
    # Find missing files
    found_names = [Path(f).name for f in available_files]
    missing_files = [f for f in expected_files if f not in found_names]
    
    return available_files, missing_files, data_dir

def load_and_explore_sample():
    """Load a sample of the dataset to explore its structure."""
    
    available_files, missing_files, data_dir = check_data_availability()
    
    print("=" * 80)
    print("CIC-IDS2017 DATA EXPLORATION")
    print("=" * 80)
    
    print(f"\n📁 Data Directory: {data_dir}")
    print(f"✅ Available Files: {len(available_files)}")
    print(f"❌ Missing Files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️  Missing files (download required):")
        for filename in missing_files:
            print(f"   - {filename}")
        
        if not available_files:
            print(f"\n🚨 No data files found. Please download the dataset first.")
            print(f"   Run: bash {data_dir / 'download_instructions.sh'}")
            return None
    
    # Load first available file for exploration
    sample_file_path = available_files[0]
    sample_file = Path(sample_file_path).name
    sample_path = Path(sample_file_path)
    
    print(f"\n📊 Loading sample data from: {sample_file}")
    
    try:
        # Load first 1000 rows to explore structure
        df_sample = pd.read_csv(sample_path, nrows=1000)
        
        print(f"   Shape: {df_sample.shape}")
        print(f"   Columns: {len(df_sample.columns)}")
        
        print(f"\n🏷️  LABEL DISTRIBUTION:")
        label_col = None
        for col in df_sample.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if label_col:
            label_counts = df_sample[label_col].value_counts()
            print(label_counts)
        else:
            print("   Label column not found")
            print("   Available columns:", df_sample.columns[-5:].tolist())
        
        print(f"\n📈 SAMPLE FEATURES (first 10):")
        feature_cols = [col for col in df_sample.columns if col != 'Label'][:10]
        for col in feature_cols:
            try:
                col_min = df_sample[col].min()
                col_max = df_sample[col].max()
                if pd.api.types.is_numeric_dtype(df_sample[col]):
                    print(f"   {col}: {df_sample[col].dtype}, range: {col_min:.2f} to {col_max:.2f}")
                else:
                    print(f"   {col}: {df_sample[col].dtype}, sample: {df_sample[col].iloc[0]}")
            except Exception as e:
                print(f"   {col}: {df_sample[col].dtype}, error analyzing: {e}")
        
        return df_sample, available_files, data_dir
        
    except Exception as e:
        print(f"❌ Error loading {sample_file}: {e}")
        return None

def create_temporal_features(df, time_col='Timestamp'):
    """Create temporal features for TDA analysis."""
    
    print(f"\n🕐 CREATING TEMPORAL FEATURES:")
    
    # Create synthetic timestamp if not available
    if time_col not in df.columns:
        print(f"   Creating synthetic timestamps...")
        df[time_col] = pd.date_range(start='2017-07-03', periods=len(df), freq='1S')
    
    # Extract temporal components
    df['Hour'] = df[time_col].dt.hour
    df['Minute'] = df[time_col].dt.minute
    df['Second'] = df[time_col].dt.second
    df['DayOfWeek'] = df[time_col].dt.dayofweek
    
    # Create time-based aggregations for TDA
    temporal_features = [
        'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 
        'Flow IAT Mean', 'Flow IAT Std', 'Total Fwd Packets',
        'Total Backward Packets', 'Fwd Packet Length Mean',
        'Bwd Packet Length Mean', 'Packet Length Std'
    ]
    
    available_features = [col for col in temporal_features if col in df.columns]
    
    print(f"   Temporal features available: {len(available_features)}")
    for feat in available_features:
        print(f"   - {feat}")
    
    return df, available_features

def prepare_tda_sequences(df, temporal_features, window_size=60, step_size=30):
    """Prepare sequences for TDA analysis."""
    
    print(f"\n🔄 PREPARING TDA SEQUENCES:")
    print(f"   Window size: {window_size} flows")
    print(f"   Step size: {step_size} flows")
    
    sequences = []
    labels = []
    
    # Create sliding windows
    for i in range(0, len(df) - window_size, step_size):
        window_data = df.iloc[i:i+window_size]
        
        # Extract feature sequence
        feature_sequence = window_data[temporal_features].values
        
        # Determine label (majority vote)
        label_col = None
        for col in window_data.columns:
            if 'label' in col.lower():
                label_col = col
                break
                
        if label_col:
            window_labels = window_data[label_col].values
            # Binary classification: BENIGN vs any attack
            is_attack = (window_labels != 'BENIGN').astype(int)
            sequence_label = 1 if np.mean(is_attack) > 0.1 else 0  # >10% attack = attack sequence
        else:
            sequence_label = 0  # Default to benign if no labels
        
        sequences.append(feature_sequence)
        labels.append(sequence_label)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"   Created {len(sequences)} sequences")
    print(f"   Sequence shape: {sequences.shape}")
    print(f"   Attack sequences: {np.sum(labels)}")
    print(f"   Benign sequences: {np.sum(labels == 0)}")
    
    return sequences, labels

def analyze_tda_suitability(sequences, labels):
    """Analyze how suitable the data is for TDA methods."""
    
    print(f"\n🔬 TDA SUITABILITY ANALYSIS:")
    
    # 1. Temporal persistence analysis
    print(f"   1. Temporal Persistence:")
    
    benign_seqs = sequences[labels == 0]
    attack_seqs = sequences[labels == 1]
    
    if len(attack_seqs) > 0:
        # Calculate sequence stability (how much sequences change over time)
        benign_stability = np.mean([np.std(seq, axis=0) for seq in benign_seqs[:10]])
        attack_stability = np.mean([np.std(seq, axis=0) for seq in attack_seqs[:10]])
        
        print(f"      Benign sequence stability: {benign_stability:.4f}")
        print(f"      Attack sequence stability: {attack_stability:.4f}")
        
        if attack_stability > benign_stability * 1.2:
            print("      ✅ Attack sequences show more variation - good for TDA")
        else:
            print("      ⚠️  Attack sequences similar to benign - may be challenging")
    
    # 2. Feature correlation analysis
    print(f"   2. Feature Relationships:")
    
    # Calculate average correlation within sequences
    avg_correlations = []
    for seq in sequences[:50]:  # Sample first 50 sequences
        if seq.shape[1] > 1:  # Need multiple features
            corr_matrix = np.corrcoef(seq.T)
            # Get upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            avg_correlations.append(np.mean(np.abs(upper_tri)))
    
    if avg_correlations:
        mean_correlation = np.mean(avg_correlations)
        print(f"      Average inter-feature correlation: {mean_correlation:.3f}")
        
        if 0.3 <= mean_correlation <= 0.7:
            print("      ✅ Good correlation balance - suitable for TDA")
        elif mean_correlation < 0.3:
            print("      ⚠️  Low correlation - features may be independent")
        else:
            print("      ⚠️  High correlation - features may be redundant")
    
    # 3. Dimensionality assessment
    print(f"   3. Dimensionality:")
    print(f"      Sequence length: {sequences.shape[1]} time points")
    print(f"      Feature count: {sequences.shape[2]} features")
    print(f"      Embedding dimension: {sequences.shape[1]} (for time delay embedding)")
    
    if sequences.shape[1] >= 10:
        print("      ✅ Sufficient temporal resolution for persistent homology")
    else:
        print("      ⚠️  Low temporal resolution - may need longer sequences")
    
    if sequences.shape[2] >= 3:
        print("      ✅ Multiple features available for topological analysis")
    else:
        print("      ⚠️  Few features - consider feature engineering")

def create_tda_analysis_plan(sequences, labels, temporal_features):
    """Create a detailed plan for TDA analysis."""
    
    print(f"\n📋 TDA ANALYSIS PLAN:")
    
    plan = {
        "data_summary": {
            "sequences": len(sequences),
            "sequence_length": sequences.shape[1],
            "features": sequences.shape[2],
            "attack_rate": np.mean(labels),
            "feature_names": temporal_features
        },
        "tda_approaches": [
            {
                "method": "Persistent Homology on Time Series",
                "description": "Apply PH to individual sequences using time-delay embeddings",
                "expected_insights": "Periodic patterns, long-term trends in individual flows",
                "implementation": "Use sliding window embeddings, analyze persistence diagrams"
            },
            {
                "method": "Mapper on Feature Space Evolution", 
                "description": "Track how feature space topology changes over time",
                "expected_insights": "Attack progression patterns, network state transitions",
                "implementation": "Apply Mapper to feature vectors, analyze graph evolution"
            },
            {
                "method": "Multi-Scale Temporal Analysis",
                "description": "Analyze different time scales (1min, 5min, 1hr windows)",
                "expected_insights": "Short vs long-term attack patterns, campaign persistence",
                "implementation": "Multiple window sizes, compare persistence across scales"
            },
            {
                "method": "Network Topology Evolution",
                "description": "TDA on connection graphs between hosts",
                "expected_insights": "Lateral movement, command & control patterns",
                "implementation": "Build adjacency matrices, track topological changes"
            }
        ],
        "success_metrics": [
            "Separation between benign/attack persistence diagrams",
            "Topological stability during normal operations",
            "Distinct topological signatures for different attack types",
            "Temporal persistence of attack patterns"
        ]
    }
    
    print(f"   📊 Data Summary:")
    print(f"      Sequences: {plan['data_summary']['sequences']:,}")
    print(f"      Length: {plan['data_summary']['sequence_length']} time points")
    print(f"      Features: {plan['data_summary']['features']}")
    print(f"      Attack Rate: {plan['data_summary']['attack_rate']:.1%}")
    
    print(f"\n   🔬 TDA Approaches:")
    for i, approach in enumerate(plan['tda_approaches'], 1):
        print(f"      {i}. {approach['method']}")
        print(f"         {approach['description']}")
        print(f"         Expected: {approach['expected_insights']}")
    
    print(f"\n   🎯 Success Metrics:")
    for metric in plan['success_metrics']:
        print(f"      - {metric}")
    
    return plan

def main():
    """Main preprocessing function."""
    
    result = load_and_explore_sample()
    
    if result is None:
        print("\n❌ Cannot proceed without data. Please download the dataset first.")
        return
    
    df_sample, available_files, data_dir = result
    
    # Create temporal features
    df_with_time, temporal_features = create_temporal_features(df_sample)
    
    # Prepare TDA sequences
    sequences, labels = prepare_tda_sequences(df_with_time, temporal_features)
    
    # Analyze TDA suitability
    analyze_tda_suitability(sequences, labels)
    
    # Create analysis plan
    plan = create_tda_analysis_plan(sequences, labels, temporal_features)
    
    # Save preprocessing results
    output_dir = Path("data/apt_datasets/cicids2017/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Save sample sequences for TDA analysis
    np.save(output_dir / "sample_sequences.npy", sequences)
    np.save(output_dir / "sample_labels.npy", labels)
    
    # Save analysis plan
    import json
    with open(output_dir / "tda_analysis_plan.json", 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    
    print(f"\n💾 PREPROCESSING COMPLETE:")
    print(f"   Sample sequences saved: {output_dir / 'sample_sequences.npy'}")
    print(f"   Labels saved: {output_dir / 'sample_labels.npy'}")
    print(f"   Analysis plan saved: {output_dir / 'tda_analysis_plan.json'}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Download complete dataset if not already done")
    print(f"   2. Run full preprocessing on all files")
    print(f"   3. Implement TDA analysis based on the plan")
    print(f"   4. Compare with baseline methods")

if __name__ == "__main__":
    main()