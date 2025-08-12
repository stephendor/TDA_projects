#!/usr/bin/env python3
"""
Full TDA Analysis on CIC-IDS2017 Dataset
Apply proper temporal TDA methods to real APT data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import our TDA modules
import sys
sys.path.append('.')
from src.core.persistent_homology import PersistentHomologyAnalyzer
from src.core.mapper import MapperAnalyzer
from src.cybersecurity.apt_detection import APTDetector

def load_full_infiltration_data():
    """Load the full infiltration dataset including attacks."""
    
    print("🔍 LOADING FULL INFILTRATION DATASET")
    print("=" * 60)
    
    file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    
    print(f"📁 Loading: {Path(file_path).name}")
    df = pd.read_csv(file_path)
    
    # Clean column names (remove leading spaces)
    df.columns = df.columns.str.strip()
    
    print(f"   Shape: {df.shape}")
    print(f"   Label distribution:")
    print(df['Label'].value_counts())
    
    attack_pct = (df['Label'] != 'BENIGN').mean() * 100
    print(f"   Attack percentage: {attack_pct:.3f}%")
    
    return df

def extract_tda_features(df, sample_size=5000):
    """Extract features suitable for TDA analysis."""
    
    print(f"\n🔧 EXTRACTING TDA FEATURES")
    print("-" * 40)
    
    # Ensure we include all attacks + random benign samples
    attacks = df[df['Label'] != 'BENIGN']
    benign = df[df['Label'] == 'BENIGN']
    
    print(f"   Total attacks available: {len(attacks)}")
    print(f"   Total benign available: {len(benign):,}")
    
    if len(attacks) > 0:
        # Include all attacks + sample benign
        benign_sample_size = min(sample_size - len(attacks), len(benign))
        benign_sample = benign.sample(n=benign_sample_size, random_state=42)
        df_sample = pd.concat([attacks, benign_sample])
        print(f"   Using ALL {len(attacks)} attacks + {len(benign_sample):,} benign samples")
    else:
        # No attacks, just sample benign
        df_sample = benign.sample(n=min(sample_size, len(benign)), random_state=42)
        print(f"   No attacks found, sampling {len(df_sample):,} benign samples")
    
    # Key features for network flow analysis
    feature_columns = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Packet Length Mean', 'Packet Length Std',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in df_sample.columns]
    print(f"   Available features: {len(available_features)}")
    
    # Extract features
    X = df_sample[available_features].copy()
    y = (df_sample['Label'] != 'BENIGN').astype(int)  # Binary: 0=benign, 1=attack
    
    # Handle missing values
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Attack samples: {y.sum()}")
    print(f"   Benign samples: {(y == 0).sum()}")
    
    return X, y, available_features

def apply_temporal_tda_analysis(X, y, features):
    """Apply TDA methods to temporal network data."""
    
    print(f"\n🔬 APPLYING TEMPORAL TDA ANALYSIS")
    print("=" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape}, attacks: {y_train.sum()}")
    print(f"   Test set: {X_test.shape}, attacks: {y_test.sum()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 TDA FEATURE EXTRACTION:")
    
    # 1. Persistent Homology Analysis
    try:
        ph = PersistentHomologyAnalyzer(maxdim=1, thresh=2.0)
        
        # Extract PH features for training data
        print("   Computing persistent homology features...")
        ph_features_train = []
        for i in range(0, len(X_train_scaled), 50):  # Process in batches
            batch = X_train_scaled[i:i+50]
            if len(batch) >= 3:  # Need minimum points for PH
                ph.fit(batch)
                batch_features = ph.extract_features()
                ph_features_train.append(batch_features)
        
        if ph_features_train:
            ph_train_matrix = np.array(ph_features_train)
            print(f"      PH features shape: {ph_train_matrix.shape}")
        
    except Exception as e:
        print(f"   ❌ PH analysis failed: {e}")
        ph_train_matrix = None
    
    # 2. Mapper Analysis
    try:
        mapper = MapperAnalyzer(n_intervals=10, overlap_frac=0.3)
        
        print("   Computing Mapper features...")
        mapper.fit(X_train_scaled[:1000])  # Sample for speed
        mapper_features_train = mapper.extract_features()
        print(f"      Mapper features shape: {mapper_features_train.shape}")
        
    except Exception as e:
        print(f"   ❌ Mapper analysis failed: {e}")
        mapper_features_train = None
    
    # 3. Baseline Comparison using original APT detector
    print(f"\n📈 BASELINE COMPARISON:")
    
    try:
        detector = APTDetector(anomaly_threshold=0.01)  # Expect 1% attacks based on data
        detector.fit(X_train_scaled)
        
        # Predictions
        y_pred = detector.predict(X_test_scaled)
        
        print("   APT Detector Performance:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        print(f"\n   📊 PERFORMANCE METRICS:")
        print(f"      Accuracy: {accuracy:.3f}")
        print(f"      Precision: {precision:.3f}")
        print(f"      Recall: {recall:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'confusion_matrix': cm,
            'ph_features_available': ph_train_matrix is not None,
            'mapper_features_available': mapper_features_train is not None
        }
        
    except Exception as e:
        print(f"   ❌ Detector analysis failed: {e}")
        return None

def main():
    """Main analysis function."""
    
    # Load real infiltration data
    df = load_full_infiltration_data()
    
    # Extract TDA features
    X, y, features = extract_tda_features(df)
    
    # Apply TDA analysis
    results = apply_temporal_tda_analysis(X, y, features)
    
    if results:
        print(f"\n🎯 ANALYSIS COMPLETE")
        print("=" * 40)
        print(f"✅ Real APT detection accuracy: {results['accuracy']:.1%}")
        print(f"✅ Attack precision: {results['precision']:.1%}")
        print(f"✅ Attack recall: {results['recall']:.1%}")
        
        if results['ph_features_available']:
            print("✅ Persistent homology features computed successfully")
        if results['mapper_features_available']:
            print("✅ Mapper features computed successfully")
        
        # Update todo status
        print(f"\n📋 Next: Compare with other attack types and optimize TDA approach")
        
    else:
        print(f"\n❌ Analysis failed - check TDA implementation")

if __name__ == "__main__":
    main()