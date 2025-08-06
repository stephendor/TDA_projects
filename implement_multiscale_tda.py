#!/usr/bin/env python3
"""
Multi-Scale Temporal TDA Implementation
Phase 1 of TDA Improvement Strategy

This implements advanced multi-scale temporal analysis to capture 
APT patterns across multiple time horizons.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# Import our TDA modules
import sys
sys.path.append('.')
from src.core.persistent_homology import PersistentHomologyAnalyzer
from src.cybersecurity.apt_detection import APTDetector

class MultiScaleTDAAnalyzer:
    """
    Advanced TDA analyzer using multiple temporal scales to capture 
    different phases of APT campaigns.
    """
    
    def __init__(self, window_sizes=None):
        """
        Initialize multi-scale analyzer.
        
        Args:
            window_sizes: List of window sizes for different temporal scales
        """
        if window_sizes is None:
            # Different scales for APT detection (adjusted for limited attack data)
            self.window_sizes = [
                5,    # Micro: Individual flows
                10,   # Tactical: Small sequences
                20,   # Operational: Medium sequences
                40,   # Strategic: Larger patterns
                60    # Campaign: Full context
            ]
        else:
            self.window_sizes = window_sizes
            
        self.tda_analyzers = {}
        self.scalers = {}
        
        print(f"🔬 Multi-Scale TDA Analyzer initialized")
        print(f"   Temporal scales: {self.window_sizes}")
        print(f"   Expected feature dimensions: {len(self.window_sizes) * 12}")  # 12 features per scale

    def extract_multiscale_features(self, X, y):
        """Extract TDA features at multiple temporal scales."""
        
        print(f"\n🔄 EXTRACTING MULTI-SCALE TDA FEATURES")
        print("=" * 60)
        
        all_features = []
        all_labels = []
        
        for scale_idx, window_size in enumerate(self.window_sizes):
            print(f"\n   📏 Scale {scale_idx + 1}: Window size {window_size}")
            
            # Create sequences for this scale
            sequences, labels = self.create_temporal_sequences(X, y, window_size)
            
            if len(sequences) == 0:
                print(f"      ⚠️ No sequences generated for window size {window_size}")
                continue
                
            print(f"      Generated {len(sequences)} sequences")
            print(f"      Attack sequences: {np.sum(labels)}")
            print(f"      Benign sequences: {np.sum(labels == 0)}")
            
            # Extract TDA features for this scale
            tda_features = self.extract_tda_features_for_scale(sequences, scale_idx)
            
            if tda_features is not None:
                all_features.append(tda_features)
                all_labels.append(labels)
                print(f"      ✅ TDA features extracted: {tda_features.shape}")
            else:
                print(f"      ❌ TDA feature extraction failed")
        
        if not all_features:
            print("\n❌ No features extracted at any scale")
            return None, None
            
        # Use the scale with the best attack preservation (usually the smallest window)
        # Find scale with highest attack rate
        attack_rates = [np.mean(labels) for labels in all_labels]
        best_scale_idx = np.argmax(attack_rates)
        
        print(f"   Attack rates by scale: {[f'{rate:.3%}' for rate in attack_rates]}")
        print(f"   Using scale {best_scale_idx + 1} (window {self.window_sizes[best_scale_idx]}) as primary")
        
        # Use features from best scale, with additional features from other scales if available
        primary_features = all_features[best_scale_idx]
        primary_labels = all_labels[best_scale_idx]
        
        # Add complementary features from other scales (same number of samples)
        n_samples = len(primary_features)
        additional_features = []
        
        for scale_idx, features in enumerate(all_features):
            if scale_idx != best_scale_idx and len(features) >= n_samples:
                # Take first n_samples to match primary scale
                additional_features.append(features[:n_samples])
        
        if additional_features:
            combined_features = np.concatenate([primary_features] + additional_features, axis=1)
            print(f"   Combined features from {len(additional_features) + 1} scales")
        else:
            combined_features = primary_features
            print(f"   Using only primary scale features")
        
        combined_labels = primary_labels
        
        print(f"\n📊 MULTI-SCALE FEATURE SUMMARY:")
        print(f"   Final feature matrix: {combined_features.shape}")
        print(f"   Total attack sequences: {np.sum(combined_labels)}")
        print(f"   Total benign sequences: {np.sum(combined_labels == 0)}")
        print(f"   Attack rate: {np.mean(combined_labels):.3%}")
        
        return combined_features, combined_labels
    
    def create_temporal_sequences(self, X, y, window_size, step_size=None):
        """Create temporal sequences for a given window size."""
        
        if step_size is None:
            step_size = max(1, window_size // 3)  # Overlap for better coverage
        
        if len(X) < window_size:
            return [], []
        
        sequences = []
        labels = []
        
        for i in range(0, len(X) - window_size + 1, step_size):
            # Extract sequence
            sequence = X.iloc[i:i+window_size].values
            
            # Determine label (majority vote with attack bias)
            window_labels = y.iloc[i:i+window_size].values
            # If ANY attack in window, label as attack (more sensitive for rare attacks)
            sequence_label = 1 if np.sum(window_labels) > 0 else 0
            
            sequences.append(sequence)
            labels.append(sequence_label)
        
        return np.array(sequences), np.array(labels)
    
    def extract_tda_features_for_scale(self, sequences, scale_idx):
        """Extract TDA features for sequences at a specific scale."""
        
        try:
            # Initialize TDA analyzer for this scale
            if scale_idx not in self.tda_analyzers:
                # Adjust parameters based on scale
                max_dim = 1 if len(sequences[0]) < 50 else 2
                thresh = 3.0 if scale_idx < 2 else 5.0  # Larger thresh for larger scales
                
                self.tda_analyzers[scale_idx] = PersistentHomologyAnalyzer(
                    maxdim=max_dim, 
                    thresh=thresh,
                    backend='ripser'
                )
            
            ph_analyzer = self.tda_analyzers[scale_idx]
            
            # Extract features in batches to handle memory
            batch_size = 50
            all_features = []
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                batch_features = []
                
                for seq in batch:
                    if len(seq) >= 3:  # Minimum points for PH
                        try:
                            ph_analyzer.fit(seq)
                            features = ph_analyzer.extract_features()
                            
                            # Ensure consistent feature length (pad if necessary)
                            if len(features) < 12:  # Expected: 6 for H0 + 6 for H1
                                padded_features = np.zeros(12)
                                padded_features[:len(features)] = features
                                features = padded_features
                            
                            batch_features.append(features[:12])  # Take first 12 features
                            
                        except Exception as e:
                            # Fallback: zero features for failed computations
                            batch_features.append(np.zeros(12))
                    else:
                        # Too few points: zero features
                        batch_features.append(np.zeros(12))
                
                if batch_features:
                    all_features.extend(batch_features)
            
            if all_features:
                feature_matrix = np.array(all_features)
                
                # Handle any remaining NaN or inf values
                feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                
                return feature_matrix
            else:
                return None
                
        except Exception as e:
            print(f"      ❌ Error in TDA feature extraction: {e}")
            return None

def load_and_prepare_data():
    """Load the infiltration dataset."""
    
    print("🔍 LOADING INFILTRATION DATASET")
    print("=" * 50)
    
    file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Get all attacks + balanced sample of benign
    attacks = df[df['Label'] != 'BENIGN']
    benign = df[df['Label'] == 'BENIGN']
    
    print(f"   Total attacks: {len(attacks)}")
    print(f"   Total benign: {len(benign):,}")
    
    # Create balanced dataset ensuring attack preservation across scales
    # Use fewer benign samples to maintain attack visibility
    benign_sample = benign.sample(n=min(5000, len(benign)), random_state=42)
    df_balanced = pd.concat([attacks, benign_sample]).sort_index()
    
    print(f"   Using {len(attacks)} attacks + {len(benign_sample):,} benign = {len(df_balanced):,} total")
    
    # Feature selection (same as validation)
    feature_columns = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Packet Length Mean', 'Packet Length Std', 'Min Packet Length',
        'Max Packet Length', 'FIN Flag Count', 'SYN Flag Count'
    ]
    
    available_features = [col for col in feature_columns if col in df_balanced.columns]
    print(f"   Available features: {len(available_features)}")
    
    X = df_balanced[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = (df_balanced['Label'] != 'BENIGN').astype(int)
    
    print(f"   Final dataset: {X.shape}")
    print(f"   Attack rate: {y.mean():.3%}")
    
    return X, y

def evaluate_multiscale_tda():
    """Main evaluation function for multi-scale TDA."""
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Initialize multi-scale analyzer
    analyzer = MultiScaleTDAAnalyzer()
    
    # Extract multi-scale features
    start_time = time.time()
    tda_features, tda_labels = analyzer.extract_multiscale_features(X, y)
    extraction_time = time.time() - start_time
    
    if tda_features is None:
        print("❌ Feature extraction failed")
        return None
    
    print(f"\n⏱️ Feature extraction completed in {extraction_time:.1f}s")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        tda_features, tda_labels, test_size=0.3, random_state=42, stratify=tda_labels
    )
    
    print(f"\n📊 EVALUATION SETUP:")
    print(f"   Training: {X_train.shape}, attacks: {y_train.sum()}")
    print(f"   Testing: {X_test.shape}, attacks: {y_test.sum()}")
    
    # Train classifier on multi-scale TDA features
    print(f"\n🚀 TRAINING MULTI-SCALE TDA CLASSIFIER")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest on TDA features
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced data
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    # Handle case where only one class exists in training
    if len(clf.classes_) > 1:
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = np.zeros(len(y_test))  # Default probabilities if only one class
    
    # Evaluation
    print(f"\n📈 MULTI-SCALE TDA RESULTS:")
    print("=" * 50)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = report['accuracy']
    precision = report['1']['precision'] if '1' in report else 0
    recall = report['1']['recall'] if '1' in report else 0
    f1_score = report['1']['f1-score'] if '1' in report else 0
    
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1_score:.3f}")
    
    print(f"\n   Confusion Matrix:")
    # Handle confusion matrix display based on available classes
    if cm.shape == (2, 2):
        print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
    else:
        print(f"   Confusion Matrix: {cm}")
        print(f"   (Only one class present in test data)")
    
    # Compare with previous results
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print("=" * 50)
    
    baseline_f1 = 0.182  # Previous single-scale TDA result
    improvement = f1_score - baseline_f1
    improvement_pct = (improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0
    
    print(f"   Previous TDA (single-scale): F1 = {baseline_f1:.3f}")
    print(f"   Multi-scale TDA: F1 = {f1_score:.3f}")
    print(f"   Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    if f1_score > 0.30:  # Target from improvement strategy
        print(f"   ✅ SUCCESS: Exceeded Phase 1 target (F1 > 30%)")
    elif f1_score > baseline_f1:
        print(f"   ✅ PROGRESS: Improvement achieved")
    else:
        print(f"   ❌ REGRESSION: Performance declined")
    
    # Feature importance analysis
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    feature_names = [f"Scale_{i+1}_Feature_{j+1}" for i in range(len(analyzer.window_sizes)) for j in range(12)]
    feature_importance = clf.feature_importances_
    
    # Top 10 most important features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    print(f"   Top 10 Most Important Features:")
    for i, idx in enumerate(top_indices):
        print(f"      {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # Scale-wise importance
    scale_importance = np.zeros(len(analyzer.window_sizes))
    for scale_idx in range(len(analyzer.window_sizes)):
        start_idx = scale_idx * 12
        end_idx = (scale_idx + 1) * 12
        scale_importance[scale_idx] = np.sum(feature_importance[start_idx:end_idx])
    
    print(f"\n   Scale-wise Feature Importance:")
    for scale_idx, (window_size, importance) in enumerate(zip(analyzer.window_sizes, scale_importance)):
        print(f"      Scale {scale_idx+1} (window {window_size}): {importance:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'scale_importance': scale_importance,
        'extraction_time': extraction_time
    }

def main():
    """Main execution function."""
    
    print("🚀 MULTI-SCALE TEMPORAL TDA IMPLEMENTATION")
    print("=" * 60)
    print("Phase 1 of TDA Improvement Strategy")
    print("Target: F1-Score >30% (baseline: 18.2%)")
    print("=" * 60)
    
    # Run evaluation
    results = evaluate_multiscale_tda()
    
    if results:
        print(f"\n🎯 PHASE 1 EVALUATION COMPLETE")
        print("=" * 60)
        
        if results['f1_score'] > 0.30:
            print(f"✅ SUCCESS: Multi-scale TDA achieved target performance!")
            print(f"   Recommended: Proceed to Phase 2 (Hybrid Ensemble)")
        elif results['improvement'] > 0:
            print(f"⚠️ PARTIAL SUCCESS: Improvement achieved but target not met")
            print(f"   Recommended: Optimize parameters and proceed to Phase 2")
        else:
            print(f"❌ FAILURE: No improvement over baseline")
            print(f"   Recommended: Debug approach or try alternative strategies")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Document results in EXPERIMENT_LOG.md")
        print(f"   2. Update VALIDATION_RESULTS.md with new performance")
        print(f"   3. Begin Phase 2: Hybrid TDA+Statistical Ensemble")

if __name__ == "__main__":
    main()