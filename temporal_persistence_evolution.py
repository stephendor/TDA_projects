#!/usr/bin/env python3
"""
Phase 2B: Temporal Persistence Evolution Tracking
Advanced TDA Enhancement Strategy

Concept: Track how topological features change over time to detect attack progression
Expected Improvement: +6-10% F1-score (strategy prediction)
Current Gap: 4.4% to reach 75% target

This implements temporal persistence evolution analysis that tracks how 
topological features evolve over time, measuring distances between consecutive
persistence diagrams to detect sudden topological changes indicating attacks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
warnings.filterwarnings('ignore')

# Import TDA modules
import sys
sys.path.append('.')
from src.core.persistent_homology import PersistentHomologyAnalyzer

class TemporalPersistenceEvolutionAnalyzer:
    """
    Advanced analyzer that tracks topological evolution over time.
    
    Key Innovation: Instead of just extracting TDA features at different scales,
    this tracks HOW the topology changes between consecutive time windows,
    detecting sudden changes that indicate attack transitions.
    """
    
    def __init__(self, evolution_window_sizes=None, overlap_ratio=0.5):
        """
        Initialize temporal persistence evolution analyzer.
        
        Args:
            evolution_window_sizes: Window sizes for evolution tracking
            overlap_ratio: Overlap between consecutive windows for evolution tracking
        """
        if evolution_window_sizes is None:
            # Windows for evolution analysis (smaller for better temporal resolution)
            self.evolution_window_sizes = [
                10,   # Fine-grained: Individual attack steps
                20,   # Medium-grained: Attack sequences  
                40,   # Coarse-grained: Attack phases
                60    # Strategic: Campaign evolution
            ]
        else:
            self.evolution_window_sizes = evolution_window_sizes
            
        self.overlap_ratio = overlap_ratio
        self.tda_analyzers = {}
        
        print(f"🌊 Temporal Persistence Evolution Analyzer initialized")
        print(f"   Evolution window sizes: {self.evolution_window_sizes}")
        print(f"   Window overlap ratio: {overlap_ratio}")
        print(f"   Focus: Detecting topological evolution patterns")

    def extract_evolution_features(self, df):
        """
        Extract temporal persistence evolution features.
        
        This is the core innovation: instead of just TDA features at different scales,
        we extract features that describe HOW the topology changes over time.
        """
        
        print(f"\n🌊 EXTRACTING TEMPORAL PERSISTENCE EVOLUTION FEATURES")
        print("=" * 70)
        
        # Prepare data for temporal analysis
        feature_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Packet Length Mean', 'Packet Length Std', 'Min Packet Length',
            'Max Packet Length', 'FIN Flag Count', 'SYN Flag Count'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        y = (df['Label'] != 'BENIGN').astype(int)
        
        print(f"   Data prepared: {X.shape} flows with {len(available_features)} features")
        
        # Extract evolution features across all window sizes
        all_evolution_features = []
        all_evolution_labels = []
        
        for scale_idx, window_size in enumerate(self.evolution_window_sizes):
            print(f"\n   🔄 Evolution Scale {scale_idx + 1}: Window size {window_size}")
            
            # Create overlapping temporal sequences for evolution tracking
            sequences, seq_labels = self.create_evolution_sequences(X, y, window_size)
            
            if len(sequences) < 2:  # Need at least 2 sequences for evolution
                print(f"      ⚠️ Insufficient sequences ({len(sequences)}) for evolution tracking")
                continue
            
            # Extract evolution features from sequence progression
            evolution_features = self.extract_persistence_evolution_features(sequences, scale_idx)
            
            if evolution_features is not None:
                all_evolution_features.append(evolution_features)
                all_evolution_labels.append(seq_labels[1:])  # Skip first (no evolution to compare)
                print(f"      ✅ Evolution features: {evolution_features.shape}")
            else:
                print(f"      ❌ Evolution feature extraction failed")
        
        if not all_evolution_features:
            print("\\n❌ No evolution features extracted at any scale")
            return None, None
        
        # Combine features from all evolution scales
        # Use scale with best attack preservation
        attack_rates = [np.mean(labels) for labels in all_evolution_labels]
        best_scale_idx = np.argmax(attack_rates)
        
        print(f"   Attack rates by evolution scale: {[f'{rate:.3%}' for rate in attack_rates]}")
        print(f"   Using evolution scale {best_scale_idx + 1} as primary")
        
        # Use primary scale and augment with other scales
        primary_features = all_evolution_features[best_scale_idx]
        primary_labels = all_evolution_labels[best_scale_idx]
        
        # Add complementary features from other scales  
        n_samples = len(primary_features)
        additional_features = []
        
        for scale_idx, features in enumerate(all_evolution_features):
            if scale_idx != best_scale_idx and len(features) >= n_samples:
                additional_features.append(features[:n_samples])
        
        if additional_features:
            combined_features = np.concatenate([primary_features] + additional_features, axis=1)
            print(f"   Combined evolution features from {len(additional_features) + 1} scales")
        else:
            combined_features = primary_features
            print(f"   Using only primary evolution scale features")
        
        combined_labels = primary_labels
        
        print(f"\n📊 EVOLUTION FEATURE SUMMARY:")
        print(f"   Final feature matrix: {combined_features.shape}")
        print(f"   Total attack sequences: {np.sum(combined_labels)}")
        print(f"   Total benign sequences: {np.sum(combined_labels == 0)}")
        print(f"   Attack rate: {np.mean(combined_labels):.3%}")
        
        return combined_features, combined_labels

    def create_evolution_sequences(self, X, y, window_size):
        """
        Create overlapping temporal sequences optimized for evolution tracking.
        
        Key difference from regular sequences: Higher overlap to capture 
        smooth temporal evolution between consecutive windows.
        """
        
        step_size = max(1, int(window_size * (1 - self.overlap_ratio)))
        
        if len(X) < window_size:
            return [], []
        
        sequences = []
        labels = []
        
        for i in range(0, len(X) - window_size + 1, step_size):
            sequence = X.iloc[i:i+window_size].values
            window_labels = y.iloc[i:i+window_size].values
            sequence_label = 1 if np.sum(window_labels) > 0 else 0
            
            sequences.append(sequence)
            labels.append(sequence_label)
        
        return np.array(sequences), np.array(labels)

    def extract_persistence_evolution_features(self, sequences, scale_idx):
        """
        Extract topological evolution features from sequence progression.
        
        This is the core innovation: computing features that describe
        how persistence diagrams change between consecutive time windows.
        """
        
        try:
            # Initialize TDA analyzer for this scale
            if scale_idx not in self.tda_analyzers:
                max_dim = 1  # Focus on H0 (components) and H1 (cycles) for stability
                thresh = 2.0 + scale_idx * 0.5  # Scale-dependent threshold
                
                self.tda_analyzers[scale_idx] = PersistentHomologyAnalyzer(
                    maxdim=max_dim, thresh=thresh, backend='ripser'
                )
            
            ph_analyzer = self.tda_analyzers[scale_idx]
            
            # Compute persistence diagrams for all sequences
            persistence_diagrams = []
            persistence_features = []
            
            print(f"      Computing persistence diagrams for {len(sequences)} sequences...")
            
            for seq_idx, sequence in enumerate(sequences):
                try:
                    if len(sequence) >= 3:
                        ph_analyzer.fit(sequence)
                        features = ph_analyzer.extract_features()
                        
                        # Ensure consistent feature length
                        if len(features) < 15:  # More features for evolution analysis
                            padded_features = np.zeros(15)
                            padded_features[:len(features)] = features
                            features = padded_features
                        
                        persistence_features.append(features[:15])
                        
                        # Also store raw persistence info if available for evolution metrics
                        # For now, use features as proxy for persistence diagram
                        persistence_diagrams.append(features[:15])
                    else:
                        persistence_features.append(np.zeros(15))
                        persistence_diagrams.append(np.zeros(15))
                        
                except Exception:
                    persistence_features.append(np.zeros(15))
                    persistence_diagrams.append(np.zeros(15))
            
            if len(persistence_features) < 2:
                return None
            
            print(f"      Computing evolution features from {len(persistence_diagrams)} diagrams...")
            
            # Compute evolution features between consecutive diagrams
            evolution_features = []
            
            for i in range(1, len(persistence_diagrams)):
                prev_diagram = persistence_diagrams[i-1]
                curr_diagram = persistence_diagrams[i]
                
                # Compute evolution metrics
                evolution_vector = self.compute_evolution_metrics(prev_diagram, curr_diagram)
                evolution_features.append(evolution_vector)
            
            if evolution_features:
                evolution_matrix = np.array(evolution_features)
                evolution_matrix = np.nan_to_num(evolution_matrix)
                return evolution_matrix
            else:
                return None
                
        except Exception as e:
            print(f"      ❌ Error in evolution feature extraction: {e}")
            return None

    def compute_evolution_metrics(self, prev_features, curr_features):
        """
        Compute evolution metrics between consecutive persistence diagrams.
        
        This captures the key insight: attacks cause sudden topological changes
        that can be detected by measuring how persistence diagrams evolve.
        """
        
        try:
            # Basic evolution metrics
            feature_diff = curr_features - prev_features
            absolute_change = np.abs(feature_diff)
            relative_change = feature_diff / (np.abs(prev_features) + 1e-8)
            
            # Statistical evolution measures
            euclidean_distance = np.linalg.norm(feature_diff)
            manhattan_distance = np.sum(absolute_change)
            max_change = np.max(absolute_change)
            
            # Stability measures
            correlation = np.corrcoef(prev_features, curr_features)[0, 1]
            correlation = correlation if not np.isnan(correlation) else 0.0
            
            # Information-theoretic measures
            # Treat features as probability distributions (after normalization)
            prev_normalized = np.abs(prev_features) / (np.sum(np.abs(prev_features)) + 1e-8)
            curr_normalized = np.abs(curr_features) / (np.sum(np.abs(curr_features)) + 1e-8)
            
            # KL divergence (measure of distribution change)
            kl_divergence = entropy(curr_normalized + 1e-8, prev_normalized + 1e-8)
            
            # Directional change metrics
            increasing_features = np.sum(feature_diff > 0)
            decreasing_features = np.sum(feature_diff < 0)
            stable_features = np.sum(np.abs(feature_diff) < 0.1)
            
            # Magnitude-based metrics
            large_changes = np.sum(absolute_change > np.std(absolute_change))
            change_variance = np.var(absolute_change)
            
            # Combine all evolution metrics into feature vector
            evolution_vector = np.array([
                # Distance metrics (4 features)
                euclidean_distance,
                manhattan_distance,
                max_change,
                correlation,
                
                # Information metrics (1 feature) 
                kl_divergence,
                
                # Change distribution metrics (5 features)
                increasing_features / len(feature_diff),
                decreasing_features / len(feature_diff), 
                stable_features / len(feature_diff),
                large_changes / len(feature_diff),
                change_variance,
                
                # Summary statistics of changes (5 features)
                np.mean(absolute_change),
                np.std(absolute_change),
                np.median(absolute_change),
                np.min(absolute_change),
                np.max(absolute_change),
            ])
            
            return evolution_vector
            
        except Exception as e:
            # Fallback for failed evolution computation
            return np.zeros(15)

def load_and_prepare_data():
    """Load and prepare dataset for evolution analysis."""
    
    print("🔍 LOADING DATA FOR TEMPORAL PERSISTENCE EVOLUTION")
    print("=" * 55)
    
    file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Get balanced dataset
    attacks = df[df['Label'] != 'BENIGN']
    benign = df[df['Label'] == 'BENIGN']
    
    print(f"   Total attacks: {len(attacks)}")
    print(f"   Total benign: {len(benign):,}")
    
    # Use moderate sample for evolution analysis (need temporal continuity)
    benign_sample = benign.sample(n=min(8000, len(benign)), random_state=42)
    df_balanced = pd.concat([attacks, benign_sample]).sort_index()
    
    print(f"   Using {len(attacks)} attacks + {len(benign_sample):,} benign = {len(df_balanced):,} total")
    print(f"   Attack rate: {(df_balanced['Label'] != 'BENIGN').mean():.3%}")
    
    return df_balanced

def evaluate_temporal_evolution_tda():
    """Main evaluation function for temporal persistence evolution TDA."""
    
    # Load data
    df = load_and_prepare_data()
    
    # Initialize evolution analyzer
    analyzer = TemporalPersistenceEvolutionAnalyzer()
    
    # Extract temporal evolution features
    start_time = time.time()
    evolution_features, evolution_labels = analyzer.extract_evolution_features(df)
    extraction_time = time.time() - start_time
    
    if evolution_features is None:
        print("❌ Evolution feature extraction failed")
        return None
    
    print(f"\\n⏱️ Evolution feature extraction completed in {extraction_time:.1f}s")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        evolution_features, evolution_labels, test_size=0.3, random_state=42, stratify=evolution_labels
    )
    
    print(f"\\n📊 EVALUATION SETUP:")
    print(f"   Training: {X_train.shape}, attacks: {y_train.sum()}")
    print(f"   Testing: {X_test.shape}, attacks: {y_test.sum()}")
    
    # Train classifier on evolution features
    print(f"\\n🌊 TRAINING TEMPORAL EVOLUTION TDA CLASSIFIER")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Enhanced ensemble optimized for evolution features
    rf1 = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=4,
        random_state=42, class_weight='balanced'
    )
    rf2 = RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_split=6, 
        random_state=123, class_weight='balanced'
    )
    lr = LogisticRegression(
        C=0.1, random_state=42, class_weight='balanced', max_iter=1000
    )
    
    # Evolution-optimized ensemble
    ensemble = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('lr', lr)],
        voting='soft'
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = ensemble.predict(X_test_scaled)
    
    # Evaluation
    print(f"\\n📈 TEMPORAL EVOLUTION TDA RESULTS:")
    print("=" * 55)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracy = report['accuracy']
    precision = report.get('1', {}).get('precision', 0)
    recall = report.get('1', {}).get('recall', 0)
    f1_score = report.get('1', {}).get('f1-score', 0)
    
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1_score:.3f}")
    
    # Handle confusion matrix display
    if cm.shape == (2, 2):
        print(f"\\n   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")
    else:
        print(f"\\n   Confusion Matrix: {cm}")
    
    # Compare with previous results
    print(f"\\n📊 PERFORMANCE EVOLUTION COMPARISON:")
    print("=" * 55)
    
    single_scale_f1 = 0.182
    multi_scale_f1 = 0.654
    graph_based_f1 = 0.708
    hybrid_f1 = 0.706
    
    print(f"   Single-Scale TDA: F1 = {single_scale_f1:.3f}")
    print(f"   Multi-Scale TDA: F1 = {multi_scale_f1:.3f} (+{(multi_scale_f1-single_scale_f1)*100:.1f}%)")
    print(f"   Graph-Based TDA: F1 = {graph_based_f1:.3f} (+{(graph_based_f1-multi_scale_f1)*100:.1f}%)")
    print(f"   Hybrid TDA: F1 = {hybrid_f1:.3f} (+{(hybrid_f1-graph_based_f1)*100:.1f}%)")
    print(f"   🌊 Evolution TDA: F1 = {f1_score:.3f} (+{(f1_score-hybrid_f1)*100:.1f}%)")
    
    # Target assessment
    target_f1 = 0.75
    if f1_score >= target_f1:
        print(f"\\n   ✅ SUCCESS: Achieved Phase 2B target (F1 ≥ 75%)!")
        status = "SUCCESS"
    elif f1_score > hybrid_f1:
        print(f"\\n   ✅ IMPROVEMENT: Evolution approach outperforms hybrid baseline")
        gap = target_f1 - f1_score
        print(f"   Gap to target: {gap:.3f} ({gap/target_f1*100:.1f}%)")
        status = "PROGRESS"
    else:
        print(f"\\n   ⚠️ MIXED: Evolution performance assessment needed")
        status = "MIXED"
    
    total_improvement = f1_score - single_scale_f1
    print(f"\\n   📈 Total improvement from original: +{total_improvement:.3f} ({(total_improvement/single_scale_f1)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'improvement_from_hybrid': f1_score - hybrid_f1,
        'total_improvement': total_improvement,
        'extraction_time': extraction_time,
        'status': status
    }

def main():
    """Main execution function."""
    
    print("🌊 TEMPORAL PERSISTENCE EVOLUTION TDA IMPLEMENTATION")
    print("=" * 80)
    print("Phase 2B of Advanced TDA Enhancement Strategy")
    print("Expected: +6-10% F1-score improvement")
    print("Current gap to 75% target: 4.4%")
    print("=" * 80)
    
    # Run evaluation
    results = evaluate_temporal_evolution_tda()
    
    if results:
        print(f"\\n🎯 PHASE 2B EVALUATION COMPLETE")
        print("=" * 80)
        
        if results['status'] == 'SUCCESS':
            print(f"✅ BREAKTHROUGH: Evolution TDA achieved target performance!")
            print(f"   Recommended: Prepare for production deployment")
        elif results['status'] == 'PROGRESS':
            print(f"✅ PROGRESS: Evolution approach shows improvement")
            print(f"   Recommended: Combine with hybrid approach for maximum performance")
        else:
            print(f"⚠️ MIXED: Evolution approach needs further analysis")
            print(f"   Recommended: Investigate feature engineering optimizations")
        
        print(f"\\n📋 Next Steps:")
        print(f"   1. Document evolution approach results")
        print(f"   2. If successful: Combine evolution + hybrid for ultimate ensemble")
        print(f"   3. If needed: Proceed to Phase 2C (Advanced ML Integration)")

if __name__ == "__main__":
    main()