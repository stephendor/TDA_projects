#!/usr/bin/env python3
"""
UNSW-NB15 TDA Validation with Leakage Prevention
===============================================

Validate our TDA approach on UNSW-NB15 dataset with careful attention to:
1. Data leakage prevention (remove duplicates)
2. Distribution shift monitoring
3. Multiple attack type testing
4. Honest accuracy reporting

This will help determine if our previous high accuracies were due to 
dataset-specific issues or legitimate TDA effectiveness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from collections import defaultdict
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import TDA infrastructure
try:
    from src.core.persistent_homology import PersistentHomologyAnalyzer
    from src.core.mapper import MapperAnalyzer
    print("✓ TDA infrastructure imported")
except ImportError as e:
    print(f"❌ Cannot import TDA infrastructure: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UNSWTDAValidator:
    """
    TDA validation on UNSW-NB15 with proper leakage prevention
    """
    
    def __init__(self):
        self.train_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/UNSW-NB15/UNSW_NB15_training-set.parquet"
        self.test_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/UNSW-NB15/UNSW_NB15_testing-set.parquet"
        
        # Feature columns (excluding labels)
        self.feature_cols = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
            'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
            'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
            'trans_depth', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports'
        ]
        
    def load_and_preprocess_data(self):
        """Load data with duplicate removal and leakage prevention"""
        
        print("="*80)
        print("UNSW-NB15 TDA VALIDATION - LEAKAGE PREVENTION MODE")
        print("="*80)
        
        # Load datasets
        print("\n📁 Loading datasets...")
        train_df = pd.read_parquet(self.train_path)
        test_df = pd.read_parquet(self.test_path)
        
        print(f"Original sizes: Train={len(train_df):,}, Test={len(test_df):,}")
        
        # Remove duplicates within each set first
        print("\n🧹 Removing intra-set duplicates...")
        train_df_clean = train_df.drop_duplicates(subset=self.feature_cols)
        test_df_clean = test_df.drop_duplicates(subset=self.feature_cols)
        
        print(f"After intra-set deduplication: Train={len(train_df_clean):,}, Test={len(test_df_clean):,}")
        print(f"Removed: Train={len(train_df)-len(train_df_clean):,} ({(len(train_df)-len(train_df_clean))/len(train_df)*100:.1f}%)")
        print(f"Removed: Test={len(test_df)-len(test_df_clean):,} ({(len(test_df)-len(test_df_clean))/len(test_df)*100:.1f}%)")
        
        # Check for cross-contamination (same features in both sets)
        print("\n🔍 Checking for train/test contamination...")
        
        # Create feature signatures for comparison
        train_signatures = train_df_clean[self.feature_cols].round(6)  # Round to avoid floating point issues
        test_signatures = test_df_clean[self.feature_cols].round(6)
        
        # Convert to string signatures for exact matching
        train_sig_strings = train_signatures.apply(lambda x: '|'.join(x.astype(str)), axis=1)
        test_sig_strings = test_signatures.apply(lambda x: '|'.join(x.astype(str)), axis=1)
        
        # Find overlapping signatures
        train_sig_set = set(train_sig_strings)
        test_sig_set = set(test_sig_strings)
        overlap = train_sig_set & test_sig_set
        
        if len(overlap) > 0:
            print(f"⚠️  CRITICAL: {len(overlap)} identical feature patterns found in both sets!")
            
            # Remove overlapping samples from test set to prevent leakage
            contaminated_mask = test_sig_strings.isin(overlap)
            test_df_final = test_df_clean[~contaminated_mask].copy()
            train_df_final = train_df_clean.copy()
            
            print(f"🔧 Removed {contaminated_mask.sum():,} contaminated samples from test set")
            print(f"Final sizes: Train={len(train_df_final):,}, Test={len(test_df_final):,}")
        else:
            print("✓ No cross-contamination detected")
            train_df_final = train_df_clean.copy()
            test_df_final = test_df_clean.copy()
        
        return train_df_final, test_df_final
    
    def create_tda_point_clouds(self, data: pd.DataFrame, method: str = "sliding_window") -> Tuple[List[np.ndarray], List[int]]:
        """
        Create point clouds with proper labeling
        """
        
        print(f"\n🔹 Creating {method} point clouds...")
        
        # Prepare numeric data
        numeric_data = data[self.feature_cols].copy()
        
        # Handle missing values and convert to numeric
        for col in self.feature_cols:
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        numeric_data = numeric_data.fillna(0)
        
        # Scale features for better point cloud construction
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        point_clouds = []
        labels = []
        
        if method == "sliding_window":
            # Create sliding windows over the data
            window_size = 30  # Points per cloud
            step_size = 15    # Overlap between windows
            
            for i in range(0, len(scaled_data) - window_size, step_size):
                window_data = scaled_data[i:i+window_size]
                window_labels = data['label'].iloc[i:i+window_size]
                
                # Use majority vote for window label
                window_label = int(window_labels.mode().iloc[0])
                
                # Reduce dimensionality to 3D for TDA
                if window_data.shape[1] > 3:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=3)
                    window_data_3d = pca.fit_transform(window_data)
                else:
                    window_data_3d = window_data
                
                point_clouds.append(window_data_3d)
                labels.append(window_label)
        
        elif method == "attack_grouping":
            # Group by attack type and create point clouds
            for attack_cat, group in data.groupby('attack_cat'):
                if len(group) < 10:  # Skip small groups
                    continue
                
                group_data = scaler.fit_transform(group[self.feature_cols].fillna(0))
                
                # Create multiple point clouds from this attack type
                for i in range(0, len(group_data) - 30, 20):
                    cloud_data = group_data[i:i+30]
                    
                    # Reduce to 3D
                    if cloud_data.shape[1] > 3:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=3)
                        cloud_data_3d = pca.fit_transform(cloud_data)
                    else:
                        cloud_data_3d = cloud_data
                    
                    point_clouds.append(cloud_data_3d)
                    # Label: 0 for Normal, 1 for any attack
                    labels.append(0 if attack_cat == 'Normal' else 1)
        
        print(f"✓ Created {len(point_clouds)} point clouds")
        print(f"  Attack ratio: {np.mean(labels):.1%}")
        
        return point_clouds, labels
    
    def extract_tda_features(self, point_clouds: List[np.ndarray]) -> np.ndarray:
        """Extract TDA features using persistent homology"""
        
        print(f"\n🔹 Extracting TDA features from {len(point_clouds)} point clouds...")
        
        ph_analyzer = PersistentHomologyAnalyzer(maxdim=2, backend='ripser')
        all_features = []
        
        for i, cloud in enumerate(point_clouds):
            if i % 50 == 0:
                print(f"  Processing cloud {i+1}/{len(point_clouds)}")
            
            features = []
            
            try:
                # Ensure we have valid point cloud dimensions
                if cloud.shape[0] <= cloud.shape[1]:
                    # Skip if not enough points
                    features.extend([0] * 15)  # 3 dimensions × 5 features
                else:
                    ph_analyzer.fit(cloud)
                    
                    if ph_analyzer.persistence_diagrams_ is not None:
                        # Extract features for each homology dimension
                        for dim in range(3):  # H0, H1, H2
                            if dim < len(ph_analyzer.persistence_diagrams_):
                                diagram = ph_analyzer.persistence_diagrams_[dim]
                                
                                if len(diagram) > 0:
                                    births = diagram[:, 0]
                                    deaths = diagram[:, 1]
                                    lifetimes = deaths - births
                                    
                                    # Filter out infinite bars
                                    finite_mask = np.isfinite(lifetimes)
                                    finite_lifetimes = lifetimes[finite_mask]
                                    
                                    if len(finite_lifetimes) > 0:
                                        features.extend([
                                            len(diagram),                    # Betti number
                                            np.sum(finite_lifetimes),       # Total persistence
                                            np.max(finite_lifetimes),       # Max persistence
                                            np.mean(finite_lifetimes),      # Mean persistence
                                            np.std(finite_lifetimes),       # Persistence variance
                                        ])
                                    else:
                                        features.extend([len(diagram), 0, 0, 0, 0])
                                else:
                                    features.extend([0, 0, 0, 0, 0])
                            else:
                                features.extend([0, 0, 0, 0, 0])
                    else:
                        features.extend([0] * 15)
                        
            except Exception as e:
                logger.debug(f"TDA failed for cloud {i}: {e}")
                features.extend([0] * 15)
            
            all_features.append(features)
        
        return np.array(all_features)
    
    def validate_binary_classification(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate binary attack detection (Normal vs Attack)"""
        
        print("\n" + "="*60)
        print("BINARY CLASSIFICATION VALIDATION (Normal vs Attack)")
        print("="*60)
        
        # Create point clouds for training
        train_clouds, train_labels = self.create_tda_point_clouds(train_df, "sliding_window")
        train_features = self.extract_tda_features(train_clouds)
        
        # Create point clouds for testing  
        test_clouds, test_labels = self.create_tda_point_clouds(test_df, "sliding_window")
        test_features = self.extract_tda_features(test_clouds)
        
        # Convert to numpy arrays
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        
        print(f"\nDataset Statistics:")
        print(f"  Training clouds: {len(train_clouds)}")
        print(f"  Testing clouds: {len(test_clouds)}")
        print(f"  Feature dimensions: {train_features.shape[1]}")
        print(f"  Train attack ratio: {train_labels.mean():.1%}")
        print(f"  Test attack ratio: {test_labels.mean():.1%}")
        
        # Check if we have both classes in both sets
        train_classes = len(np.unique(train_labels))
        test_classes = len(np.unique(test_labels))
        
        if train_classes < 2 or test_classes < 2:
            print(f"❌ Insufficient classes: Train={train_classes}, Test={test_classes}")
            return None
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Train classifier
        print(f"\n🤖 Training RandomForest classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        clf.fit(train_features_scaled, train_labels)
        
        # Evaluate on test set
        test_pred = clf.predict(test_features_scaled)
        test_accuracy = accuracy_score(test_labels, test_pred)
        
        print(f"\n{'='*60}")
        print(f"UNSW-NB15 BINARY TDA RESULTS")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_accuracy:.1%}")
        print(f"Train Attack Ratio: {train_labels.mean():.1%}")
        print(f"Test Attack Ratio: {test_labels.mean():.1%}")
        print(f"\nClassification Report:")
        print(classification_report(test_labels, test_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(test_labels, test_pred))
        
        return test_accuracy
    
    def validate_multiclass_detection(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                    target_attacks: List[str] | None = None):
        """Validate specific attack type detection"""
        
        if target_attacks is None:
            target_attacks = ['DoS', 'Exploits', 'Fuzzers', 'Generic']
        
        results = {}
        
        for attack_type in target_attacks:
            print(f"\n" + "="*60)
            print(f"ATTACK-SPECIFIC VALIDATION: {attack_type}")
            print("="*60)
            
            # Filter for binary classification: target attack vs normal
            train_binary = train_df[
                (train_df['attack_cat'] == attack_type) | 
                (train_df['attack_cat'] == 'Normal')
            ].copy()
            
            test_binary = test_df[
                (test_df['attack_cat'] == attack_type) | 
                (test_df['attack_cat'] == 'Normal')
            ].copy()
            
            if len(train_binary) < 100 or len(test_binary) < 50:
                print(f"❌ Insufficient data for {attack_type}")
                results[attack_type] = None
                continue
            
            # Create binary labels (0=Normal, 1=Attack)
            train_binary['binary_label'] = (train_binary['attack_cat'] == attack_type).astype(int)
            test_binary['binary_label'] = (test_binary['attack_cat'] == attack_type).astype(int)
            
            print(f"Dataset sizes: Train={len(train_binary)}, Test={len(test_binary)}")
            print(f"Attack ratios: Train={train_binary['binary_label'].mean():.1%}, Test={test_binary['binary_label'].mean():.1%}")
            
            # Create point clouds
            train_clouds, train_labels = self.create_tda_point_clouds(train_binary, "attack_grouping")
            test_clouds, test_labels = self.create_tda_point_clouds(test_binary, "attack_grouping")
            
            if len(train_clouds) < 10 or len(test_clouds) < 5:
                print(f"❌ Insufficient point clouds for {attack_type}")
                results[attack_type] = None
                continue
            
            # Extract TDA features
            train_features = self.extract_tda_features(train_clouds)
            test_features = self.extract_tda_features(test_clouds)
            
            # Scale and train
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            clf.fit(train_features_scaled, train_labels)
            
            # Evaluate
            test_pred = clf.predict(test_features_scaled)
            accuracy = accuracy_score(test_labels, test_pred)
            
            print(f"\n{attack_type} TDA Results:")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Point clouds: Train={len(train_clouds)}, Test={len(test_clouds)}")
            print(f"  Classification Report:")
            print(classification_report(test_labels, test_pred))
            
            results[attack_type] = accuracy
        
        return results

def main():
    """Run comprehensive TDA validation on UNSW-NB15"""
    
    validator = UNSWTDAValidator()
    
    # Load and preprocess data with leakage prevention
    train_df, test_df = validator.load_and_preprocess_data()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("❌ No clean data available after preprocessing")
        return
    
    # Run binary classification validation
    binary_accuracy = validator.validate_binary_classification(train_df, test_df)
    
    # Run multi-class validations
    attack_results = validator.validate_multiclass_detection(train_df, test_df)
    
    # Summary
    print("\n" + "="*80)
    print("UNSW-NB15 TDA VALIDATION SUMMARY")
    print("="*80)
    print(f"Binary Classification Accuracy: {binary_accuracy:.1%}" if binary_accuracy else "Binary: Failed")
    
    print(f"\nAttack-Specific Results:")
    for attack, acc in attack_results.items():
        if acc is not None:
            print(f"  {attack:15s}: {acc:.1%}")
        else:
            print(f"  {attack:15s}: Failed")
    
    # Reality check
    if binary_accuracy and binary_accuracy > 0.90:
        print(f"\n⚠️  HIGH ACCURACY WARNING:")
        print(f"   Binary accuracy of {binary_accuracy:.1%} may indicate:")
        print(f"   - Remaining data leakage")
        print(f"   - Dataset artifacts")
        print(f"   - Overfitting")
        print(f"   - Need for more rigorous validation")
    
    print("="*80)

if __name__ == "__main__":
    main()
