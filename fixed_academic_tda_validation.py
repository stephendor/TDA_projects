#!/usr/bin/env python3
"""
FIXED Academic TDA Validation
============================

Fixing the point cloud construction issues and data loading problems.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import logging
import json

# Import TDA infrastructure
try:
    from src.core.persistent_homology import PersistentHomologyAnalyzer
    print("✓ TDA infrastructure imported")
except ImportError as e:
    print(f"❌ Cannot import TDA infrastructure: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedTDAFramework:
    """
    Fixed TDA implementation that creates proper point clouds
    """
    
    def __init__(self):
        # Load attack location map
        with open('/home/stephen-dorman/dev/TDA_projects/attack_location_map.json', 'r') as f:
            self.attack_map = json.load(f)
        
        self.data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    def load_mixed_chunk_data(self, target_chunks: list, max_samples: int = 1000) -> pd.DataFrame:
        """Load data from mixed chunks that contain both attacks and benign"""
        
        logger.info(f"Loading data from mixed chunks: {target_chunks[:5]}...")
        
        collected_data = []
        samples_collected = 0
        chunk_size = self.attack_map['chunk_size']
        
        for chunk_info in target_chunks:
            if samples_collected >= max_samples:
                break
                
            chunk_num = chunk_info[0]  # chunk number
            
            # Calculate proper row range
            start_row = chunk_num * chunk_size + 1  # +1 to skip header
            
            try:
                # Read chunk with header from original file
                chunk = pd.read_csv(
                    self.data_path,
                    skiprows=range(1, start_row + 1) if start_row > 1 else None,
                    nrows=chunk_size
                )
                
                if 'Attack' not in chunk.columns:
                    logger.warning(f"No Attack column in chunk {chunk_num}")
                    continue
                
                remaining = max_samples - samples_collected
                sample = chunk.head(remaining)
                
                if len(sample) > 0:
                    collected_data.append(sample)
                    samples_collected += len(sample)
                    logger.info(f"Loaded {len(sample)} samples from chunk {chunk_num}")
                
            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_num}: {e}")
                continue
        
        if collected_data:
            result = pd.concat(collected_data, ignore_index=True)
            logger.info(f"Total samples loaded: {len(result)}")
            
            # Show attack distribution
            attack_dist = result['Attack'].value_counts()
            logger.info("Attack distribution:")
            for attack, count in attack_dist.items():
                logger.info(f"  {attack}: {count}")
            
            return result
        else:
            logger.error("No data loaded")
            return pd.DataFrame()
    
    def create_proper_point_clouds(self, data: pd.DataFrame, window_size: int = 30) -> tuple:
        """Create proper point clouds with more points than dimensions"""
        
        logger.info("Creating proper point clouds...")
        
        # Get numeric columns
        numeric_cols = [col for col in data.columns 
                       if col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack']]
        
        # Convert to numeric
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.fillna(0)
        
        # Sort by time if available
        if 'FLOW_START_MILLISECONDS' in data.columns:
            data = data.sort_values('FLOW_START_MILLISECONDS')
        
        point_clouds = []
        labels = []
        
        # Create sliding windows
        for i in range(0, len(data) - window_size, window_size // 2):
            window = data.iloc[i:i+window_size]
            
            if len(window) < 10:
                continue
            
            # Get features for this window
            features = window[numeric_cols].values
            
            # Ensure we have valid numeric data
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue
            
            # Critical fix: Create proper point cloud dimensions
            n_points, n_features = features.shape
            
            if n_points < 3:  # Need minimum points for topology
                continue
            
            if n_features > n_points:
                # Too many features - use PCA to reduce
                n_components = min(3, n_points - 1)
                if n_components < 2:
                    continue
                pca = PCA(n_components=n_components)
                features = pca.fit_transform(features)
            elif n_features > 3:
                # Reduce to 3D for computational efficiency
                pca = PCA(n_components=3)
                features = pca.fit_transform(features)
            
            # Final validation: points > dimensions and at least 2D
            if features.shape[0] > features.shape[1] and features.shape[1] >= 2:
                point_clouds.append(features)
                
                # Label this window based on majority class
                window_attacks = window[window['Attack'] != 'Benign']
                is_attack = len(window_attacks) > len(window) * 0.1  # 10% threshold
                labels.append(1 if is_attack else 0)
        
        logger.info(f"Created {len(point_clouds)} valid point clouds")
        if len(point_clouds) > 0:
            logger.info(f"Point cloud shape example: {point_clouds[0].shape}")
            logger.info(f"Attack window ratio: {np.mean(labels):.3f}")
        
        return point_clouds, np.array(labels)
    
    def extract_tda_features(self, point_clouds: list) -> np.ndarray:
        """Extract TDA features using persistent homology"""
        
        logger.info("Extracting TDA features...")
        
        ph_analyzer = PersistentHomologyAnalyzer(maxdim=2, backend='ripser')
        all_features = []
        
        for i, cloud in enumerate(point_clouds):
            if i % 10 == 0:
                logger.info(f"Processing cloud {i+1}/{len(point_clouds)}")
            
            features = []
            
            try:
                # Validate point cloud shape
                if cloud.shape[0] <= cloud.shape[1]:
                    logger.warning(f"Invalid cloud shape: {cloud.shape}")
                    features.extend([0] * 15)  # 3 dimensions × 5 features
                else:
                    ph_analyzer.fit(cloud)
                    
                    if ph_analyzer.persistence_diagrams_ is not None:
                        # Extract features for each dimension
                        for dim in range(3):  # H0, H1, H2
                            if dim < len(ph_analyzer.persistence_diagrams_):
                                diagram = ph_analyzer.persistence_diagrams_[dim]
                                
                                if len(diagram) > 0:
                                    births = diagram[:, 0]
                                    deaths = diagram[:, 1]
                                    lifetimes = deaths - births
                                    
                                    # Remove infinite bars
                                    finite_mask = np.isfinite(lifetimes)
                                    finite_lifetimes = lifetimes[finite_mask]
                                    
                                    if len(finite_lifetimes) > 0:
                                        features.extend([
                                            len(diagram),  # Betti number
                                            np.sum(finite_lifetimes),  # Total persistence
                                            np.max(finite_lifetimes),  # Max persistence
                                            np.mean(finite_lifetimes),  # Mean persistence
                                            np.std(finite_lifetimes),   # Persistence variance
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
    
    def validate_attack_detection(self, attack_type: str):
        """Validate attack detection using fixed TDA approach"""
        
        logger.info(f"Validating {attack_type} detection...")
        
        # Get mixed chunks for this attack type
        recommended_chunks = self.attack_map['recommended_chunks']
        
        # Find chunks with this attack type
        target_chunks = []
        for chunk_info in recommended_chunks:
            chunk_num, ratio, attacks = chunk_info
            attack_names = [name for name, _ in attacks]
            if attack_type in attack_names:
                target_chunks.append(chunk_info)
        
        if not target_chunks:
            logger.error(f"No mixed chunks found for {attack_type}")
            return None
        
        logger.info(f"Found {len(target_chunks)} mixed chunks for {attack_type}")
        
        # Load data
        data = self.load_mixed_chunk_data(target_chunks[:5], max_samples=2000)
        
        if len(data) == 0:
            logger.error(f"No data loaded for {attack_type}")
            return None
        
        # Create point clouds
        point_clouds, labels = self.create_proper_point_clouds(data)
        
        if len(point_clouds) == 0:
            logger.error(f"No valid point clouds created")
            return None
        
        # Check if we have both classes
        if len(np.unique(labels)) < 2:
            logger.warning(f"Only one class present - cannot evaluate")
            return None
        
        # Extract TDA features
        tda_features = self.extract_tda_features(point_clouds)
        
        if tda_features.size == 0:
            logger.error(f"No TDA features extracted")
            return None
        
        # Train and evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            tda_features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*80}")
        print(f"FIXED ACADEMIC TDA VALIDATION: {attack_type}")
        print(f"{'='*80}")
        print(f"Point clouds: {len(point_clouds)}")
        print(f"TDA features: {tda_features.shape[1]}")
        print(f"Attack ratio: {labels.mean():.1%}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"{'='*80}")
        
        return {
            'attack_type': attack_type,
            'accuracy': accuracy,
            'point_clouds': len(point_clouds),
            'features': tda_features.shape[1],
            'attack_ratio': labels.mean()
        }

def main():
    """Test fixed TDA approach"""
    print("FIXED ACADEMIC TDA VALIDATION")
    print("Properly constructing point clouds with points > dimensions")
    print("="*80)
    
    framework = FixedTDAFramework()
    
    # Test attack types that have mixed chunks
    attack_types = ['SSH-Bruteforce', 'DoS_attacks-SlowHTTPTest', 'Bot']
    
    results = []
    
    for attack_type in attack_types:
        print(f"\n🔍 Testing {attack_type}...")
        result = framework.validate_attack_detection(attack_type)
        if result:
            results.append(result)
        else:
            print(f"❌ Failed: {attack_type}")
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("FIXED TDA RESULTS SUMMARY")
        print(f"{'='*80}")
        for result in results:
            print(f"{result['attack_type']:25s}: {result['accuracy']:.1%} accuracy "
                  f"({result['point_clouds']} clouds, attack ratio: {result['attack_ratio']:.1%})")
        print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    main()
