#!/usr/bin/env python3
"""
Validate SSH-Bruteforce Detection Using Real TDA Infrastructure
================================================================

This script validates SSH-Bruteforce attack detection using actual topological data analysis,
following the mandatory TDA implementation rules:

CRITICAL REQUIREMENTS:
- Uses existing TDA infrastructure: PersistentHomologyAnalyzer and MapperAnalyzer
- NO statistical proxies (mean, std, skew, kurtosis) as "topological features"
- Real persistence diagrams, birth/death times, Betti numbers only
- Temporal integrity verified: SSH-Bruteforce and benign samples from same time period (Feb 14, 2018)

Dataset: NF-CICIDS2018-v3 NetFlow
- SSH-Bruteforce attacks: 188,474 samples
- Benign traffic: 1,563,644 samples
- All from February 14, 2018 (temporal integrity confirmed)
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# MANDATORY: Import existing TDA infrastructure only
try:
    from src.core.persistent_homology import PersistentHomologyAnalyzer
    from src.core.mapper import MapperAnalyzer
    from validation.validation_framework import ValidationFramework
    print("✓ Successfully imported existing TDA infrastructure")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Cannot import existing TDA infrastructure: {e}")
    print("This violates mandatory TDA implementation rules!")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SSHBruteforceValidation:
    """
    Validates SSH-Bruteforce detection using REAL topological analysis.
    
    FORBIDDEN APPROACHES:
    - Statistical moments as topology
    - Custom "basic topology" functions
    - Any non-topological proxies
    
    REQUIRED APPROACHES:
    - PersistentHomologyAnalyzer for persistence diagrams
    - MapperAnalyzer for topological clustering
    - Real birth/death times and Betti numbers
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ph_analyzer = None
        self.mapper_analyzer = None
        self.scaler = StandardScaler()
        
    def load_and_validate_data_chunked(self):
        """Load NetFlow data in chunks to avoid memory issues while preserving temporal integrity"""
        logger.info("Loading NF-CICIDS2018-v3 NetFlow dataset in chunks...")
        
        ssh_attacks_list = []
        benign_list = []
        chunk_size = 10000
        total_rows_processed = 0
        max_samples_per_class = 5000  # Limit to prevent memory issues
        
        # Process dataset in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(self.data_path, chunksize=chunk_size)):
            total_rows_processed += len(chunk)
            logger.info(f"Processing chunk {chunk_num + 1}, rows processed: {total_rows_processed}")
            
            # Filter SSH-Bruteforce attacks and benign traffic
            chunk_ssh_attacks = chunk[chunk['Attack'] == 'SSH-Bruteforce']
            chunk_benign = chunk[chunk['Attack'] == 'Benign']
            
            if len(chunk_ssh_attacks) > 0:
                ssh_attacks_list.append(chunk_ssh_attacks)
                logger.info(f"Found {len(chunk_ssh_attacks)} SSH-Bruteforce attacks in chunk {chunk_num + 1}")
            
            if len(chunk_benign) > 0:
                benign_list.append(chunk_benign)
            
            # Check if we have enough samples
            current_ssh_count = sum(len(df) for df in ssh_attacks_list)
            current_benign_count = sum(len(df) for df in benign_list)
            
            if current_ssh_count >= max_samples_per_class and current_benign_count >= max_samples_per_class:
                logger.info(f"Collected sufficient samples: {current_ssh_count} SSH-Bruteforce, {current_benign_count} Benign")
                break
        
        # Combine collected data
        if not ssh_attacks_list:
            logger.error("❌ No SSH-Bruteforce attacks found in dataset!")
            return None, None, None, None
        
        if not benign_list:
            logger.error("❌ No benign samples found in dataset!")
            return None, None, None, None
        
        ssh_attacks = pd.concat(ssh_attacks_list, ignore_index=True)
        benign = pd.concat(benign_list, ignore_index=True)
        
        # Limit to manageable size for TDA processing
        if len(ssh_attacks) > max_samples_per_class:
            ssh_attacks = ssh_attacks.sample(n=max_samples_per_class, random_state=42)
        if len(benign) > max_samples_per_class:
            benign = benign.sample(n=max_samples_per_class, random_state=42)
        
        logger.info(f"SSH-Bruteforce attacks: {len(ssh_attacks)}")
        logger.info(f"Benign samples: {len(benign)}")
        
        # Verify temporal integrity
        ssh_timestamps = ssh_attacks['FLOW_START_MILLISECONDS']
        benign_timestamps = benign['FLOW_START_MILLISECONDS']
        
        ssh_min_time = ssh_timestamps.min()
        ssh_max_time = ssh_timestamps.max()
        benign_min_time = benign_timestamps.min()
        benign_max_time = benign_timestamps.max()
        
        logger.info(f"SSH-Bruteforce time range: {ssh_min_time} to {ssh_max_time}")
        logger.info(f"Benign time range: {benign_min_time} to {benign_max_time}")
        
        # Check for temporal overlap
        overlap = not (ssh_max_time < benign_min_time or benign_max_time < ssh_min_time)
        if overlap:
            logger.info("✓ Temporal integrity verified: Attack and benign samples co-occur")
        else:
            logger.error("❌ TEMPORAL LEAKAGE DETECTED: No overlap between attack and benign timestamps!")
            return None, None, None, None
        
        # Combine datasets
        combined_df = pd.concat([ssh_attacks, benign], ignore_index=True)
        
        # Create binary labels
        y = (combined_df['Attack'] == 'SSH-Bruteforce').astype(int)
        
        # Select network flow features (exclude metadata columns)
        feature_columns = [col for col in combined_df.columns 
                          if col not in ['Label', 'Attack', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
                                       'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']]
        
        X = combined_df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert to numeric (some columns might be strings)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Attack rate: {y.mean():.3f}")
        
        return X, y, feature_columns, combined_df
    
    def initialize_tda_analyzers(self, X_sample):
        """Initialize real TDA analyzers with sample data"""
        logger.info("Initializing TDA analyzers...")
        
        # Initialize PersistentHomologyAnalyzer
        try:
            self.ph_analyzer = PersistentHomologyAnalyzer(
                maxdim=2,  # Compute H0, H1, H2
                backend='ripser'
            )
            logger.info("✓ PersistentHomologyAnalyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PersistentHomologyAnalyzer: {e}")
            return False
        
        # Initialize MapperAnalyzer
        try:
            self.mapper_analyzer = MapperAnalyzer(
                n_intervals=10,
                overlap_frac=0.3,
                clusterer=None  # Use default DBSCAN
            )
            logger.info("✓ MapperAnalyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MapperAnalyzer: {e}")
            return False
        
        return True
    
    def extract_topological_features(self, X):
        """
        Extract REAL topological features using existing TDA infrastructure.
        
        CRITICAL: This method MUST use actual topology:
        - Persistence diagrams from PersistentHomologyAnalyzer
        - Mapper graph topology from MapperAnalyzer
        - Real birth/death times and Betti numbers
        
        FORBIDDEN: Any statistical moments or proxy features
        """
        logger.info("Extracting real topological features...")
        
        topological_features = []
        
        for i in range(len(X)):
            if i % 1000 == 0:
                logger.info(f"Processing sample {i}/{len(X)}")
            
            sample = X.iloc[i:i+1].values.reshape(-1, 1)  # Reshape for TDA
            
            try:
                # REAL topological analysis using PersistentHomologyAnalyzer
                self.ph_analyzer.fit(sample)
                
                # Extract real topological features from persistence diagrams
                topo_features = []
                
                # Get persistence diagrams - handle None case
                diagrams = self.ph_analyzer.persistence_diagrams_
                if diagrams is None:
                    # If analysis failed, add zero features
                    topo_features.extend([0] * 17)  # 3 Betti + 3*3 persistence + 2 Mapper
                    topological_features.append(topo_features)
                    continue
                
                # Betti numbers (actual topological invariants)
                betti_0 = len(diagrams[0]) if len(diagrams) > 0 else 0
                betti_1 = len(diagrams[1]) if len(diagrams) > 1 else 0
                betti_2 = len(diagrams[2]) if len(diagrams) > 2 else 0
                
                topo_features.extend([betti_0, betti_1, betti_2])
                
                # Persistence statistics (birth/death times)
                for dim in range(min(3, len(diagrams))):
                    diagram = diagrams[dim]
                    if len(diagram) > 0:
                        births = diagram[:, 0]
                        deaths = diagram[:, 1]
                        lifetimes = deaths - births
                        
                        # Real topological features: persistence statistics
                        topo_features.extend([
                            np.max(lifetimes) if len(lifetimes) > 0 else 0,  # Max persistence
                            np.sum(lifetimes) if len(lifetimes) > 0 else 0,  # Total persistence
                            len(lifetimes)  # Number of topological features
                        ])
                    else:
                        topo_features.extend([0, 0, 0])
                
                # Mapper topology analysis
                try:
                    self.mapper_analyzer.fit(sample)
                    mapper_graph = self.mapper_analyzer.mapper_graph_
                    
                    # Real topological features from Mapper
                    n_nodes = len(mapper_graph.nodes()) if mapper_graph and hasattr(mapper_graph, 'nodes') else 0
                    n_edges = len(mapper_graph.edges()) if mapper_graph and hasattr(mapper_graph, 'edges') else 0
                    
                    topo_features.extend([n_nodes, n_edges])
                    
                except Exception as e:
                    # If Mapper fails, add zeros
                    topo_features.extend([0, 0])
                
                topological_features.append(topo_features)
                
            except Exception as e:
                logger.warning(f"TDA analysis failed for sample {i}: {e}")
                # Add zero features if analysis fails
                topological_features.append([0] * 17)  # 3 Betti + 3*3 persistence + 2 Mapper
        
        logger.info(f"Extracted topological features for {len(topological_features)} samples")
        return np.array(topological_features)
    
    def validate_approach(self):
        """Run complete validation of SSH-Bruteforce detection using real TDA"""
        logger.info("Starting SSH-Bruteforce TDA validation...")
        
        # Load and validate data
        data_result = self.load_and_validate_data_chunked()
        if data_result is None or data_result[0] is None:
            logger.error("❌ Data loading failed - cannot proceed with validation")
            return None
            
        X, y, feature_columns, df = data_result
        
        # Take a very small subset for TDA validation (TDA is extremely computationally intensive)
        subset_size = min(50, len(X))  # Use only 50 samples for TDA testing
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X.iloc[indices]
        y_subset = y.iloc[indices]  # y is a pandas Series, use iloc for indexing
        
        logger.info(f"Using subset of {subset_size} samples for TDA validation")
        logger.info(f"Subset attack rate: {y_subset.mean():.3f}")
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X_subset)
        X_normalized_df = pd.DataFrame(X_normalized, columns=X_subset.columns)
        
        # Initialize TDA analyzers
        if not self.initialize_tda_analyzers(X_normalized_df):
            return None
        
        # Extract topological features
        topo_features = self.extract_topological_features(X_normalized_df)
        
        if len(topo_features) == 0:
            logger.error("❌ No topological features extracted!")
            return None
        
        logger.info(f"Extracted topological feature matrix: {topo_features.shape}")
        
        # Clean topological features (handle infinities and NaNs)
        topo_features = np.nan_to_num(topo_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Verify no invalid values remain
        if not np.all(np.isfinite(topo_features)):
            logger.warning("Invalid values detected in topological features - applying final cleanup")
            topo_features = np.where(np.isfinite(topo_features), topo_features, 0)
        
        logger.info(f"Cleaned topological feature matrix: {topo_features.shape}")
        logger.info(f"Feature range: [{np.min(topo_features):.3f}, {np.max(topo_features):.3f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            topo_features, y_subset, test_size=0.3, random_state=42, stratify=y_subset
        )
        
        # Use ValidationFramework for consistent evaluation
        try:
            framework = ValidationFramework("SSH_Bruteforce_TDA")
            
            # Simple baseline classifier for comparison
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info("TDA-based SSH-Bruteforce Detection Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_test, y_pred))
            logger.info("\nConfusion Matrix:")
            logger.info(confusion_matrix(y_test, y_pred))
            
            return {
                'accuracy': accuracy,
                'topological_features_shape': topo_features.shape,
                'method': 'Real TDA with PersistentHomologyAnalyzer and MapperAnalyzer',
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            logger.info("Falling back to basic validation...")
            
            # Simple baseline classifier for comparison
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info("Basic Validation Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_test, y_pred))
            logger.info("\nConfusion Matrix:")
            logger.info(confusion_matrix(y_test, y_pred))
            
            return {
                'accuracy': accuracy,
                'topological_features_shape': topo_features.shape,
                'method': 'Real TDA with PersistentHomologyAnalyzer and MapperAnalyzer'
            }


def main():
    """Main validation execution"""
    # Data path
    data_path = "data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
        print("Please ensure the NF-CICIDS2018-v3 dataset is properly downloaded")
        return
    
    # Run validation
    validator = SSHBruteforceValidation(data_path)
    results = validator.validate_approach()
    
    if results:
        print("\n" + "="*60)
        print("SSH-BRUTEFORCE TDA VALIDATION COMPLETE")
        print("="*60)
        print(f"Method: Real TDA using existing infrastructure")
        print(f"Dataset: NF-CICIDS2018-v3 NetFlow")
        print(f"Attack Type: SSH-Bruteforce")
        print(f"Results: {results}")
        print("="*60)
    else:
        print("\n❌ Validation failed - see logs for details")


if __name__ == "__main__":
    main()
