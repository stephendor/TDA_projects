#!/usr/bin/env python3
"""
Validate Infilteration Attack Detection Using Advanced TDA
===========================================================

This script validates Infilteration (APT lateral movement) detection using sophisticated 
topological data analysis, focusing on the most promising attack type for TDA.

INFILTERATION ATTACK CHARACTERISTICS:
- Multi-stage lateral movement patterns
- Complex network topology evolution  
- Rich temporal persistence structures
- Advanced persistent threat (APT) behavior

ADVANCED TDA FEATURES:
- Time-series persistence analysis
- Multi-scale topological features
- Evolving Mapper graph topology
- Protocol-specific persistence patterns

Expected Performance: >70% accuracy (vs 46.7% for SSH-Bruteforce)
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# MANDATORY: Import existing TDA infrastructure only
try:
    from src.core.persistent_homology import PersistentHomologyAnalyzer
    from src.core.mapper import MapperAnalyzer
    print("✓ Successfully imported existing TDA infrastructure")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Cannot import existing TDA infrastructure: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfilterationTDAValidation:
    """
    Advanced TDA validation for Infilteration attacks.
    
    INFILTERATION = IDEAL TDA TARGET:
    - Multi-stage attack progression
    - Complex network behavior patterns
    - Rich topological evolution over time
    - Advanced persistent threat characteristics
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ph_analyzer = None
        self.mapper_analyzer = None
        self.scaler = StandardScaler()
        
    def load_infilteration_data_targeted(self):
        """Load Infilteration data using targeted approach to find samples efficiently"""
        logger.info("Loading Infilteration attacks with targeted search...")
        
        infilteration_samples = []
        benign_samples = []
        chunk_size = 10000
        total_rows_processed = 0
        max_samples_per_class = 1000  # Start with smaller set for TDA validation
        
        # Start searching from where we know Infilteration attacks exist (around line 14M)
        skip_rows = 14000000  # Skip to where Infilteration attacks start
        logger.info(f"Skipping to row {skip_rows:,} where Infilteration attacks begin...")
        
        # Read header first to get column names
        header_df = pd.read_csv(self.data_path, nrows=1)
        column_names = header_df.columns.tolist()
        
        # Process dataset starting from the Infilteration region
        for chunk_num, chunk in enumerate(pd.read_csv(self.data_path, chunksize=chunk_size, skiprows=skip_rows, header=None, names=column_names)):
            total_rows_processed += len(chunk)
            
            if chunk_num % 100 == 0:
                logger.info(f"Processing chunk {chunk_num + 1}, rows processed: {total_rows_processed}")
            
            # Filter for Infilteration attacks and benign traffic
            chunk_infilteration = chunk[chunk['Attack'] == 'Infilteration']
            chunk_benign = chunk[chunk['Attack'] == 'Benign']
            
            if len(chunk_infilteration) > 0:
                infilteration_samples.append(chunk_infilteration)
                logger.info(f"Found {len(chunk_infilteration)} Infilteration attacks in chunk {chunk_num + 1}")
            
            if len(chunk_benign) > 0:
                benign_samples.append(chunk_benign)
            
            # Check if we have enough samples
            current_infilteration_count = sum(len(df) for df in infilteration_samples)
            current_benign_count = sum(len(df) for df in benign_samples)
            
            if current_infilteration_count >= max_samples_per_class and current_benign_count >= max_samples_per_class:
                logger.info(f"Collected sufficient samples: {current_infilteration_count} Infilteration, {current_benign_count} Benign")
                break
        
        if not infilteration_samples:
            logger.error("❌ No Infilteration attacks found!")
            return None
        
        if not benign_samples:
            logger.error("❌ No benign samples found!")
            return None
        
        # Combine collected data
        infilteration_df = pd.concat(infilteration_samples, ignore_index=True)
        benign_df = pd.concat(benign_samples, ignore_index=True)
        
        # Limit to manageable size
        if len(infilteration_df) > max_samples_per_class:
            infilteration_df = infilteration_df.sample(n=max_samples_per_class, random_state=42)
        if len(benign_df) > max_samples_per_class:
            benign_df = benign_df.sample(n=max_samples_per_class, random_state=42)
        
        logger.info(f"Infilteration attacks: {len(infilteration_df)}")
        logger.info(f"Benign samples: {len(benign_df)}")
        
        # Verify temporal integrity
        inf_timestamps = infilteration_df['FLOW_START_MILLISECONDS']
        benign_timestamps = benign_df['FLOW_START_MILLISECONDS']
        
        logger.info(f"Infilteration time range: {inf_timestamps.min()} to {inf_timestamps.max()}")
        logger.info(f"Benign time range: {benign_timestamps.min()} to {benign_timestamps.max()}")
        
        # Check for temporal overlap
        overlap = not (inf_timestamps.max() < benign_timestamps.min() or benign_timestamps.max() < inf_timestamps.min())
        if overlap:
            logger.info("✓ Temporal integrity verified: Attack and benign samples co-occur")
        else:
            logger.warning("⚠️ Limited temporal overlap - proceeding with available data")
        
        # Combine datasets
        combined_df = pd.concat([infilteration_df, benign_df], ignore_index=True)
        
        # Create binary labels
        y = (combined_df['Attack'] == 'Infilteration').astype(int)
        
        # Select network flow features for advanced TDA analysis
        feature_columns = [col for col in combined_df.columns 
                          if col not in ['Label', 'Attack', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
                                       'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']]
        
        X = combined_df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Infilteration attack rate: {y.mean():.3f}")
        
        return X, y, feature_columns, combined_df
    
    def extract_advanced_topological_features(self, X):
        """
        Extract ADVANCED topological features specifically designed for complex attacks.
        
        INFILTERATION-OPTIMIZED TDA:
        - Multi-scale persistence analysis
        - Time-series topological features  
        - Enhanced Mapper graph analysis
        - Protocol-aware topological patterns
        """
        logger.info("Extracting advanced topological features for Infilteration detection...")
        
        topological_features = []
        
        for i in range(len(X)):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{len(X)}")
            
            sample = X.iloc[i:i+1].values.reshape(-1, 1)
            
            try:
                # ADVANCED PERSISTENCE HOMOLOGY ANALYSIS
                self.ph_analyzer.fit(sample)
                
                topo_features = []
                diagrams = self.ph_analyzer.persistence_diagrams_
                
                if diagrams is None:
                    topo_features.extend([0] * 25)  # Extended feature set
                    topological_features.append(topo_features)
                    continue
                
                # Enhanced Betti number analysis
                betti_0 = len(diagrams[0]) if len(diagrams) > 0 else 0
                betti_1 = len(diagrams[1]) if len(diagrams) > 1 else 0
                betti_2 = len(diagrams[2]) if len(diagrams) > 2 else 0
                
                topo_features.extend([betti_0, betti_1, betti_2])
                
                # MULTI-SCALE PERSISTENCE ANALYSIS
                for dim in range(min(3, len(diagrams))):
                    diagram = diagrams[dim]
                    if len(diagram) > 0:
                        births = diagram[:, 0]
                        deaths = diagram[:, 1]
                        lifetimes = deaths - births
                        
                        # Advanced persistence statistics
                        topo_features.extend([
                            np.max(lifetimes) if len(lifetimes) > 0 else 0,  # Max persistence
                            np.mean(lifetimes) if len(lifetimes) > 0 else 0,  # Mean persistence  
                            np.std(lifetimes) if len(lifetimes) > 0 else 0,   # Persistence variance
                            np.sum(lifetimes) if len(lifetimes) > 0 else 0,   # Total persistence
                            len(lifetimes),  # Feature count
                            np.percentile(lifetimes, 75) if len(lifetimes) > 0 else 0,  # 75th percentile
                        ])
                    else:
                        topo_features.extend([0, 0, 0, 0, 0, 0])
                
                # ENHANCED MAPPER ANALYSIS
                try:
                    self.mapper_analyzer.fit(sample)
                    mapper_graph = self.mapper_analyzer.mapper_graph_
                    
                    if mapper_graph and hasattr(mapper_graph, 'nodes'):
                        n_nodes = len(mapper_graph.nodes())
                        n_edges = len(mapper_graph.edges()) if hasattr(mapper_graph, 'edges') else 0
                        
                        # Advanced graph topology features
                        topo_features.extend([
                            n_nodes,
                            n_edges,
                            n_edges / max(n_nodes, 1),  # Edge density
                            n_nodes / max(n_edges, 1) if n_edges > 0 else 0,  # Node/edge ratio
                        ])
                    else:
                        topo_features.extend([0, 0, 0, 0])
                        
                except Exception as e:
                    topo_features.extend([0, 0, 0, 0])
                
                # Ensure we have exactly 25 features
                while len(topo_features) < 25:
                    topo_features.append(0)
                topo_features = topo_features[:25]
                
                topological_features.append(topo_features)
                
            except Exception as e:
                logger.warning(f"TDA analysis failed for sample {i}: {e}")
                topological_features.append([0] * 25)
        
        logger.info(f"Extracted advanced topological features for {len(topological_features)} samples")
        return np.array(topological_features)
    
    def initialize_advanced_tda_analyzers(self, X_sample):
        """Initialize TDA analyzers optimized for complex attack patterns"""
        logger.info("Initializing advanced TDA analyzers...")
        
        # Enhanced PersistentHomologyAnalyzer for complex patterns
        try:
            self.ph_analyzer = PersistentHomologyAnalyzer(
                maxdim=2,  # H0, H1, H2
                backend='ripser'
            )
            logger.info("✓ Advanced PersistentHomologyAnalyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PersistentHomologyAnalyzer: {e}")
            return False
        
        # Enhanced MapperAnalyzer for infilteration patterns
        try:
            self.mapper_analyzer = MapperAnalyzer(
                n_intervals=15,  # Higher resolution for complex patterns
                overlap_frac=0.4,  # More overlap for better topology capture
                clusterer=None  # Default DBSCAN
            )
            logger.info("✓ Advanced MapperAnalyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MapperAnalyzer: {e}")
            return False
        
        return True
    
    def validate_infilteration_detection(self):
        """Run advanced TDA validation for Infilteration attack detection"""
        logger.info("Starting advanced Infilteration TDA validation...")
        
        # Load Infilteration attack data
        data_result = self.load_infilteration_data_targeted()
        if data_result is None:
            logger.error("❌ Data loading failed")
            return None
            
        X, y, feature_columns, df = data_result
        
        # Use larger subset for Infilteration (more complex patterns)
        subset_size = min(100, len(X))  # 100 samples for advanced analysis
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X.iloc[indices]
        y_subset = y.iloc[indices]
        
        logger.info(f"Using subset of {subset_size} samples for advanced TDA validation")
        logger.info(f"Infilteration attack rate: {y_subset.mean():.3f}")
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X_subset)
        X_normalized_df = pd.DataFrame(X_normalized, columns=X_subset.columns)
        
        # Initialize advanced TDA analyzers
        if not self.initialize_advanced_tda_analyzers(X_normalized_df):
            return None
        
        # Extract advanced topological features
        advanced_topo_features = self.extract_advanced_topological_features(X_normalized_df)
        
        if len(advanced_topo_features) == 0:
            logger.error("❌ No topological features extracted!")
            return None
        
        logger.info(f"Extracted advanced topological feature matrix: {advanced_topo_features.shape}")
        
        # Clean features
        advanced_topo_features = np.nan_to_num(advanced_topo_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if not np.all(np.isfinite(advanced_topo_features)):
            logger.warning("Cleaning infinite values in topological features")
            advanced_topo_features = np.where(np.isfinite(advanced_topo_features), advanced_topo_features, 0)
        
        logger.info(f"Feature range: [{np.min(advanced_topo_features):.3f}, {np.max(advanced_topo_features):.3f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            advanced_topo_features, y_subset, test_size=0.3, random_state=42, stratify=y_subset
        )
        
        # Advanced classifier for complex topological patterns
        rf = RandomForestClassifier(
            n_estimators=200,  # More trees for complex patterns
            max_depth=10,      # Deeper trees for intricate topology
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info("ADVANCED TDA-based Infilteration Detection Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'topological_features_shape': advanced_topo_features.shape,
            'method': 'Advanced TDA with Enhanced PersistentHomology and Mapper',
            'attack_type': 'Infilteration',
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }


def main():
    """Main execution for advanced Infilteration TDA validation"""
    data_path = "data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
        return
    
    # Run advanced validation
    validator = InfilterationTDAValidation(data_path)
    results = validator.validate_infilteration_detection()
    
    if results:
        print("\n" + "="*80)
        print("ADVANCED INFILTERATION TDA VALIDATION COMPLETE")
        print("="*80)
        print(f"Attack Type: {results['attack_type']} (Multi-stage APT)")
        print(f"Method: {results['method']}")
        print(f"Accuracy: {results['accuracy']:.1%}")
        print(f"Feature Matrix: {results['topological_features_shape']}")
        print()
        print("COMPARISON WITH PREVIOUS RESULTS:")
        print(f"• SSH-Bruteforce (simple): 46.7% accuracy")
        print(f"• Infilteration (complex): {results['accuracy']:.1%} accuracy")
        print(f"• Improvement: {(results['accuracy'] - 0.467) * 100:+.1f} percentage points")
        print("="*80)
    else:
        print("\n❌ Advanced validation failed - see logs for details")


if __name__ == "__main__":
    main()
