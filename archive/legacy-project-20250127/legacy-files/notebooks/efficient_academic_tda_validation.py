#!/usr/bin/env python3
"""
Academic TDA Implementation with Proper Data Loading
===================================================

Now using efficient targeted data loading with all attack types properly identified.
Based on "A Review of Topological Data Analysis for Cybersecurity" methodologies.
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
import networkx as nx
import json

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

class EfficientTDAFramework:
    """
    Efficient TDA implementation using targeted data loading and academic methodologies
    """
    
    def __init__(self):
        # Load attack location map
        with open('/home/stephen-dorman/dev/TDA_projects/attack_location_map.json', 'r') as f:
            self.attack_map = json.load(f)
        
        self.data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
        self.scaler = StandardScaler()
        
        # Available attack types from dataset analysis
        self.available_attacks = [
            'SSH-Bruteforce', 'Bot', 'FTP-BruteForce', 'Infilteration',
            'DDOS_attack-HOIC', 'DDoS_attacks-LOIC-HTTP', 'DoS_attacks-SlowHTTPTest',
            'DoS_attacks-Hulk', 'DoS_attacks-GoldenEye', 'DoS_attacks-Slowloris'
        ]
        
    def load_attack_data_efficiently(self, attack_type: str, n_samples: int = 2000) -> pd.DataFrame:
        """Load specific attack data efficiently using attack location map"""
        
        if attack_type not in self.attack_map['attack_locations']:
            logger.error(f"Attack type {attack_type} not found in dataset")
            return pd.DataFrame()
        
        # Get chunks containing this attack
        attack_chunks = self.attack_map['attack_locations'][attack_type]
        chunk_size = self.attack_map['chunk_size']
        
        collected_data = []
        samples_collected = 0
        
        logger.info(f"Loading {attack_type} from {len(attack_chunks)} chunks...")
        
        for chunk_num, chunk_count in attack_chunks:
            if samples_collected >= n_samples:
                break
            
            # Load specific chunk
            skip_rows = chunk_num * chunk_size + 1  # +1 for header
            
            try:
                # Read with proper column names since we're skipping header
                chunk = pd.read_csv(
                    self.data_path, 
                    skiprows=range(1, skip_rows + 1),  # Skip to the right position
                    nrows=chunk_size
                )
                
                # Check if 'Attack' column exists
                if 'Attack' not in chunk.columns:
                    logger.warning(f"Attack column not found in chunk {chunk_num}")
                    continue
                
                # Filter for target attack
                attack_data = chunk[chunk['Attack'] == attack_type]
                benign_data = chunk[chunk['Attack'] == 'Benign']
                
                # Take what we need
                remaining = n_samples - samples_collected
                attack_sample = attack_data.head(remaining // 2)  # Half attack
                benign_sample = benign_data.head(remaining // 2)  # Half benign
                
                if len(attack_sample) > 0:
                    collected_data.append(attack_sample)
                    samples_collected += len(attack_sample)
                
                if len(benign_sample) > 0:
                    collected_data.append(benign_sample)
                    samples_collected += len(benign_sample)
                
            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_num}: {e}")
                continue
        
        if collected_data:
            result = pd.concat(collected_data, ignore_index=True)
            logger.info(f"Loaded {len(result)} samples for {attack_type}")
            return result
        else:
            logger.error(f"No data loaded for {attack_type}")
            return pd.DataFrame()
    
    def create_academic_point_clouds(self, data: pd.DataFrame, method: str = "flow_sequence") -> List[np.ndarray]:
        """
        Create point clouds using academic methodologies from TDA review
        
        Methods:
        - flow_sequence: Time-ordered flow sequences (Bruillard et al.)
        - network_embedding: Network structure embedding (Collins et al.)
        - behavioral_features: Behavioral pattern embedding (Aksoy et al.)
        """
        
        numeric_cols = [col for col in data.columns 
                       if col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack']]
        
        # Convert to numeric
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.fillna(0)
        
        if method == "flow_sequence":
            return self._create_flow_sequence_clouds(data, numeric_cols)
        elif method == "network_embedding":
            return self._create_network_embedding_clouds(data, numeric_cols)
        elif method == "behavioral_features":
            return self._create_behavioral_clouds(data, numeric_cols)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_flow_sequence_clouds(self, data: pd.DataFrame, numeric_cols: List[str]) -> List[np.ndarray]:
        """Create point clouds from flow sequences (Bruillard et al. 2016)"""
        
        logger.info("Creating flow sequence point clouds...")
        
        # Sort by time to create sequences
        data_sorted = data.sort_values('FLOW_START_MILLISECONDS')
        
        point_clouds = []
        window_size = 50  # Flows per point cloud
        
        for i in range(0, len(data_sorted) - window_size, window_size // 2):
            window = data_sorted.iloc[i:i+window_size]
            
            if len(window) < 10:  # Need minimum flows
                continue
                
            # Create point cloud from flow features
            features = window[numeric_cols].values
            
            # CRITICAL FIX: Ensure we have more points than dimensions
            if features.shape[0] < features.shape[1]:
                # Use PCA to reduce dimensions to fit point cloud requirements
                from sklearn.decomposition import PCA
                n_components = min(3, features.shape[0] - 1)  # Max 3D, but respect point count
                if n_components >= 2:  # Need at least 2D for meaningful topology
                    pca = PCA(n_components=n_components)
                    features = pca.fit_transform(features)
                else:
                    continue  # Skip if we can't create valid point cloud
            elif features.shape[1] > 3:
                # Reduce to 3D even when we have enough points
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                features = pca.fit_transform(features)
            
            # Final check: must have more points than dimensions
            if features.shape[0] > features.shape[1] and features.shape[1] >= 2:
                point_clouds.append(features)
        
        logger.info(f"Created {len(point_clouds)} flow sequence point clouds")
        return point_clouds
    
    def _create_network_embedding_clouds(self, data: pd.DataFrame, numeric_cols: List[str]) -> List[np.ndarray]:
        """Create point clouds from network structure (Collins et al. 2020)"""
        
        logger.info("Creating network embedding point clouds...")
        
        point_clouds = []
        
        # Group by time windows
        data_sorted = data.sort_values('FLOW_START_MILLISECONDS')
        window_size_ms = 60000  # 1 minute windows
        
        min_time = data_sorted['FLOW_START_MILLISECONDS'].min()
        max_time = data_sorted['FLOW_START_MILLISECONDS'].max()
        
        current_time = min_time
        while current_time + window_size_ms <= max_time:
            window = data_sorted[
                (data_sorted['FLOW_START_MILLISECONDS'] >= current_time) &
                (data_sorted['FLOW_START_MILLISECONDS'] < current_time + window_size_ms)
            ]
            
            if len(window) > 5:
                # Create network graph
                G = nx.Graph()
                
                for _, flow in window.iterrows():
                    src = flow['IPV4_SRC_ADDR']
                    dst = flow['IPV4_DST_ADDR']
                    
                    if not G.has_edge(src, dst):
                        G.add_edge(src, dst, weight=1, flows=[])
                    G[src][dst]['weight'] += 1
                    G[src][dst]['flows'].append(flow[numeric_cols].values)
                
                # Create point cloud from network structure
                if len(G.nodes()) > 3:
                    # Use network properties as point cloud
                    node_features = []
                    for node in G.nodes():
                        degree = len(list(G.neighbors(node)))
                        clustering = nx.clustering(G, node)
                        centrality = nx.degree_centrality(G)[node]
                        node_features.append([degree, clustering, centrality])
                    
                    if len(node_features) > 3:
                        point_clouds.append(np.array(node_features))
            
            current_time += window_size_ms // 2
        
        logger.info(f"Created {len(point_clouds)} network embedding point clouds")
        return point_clouds
    
    def _create_behavioral_clouds(self, data: pd.DataFrame, numeric_cols: List[str]) -> List[np.ndarray]:
        """Create point clouds from behavioral patterns (Aksoy et al. 2019)"""
        
        logger.info("Creating behavioral pattern point clouds...")
        
        # Group by source IP to capture behaviors
        point_clouds = []
        
        for src_ip, src_data in data.groupby('IPV4_SRC_ADDR'):
            if len(src_data) < 5:
                continue
            
            # Create behavioral feature vectors
            behavioral_features = []
            
            # Temporal patterns
            if 'FLOW_START_MILLISECONDS' in src_data.columns:
                times = src_data['FLOW_START_MILLISECONDS'].values
                time_diffs = np.diff(times) if len(times) > 1 else np.array([0])
                
                behavioral_features.extend([
                    len(src_data),  # Number of flows
                    np.mean(time_diffs) if time_diffs else 0,  # Average time between flows
                    np.std(time_diffs) if len(time_diffs) > 1 else 0,  # Time variance
                ])
            
            # Port patterns
            if 'L4_DST_PORT' in src_data.columns:
                unique_ports = src_data['L4_DST_PORT'].nunique()
                port_entropy = -np.sum(
                    (src_data['L4_DST_PORT'].value_counts() / len(src_data)) * 
                    np.log2(src_data['L4_DST_PORT'].value_counts() / len(src_data) + 1e-10)
                )
                
                behavioral_features.extend([
                    unique_ports,  # Port diversity
                    port_entropy,  # Port scanning pattern
                ])
            
            # Traffic patterns
            if 'IN_BYTES' in src_data.columns:
                behavioral_features.extend([
                    src_data['IN_BYTES'].mean(),
                    src_data['IN_BYTES'].std(),
                    src_data['OUT_BYTES'].mean() if 'OUT_BYTES' in src_data.columns else 0,
                ])
            
            # Protocol patterns
            if 'PROTOCOL' in src_data.columns:
                protocol_diversity = src_data['PROTOCOL'].nunique()
                behavioral_features.append(protocol_diversity)
            
            # Create point cloud from behavioral vectors
            if len(behavioral_features) >= 3:
                # Use sliding window over behavioral features
                for i in range(len(src_data) - 3):
                    point = behavioral_features[:8]  # Take first 8 features
                    if len(point) >= 3:
                        point_clouds.append(np.array([point]))
        
        # Combine into larger point clouds
        if len(point_clouds) > 10:
            combined_clouds = []
            for i in range(0, len(point_clouds), 20):
                batch = point_clouds[i:i+20]
                if len(batch) > 3:
                    combined = np.vstack([cloud[0] if len(cloud.shape) > 1 else cloud for cloud in batch])
                    combined_clouds.append(combined)
            point_clouds = combined_clouds
        
        logger.info(f"Created {len(point_clouds)} behavioral pattern point clouds")
        return point_clouds
    
    def extract_tda_features(self, point_clouds: List[np.ndarray]) -> np.ndarray:
        """Extract TDA features using persistent homology"""
        
        logger.info("Extracting TDA features from point clouds...")
        
        ph_analyzer = PersistentHomologyAnalyzer(maxdim=2, backend='ripser')
        all_features = []
        
        for i, cloud in enumerate(point_clouds):
            if i % 20 == 0:
                logger.info(f"Processing cloud {i+1}/{len(point_clouds)}")
            
            features = []
            
            try:
                if len(cloud) >= 3:
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
                        features.extend([0] * 15)  # 3 dimensions × 5 features
                else:
                    features.extend([0] * 15)
                    
            except Exception as e:
                logger.debug(f"TDA failed for cloud {i}: {e}")
                features.extend([0] * 15)
            
            all_features.append(features)
        
        return np.array(all_features)
    
    def validate_attack_detection(self, attack_type: str, method: str = "flow_sequence"):
        """Validate attack detection using academic TDA methods"""
        
        logger.info(f"Validating {attack_type} detection using {method} method...")
        
        # Load data efficiently
        data = self.load_attack_data_efficiently(attack_type, n_samples=1000)
        
        if len(data) == 0:
            logger.error(f"No data loaded for {attack_type}")
            return None
        
        # Create point clouds using academic method
        point_clouds = self.create_academic_point_clouds(data, method)
        
        if len(point_clouds) == 0:
            logger.error(f"No point clouds created for {attack_type}")
            return None
        
        # Extract TDA features
        tda_features = self.extract_tda_features(point_clouds)
        
        if tda_features.size == 0:
            logger.error(f"No TDA features extracted for {attack_type}")
            return None
        
        # Create labels (attack vs benign)
        labels = []
        for _, row in data.iterrows():
            if row['Attack'] == attack_type:
                labels.append(1)  # Attack
            else:
                labels.append(0)  # Benign
        
        # Match labels to point clouds (approximate)
        n_clouds = len(point_clouds)
        n_labels = len(labels)
        
        if n_clouds <= n_labels:
            cloud_labels = labels[:n_clouds]
        else:
            # Replicate labels to match clouds
            cloud_labels = (labels * (n_clouds // n_labels + 1))[:n_clouds]
        
        cloud_labels = np.array(cloud_labels)
        
        logger.info(f"TDA features: {tda_features.shape}")
        logger.info(f"Attack ratio: {cloud_labels.mean():.3f}")
        
        # Check if we have both classes
        if len(np.unique(cloud_labels)) < 2:
            logger.warning("Only one class present - cannot evaluate")
            return None
        
        # Train and evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            tda_features, cloud_labels, test_size=0.3, random_state=42, stratify=cloud_labels
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
        print(f"ACADEMIC TDA VALIDATION: {attack_type}")
        print(f"{'='*80}")
        print(f"Method: {method}")
        print(f"Point clouds: {len(point_clouds)}")
        print(f"TDA features: {tda_features.shape[1]}")
        print(f"Attack ratio: {cloud_labels.mean():.1%}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"{'='*80}")
        
        return {
            'attack_type': attack_type,
            'method': method,
            'accuracy': accuracy,
            'point_clouds': len(point_clouds),
            'features': tda_features.shape[1],
            'attack_ratio': cloud_labels.mean()
        }

def main():
    """Test academic TDA methods on multiple attack types"""
    print("ACADEMIC TDA VALIDATION WITH EFFICIENT DATA LOADING")
    print("Using complete dataset analysis and targeted loading")
    print("="*80)
    
    framework = EfficientTDAFramework()
    
    # Test multiple attack types and methods
    attack_types = ['SSH-Bruteforce', 'Bot', 'FTP-BruteForce', 'Infilteration']
    methods = ['flow_sequence', 'network_embedding', 'behavioral_features']
    
    results = []
    
    for attack_type in attack_types:
        for method in methods:
            print(f"\n🔍 Testing {attack_type} with {method} method...")
            result = framework.validate_attack_detection(attack_type, method)
            if result:
                results.append(result)
            else:
                print(f"❌ Failed: {attack_type} + {method}")
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY OF ACADEMIC TDA RESULTS")
        print(f"{'='*80}")
        for result in results:
            print(f"{result['attack_type']:15s} + {result['method']:20s}: "
                  f"{result['accuracy']:.1%} accuracy ({result['point_clouds']} clouds)")
        print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    main()
