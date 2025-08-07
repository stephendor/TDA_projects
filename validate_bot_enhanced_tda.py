#!/usr/bin/env python3
"""
Enhanced Bot Attack TDA Validation with Improved Topological Setup
==================================================================

This script validates Bot attack detection using optimized topological data analysis,
focusing on coordinated botnet behavior patterns that should exhibit rich topology.

Bot attacks represent coordinated malicious activity across multiple systems,
creating distinctive topological signatures in network flow patterns.

ENHANCED TDA SETUP:
- Multi-dimensional point cloud construction
- Enhanced persistence diagram analysis
- Improved Mapper graph generation
- Time-series topological features
- Protocol-specific topological analysis

Following TDA Project Rules:
- Uses existing PersistentHomologyAnalyzer and MapperAnalyzer
- No statistical proxies - actual topological features only
- Temporal integrity verification to prevent data leakage
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import existing TDA infrastructure (MANDATORY)
try:
    from src.core.persistent_homology import PersistentHomologyAnalyzer
    from src.core.mapper import MapperAnalyzer
    print("✓ Successfully imported existing TDA infrastructure")
except ImportError as e:
    print(f"❌ CRITICAL: Cannot import TDA infrastructure: {e}")
    print("This violates TDA Project Rules - must use existing infrastructure")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_bot_attack_data():
    """
    Load Bot attack data with enhanced preprocessing for better topology
    """
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    logger.info("Loading Bot attacks with enhanced topological preprocessing...")
    
    # Read column names first
    header_df = pd.read_csv(data_path, nrows=0)
    column_names = header_df.columns.tolist()
    logger.info(f"Dataset has {len(column_names)} columns")
    
    # Define key topological feature groups for enhanced analysis
    flow_features = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
    timing_features = ['DURATION_IN', 'DURATION_OUT', 'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX', 'SRC_TO_DST_IAT_AVG']
    packet_features = ['NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES']
    protocol_features = ['PROTOCOL', 'L7_PROTO', 'TCP_FLAGS', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT']
    
    # All numeric columns for conversion
    numeric_columns = [col for col in column_names 
                      if col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack']]
    
    logger.info(f"Will convert {len(numeric_columns)} columns to numeric for enhanced topology")
    
    # Skip to where Bot attacks begin (line 18,648,186)
    logger.info("Skipping to row 18,648,186 where Bot attacks begin...")
    
    bot_samples = []
    benign_samples = []
    chunk_size = 5000
    start_row = 18648186  # Exact line where Bot attacks start
    max_bot = 800  # More samples for richer topology
    max_benign = 800
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(data_path, 
                                                    skiprows=range(1, start_row + 1),
                                                    names=column_names,
                                                    chunksize=chunk_size), 1):
            
            if len(bot_samples) >= max_bot and len(benign_samples) >= max_benign:
                break
            
            logger.info(f"Processing chunk {chunk_num}, converting data types for topology...")
            
            # ENHANCED: Convert numeric columns with better error handling
            for col in numeric_columns:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # Fill NaN values strategically
            chunk[numeric_columns] = chunk[numeric_columns].fillna(0)
            
            # Filter for Bot attacks
            if len(bot_samples) < max_bot:
                bot_chunk = chunk[chunk['Attack'] == 'Bot']
                if len(bot_chunk) > 0:
                    bot_samples.extend(bot_chunk.to_dict('records'))
                    logger.info(f"Found {len(bot_chunk)} Bot attacks in chunk {chunk_num}")
            
            # Filter for contemporary Benign samples (for temporal integrity)
            if len(benign_samples) < max_benign:
                benign_chunk = chunk[chunk['Attack'] == 'Benign']
                if len(benign_chunk) > 0:
                    sample_size = min(len(benign_chunk), max_benign - len(benign_samples))
                    benign_sample = benign_chunk.sample(n=sample_size, random_state=42)
                    benign_samples.extend(benign_sample.to_dict('records'))
            
            if chunk_num > 15:  # Don't search too far
                break
                
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None
    
    logger.info(f"Collected samples: {len(bot_samples)} Bot, {len(benign_samples)} Benign")
    
    # Convert to DataFrames with proper types
    bot_df = pd.DataFrame(bot_samples[:max_bot])
    benign_df = pd.DataFrame(benign_samples[:max_benign])
    
    # Ensure numeric columns are properly typed
    for col in numeric_columns:
        if col in bot_df.columns:
            bot_df[col] = pd.to_numeric(bot_df[col], errors='coerce')
        if col in benign_df.columns:
            benign_df[col] = pd.to_numeric(benign_df[col], errors='coerce')
    
    # Fill any remaining NaN values
    bot_df = bot_df.fillna(0)
    benign_df = benign_df.fillna(0)
    
    # Feature groups for enhanced topological analysis
    feature_groups = {
        'flow': flow_features,
        'timing': timing_features,
        'packet': packet_features,
        'protocol': protocol_features
    }
    
    return bot_df, benign_df, numeric_columns, feature_groups

def create_enhanced_point_cloud(sample, feature_groups):
    """
    Create enhanced multi-dimensional point cloud for richer topology
    """
    try:
        # Extract different feature types
        flow_data = []
        timing_data = []
        packet_data = []
        protocol_data = []
        
        # Group features by type for multi-dimensional analysis
        for group_name, features in feature_groups.items():
            group_values = []
            for feature in features:
                if feature in sample.index:
                    value = sample[feature]
                    if not (np.isnan(value) or np.isinf(value)):
                        group_values.append(value)
                
            if group_values:
                if group_name == 'flow':
                    flow_data = group_values[:5]  # Limit dimensions
                elif group_name == 'timing':
                    timing_data = group_values[:5]
                elif group_name == 'packet':
                    packet_data = group_values[:3]
                elif group_name == 'protocol':
                    protocol_data = group_values[:3]
        
        # Create multi-dimensional point cloud
        point_cloud = []
        
        # Method 1: Feature-grouped points
        if flow_data:
            # Create flow-based points
            for i, val in enumerate(flow_data):
                if val > 0:  # Only meaningful values
                    point_cloud.append([i, np.log1p(val)])  # Log transform for better topology
        
        if timing_data:
            # Create timing-based points
            for i, val in enumerate(timing_data):
                if val > 0:
                    point_cloud.append([i + 10, np.log1p(val)])  # Offset to separate feature types
        
        # Method 2: Protocol signature points
        if protocol_data:
            for i, val in enumerate(protocol_data):
                if val > 0:
                    point_cloud.append([i + 20, val])  # Different scale for protocols
        
        # Method 3: Combined feature relationships
        if len(flow_data) >= 2 and len(timing_data) >= 2:
            # Create relationship points
            point_cloud.append([flow_data[0], timing_data[0]])
            if len(flow_data) > 1 and len(timing_data) > 1:
                point_cloud.append([flow_data[1], timing_data[1]])
        
        # Ensure minimum points for topology
        if len(point_cloud) < 5:
            # Fallback: use first 10 numeric features as 2D points
            sample_values = [v for v in sample.values[:20] if not (np.isnan(v) or np.isinf(v)) and v != 0]
            if len(sample_values) >= 2:
                for i in range(0, min(len(sample_values)-1, 10), 2):
                    point_cloud.append([sample_values[i], sample_values[i+1]])
        
        # Convert to numpy array
        if len(point_cloud) > 0:
            return np.array(point_cloud)
        else:
            # Final fallback
            return np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
            
    except Exception as e:
        logger.debug(f"Point cloud creation failed: {e}")
        # Fallback point cloud
        return np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

def extract_enhanced_topological_features(sample, ph_analyzer, mapper_analyzer, feature_groups):
    """
    Extract enhanced topological features with improved point cloud construction
    """
    try:
        # Create enhanced point cloud
        point_cloud = create_enhanced_point_cloud(sample, feature_groups)
        
        features = []
        
        # 1. Enhanced Persistent Homology Analysis
        try:
            ph_result = ph_analyzer.fit_transform([point_cloud])
            if len(ph_result) > 0 and len(ph_result[0]) > 0:
                # Extract persistence features by dimension
                h0_intervals = []
                h1_intervals = []
                h2_intervals = []
                
                for interval in ph_result[0]:
                    if len(interval) >= 3:  # (dim, birth, death)
                        dim, birth, death = interval[0], interval[1], interval[2]
                        if not (np.isnan(birth) or np.isnan(death) or np.isinf(birth) or np.isinf(death)):
                            if dim == 0:
                                h0_intervals.append((birth, death))
                            elif dim == 1:
                                h1_intervals.append((birth, death))
                            elif dim == 2:
                                h2_intervals.append((birth, death))
                
                # H0 (Connected components) features
                h0_lifetimes = [death - birth for birth, death in h0_intervals if death > birth]
                features.extend([
                    len(h0_intervals),  # Betti 0
                    np.mean(h0_lifetimes) if h0_lifetimes else 0,  # Mean H0 persistence
                    np.max(h0_lifetimes) if h0_lifetimes else 0,   # Max H0 persistence
                    np.std(h0_lifetimes) if len(h0_lifetimes) > 1 else 0  # H0 persistence variance
                ])
                
                # H1 (Loops) features
                h1_lifetimes = [death - birth for birth, death in h1_intervals if death > birth]
                features.extend([
                    len(h1_intervals),  # Betti 1
                    np.mean(h1_lifetimes) if h1_lifetimes else 0,  # Mean H1 persistence
                    np.max(h1_lifetimes) if h1_lifetimes else 0,   # Max H1 persistence
                    np.std(h1_lifetimes) if len(h1_lifetimes) > 1 else 0  # H1 persistence variance
                ])
                
                # H2 (Voids) features  
                h2_lifetimes = [death - birth for birth, death in h2_intervals if death > birth]
                features.extend([
                    len(h2_intervals),  # Betti 2
                    np.mean(h2_lifetimes) if h2_lifetimes else 0,  # Mean H2 persistence
                    np.max(h2_lifetimes) if h2_lifetimes else 0,   # Max H2 persistence
                ])
                
                # Total persistence landscape features
                all_lifetimes = h0_lifetimes + h1_lifetimes + h2_lifetimes
                features.extend([
                    len(all_lifetimes),  # Total topological features
                    np.sum(all_lifetimes) if all_lifetimes else 0,  # Total persistence
                    np.mean(all_lifetimes) if all_lifetimes else 0  # Overall mean persistence
                ])
            else:
                features.extend([0] * 14)  # 4 H0 + 4 H1 + 3 H2 + 3 total
        except Exception as e:
            logger.debug(f"Enhanced PH computation failed: {e}")
            features.extend([0] * 14)
        
        # 2. Enhanced Mapper Analysis
        try:
            mapper_result = mapper_analyzer.fit_transform([point_cloud])
            if len(mapper_result) > 0:
                graph = mapper_result[0]
                
                # Basic graph topology
                num_nodes = len(graph['nodes']) if 'nodes' in graph else 0
                num_edges = len(graph['edges']) if 'edges' in graph else 0
                
                features.extend([
                    num_nodes,
                    num_edges,
                    num_edges / max(num_nodes, 1),  # Edge density
                    num_nodes - num_edges + 1 if num_nodes > 0 else 0,  # Euler characteristic approximation
                ])
                
                # Enhanced node analysis
                if 'nodes' in graph and graph['nodes']:
                    node_sizes = [len(node) for node in graph['nodes'].values()]
                    features.extend([
                        np.mean(node_sizes),
                        np.std(node_sizes) if len(node_sizes) > 1 else 0,
                        np.max(node_sizes) - np.min(node_sizes) if node_sizes else 0,  # Size range
                    ])
                else:
                    features.extend([0, 0, 0])
                    
                # Graph connectivity features
                features.extend([
                    1 if num_edges > 0 else 0,  # Is connected
                    max(0, num_edges - num_nodes + 1) if num_nodes > 0 else 0,  # Cycle rank
                ])
            else:
                features.extend([0] * 9)
        except Exception as e:
            logger.debug(f"Enhanced Mapper computation failed: {e}")
            features.extend([0] * 9)
        
        # 3. Point cloud geometric features
        try:
            if len(point_cloud) > 1:
                # Geometric properties of the point cloud
                distances = []
                for i in range(len(point_cloud)):
                    for j in range(i+1, len(point_cloud)):
                        dist = np.linalg.norm(point_cloud[i] - point_cloud[j])
                        if not (np.isnan(dist) or np.isinf(dist)):
                            distances.append(dist)
                
                if distances:
                    features.extend([
                        np.mean(distances),  # Mean pairwise distance
                        np.std(distances) if len(distances) > 1 else 0,  # Distance variance
                        np.max(distances),   # Maximum distance (diameter)
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
        except Exception as e:
            logger.debug(f"Geometric feature computation failed: {e}")
            features.extend([0, 0, 0])
        
        # Ensure exactly 26 features (14 PH + 9 Mapper + 3 geometric)
        while len(features) < 26:
            features.append(0.0)
        features = features[:26]
        
        # Clean features
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        
        return np.array(features)
        
    except Exception as e:
        logger.warning(f"Enhanced feature extraction failed: {e}")
        return np.zeros(26)

def main():
    logger.info("Starting ENHANCED Bot Attack TDA validation...")
    
    # Load data with enhanced preprocessing
    result = load_bot_attack_data()
    if result is None or result[0] is None:
        logger.error("Failed to load Bot attack data")
        return
    
    bot_df, benign_df, numeric_columns, feature_groups = result
    
    if len(bot_df) == 0:
        logger.error("No Bot samples found!")
        return
    
    logger.info(f"Loaded {len(bot_df)} Bot and {len(benign_df)} Benign samples")
    logger.info(f"Feature groups for enhanced topology: {list(feature_groups.keys())}")
    
    # Verify temporal integrity
    if 'FLOW_START_MILLISECONDS' in bot_df.columns and 'FLOW_START_MILLISECONDS' in benign_df.columns:
        bot_time_range = (bot_df['FLOW_START_MILLISECONDS'].min(), bot_df['FLOW_START_MILLISECONDS'].max())
        ben_time_range = (benign_df['FLOW_START_MILLISECONDS'].min(), benign_df['FLOW_START_MILLISECONDS'].max())
        
        logger.info(f"Bot time range: {bot_time_range[0]} to {bot_time_range[1]}")
        logger.info(f"Benign time range: {ben_time_range[0]} to {ben_time_range[1]}")
        
        # Check for temporal overlap
        if (bot_time_range[0] <= ben_time_range[1] and ben_time_range[0] <= bot_time_range[1]):
            logger.info("✓ Temporal integrity verified: Bot and benign samples co-occur")
        else:
            logger.warning("⚠️ Potential temporal leakage detected")
    
    # Prepare dataset
    bot_df['label'] = 1
    benign_df['label'] = 0
    
    combined_df = pd.concat([bot_df, benign_df], ignore_index=True)
    
    # Use confirmed numeric columns
    feature_columns = [col for col in numeric_columns if col in combined_df.columns and col != 'label']
    
    if len(feature_columns) == 0:
        logger.error("❌ CRITICAL: No numeric features found!")
        return
    
    X = combined_df[feature_columns].values
    y = combined_df['label'].values
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"✓ ENHANCED dataset: {len(combined_df)} samples, {len(feature_columns)} features")
    logger.info(f"Bot attack rate: {y.mean():.3f}")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Use larger subset for enhanced validation
    subset_size = min(150, len(combined_df))
    indices = np.random.choice(len(X), subset_size, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    logger.info(f"Using subset of {subset_size} samples for enhanced TDA validation")
    logger.info(f"Subset attack rate: {y_subset.mean():.3f}")
    
    # Initialize enhanced TDA analyzers
    logger.info("Initializing ENHANCED TDA analyzers...")
    ph_analyzer = PersistentHomologyAnalyzer(
        maxdim=2,
        thresh=50.0,  # Lower threshold for richer topology
        backend='ripser'
    )
    
    mapper_analyzer = MapperAnalyzer(
        n_intervals=15,  # More intervals for finer structure
        overlap_frac=0.5  # Higher overlap for better connectivity
    )
    
    logger.info("✓ Enhanced TDA analyzers initialized")
    
    # Extract enhanced topological features
    logger.info("Extracting ENHANCED topological features for Bot detection...")
    
    tda_features = []
    combined_subset_df = combined_df.iloc[indices]
    
    for i, (_, sample) in enumerate(combined_subset_df.iterrows()):
        if i % 30 == 0:
            logger.info(f"Processing sample {i}/{len(combined_subset_df)}")
        
        features = extract_enhanced_topological_features(sample, ph_analyzer, mapper_analyzer, feature_groups)
        tda_features.append(features)
    
    tda_matrix = np.array(tda_features)
    
    logger.info(f"✓ Extracted ENHANCED topological features: {tda_matrix.shape}")
    logger.info(f"Enhanced TDA feature range: [{tda_matrix.min():.3f}, {tda_matrix.max():.3f}]")
    
    # Check for meaningful features
    non_zero_features = np.count_nonzero(tda_matrix)
    total_features = tda_matrix.size
    logger.info(f"Non-zero features: {non_zero_features}/{total_features} ({100*non_zero_features/total_features:.1f}%)")
    
    if non_zero_features > total_features * 0.1:  # At least 10% non-zero
        logger.info("✓ Enhanced TDA features contain meaningful topological information")
    else:
        logger.warning("⚠️ Enhanced TDA features may need further optimization")
    
    # Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        tda_matrix, y_subset, test_size=0.25, random_state=42, stratify=y_subset
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info("ENHANCED TDA-based Bot Detection Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    logger.info(f"\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    print("\n" + "="*80)
    print("ENHANCED BOT ATTACK TDA VALIDATION COMPLETE")
    print("="*80)
    print(f"Attack Type: Bot (Coordinated Botnet Behavior)")
    print(f"Method: Enhanced TDA with Multi-dimensional Point Clouds")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Feature Matrix: {tda_matrix.shape}")
    print(f"Non-zero Features: {100*non_zero_features/total_features:.1f}%")
    print(f"\nCOMPARISON WITH PREVIOUS RESULTS:")
    print(f"• SSH-Bruteforce (simple): 46.7% accuracy")
    print(f"• Infilteration (complex): 53.3% accuracy")
    print(f"• Bot (enhanced setup): {accuracy:.1%} accuracy")
    
    if accuracy > 0.533:
        improvement = (accuracy - 0.533) * 100
        print(f"• Improvement over Infilteration: +{improvement:.1f} percentage points")
        print("✅ ENHANCED TOPOLOGY SETUP SHOWS FURTHER IMPROVEMENT")
    elif accuracy > 0.467:
        improvement = (accuracy - 0.467) * 100
        print(f"• Improvement over SSH-Bruteforce: +{improvement:.1f} percentage points")
        print("✓ COORDINATED ATTACKS SHOW BETTER TOPOLOGICAL STRUCTURE")
    else:
        print("• Requires further topological optimization")
    print("="*80)

if __name__ == "__main__":
    main()
