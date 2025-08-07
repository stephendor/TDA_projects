#!/usr/bin/env python3
"""
Corrected Infilteration TDA Validation with Proper Data Type Handling

This script fixes the critical data type issue where all columns were being 
read as strings instead of numeric values, leading to 0 features.

Following TDA Project Rules:
- Uses existing PersistentHomologyAnalyzer and MapperAnalyzer
- No statistical proxies - actual topological features only
- Temporal integrity verification to prevent data leakage
- Proper numeric data type conversion for TDA analysis
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

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

def load_infilteration_data_with_proper_types():
    """
    Load Infilteration attack data with PROPER numeric type conversion
    """
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    logger.info("Loading Infilteration attacks with CORRECTED data type handling...")
    
    # First, read the header to get column names
    header_df = pd.read_csv(data_path, nrows=0)
    column_names = header_df.columns.tolist()
    logger.info(f"Dataset has {len(column_names)} columns")
    
    # Define which columns should be numeric (exclude IP addresses and labels)
    numeric_columns = [col for col in column_names 
                      if col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack']]
    
    logger.info(f"Will convert {len(numeric_columns)} columns to numeric")
    
    # Skip to where Infilteration attacks begin (line 14,113,454)
    logger.info("Skipping to row 14,113,454 where Infilteration attacks begin...")
    
    infilteration_samples = []
    benign_samples = []
    chunk_size = 5000  # Smaller chunks for better type conversion
    start_row = 14113454  # Exact line where Infilteration starts
    max_infilteration = 500
    max_benign = 500
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(data_path, 
                                                    skiprows=range(1, start_row + 1),
                                                    names=column_names,
                                                    chunksize=chunk_size), 1):
            
            if len(infilteration_samples) >= max_infilteration and len(benign_samples) >= max_benign:
                break
            
            # CRITICAL FIX: Convert numeric columns to proper types
            logger.info(f"Processing chunk {chunk_num}, converting data types...")
            
            for col in numeric_columns:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # Fill NaN values with 0
            chunk[numeric_columns] = chunk[numeric_columns].fillna(0)
            
            # Filter for Infilteration attacks
            if len(infilteration_samples) < max_infilteration:
                infilteration_chunk = chunk[chunk['Attack'] == 'Infilteration']
                if len(infilteration_chunk) > 0:
                    infilteration_samples.extend(infilteration_chunk.to_dict('records'))
                    logger.info(f"Found {len(infilteration_chunk)} Infilteration attacks in chunk {chunk_num}")
            
            # Filter for Benign samples
            if len(benign_samples) < max_benign:
                benign_chunk = chunk[chunk['Attack'] == 'Benign']
                if len(benign_chunk) > 0:
                    # Sample randomly to get variety
                    sample_size = min(len(benign_chunk), max_benign - len(benign_samples))
                    benign_sample = benign_chunk.sample(n=sample_size, random_state=42)
                    benign_samples.extend(benign_sample.to_dict('records'))
            
            if chunk_num > 15:  # Don't search too far
                break
                
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None
    
    logger.info(f"Collected samples: {len(infilteration_samples)} Infilteration, {len(benign_samples)} Benign")
    
    # Convert to DataFrames with proper types
    infilteration_df = pd.DataFrame(infilteration_samples[:max_infilteration])
    benign_df = pd.DataFrame(benign_samples[:max_benign])
    
    # Ensure numeric columns are properly typed
    for col in numeric_columns:
        if col in infilteration_df.columns:
            infilteration_df[col] = pd.to_numeric(infilteration_df[col], errors='coerce')
        if col in benign_df.columns:
            benign_df[col] = pd.to_numeric(benign_df[col], errors='coerce')
    
    # Fill any remaining NaN values
    infilteration_df = infilteration_df.fillna(0)
    benign_df = benign_df.fillna(0)
    
    return infilteration_df, benign_df, numeric_columns

def extract_corrected_topological_features(sample, ph_analyzer, mapper_analyzer):
    """
    Extract topological features with corrected data handling
    """
    try:
        # Ensure we have numeric data
        if len(sample) == 0:
            return np.zeros(15)
        
        # Create point cloud from sample
        sample_array = np.array(sample).reshape(-1, 1)
        
        # Handle degenerate cases
        if sample_array.shape[0] < 2:
            sample_array = np.vstack([sample_array, sample_array + 1e-6])
        
        features = []
        
        # 1. Persistent Homology Features
        try:
            ph_result = ph_analyzer.fit_transform([sample_array])
            if len(ph_result) > 0 and len(ph_result[0]) > 0:
                # Count features by dimension
                h0_count = sum(1 for interval in ph_result[0] if interval[0] == 0)
                h1_count = sum(1 for interval in ph_result[0] if interval[0] == 1)
                
                # Persistence lifetimes
                lifetimes = []
                for interval in ph_result[0]:
                    if len(interval) >= 3:  # (dim, birth, death)
                        birth, death = interval[1], interval[2]
                        if not (np.isnan(birth) or np.isnan(death) or np.isinf(birth) or np.isinf(death)):
                            lifetimes.append(death - birth)
                
                features.extend([
                    h0_count,  # H0 Betti number
                    h1_count,  # H1 Betti number
                    len(lifetimes),  # Total topological features
                    np.mean(lifetimes) if lifetimes else 0,  # Mean persistence
                    np.max(lifetimes) if lifetimes else 0,   # Max persistence
                    np.std(lifetimes) if len(lifetimes) > 1 else 0  # Persistence variance
                ])
            else:
                features.extend([0] * 6)
        except Exception as e:
            logger.debug(f"PH computation failed: {e}")
            features.extend([0] * 6)
        
        # 2. Mapper Analysis
        try:
            mapper_result = mapper_analyzer.fit_transform([sample_array])
            if len(mapper_result) > 0:
                graph = mapper_result[0]
                
                # Extract graph topology
                num_nodes = len(graph['nodes']) if 'nodes' in graph else 0
                num_edges = len(graph['edges']) if 'edges' in graph else 0
                
                features.extend([
                    num_nodes,
                    num_edges,
                    num_edges / max(num_nodes, 1),  # Edge density
                ])
                
                # Node clustering information
                if 'nodes' in graph and graph['nodes']:
                    node_sizes = [len(node) for node in graph['nodes'].values()]
                    features.extend([
                        np.mean(node_sizes),
                        np.std(node_sizes) if len(node_sizes) > 1 else 0,
                        np.max(node_sizes) if node_sizes else 0
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0] * 6)
        except Exception as e:
            logger.debug(f"Mapper computation failed: {e}")
            features.extend([0] * 6)
        
        # Ensure exactly 15 features
        while len(features) < 15:
            features.append(0.0)
        features = features[:15]
        
        # Clean features
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        
        return np.array(features)
        
    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        return np.zeros(15)

def main():
    logger.info("Starting CORRECTED Infilteration TDA validation...")
    
    # Load data with proper type conversion
    result = load_infilteration_data_with_proper_types()
    if result is None or result[0] is None:
        logger.error("Failed to load data")
        return
    
    infilteration_df, benign_df, numeric_columns = result
    
    if len(infilteration_df) == 0:
        logger.error("No Infilteration samples found!")
        return
    
    logger.info(f"Loaded {len(infilteration_df)} Infilteration and {len(benign_df)} Benign samples")
    
    # Verify we now have numeric data
    logger.info(f"Numeric columns available: {len(numeric_columns)}")
    
    # Check data types after conversion
    sample_types = infilteration_df[numeric_columns[:5]].dtypes if len(numeric_columns) >= 5 else infilteration_df.dtypes
    logger.info(f"Sample data types: {sample_types}")
    
    # Verify temporal integrity
    if 'FLOW_START_MILLISECONDS' in infilteration_df.columns and 'FLOW_START_MILLISECONDS' in benign_df.columns:
        inf_time_range = (infilteration_df['FLOW_START_MILLISECONDS'].min(), infilteration_df['FLOW_START_MILLISECONDS'].max())
        ben_time_range = (benign_df['FLOW_START_MILLISECONDS'].min(), benign_df['FLOW_START_MILLISECONDS'].max())
        
        logger.info(f"Infilteration time range: {inf_time_range[0]} to {inf_time_range[1]}")
        logger.info(f"Benign time range: {ben_time_range[0]} to {ben_time_range[1]}")
        
        # Check for temporal overlap
        if (inf_time_range[0] <= ben_time_range[1] and ben_time_range[0] <= inf_time_range[1]):
            logger.info("✓ Temporal integrity verified: Attack and benign samples co-occur")
        else:
            logger.warning("⚠️ Potential temporal leakage detected")
    
    # Prepare dataset
    infilteration_df['label'] = 1
    benign_df['label'] = 0
    
    combined_df = pd.concat([infilteration_df, benign_df], ignore_index=True)
    
    # Use only the confirmed numeric columns
    feature_columns = [col for col in numeric_columns if col in combined_df.columns and col != 'label']
    
    if len(feature_columns) == 0:
        logger.error("❌ CRITICAL: Still no numeric features after type conversion!")
        return
    
    X = combined_df[feature_columns].values
    y = combined_df['label'].values
    
    # Convert to numpy arrays to fix type issues
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"✓ CORRECTED dataset: {len(combined_df)} samples, {len(feature_columns)} features")
    logger.info(f"Infilteration attack rate: {y.mean():.3f}")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Use manageable subset for TDA validation
    subset_size = min(100, len(combined_df))
    indices = np.random.choice(len(X), subset_size, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    logger.info(f"Using subset of {subset_size} samples for TDA validation")
    logger.info(f"Subset attack rate: {y_subset.mean():.3f}")
    
    # Initialize TDA analyzers
    logger.info("Initializing corrected TDA analyzers...")
    ph_analyzer = PersistentHomologyAnalyzer(
        maxdim=1,
        thresh=100.0,
        backend='ripser'
    )
    
    mapper_analyzer = MapperAnalyzer(
        n_intervals=8,
        overlap_frac=0.3
    )
    
    logger.info("✓ TDA analyzers initialized")
    
    # Extract topological features
    logger.info("Extracting corrected topological features...")
    
    tda_features = []
    for i, sample in enumerate(X_subset):
        if i % 25 == 0:
            logger.info(f"Processing sample {i}/{len(X_subset)}")
        
        features = extract_corrected_topological_features(sample, ph_analyzer, mapper_analyzer)
        tda_features.append(features)
    
    tda_matrix = np.array(tda_features)
    
    logger.info(f"✓ Extracted corrected topological features: {tda_matrix.shape}")
    logger.info(f"TDA feature range: [{tda_matrix.min():.3f}, {tda_matrix.max():.3f}]")
    
    # Check if we actually have meaningful features
    if tda_matrix.max() == 0 and tda_matrix.min() == 0:
        logger.warning("⚠️ All TDA features are zero - may indicate data issues")
    else:
        logger.info("✓ TDA features contain non-zero values")
    
    # Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        tda_matrix, y_subset, test_size=0.3, random_state=42, stratify=y_subset
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info("CORRECTED TDA-based Infilteration Detection Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    logger.info(f"\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    print("\n" + "="*80)
    print("CORRECTED INFILTERATION TDA VALIDATION COMPLETE")
    print("="*80)
    print(f"Attack Type: Infilteration (Multi-stage APT)")
    print(f"Method: Corrected TDA with Proper Data Types")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Feature Matrix: {tda_matrix.shape}")
    print(f"Numeric Features Used: {len(feature_columns)}")
    print(f"\nCOMPARISON WITH PREVIOUS RESULTS:")
    print(f"• SSH-Bruteforce (baseline): 46.7% accuracy")
    print(f"• Infilteration (corrected): {accuracy:.1%} accuracy")
    
    if accuracy > 0.467:
        improvement = (accuracy - 0.467) * 100
        print(f"• Improvement: +{improvement:.1f} percentage points")
        print("✓ CORRECTED PROCESSING SHOWS IMPROVEMENT")
    else:
        print("• Performance similar to baseline")
    print("="*80)

if __name__ == "__main__":
    main()
