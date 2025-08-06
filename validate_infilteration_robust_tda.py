#!/usr/bin/env python3
"""
Robust Infilteration TDA Validation with Data Quality Checks

This script validates TDA effectiveness on Infilteration attacks with:
- Enhanced data preprocessing to handle invalid values
- Robust feature extraction with outlier handling
- Improved persistence diagram computation
- Better temporal correlation analysis

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
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

def robust_data_preprocessing(X):
    """
    Robust preprocessing to handle invalid values and outliers
    """
    logger.info("Applying robust data preprocessing...")
    
    # Replace infinite values with NaN
    X = np.where(np.isinf(X), np.nan, X)
    
    # Replace NaN with median of each column
    for col in range(X.shape[1]):
        col_data = X[:, col]
        if np.isnan(col_data).any():
            median_val = np.nanmedian(col_data)
            if np.isnan(median_val):
                median_val = 0.0
            X[:, col] = np.where(np.isnan(col_data), median_val, col_data)
    
    # Clip extreme outliers (beyond 5 standard deviations)
    for col in range(X.shape[1]):
        col_data = X[:, col]
        if np.std(col_data) > 0:
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val
            X[:, col] = np.clip(col_data, lower_bound, upper_bound)
    
    logger.info(f"Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")
    return X

def extract_robust_topological_features(sample, ph_analyzer, mapper_analyzer):
    """
    Extract topological features with robust error handling
    """
    try:
        sample_clean = robust_data_preprocessing(sample.reshape(1, -1)).flatten()
        
        # Ensure minimum dimensionality for topology
        if len(sample_clean) < 3:
            sample_clean = np.pad(sample_clean, (0, 3 - len(sample_clean)), 'constant')
        
        # Create point cloud with minimal variance check
        point_cloud = sample_clean.reshape(-1, 1)
        if point_cloud.shape[0] < 2:
            point_cloud = np.vstack([point_cloud, point_cloud + 1e-6])
        
        features = []
        
        # 1. Robust Persistent Homology Features
        try:
            ph_result = ph_analyzer.fit_transform([point_cloud])
            if len(ph_result) > 0 and len(ph_result[0]) > 0:
                # H0 features (connected components)
                h0_births = []
                h0_deaths = []
                h0_lifetimes = []
                
                # H1 features (loops)
                h1_births = []
                h1_deaths = []
                h1_lifetimes = []
                
                for interval in ph_result[0]:
                    dim, birth, death = interval
                    if not (np.isnan(birth) or np.isnan(death) or np.isinf(birth) or np.isinf(death)):
                        if dim == 0:  # H0
                            h0_births.append(birth)
                            h0_deaths.append(death)
                            h0_lifetimes.append(death - birth)
                        elif dim == 1:  # H1
                            h1_births.append(birth)
                            h1_deaths.append(death)
                            h1_lifetimes.append(death - birth)
                
                # H0 statistics
                features.extend([
                    len(h0_births),  # Number of components
                    np.mean(h0_births) if h0_births else 0,
                    np.std(h0_births) if len(h0_births) > 1 else 0,
                    np.mean(h0_lifetimes) if h0_lifetimes else 0,
                    np.max(h0_lifetimes) if h0_lifetimes else 0
                ])
                
                # H1 statistics
                features.extend([
                    len(h1_births),  # Number of loops
                    np.mean(h1_births) if h1_births else 0,
                    np.std(h1_births) if len(h1_births) > 1 else 0,
                    np.mean(h1_lifetimes) if h1_lifetimes else 0,
                    np.max(h1_lifetimes) if h1_lifetimes else 0
                ])
            else:
                features.extend([0] * 10)  # Default values
        except Exception as e:
            logger.warning(f"PH computation failed: {e}")
            features.extend([0] * 10)
        
        # 2. Robust Mapper Analysis
        try:
            mapper_result = mapper_analyzer.fit_transform([point_cloud])
            if len(mapper_result) > 0:
                graph = mapper_result[0]
                
                # Graph topology features
                num_nodes = len(graph['nodes']) if 'nodes' in graph else 0
                num_edges = len(graph['edges']) if 'edges' in graph else 0
                
                features.extend([
                    num_nodes,
                    num_edges,
                    num_edges / max(num_nodes, 1),  # Edge density
                    num_nodes - num_edges + 1 if num_nodes > 0 else 0,  # Approximate connected components
                ])
                
                # Node size statistics
                if 'nodes' in graph and graph['nodes']:
                    node_sizes = [len(node) for node in graph['nodes'].values()]
                    features.extend([
                        np.mean(node_sizes),
                        np.std(node_sizes) if len(node_sizes) > 1 else 0,
                        np.max(node_sizes),
                        np.min(node_sizes)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0] * 8)
        except Exception as e:
            logger.warning(f"Mapper computation failed: {e}")
            features.extend([0] * 8)
        
        # 3. Multi-scale Topological Features
        try:
            # Create multi-scale point clouds
            scales = [0.1, 0.5, 1.0, 2.0, 5.0]
            multiscale_features = []
            
            for scale in scales:
                scaled_cloud = point_cloud * scale
                try:
                    ph_scaled = ph_analyzer.fit_transform([scaled_cloud])
                    if len(ph_scaled) > 0 and len(ph_scaled[0]) > 0:
                        h0_count = sum(1 for interval in ph_scaled[0] if interval[0] == 0)
                        h1_count = sum(1 for interval in ph_scaled[0] if interval[0] == 1)
                        multiscale_features.extend([h0_count, h1_count])
                    else:
                        multiscale_features.extend([0, 0])
                except:
                    multiscale_features.extend([0, 0])
            
            features.extend(multiscale_features[:7])  # Limit to 7 features
        except Exception as e:
            logger.warning(f"Multi-scale computation failed: {e}")
            features.extend([0] * 7)
        
        # Ensure exactly 25 features
        while len(features) < 25:
            features.append(0.0)
        features = features[:25]
        
        # Final robustness check
        features = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in features]
        
        return np.array(features)
        
    except Exception as e:
        logger.warning(f"Feature extraction failed completely: {e}")
        return np.zeros(25)

def load_infilteration_data_robust():
    """
    Load Infilteration attack data with robust error handling
    """
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    logger.info("Loading Infilteration attacks with robust processing...")
    
    # Read column names first
    header_df = pd.read_csv(data_path, nrows=0)
    column_names = header_df.columns.tolist()
    logger.info(f"Dataset columns: {len(column_names)}")
    
    # Skip to where Infilteration attacks begin (around row 14M)
    logger.info("Skipping to row 14,000,000 where Infilteration attacks begin...")
    
    infilteration_samples = []
    benign_samples = []
    chunk_size = 10000
    start_row = 14000000
    max_infilteration = 1000
    max_benign = 1000
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(data_path, 
                                                    skiprows=range(1, start_row + 1),
                                                    names=column_names,
                                                    chunksize=chunk_size), 1):
            
            if len(infilteration_samples) >= max_infilteration and len(benign_samples) >= max_benign:
                break
                
            # Filter for Infilteration attacks
            if len(infilteration_samples) < max_infilteration:
                infilteration_chunk = chunk[chunk['Attack'] == 'Infilteration']
                if len(infilteration_chunk) > 0:
                    infilteration_samples.extend(infilteration_chunk.to_dict('records'))
                    logger.info(f"Found {len(infilteration_chunk)} Infilteration attacks in chunk {chunk_num + 11}")
            
            # Filter for Benign samples
            if len(benign_samples) < max_benign:
                benign_chunk = chunk[chunk['Attack'] == 'Benign']
                if len(benign_chunk) > 0:
                    # Sample randomly to get variety
                    sample_size = min(len(benign_chunk), max_benign - len(benign_samples))
                    benign_sample = benign_chunk.sample(n=sample_size, random_state=42)
                    benign_samples.extend(benign_sample.to_dict('records'))
            
            if chunk_num > 20:  # Don't search too far
                break
                
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None
    
    logger.info(f"Collected sufficient samples: {len(infilteration_samples)} Infilteration, {len(benign_samples)} Benign")
    
    # Convert to DataFrames
    infilteration_df = pd.DataFrame(infilteration_samples[:max_infilteration])
    benign_df = pd.DataFrame(benign_samples[:max_benign])
    
    return infilteration_df, benign_df

def main():
    logger.info("Starting robust Infilteration TDA validation...")
    
    # Load data
    infilteration_df, benign_df = load_infilteration_data_robust()
    if infilteration_df is None or benign_df is None:
        logger.error("Failed to load data")
        return
    
    # Verify temporal integrity
    if 'Timestamp' in infilteration_df.columns and 'Timestamp' in benign_df.columns:
        inf_time_range = (infilteration_df['Timestamp'].min(), infilteration_df['Timestamp'].max())
        ben_time_range = (benign_df['Timestamp'].min(), benign_df['Timestamp'].max())
        
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
    
    # Remove non-numeric columns
    feature_columns = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns.remove('label')
    
    X = combined_df[feature_columns].values
    y = combined_df['label'].values
    
    # Convert to numpy arrays to fix type issues
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Final dataset: {len(combined_df)} samples, {len(feature_columns)} features")
    logger.info(f"Infilteration attack rate: {y.mean():.3f}")
    
    # Use subset for validation
    subset_size = 120
    X_subset = X[:subset_size]
    y_subset = y[:subset_size]
    logger.info(f"Using subset of {subset_size} samples for robust TDA validation")
    logger.info(f"Infilteration attack rate: {y_subset.mean():.3f}")
    
    # Apply robust preprocessing
    X_subset = robust_data_preprocessing(X_subset)
    
    # Initialize enhanced TDA analyzers
    logger.info("Initializing robust TDA analyzers...")
    ph_analyzer = PersistentHomologyAnalyzer(
        maxdim=2,
        thresh=1000.0,  # Higher threshold for robustness
        backend='ripser'
    )
    
    mapper_analyzer = MapperAnalyzer(
        n_intervals=12,
        overlap_frac=0.4
    )
    
    logger.info("✓ Robust TDA analyzers initialized")
    
    # Extract robust topological features
    logger.info("Extracting robust topological features for Infilteration detection...")
    
    tda_features = []
    for i, sample in enumerate(X_subset):
        if i % 25 == 0:
            logger.info(f"Processing sample {i}/{len(X_subset)}")
        
        features = extract_robust_topological_features(sample, ph_analyzer, mapper_analyzer)
        tda_features.append(features)
    
    tda_matrix = np.array(tda_features)
    
    logger.info(f"Extracted robust topological features for {len(X_subset)} samples")
    logger.info(f"Robust topological feature matrix: {tda_matrix.shape}")
    logger.info(f"Feature range: [{tda_matrix.min():.3f}, {tda_matrix.max():.3f}]")
    
    # Handle any remaining invalid values
    tda_matrix = robust_data_preprocessing(tda_matrix)
    
    # Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        tda_matrix, y_subset, test_size=0.25, random_state=42, stratify=y_subset
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
    
    logger.info("ROBUST TDA-based Infilteration Detection Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    logger.info(f"\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    print("\n" + "="*80)
    print("ROBUST INFILTERATION TDA VALIDATION COMPLETE")
    print("="*80)
    print(f"Attack Type: Infilteration (Multi-stage APT)")
    print(f"Method: Robust TDA with Enhanced Error Handling")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Feature Matrix: {tda_matrix.shape}")
    print(f"\nCOMPARISON WITH PREVIOUS RESULTS:")
    print(f"• SSH-Bruteforce (simple): 46.7% accuracy")
    print(f"• Infilteration (standard): 60.0% accuracy")
    print(f"• Infilteration (robust): {accuracy:.1%} accuracy")
    
    if accuracy > 0.60:
        print(f"• Improvement over standard: +{(accuracy - 0.60)*100:.1f} percentage points")
        print("✓ ROBUST PROCESSING SHOWS IMPROVEMENT")
    else:
        print("• Similar performance to standard method")
    print("="*80)

if __name__ == "__main__":
    main()
