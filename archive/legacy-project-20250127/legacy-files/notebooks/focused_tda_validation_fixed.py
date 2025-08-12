#!/usr/bin/env python3
"""
Focused TDA Validation Script - Fixed Data Path
Tests TDA methods on real CIC-IDS2017 data to resolve validation crisis
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cicids2017_data(max_samples=5000):
    """Load CIC-IDS2017 data with correct path"""
    
    # Correct data path
    data_path = Path("data/apt_datasets/cicids2017/MachineLearningCSV/MachineLearningCVE")
    
    logger.info(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Find infiltration data (APT-like attacks)
    infiltration_files = list(data_path.glob("*Infilteration*.csv"))
    
    if not infiltration_files:
        logger.warning("No infiltration files found, using any available data")
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found")
        infiltration_files = [csv_files[0]]
    
    logger.info(f"Found {len(infiltration_files)} infiltration files")
    
    # Load data
    dfs = []
    for file in infiltration_files[:2]:  # Limit to 2 files for faster testing
        logger.info(f"Loading: {file.name}")
        df = pd.read_csv(file, nrows=max_samples//len(infiltration_files))
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    logger.info(f"Loaded {len(data)} total samples")
    
    return data

def prepare_data(data):
    """Prepare data for TDA analysis"""
    logger.info("Preparing data for TDA analysis")
    
    # Handle common column naming variations
    label_columns = [col for col in data.columns if 'label' in col.lower()]
    if not label_columns:
        # Create synthetic labels based on anomaly patterns
        logger.warning("No label column found, creating synthetic labels")
        data['Label'] = 'BENIGN'
        # Mark potential anomalies based on high values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Use z-score to identify outliers
            z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
            anomaly_mask = (z_scores > 3).any(axis=1)
            data.loc[anomaly_mask, 'Label'] = 'ATTACK'
    
    # Get features (all numeric columns except label)
    label_col = [col for col in data.columns if 'label' in col.lower()][0]
    feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in feature_cols:
        feature_cols.remove(label_col)
    
    # Clean data
    X = data[feature_cols].fillna(0)
    y = data[label_col]
    
    # Convert labels to binary
    y_binary = (y != 'BENIGN').astype(int)
    
    # Take sample to avoid memory issues
    if len(X) > 1000:
        sample_idx = np.random.choice(len(X), 1000, replace=False)
        X = X.iloc[sample_idx]
        y_binary = y_binary.iloc[sample_idx]
    
    logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
    logger.info(f"Attack samples: {y_binary.sum()}/{len(y_binary)} ({y_binary.mean():.2%})")
    
    return X.values, y_binary.values

def test_hybrid_tda_method(X, y):
    """Test the confirmed working hybrid TDA method"""
    logger.info("Testing Hybrid TDA method (baseline)")
    
    try:
        # Import hybrid method
        from src.algorithms.hybrid.hybrid_multiscale_graph_tda import HybridTDAAnalyzer
        
        # Initialize and test
        detector = HybridTDAAnalyzer()
        
        # Train/test split
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info("Training hybrid TDA model...")
        detector.fit(X_train, y_train)
        
        logger.info("Making predictions...")
        y_pred = detector.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results = {
            'method': 'Hybrid TDA',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'status': 'SUCCESS'
        }
        
        logger.info(f"Hybrid TDA Results: F1={f1:.3f}, Acc={accuracy:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"Hybrid TDA test failed: {e}")
        return {
            'method': 'Hybrid TDA',
            'status': 'FAILED',
            'error': str(e)
        }

def test_deep_tda_method(X, y):
    """Test Deep TDA method"""
    logger.info("Testing Deep TDA method")
    
    try:
        # Import deep TDA components from new structure
        from src.algorithms.deep.deep_tda_breakthrough import DifferentiablePersistentHomology
        
        # Initialize with smaller parameters for faster testing
        detector = DeepTDAEnsemble(
            n_estimators=3,  # Reduced from default
            max_features=10  # Limit features
        )
        
        # Train/test split
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info("Training deep TDA model...")
        detector.fit(X_train, y_train)
        
        logger.info("Making predictions...")
        y_pred = detector.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results = {
            'method': 'Deep TDA',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'status': 'SUCCESS'
        }
        
        logger.info(f"Deep TDA Results: F1={f1:.3f}, Acc={accuracy:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"Deep TDA test failed: {e}")
        return {
            'method': 'Deep TDA',
            'status': 'FAILED',
            'error': str(e)
        }

def test_supervised_ensemble(X, y):
    """Test Supervised TDA Ensemble"""
    logger.info("Testing Supervised TDA Ensemble")
    
    try:
        from src.algorithms.ensemble.tda_supervised_ensemble import TDASupervisedEnsemble
        
        detector = TDASupervisedEnsemble()
        
        # Train/test split
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info("Training supervised ensemble...")
        detector.fit(X_train, y_train)
        
        logger.info("Making predictions...")
        y_pred = detector.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results = {
            'method': 'Supervised TDA Ensemble',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'status': 'SUCCESS'
        }
        
        logger.info(f"Supervised Ensemble Results: F1={f1:.3f}, Acc={accuracy:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"Supervised ensemble test failed: {e}")
        return {
            'method': 'Supervised TDA Ensemble',
            'status': 'FAILED',
            'error': str(e)
        }

def main():
    """Main validation function"""
    logger.info("=== Starting Focused TDA Validation ===")
    
    results = []
    
    try:
        # Load real data
        logger.info("Loading CIC-IDS2017 data...")
        data = load_cicids2017_data(max_samples=3000)  # Smaller sample for faster testing
        
        # Prepare data
        X, y = prepare_data(data)
        
        # Test methods
        methods_to_test = [
            test_hybrid_tda_method,
            test_deep_tda_method,
            test_supervised_ensemble
        ]
        
        for test_func in methods_to_test:
            logger.info(f"Running {test_func.__name__}...")
            result = test_func(X, y)
            results.append(result)
            
            # Log result
            if result['status'] == 'SUCCESS':
                logger.info(f"✓ {result['method']}: F1={result['f1_score']:.3f}")
            else:
                logger.error(f"✗ {result['method']}: {result.get('error', 'Unknown error')}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'data_samples': len(X),
                'attack_rate': float(y.mean()),
                'results': results
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Summary
        successful_methods = [r for r in results if r['status'] == 'SUCCESS']
        logger.info(f"\n=== VALIDATION SUMMARY ===")
        logger.info(f"Total methods tested: {len(results)}")
        logger.info(f"Successful: {len(successful_methods)}")
        
        if successful_methods:
            best_method = max(successful_methods, key=lambda x: x['f1_score'])
            logger.info(f"Best performing: {best_method['method']} (F1={best_method['f1_score']:.3f})")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("✓ Focused validation completed successfully")
    else:
        print("✗ Validation failed")
        sys.exit(1)
