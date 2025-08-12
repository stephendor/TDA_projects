#!/usr/bin/env python3
"""
Quick debug to find the hanging issue in validation scripts
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_data_loading():
    """Test if data loading is the issue"""
    print("Starting data loading test...")
    
    # Use the ACTUAL data path where CSV files exist
    infiltration_file = "data/apt_datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    
    print(f"Testing real infiltration data: {infiltration_file}")
    print(f"File exists: {Path(infiltration_file).exists()}")
    
    if Path(infiltration_file).exists():
        try:
            print("Loading infiltration dataset (first 1000 rows)...")
            df = pd.read_csv(infiltration_file, nrows=1000)
            print(f"✓ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"✓ Labels found: {df['Label'].unique()}")
            print(f"✓ Attack rate: {(df['Label'] != 'BENIGN').mean():.3%}")
            
            # Test feature extraction
            feature_cols = [col for col in df.columns if col != 'Label']
            X = df[feature_cols].select_dtypes(include=[np.number])
            print(f"✓ Numeric features: {X.shape[1]} dimensions")
            
            print("Real data loading test PASSED")
            return True
            
        except Exception as e:
            print(f"Real data loading FAILED: {e}")
            return False
    else:
        print("✗ Real infiltration data file not found")
        return False

def test_imports():
    """Test if imports are causing the hang"""
    print("Testing imports...")
    
    try:
        print("Importing basic packages...")
        import sklearn
        print("✓ sklearn imported")
        
        import gudhi
        print("✓ gudhi imported")
        
        import ripser
        print("✓ ripser imported")
        
        print("All imports successful!")
        return True
        
    except Exception as e:
        print(f"Import failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TDA Validation Debug ===")
    
    # Test 1: Imports
    import_success = test_imports()
    
    # Test 2: Data loading
    data_success = test_data_loading()
    
    print(f"\nResults:")
    print(f"Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"Data loading: {'PASS' if data_success else 'FAIL'}")
    
    if import_success and data_success:
        print("✓ Basic components working - issue is likely in TDA computation")
    else:
        print("✗ Found root cause of hanging issue")
