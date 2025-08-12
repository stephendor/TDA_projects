#!/usr/bin/env python3
"""
Debug data loading issue for Infilteration TDA validation
"""

import pandas as pd
import numpy as np

def debug_data_loading():
    """
    Debug why we're getting 0 features in the dataset
    """
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    print("=== DEBUGGING DATA LOADING ISSUE ===")
    
    # Read column names first
    header_df = pd.read_csv(data_path, nrows=0)
    column_names = header_df.columns.tolist()
    print(f"Total columns: {len(column_names)}")
    print(f"Column names: {column_names}")
    
    # Load a small sample from the Infilteration section
    print("\n=== LOADING SAMPLE DATA ===")
    start_row = 14000000
    sample_chunk = pd.read_csv(data_path, 
                              skiprows=range(1, start_row + 1),
                              names=column_names,
                              nrows=1000)
    
    print(f"Sample chunk shape: {sample_chunk.shape}")
    print(f"Sample chunk columns: {sample_chunk.columns.tolist()}")
    
    # Check attack distribution
    print(f"\nAttack distribution:")
    print(sample_chunk['Attack'].value_counts())
    
    # Check data types
    print(f"\nData types:")
    print(sample_chunk.dtypes)
    
    # Select numeric columns CORRECTLY
    print(f"\n=== NUMERIC COLUMN SELECTION ===")
    
    # Method 1: select_dtypes
    numeric_columns_1 = sample_chunk.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Method 1 (select_dtypes): {len(numeric_columns_1)} columns")
    print(f"Numeric columns: {numeric_columns_1}")
    
    # Method 2: pandas numeric conversion
    numeric_columns_2 = []
    for col in sample_chunk.columns:
        if col not in ['Label', 'Attack']:
            try:
                pd.to_numeric(sample_chunk[col], errors='raise')
                numeric_columns_2.append(col)
            except:
                print(f"Non-numeric column: {col}")
    
    print(f"Method 2 (conversion): {len(numeric_columns_2)} columns")
    
    # Filter for Infilteration samples
    print(f"\n=== INFILTERATION FILTERING ===")
    infilteration_samples = sample_chunk[sample_chunk['Attack'] == 'Infilteration']
    benign_samples = sample_chunk[sample_chunk['Attack'] == 'Benign']
    
    print(f"Infilteration samples: {len(infilteration_samples)}")
    print(f"Benign samples: {len(benign_samples)}")
    
    if len(infilteration_samples) > 0:
        print(f"\nInfilteration sample:")
        print(infilteration_samples.iloc[0])
        
        # Test feature matrix creation
        if len(numeric_columns_2) > 0:
            numeric_columns_2 = [col for col in numeric_columns_2 if col != 'label']
            
            # Combine samples
            infilteration_samples['label'] = 1
            benign_samples['label'] = 0
            combined_df = pd.concat([infilteration_samples[:10], benign_samples[:10]], ignore_index=True)
            
            X = combined_df[numeric_columns_2].values
            y = combined_df['label'].values
            
            print(f"\nFeature matrix shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Feature range: [{X.min():.3f}, {X.max():.3f}]")
            
        else:
            print("ERROR: No numeric features found!")

if __name__ == "__main__":
    debug_data_loading()
