#!/usr/bin/env python3
"""
Debug UNSW-NB15 label creation
"""

import pandas as pd
import numpy as np

def debug_unsw_labels():
    """Debug UNSW-NB15 label issues"""
    print("🔍 Debugging UNSW-NB15 labels...")
    
    # Load data
    train_path = "data/apt_datasets/UNSW-NB15/UNSW_NB15_training-set.parquet"
    df = pd.read_parquet(train_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check label columns
    if 'label' in df.columns:
        print(f"\n'label' column:")
        print(f"  Type: {df['label'].dtype}")
        print(f"  Unique values: {df['label'].unique()}")
        print(f"  Value counts:\n{df['label'].value_counts()}")
    
    if 'attack_cat' in df.columns:
        print(f"\n'attack_cat' column:")
        print(f"  Type: {df['attack_cat'].dtype}")
        print(f"  Unique values: {df['attack_cat'].unique()[:10]}")
        print(f"  Value counts:\n{df['attack_cat'].value_counts().head(10)}")
    
    # Test label creation
    print(f"\nTesting label creation methods:")
    
    if 'label' in df.columns:
        y1 = (df['label'] != 0).astype(int)
        print(f"Method 1 (label != 0): Attack rate = {y1.mean()*100:.1f}%")
        
        y2 = (df['label'] == 1).astype(int)
        print(f"Method 2 (label == 1): Attack rate = {y2.mean()*100:.1f}%")
        
        # Sample and check
        sample_df = df.sample(n=1000, random_state=42)
        y_sample = (sample_df['label'] != 0).astype(int)
        print(f"Sample (1000): Attack rate = {y_sample.mean()*100:.1f}%")

if __name__ == "__main__":
    debug_unsw_labels()