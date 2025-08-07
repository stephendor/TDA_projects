#!/usr/bin/env python3
"""
UNSW-NB15 Dataset Analysis
==========================

Examine the UNSW-NB15 dataset structure to understand:
1. Attack types and distribution
2. Feature set and data format
3. Temporal characteristics
4. Data quality and completeness

This will help us validate our TDA approach on a different dataset
to check for overfitting or data leakage issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UNSWDatasetAnalyzer:
    """Analyze UNSW-NB15 dataset structure and characteristics"""
    
    def __init__(self):
        self.train_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/UNSW-NB15/UNSW_NB15_training-set.parquet"
        self.test_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/UNSW-NB15/UNSW_NB15_testing-set.parquet"
        
    def load_and_analyze_basic_stats(self):
        """Load datasets and analyze basic statistics"""
        
        print("="*80)
        print("UNSW-NB15 DATASET ANALYSIS")
        print("="*80)
        
        # Load training set
        print("\n📁 Loading training set...")
        try:
            train_df = pd.read_parquet(self.train_path)
            print(f"✓ Training set loaded: {train_df.shape}")
        except Exception as e:
            print(f"❌ Error loading training set: {e}")
            return None, None
        
        # Load testing set
        print("\n📁 Loading testing set...")
        try:
            test_df = pd.read_parquet(self.test_path)
            print(f"✓ Testing set loaded: {test_df.shape}")
        except Exception as e:
            print(f"❌ Error loading testing set: {e}")
            return train_df, None
        
        return train_df, test_df
    
    def analyze_columns_and_types(self, train_df, test_df):
        """Analyze column structure and data types"""
        
        print("\n" + "="*50)
        print("COLUMN ANALYSIS")
        print("="*50)
        
        print(f"\nTraining set columns ({len(train_df.columns)}):")
        for i, col in enumerate(train_df.columns):
            print(f"{i+1:2d}. {col:20s} - {train_df[col].dtype}")
        
        print(f"\nTesting set columns ({len(test_df.columns)}):")
        for i, col in enumerate(test_df.columns):
            print(f"{i+1:2d}. {col:20s} - {test_df[col].dtype}")
        
        # Check for column consistency
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        if train_cols == test_cols:
            print("\n✓ Column sets are identical between train/test")
        else:
            print("\n❌ Column mismatch between train/test:")
            print(f"  Only in train: {train_cols - test_cols}")
            print(f"  Only in test: {test_cols - train_cols}")
    
    def analyze_attack_types(self, train_df, test_df):
        """Analyze attack types and their distribution"""
        
        print("\n" + "="*50)
        print("ATTACK TYPE ANALYSIS")
        print("="*50)
        
        # Check for label columns
        label_candidates = [col for col in train_df.columns 
                          if 'label' in col.lower() or 'attack' in col.lower() 
                          or 'class' in col.lower() or 'category' in col.lower()]
        
        print(f"\nPotential label columns: {label_candidates}")
        
        # Analyze each potential label column
        for col in label_candidates:
            print(f"\n--- Analysis of '{col}' column ---")
            
            print("Training set distribution:")
            train_counts = train_df[col].value_counts()
            print(train_counts)
            print(f"Training set unique values: {train_df[col].nunique()}")
            
            print("\nTesting set distribution:")
            test_counts = test_df[col].value_counts()
            print(test_counts)
            print(f"Testing set unique values: {test_df[col].nunique()}")
            
            # Check for overlap
            train_values = set(train_df[col].unique())
            test_values = set(test_df[col].unique())
            
            overlap = train_values & test_values
            train_only = train_values - test_values
            test_only = test_values - train_values
            
            print(f"\nLabel overlap analysis:")
            print(f"  Common labels: {len(overlap)} - {sorted(overlap)}")
            if train_only:
                print(f"  Only in train: {sorted(train_only)}")
            if test_only:
                print(f"  Only in test: {sorted(test_only)}")
    
    def analyze_temporal_characteristics(self, train_df, test_df):
        """Look for temporal features that could cause data leakage"""
        
        print("\n" + "="*50)
        print("TEMPORAL ANALYSIS")
        print("="*50)
        
        # Look for time-related columns
        time_candidates = [col for col in train_df.columns 
                         if any(keyword in col.lower() for keyword in 
                               ['time', 'start', 'end', 'duration', 'timestamp'])]
        
        print(f"\nPotential temporal columns: {time_candidates}")
        
        for col in time_candidates:
            print(f"\n--- Temporal analysis of '{col}' ---")
            
            if train_df[col].dtype in ['int64', 'float64']:
                print(f"Training {col}:")
                print(f"  Min: {train_df[col].min()}")
                print(f"  Max: {train_df[col].max()}")
                print(f"  Range: {train_df[col].max() - train_df[col].min()}")
                
                print(f"Testing {col}:")
                print(f"  Min: {test_df[col].min()}")
                print(f"  Max: {test_df[col].max()}")
                print(f"  Range: {test_df[col].max() - test_df[col].min()}")
                
                # Check for temporal overlap
                train_min, train_max = train_df[col].min(), train_df[col].max()
                test_min, test_max = test_df[col].min(), test_df[col].max()
                
                overlap_start = max(train_min, test_min)
                overlap_end = min(train_max, test_max)
                
                if overlap_end > overlap_start:
                    print(f"  ⚠️  TEMPORAL OVERLAP DETECTED!")
                    print(f"      Overlap range: {overlap_start} to {overlap_end}")
                    print(f"      This could cause data leakage!")
                else:
                    print(f"  ✓ No temporal overlap detected")
    
    def analyze_feature_characteristics(self, train_df, test_df):
        """Analyze feature distributions and potential for TDA"""
        
        print("\n" + "="*50)
        print("FEATURE ANALYSIS FOR TDA")
        print("="*50)
        
        # Identify numeric features
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove label columns
        label_cols = [col for col in numeric_cols 
                     if 'label' in col.lower() or 'attack' in col.lower()]
        feature_cols = [col for col in numeric_cols if col not in label_cols]
        
        print(f"\nNumeric features for TDA ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols[:20]):  # Show first 20
            print(f"{i+1:2d}. {col}")
        if len(feature_cols) > 20:
            print(f"    ... and {len(feature_cols) - 20} more")
        
        # Analyze feature distributions
        print(f"\nFeature distribution analysis:")
        
        sample_features = feature_cols[:10]  # Analyze first 10 features
        
        for col in sample_features:
            train_vals = train_df[col].dropna()
            test_vals = test_df[col].dropna()
            
            print(f"\n{col}:")
            print(f"  Train: mean={train_vals.mean():.3f}, std={train_vals.std():.3f}")
            print(f"  Test:  mean={test_vals.mean():.3f}, std={test_vals.std():.3f}")
            
            # Check for major distribution shifts
            train_mean, test_mean = train_vals.mean(), test_vals.mean()
            if abs(train_mean - test_mean) > 0.1 * abs(train_mean):
                print(f"  ⚠️  Potential distribution shift detected!")
    
    def check_data_quality(self, train_df, test_df):
        """Check for data quality issues"""
        
        print("\n" + "="*50)
        print("DATA QUALITY ANALYSIS")
        print("="*50)
        
        print(f"\nMissing values analysis:")
        print(f"Training set missing values:")
        train_missing = train_df.isnull().sum()
        train_missing = train_missing[train_missing > 0].sort_values(ascending=False)
        if len(train_missing) > 0:
            print(train_missing.head(10))
        else:
            print("  No missing values found")
        
        print(f"\nTesting set missing values:")
        test_missing = test_df.isnull().sum()
        test_missing = test_missing[test_missing > 0].sort_values(ascending=False)
        if len(test_missing) > 0:
            print(test_missing.head(10))
        else:
            print("  No missing values found")
        
        # Check for duplicates
        train_dups = train_df.duplicated().sum()
        test_dups = test_df.duplicated().sum()
        
        print(f"\nDuplicate rows:")
        print(f"  Training set: {train_dups} duplicates ({train_dups/len(train_df)*100:.2f}%)")
        print(f"  Testing set: {test_dups} duplicates ({test_dups/len(test_df)*100:.2f}%)")
    
    def run_complete_analysis(self):
        """Run complete dataset analysis"""
        
        # Load data
        train_df, test_df = self.load_and_analyze_basic_stats()
        
        if train_df is None:
            print("❌ Failed to load training data")
            return None
        
        if test_df is None:
            print("❌ Failed to load testing data")
            return None
        
        # Run all analyses
        self.analyze_columns_and_types(train_df, test_df)
        self.analyze_attack_types(train_df, test_df)
        self.analyze_temporal_characteristics(train_df, test_df)
        self.analyze_feature_characteristics(train_df, test_df)
        self.check_data_quality(train_df, test_df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return train_df, test_df

def main():
    """Run UNSW-NB15 dataset analysis"""
    
    analyzer = UNSWDatasetAnalyzer()
    train_df, test_df = analyzer.run_complete_analysis()
    
    if train_df is not None and test_df is not None:
        print(f"\n✓ Dataset analysis complete")
        print(f"  Training samples: {len(train_df):,}")
        print(f"  Testing samples: {len(test_df):,}")
        print(f"  Total features: {len(train_df.columns)}")
        
        return train_df, test_df
    else:
        print(f"❌ Dataset analysis failed")
        return None, None

if __name__ == "__main__":
    main()
