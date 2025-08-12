#!/usr/bin/env python3
"""
Investigate suspiciously high F1 scores in UNSW-NB15 validation
Check for data leakage, trivial separators, and other issues
"""

import pandas as pd
import numpy as np
from pathlib import Path

def investigate_suspicious_scores():
    """Investigate potential causes of unrealistically high scores"""
    
    print("🕵️ INVESTIGATING SUSPICIOUS SCORES")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_parquet('data/apt_datasets/UNSW-NB15/UNSW_NB15_training-set.parquet')
    print(f"Full dataset shape: {train_df.shape}")
    
    # Sample strategically to include both attacks and normal traffic
    attacks = train_df[train_df['label'] == 1].sample(min(2000, len(train_df[train_df['label'] == 1])), random_state=42)
    normal = train_df[train_df['label'] == 0].sample(min(2000, len(train_df[train_df['label'] == 0])), random_state=42)
    sample_df = pd.concat([attacks, normal])
    
    print(f"Sample shape: {sample_df.shape}")
    print(f"Sample label distribution:")
    print(sample_df['label'].value_counts())
    print(f"Sample attack types:")
    print(sample_df['attack_cat'].value_counts())
    
    print("\n" + "=" * 60)
    print("ISSUE 1: PERFECT LABEL-ATTACK_CAT CORRELATION")
    print("=" * 60)
    
    # Check if attack_cat perfectly predicts label
    crosstab = pd.crosstab(sample_df['attack_cat'], sample_df['label'], margins=True)
    print("Attack_cat vs Label crosstab:")
    print(crosstab)
    
    # Check the relationship more explicitly
    normal_labels = sample_df[sample_df['attack_cat'] == 'Normal']['label'].unique()
    attack_labels = sample_df[sample_df['attack_cat'] != 'Normal']['label'].unique()
    
    print(f"\nNormal traffic labels: {normal_labels}")
    print(f"Attack traffic labels: {attack_labels}")
    
    # This is the smoking gun - if Normal == 0 and everything else == 1
    if (len(normal_labels) == 1 and normal_labels[0] == 0 and 
        len(attack_labels) == 1 and attack_labels[0] == 1):
        print("\n🚨 CRITICAL ISSUE FOUND:")
        print("   attack_cat PERFECTLY predicts label!")
        print("   - If attack_cat == 'Normal' → label = 0")
        print("   - If attack_cat != 'Normal' → label = 1")
        print("   - This is data leakage - the model can trivially achieve perfect scores")
        issue_1_found = True
    else:
        print("\n✅ No perfect correlation between attack_cat and label")
        issue_1_found = False
    
    print("\n" + "=" * 60)
    print("ISSUE 2: CATEGORICAL FEATURE LEAKAGE")
    print("=" * 60)
    
    categorical_cols = [col for col in sample_df.columns if sample_df[col].dtype == 'object']
    print(f"Categorical columns: {categorical_cols}")
    
    leakage_found = False
    
    for col in ['proto', 'service', 'state']:
        if col in sample_df.columns:
            print(f"\n--- {col.upper()} ANALYSIS ---")
            
            # Values in normal vs attack traffic
            normal_values = set(sample_df[sample_df['attack_cat'] == 'Normal'][col].unique())
            attack_values = set(sample_df[sample_df['attack_cat'] != 'Normal'][col].unique())
            
            exclusive_to_normal = normal_values - attack_values
            exclusive_to_attacks = attack_values - normal_values
            shared_values = normal_values & attack_values
            
            print(f"Values exclusive to Normal: {exclusive_to_normal}")
            print(f"Values exclusive to Attacks: {exclusive_to_attacks}")
            print(f"Shared values: {len(shared_values)} out of {len(normal_values | attack_values)} total")
            
            if exclusive_to_attacks:
                print(f"🚨 POTENTIAL LEAKAGE: {col} has attack-exclusive values")
                leakage_found = True
                
                # Show the breakdown
                attack_breakdown = pd.crosstab(sample_df[col], sample_df['label'])
                print("Distribution by label:")
                print(attack_breakdown)
    
    print("\n" + "=" * 60)
    print("ISSUE 3: FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Look at numeric features for obvious separators
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['label', 'attack_cat']]
    
    print("Checking first 10 numeric features for separation...")
    
    separation_issues = []
    
    for col in numeric_cols[:10]:
        normal_vals = sample_df[sample_df['label'] == 0][col]
        attack_vals = sample_df[sample_df['label'] == 1][col]
        
        # Check for clear separation
        normal_range = (normal_vals.min(), normal_vals.max())
        attack_range = (attack_vals.min(), attack_vals.max())
        
        # Check if ranges don't overlap
        if normal_range[1] < attack_range[0] or attack_range[1] < normal_range[0]:
            separation_issues.append({
                'feature': col,
                'normal_range': normal_range,
                'attack_range': attack_range
            })
            print(f"🚨 {col}: Normal [{normal_range[0]:.2f}, {normal_range[1]:.2f}], Attack [{attack_range[0]:.2f}, {attack_range[1]:.2f}] - NO OVERLAP")
        else:
            normal_mean = normal_vals.mean()
            attack_mean = attack_vals.mean()
            effect_size = abs(normal_mean - attack_mean) / max(normal_vals.std(), attack_vals.std(), 0.001)
            
            if effect_size > 2.0:  # Large effect size
                print(f"⚠️ {col}: Large effect size {effect_size:.1f} - might be too easy")
    
    print(f"\nFeatures with perfect separation: {len(separation_issues)}")
    
    print("\n" + "=" * 60)
    print("ISSUE 4: SIMULATING OUR VALIDATION PROCESS")
    print("=" * 60)
    
    # Simulate what our validation is doing
    print("Reproducing the validation approach to find the issue...")
    
    # Select the same features our validation uses
    numeric_features = [col for col in sample_df.select_dtypes(include=[np.number]).columns 
                       if col not in ['label']][:25]  # Same limit as validation
    
    print(f"Features being used in validation: {len(numeric_features)}")
    print(f"First 10: {numeric_features[:10]}")
    
    # Check if any of these features contain the attack_cat information
    if 'attack_cat' in numeric_features:
        print("🚨 CRITICAL: attack_cat is being included in numeric features!")
    else:
        print("✅ attack_cat not directly included in features")
    
    # Check feature matrix
    X = sample_df[numeric_features]
    y = sample_df['label']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Look for features with perfect correlation to labels
    perfect_correlations = []
    for col in numeric_features[:10]:  # Check first 10
        correlation = abs(X[col].corr(y))
        if correlation > 0.95:
            perfect_correlations.append((col, correlation))
            print(f"🚨 {col}: correlation with label = {correlation:.3f}")
    
    print(f"\nFeatures with >0.95 correlation to labels: {len(perfect_correlations)}")
    
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    
    total_issues = 0
    
    if issue_1_found:
        print("❌ CRITICAL: Perfect attack_cat → label correlation (data leakage)")
        total_issues += 1
        
    if leakage_found:
        print("❌ CRITICAL: Categorical feature leakage detected")
        total_issues += 1
        
    if separation_issues:
        print(f"❌ WARNING: {len(separation_issues)} features with perfect separation")
        total_issues += 1
        
    if perfect_correlations:
        print(f"❌ CRITICAL: {len(perfect_correlations)} features perfectly correlated with labels")
        total_issues += 1
    
    if total_issues == 0:
        print("✅ No obvious data leakage found - scores might be legitimate")
        print("   (Though still surprisingly high - consider other factors)")
    else:
        print(f"\n🚨 TOTAL ISSUES FOUND: {total_issues}")
        print("   This explains the unrealistically high scores!")
        print("   The model is not learning to detect attacks - it's exploiting data leakage")
    
    return total_issues > 0

if __name__ == "__main__":
    has_issues = investigate_suspicious_scores()
    if has_issues:
        print("\n💡 RECOMMENDATIONS:")
        print("1. Exclude attack_cat from any feature engineering")
        print("2. Use only raw network flow features")
        print("3. Ensure train/test split doesn't leak information")
        print("4. Validate on truly held-out data")
        print("5. Expect more realistic scores (0.6-0.8 range)")