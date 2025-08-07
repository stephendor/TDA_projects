#!/usr/bin/env python3
"""
Validate the 70.6% F1-score baseline using proper ValidationFramework
This will show actual console output and real plots like the example
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from validation.validation_framework import ValidationFramework
from src.algorithms.hybrid.hybrid_multiscale_graph_tda import HybridTDAAnalyzer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

def validate_hybrid_baseline_with_framework():
    """Use the ValidationFramework to validate the 70.6% baseline claim"""
    
    # Initialize ValidationFramework  
    validator = ValidationFramework("hybrid_baseline_real_validation", random_seed=42)
    
    with validator.capture_console_output():
        print("🔬 HYBRID TDA BASELINE VALIDATION - REAL CIC-IDS2017")
        print("=" * 60)
        print("Verifying claimed 70.6% F1-score on real infiltration data")
        
        # Load real CIC-IDS2017 infiltration data
        file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
        
        print(f"Loading CIC-IDS2017 infiltration data...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Extract attacks and benign
        attacks = df[df['Label'] == 'Infiltration']
        benign = df[df['Label'] == 'BENIGN']
        
        print(f"Real infiltration attacks found: {len(attacks)}")
        print(f"Real benign samples: {len(benign):,}")
        
        if len(attacks) == 0:
            raise ValueError("No infiltration attacks found")
        
        # Sample larger dataset for hybrid method (needs more data)
        benign_sample = benign.sample(n=min(5000, len(benign)), random_state=42)
        df_balanced = pd.concat([attacks, benign_sample])
        
        print(f"Dataset for validation: {len(df_balanced)} samples")
        print(f"Attack rate: {len(attacks)/len(df_balanced)*100:.2f}%")
        
        # Initialize hybrid analyzer
        print(f"\nInitializing Hybrid TDA Analyzer...")
        analyzer = HybridTDAAnalyzer()
        
        # Extract hybrid features
        print(f"Extracting hybrid TDA features...")
        features, labels = analyzer.extract_hybrid_features(df_balanced)
        
        if features is None:
            raise ValueError("Feature extraction failed")
        
        print(f"Features extracted: {features.shape}")
        print(f"Labels: {labels.shape}, Attacks: {labels.sum()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        print(f"Training set: {X_train.shape}, attacks: {y_train.sum()}")
        print(f"Test set: {X_test.shape}, attacks: {y_test.sum()}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create ensemble (same as claimed baseline)
        print(f"\nTraining hybrid ensemble...")
        rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                   random_state=42, class_weight='balanced')
        rf2 = RandomForestClassifier(n_estimators=150, max_depth=15, 
                                   random_state=123, class_weight='balanced')
        lr = LogisticRegression(C=0.1, random_state=42, 
                              class_weight='balanced', max_iter=1000)
        
        ensemble = VotingClassifier(
            estimators=[('rf1', rf1), ('rf2', rf2), ('lr', lr)],
            voting='soft'
        )
        
        # Train
        ensemble.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = ensemble.predict(X_test_scaled)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)
        
        # Handle single class probability output
        if y_pred_proba.shape[1] == 1:
            # Only one class predicted, create dummy probabilities
            y_pred_proba = np.column_stack([1 - y_pred_proba[:, 0], y_pred_proba[:, 0]])
        
        y_pred_proba = y_pred_proba[:, 1]  # Attack class probabilities
        
        print(f"Training completed")
        print(f"Predictions: {y_pred.shape}, attacks predicted: {y_pred.sum()}")
        print(f"Probabilities: {y_pred_proba.shape}")
    
    # Use ValidationFramework to validate results
    results = validator.validate_classification_results(
        y_true=y_test,
        y_pred=y_pred, 
        y_pred_proba=y_pred_proba,
        class_names=['Benign', 'Attack']
    )
    
    # Verify against claimed baseline
    claimed_f1 = 0.706
    claim_verified = validator.verify_claim(claimed_f1, tolerance=0.05)
    
    print(f"\nBaseline validation results:")
    print(f"Achieved F1-score: {results['f1_score']:.4f}")
    print(f"Claimed F1-score: {claimed_f1:.4f}")
    print(f"Difference: {results['f1_score'] - claimed_f1:+.4f}")
    print(f"Claim verified (±5%): {claim_verified}")
    
    return validator, results

if __name__ == "__main__":
    try:
        validator, results = validate_hybrid_baseline_with_framework()
        print("\n" + "="*60)
        print(f"HYBRID BASELINE VALIDATION COMPLETE")
        print(f"Results directory: {validator.output_dir}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Plots and console output saved")
        print("="*60)
        
    except Exception as e:
        print(f"Baseline validation failed: {e}")
        import traceback
        traceback.print_exc()