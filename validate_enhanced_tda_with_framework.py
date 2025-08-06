#!/usr/bin/env python3
"""
Validate Enhanced Deep TDA using the proper ValidationFramework
This will generate actual console output, confusion matrix, ROC curves, etc.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from validation.validation_framework import ValidationFramework
from src.algorithms.deep.enhanced_deep_tda import EnhancedDeepTDAAnalyzer
import pandas as pd
import numpy as np
import torch

def validate_enhanced_deep_tda_with_framework():
    """Use the actual ValidationFramework to validate Enhanced Deep TDA"""
    
    # Initialize ValidationFramework
    validator = ValidationFramework("enhanced_deep_tda_real_validation", random_seed=42)
    
    with validator.capture_console_output():
        print("🔬 ENHANCED DEEP TDA VALIDATION WITH REAL CIC-IDS2017")
        print("=" * 60)
        
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
        
        # Balance the dataset properly for ML training
        if len(attacks) > 0:
            # Take all attacks, sample equivalent benign
            benign_sample = benign.sample(n=len(attacks) * 10, random_state=42)  # 10:1 ratio
            df_balanced = pd.concat([attacks, benign_sample])
        else:
            raise ValueError("No infiltration attacks found")
        
        print(f"Balanced dataset: {len(df_balanced)} samples")
        print(f"Attack rate: {len(attacks)/len(df_balanced)*100:.2f}%")
        
        # Prepare features using common network flow features
        feature_cols = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Packet Length Mean', 'Packet Length Std', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'URG Flag Count', 'Average Packet Size'
        ]
        
        # Get available features
        available_features = [col for col in feature_cols if col in df_balanced.columns]
        print(f"Available features: {len(available_features)}")
        
        # Extract features and labels
        X = df_balanced[available_features].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Create binary labels
        y = (df_balanced['Label'] == 'Infiltration').astype(int)
        
        print(f"Feature matrix: {X.shape}")
        print(f"Labels: {y.shape}, Attacks: {y.sum()}")
        
        # Initialize Enhanced Deep TDA with correct input dimension
        print(f"\nInitializing Enhanced Deep TDA...")
        analyzer = EnhancedDeepTDAAnalyzer(
            input_dim=80,   # Fixed dimension that the model expects
            embed_dim=64,   # Smaller for real data
            num_layers=2,   # Simpler architecture
            num_heads=4
        )
        
        # Prepare data for enhanced model
        attack_datasets = {'infiltration': df_balanced}
        X_multi, y_binary, y_attack_type, y_phase = analyzer.prepare_multi_attack_data(
            attack_datasets, max_samples_per_attack=len(df_balanced)
        )
        
        print(f"Multi-attack data prepared: {X_multi.shape}")
        
        # Train model
        print(f"\nTraining Enhanced Deep TDA...")
        best_f1 = analyzer.train(
            X=X_multi,
            y_binary=y_binary,
            y_attack_type=y_attack_type,
            y_phase=y_phase,
            epochs=5,  # Quick training for real validation
            batch_size=16,
            learning_rate=1e-3
        )
        
        print(f"Training completed. Best F1: {best_f1:.4f}")
        
        # Get predictions for validation framework
        analyzer.model.eval()
        X_tensor = torch.FloatTensor(X_multi).to(analyzer.device)
        
        with torch.no_grad():
            outputs = analyzer.model(X_tensor)
            y_pred_logits = outputs['binary_logits']
            y_pred_proba = torch.softmax(y_pred_logits, dim=1)[:, 1].cpu().numpy()
            y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        
        print(f"\nPredictions generated:")
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Probabilities shape: {y_pred_proba.shape}")
        print(f"Predicted attacks: {y_pred.sum()}")
        print(f"Actual attacks: {y_binary.sum()}")
        
    # Use ValidationFramework to validate results (generates real plots)
    results = validator.validate_classification_results(
        y_true=y_binary,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        class_names=['Benign', 'Attack']
    )
    
    # Verify against baseline
    baseline_f1 = 0.706
    claim_verified = validator.verify_claim(baseline_f1, tolerance=0.05)
    
    print(f"\nValidation complete:")
    print(f"Enhanced Deep TDA F1-score: {results['f1_score']:.4f}")
    print(f"Baseline F1-score: {baseline_f1:.4f}")
    print(f"Improvement: {results['f1_score'] - baseline_f1:+.4f}")
    print(f"Baseline claim verified: {claim_verified}")
    
    return validator, results

if __name__ == "__main__":
    try:
        validator, results = validate_enhanced_deep_tda_with_framework()
        print("\n" + "="*60)
        print(f"ENHANCED DEEP TDA VALIDATION COMPLETE")
        print(f"Results saved to: {validator.output_dir}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()