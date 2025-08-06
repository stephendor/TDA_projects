#!/usr/bin/env python3
"""
Validate Enhanced Deep TDA on CIC-IDS2017 PortScan with proper attack/benign balance
Using sufficient data for meaningful validation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from validation.validation_framework import ValidationFramework
from src.algorithms.deep.enhanced_deep_tda import EnhancedDeepTDAAnalyzer
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def validate_enhanced_tda_portscan_proper():
    """Validate Enhanced Deep TDA on PortScan with proper balance"""
    
    # Initialize ValidationFramework
    validator = ValidationFramework("enhanced_tda_portscan_proper", random_seed=42)
    
    with validator.capture_console_output():
        print("🔬 ENHANCED DEEP TDA - CIC-IDS2017 PORTSCAN PROPER VALIDATION")
        print("=" * 70)
        print("Dataset: Friday-WorkingHours-Afternoon-PortScan")
        print("Using balanced subset with sufficient attacks for proper ML validation")
        
        # Load PortScan dataset
        file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
        
        print(f"Loading PortScan dataset...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        print(f"Full dataset: {len(df):,} samples")
        
        # Analyze attack distribution
        attack_mask = df['Label'] != 'BENIGN'
        attacks = df[attack_mask]
        benign = df[~attack_mask]
        
        print(f"Total PortScan attacks: {len(attacks):,}")
        print(f"Total benign samples: {len(benign):,}")
        
        # Use balanced subset - take 10K attacks + 10K benign for proper validation
        n_attacks_to_use = 10000
        n_benign_to_use = 10000
        
        print(f"\nUsing balanced subset:")
        print(f"  Attacks: {n_attacks_to_use:,}")
        print(f"  Benign: {n_benign_to_use:,}")
        
        # Sample attacks and benign
        attacks_sample = attacks.sample(n=n_attacks_to_use, random_state=42)
        benign_sample = benign.sample(n=n_benign_to_use, random_state=42)
        
        # Combine for balanced dataset
        df_balanced = pd.concat([attacks_sample, benign_sample])
        
        print(f"Balanced dataset: {len(df_balanced):,} samples")
        print(f"Attack rate: {len(attacks_sample)/len(df_balanced)*100:.1f}%")
        
        # Initialize Enhanced Deep TDA
        print(f"\nInitializing Enhanced Deep TDA...")
        analyzer = EnhancedDeepTDAAnalyzer(
            input_dim=80,
            embed_dim=128,
            num_layers=3,
            num_heads=4
        )
        
        print(f"Model parameters: {sum(p.numel() for p in analyzer.model.parameters()):,}")
        
        # Prepare data for enhanced model
        attack_datasets = {'portscan': df_balanced}
        print(f"\nPreparing multi-attack data...")
        
        X, y_binary, y_attack_type, y_phase = analyzer.prepare_multi_attack_data(
            attack_datasets, max_samples_per_attack=len(df_balanced)
        )
        
        print(f"Prepared dataset: {X.shape}")
        print(f"Attack rate: {np.mean(y_binary)*100:.1f}%")
        print(f"Total attacks: {y_binary.sum():,}")
        print(f"Total benign: {(y_binary == 0).sum():,}")
        
        # Train-test split
        X_train, X_test, y_train_bin, y_test_bin = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        print(f"\nDataset split:")
        print(f"Training: {X_train.shape[0]:,} samples ({y_train_bin.sum():,} attacks)")
        print(f"Testing: {X_test.shape[0]:,} samples ({y_test_bin.sum():,} attacks)")
        
        # Prepare other labels for training
        _, _, y_train_type, y_test_type = train_test_split(
            X, y_attack_type, test_size=0.2, random_state=42, stratify=y_binary
        )
        _, _, y_train_phase, y_test_phase = train_test_split(
            X, y_phase, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train model
        print(f"\nTraining Enhanced Deep TDA...")
        best_f1 = analyzer.train(
            X=X_train,
            y_binary=y_train_bin,
            y_attack_type=y_train_type,
            y_phase=y_train_phase,
            epochs=15,
            batch_size=64,
            learning_rate=1e-4
        )
        
        print(f"Training completed. Best validation F1: {best_f1:.4f}")
        
        # Get predictions on test set
        print(f"\nGenerating predictions on test set...")
        analyzer.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(analyzer.device)
        
        with torch.no_grad():
            outputs = analyzer.model(X_test_tensor)
            y_pred_logits = outputs['binary_logits']
            y_pred_proba = torch.softmax(y_pred_logits, dim=1)[:, 1].cpu().numpy()
            y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        
        print(f"Test set predictions:")
        print(f"  Predicted attacks: {y_pred.sum():,}")
        print(f"  Actual attacks: {y_test_bin.sum():,}")
        print(f"  True Positives: {np.sum((y_pred == 1) & (y_test_bin == 1)):,}")
        print(f"  False Positives: {np.sum((y_pred == 1) & (y_test_bin == 0)):,}")
        print(f"  False Negatives: {np.sum((y_pred == 0) & (y_test_bin == 1)):,}")
        print(f"  True Negatives: {np.sum((y_pred == 0) & (y_test_bin == 0)):,}")
    
    # Validate using ValidationFramework
    results = validator.validate_classification_results(
        y_true=y_test_bin,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        class_names=['Benign', 'PortScan']
    )
    
    return results

if __name__ == "__main__":
    try:
        results = validate_enhanced_tda_portscan_proper()
        
        print("\n" + "="*70)
        print("ENHANCED DEEP TDA PORTSCAN VALIDATION COMPLETE")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()