#!/usr/bin/env python3
"""
Validate Enhanced Deep TDA on FULL CIC-IDS2017 PortScan dataset
Using the proper ValidationFramework with 158,930 real attacks
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

def validate_enhanced_tda_full_portscan():
    """Validate Enhanced Deep TDA on FULL PortScan dataset"""
    
    # Initialize ValidationFramework
    validator = ValidationFramework("enhanced_tda_full_portscan", random_seed=42)
    
    with validator.capture_console_output():
        print("🔬 ENHANCED DEEP TDA - FULL CIC-IDS2017 PORTSCAN VALIDATION")
        print("=" * 70)
        print("Dataset: Friday-WorkingHours-Afternoon-PortScan (158,930 attacks)")
        print("Using ENTIRE dataset - NO SAMPLING")
        
        # Load FULL PortScan dataset
        file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
        
        print(f"Loading FULL PortScan dataset...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        print(f"Total samples loaded: {len(df):,}")
        
        # Analyze attack distribution
        attack_mask = df['Label'] != 'BENIGN'
        attacks = df[attack_mask]
        benign = df[~attack_mask]
        
        print(f"PortScan attacks: {len(attacks):,}")
        print(f"Benign samples: {len(benign):,}")
        print(f"Attack rate: {len(attacks)/len(df)*100:.1f}%")
        
        # Check attack types
        attack_types = attacks['Label'].value_counts()
        print(f"Attack type breakdown:")
        for attack_type, count in attack_types.items():
            print(f"  {attack_type}: {count:,}")
        
        # Use FULL dataset - no sampling
        print(f"\nUsing ENTIRE dataset for validation ({len(df):,} samples)")
        
        # Initialize Enhanced Deep TDA
        print(f"\nInitializing Enhanced Deep TDA...")
        analyzer = EnhancedDeepTDAAnalyzer(
            input_dim=80,
            embed_dim=128,
            num_layers=3,
            num_heads=4
        )
        
        print(f"Model parameters: {sum(p.numel() for p in analyzer.model.parameters()):,}")
        
        # Prepare data for enhanced model using FULL dataset
        attack_datasets = {'portscan': df}
        print(f"\nPreparing multi-attack data from FULL dataset...")
        
        X, y_binary, y_attack_type, y_phase = analyzer.prepare_multi_attack_data(
            attack_datasets, max_samples_per_attack=len(df)  # Use ALL samples
        )
        
        print(f"Prepared dataset: {X.shape}")
        print(f"Attack rate: {np.mean(y_binary)*100:.1f}%")
        print(f"Total attacks: {y_binary.sum():,}")
        print(f"Total benign: {(y_binary == 0).sum():,}")
        
        # Train-test split on the full prepared dataset
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
        
        # Train model on full data
        print(f"\nTraining Enhanced Deep TDA on FULL dataset...")
        best_f1 = analyzer.train(
            X=X_train,
            y_binary=y_train_bin,
            y_attack_type=y_train_type,
            y_phase=y_train_phase,
            epochs=10,
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
        print(f"  Predicted benign: {(y_pred == 0).sum():,}")
        print(f"  Actual benign: {(y_test_bin == 0).sum():,}")
    
    # Validate using ValidationFramework
    results = validator.validate_classification_results(
        y_true=y_test_bin,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        class_names=['Benign', 'PortScan']
    )
    
    # Compare to baseline
    baseline_f1 = 0.706
    improvement = results['f1_score'] - baseline_f1
    
    print(f"\nFULL DATASET VALIDATION RESULTS:")
    print(f"Enhanced Deep TDA F1-score: {results['f1_score']:.4f}")
    print(f"Baseline F1-score: {baseline_f1:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    print(f"Test set size: {len(y_test_bin):,} samples")
    print(f"Attack detection rate: {y_pred.sum()}/{y_test_bin.sum()}")
    
    return results

if __name__ == "__main__":
    try:
        results = validate_enhanced_tda_full_portscan()
        
        print("\n" + "="*70)
        print("ENHANCED DEEP TDA FULL PORTSCAN VALIDATION COMPLETE")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Real validation plots generated in:")
        print("validation/enhanced_tda_full_portscan/plots_*/")
        print("="*70)
        
    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()