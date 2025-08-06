#!/usr/bin/env python3
"""
Validate Enhanced Deep TDA on NF-CICIDS2018-v3 NetFlo    # Prepare features (exclude non-numeric and label columns)
    feature_columns = [col for col in combined_df.columns 
                      if col not in ['Label', 'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']]
    
    X = combined_df[feature_columns].fillna(0)
    y = combined_df['Label'].values.astype(int)  # 0=benign, 1=attack
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"\n🔢 FEATURE MATRIX")
    print(f"   Shape: {X.shape}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Labels: {len(y)} (Attacks: {np.sum(y)}, Benign: {len(y)-np.sum(y)})")
    
    return X.values, y, feature_columns, combined_df['Attack'].valuesthe proper ValidationFramework with multiple attack types
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from validation.validation_framework import ValidationFramework
from src.algorithms.deep.enhanced_deep_tda import EnhancedDeepTDAAnalyzer

def load_netflow_data(max_samples=50000):
    """
    Load balanced NetFlow data with multiple attack types
    """
    print("🔧 LOADING NF-CICIDS2018-v3 NETFLOW DATA")
    print("-" * 60)
    
    data_path = "data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    # Load data in chunks to manage memory
    chunk_size = 10000
    all_attacks = []
    all_benign = []
    
    print(f"Loading data in chunks of {chunk_size:,}...")
    chunks_processed = 0
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        chunks_processed += 1
        
        # Separate attacks and benign
        attacks = chunk[chunk['Attack'] != 'Benign']
        benign = chunk[chunk['Attack'] == 'Benign']
        
        if len(attacks) > 0:
            all_attacks.append(attacks)
            print(f"  Chunk {chunks_processed}: Found {len(attacks)} attacks")
        
        if len(benign) > 0:
            all_benign.append(benign)
        
        # Stop when we have enough data
        total_attacks = sum(len(df) for df in all_attacks)
        total_benign = sum(len(df) for df in all_benign)
        
        if total_attacks >= max_samples//2 and total_benign >= max_samples//2:
            break
    
    # Combine and sample
    attacks_df = pd.concat(all_attacks, ignore_index=True)
    benign_df = pd.concat(all_benign, ignore_index=True)
    
    print(f"\\n📊 DATA SUMMARY")
    print(f"   Total attacks loaded: {len(attacks_df):,}")
    print(f"   Total benign loaded: {len(benign_df):,}")
    print(f"   Attack types: {attacks_df['Attack'].value_counts().to_dict()}")
    
    # Balance the dataset
    n_attacks = min(len(attacks_df), max_samples//2)
    n_benign = min(len(benign_df), max_samples//2)
    
    attacks_sample = attacks_df.sample(n=n_attacks, random_state=42)
    benign_sample = benign_df.sample(n=n_benign, random_state=42)
    
    # Combine
    combined_df = pd.concat([attacks_sample, benign_sample], ignore_index=True)
    
    print(f"\\n✅ BALANCED DATASET CREATED")
    print(f"   Attack samples: {len(attacks_sample):,}")
    print(f"   Benign samples: {len(benign_sample):,}")
    print(f"   Total samples: {len(combined_df):,}")
    print(f"   Attack rate: {len(attacks_sample)/len(combined_df)*100:.1f}%")
    
    # Prepare features (exclude non-numeric and label columns)
    feature_columns = [col for col in combined_df.columns 
                      if col not in ['Label', 'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']]
    
    X = combined_df[feature_columns].fillna(0)
    y = combined_df['Label'].values  # 0=benign, 1=attack
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"\\n🔢 FEATURE MATRIX")
    print(f"   Shape: {X.shape}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Labels: {len(y)} (Attacks: {y.sum()}, Benign: {len(y)-y.sum()})")
    
    return X.values, y, feature_columns, combined_df['Attack'].values

def validate_enhanced_deep_tda_netflow():
    """
    Main validation function using proper ValidationFramework
    """
    # Initialize validation framework
    validator = ValidationFramework(
        experiment_name="enhanced_deep_tda_netflow",
        random_seed=42
    )
    
    # Capture console output
    with validator.capture_console_output():
        print("🔬 ENHANCED DEEP TDA VALIDATION - NF-CICIDS2018-v3")
        print("=" * 80)
        print("Multi-attack NetFlow validation with proper ValidationFramework")
        print("Target: Real performance metrics with actual plots")
        print("=" * 80)
        
        # Load data
        X, y, feature_names, attack_types = load_netflow_data(max_samples=20000)
        
        # Initialize Enhanced Deep TDA with appropriate dimensions
        print(f"\\n🧠 INITIALIZING ENHANCED DEEP TDA")
        print("-" * 60)
        
        analyzer = EnhancedDeepTDAAnalyzer(
            input_dim=X.shape[1],  # Use actual feature count
            embed_dim=128,         # Reasonable size
            num_layers=4,          # Moderate depth
            num_heads=8            # Multi-head attention
        )
        
        # Create attack datasets for multi-attack training
        print(f"\\n📊 PREPARING MULTI-ATTACK DATA")
        print("-" * 60)
        
        # Convert to DataFrame for analyzer
        df = pd.DataFrame(X, columns=feature_names)
        df['Label'] = y
        df['Attack'] = attack_types
        
        # Group by attack type
        attack_datasets = {}
        for attack_type in df['Attack'].unique():
            if attack_type != 'Benign':
                attack_data = df[df['Attack'] == attack_type]
                benign_data = df[df['Attack'] == 'Benign'].sample(
                    n=min(len(attack_data), 1000), random_state=42
                )
                combined = pd.concat([attack_data, benign_data])
                attack_datasets[attack_type.lower().replace('-', '_')] = combined
                print(f"   {attack_type}: {len(attack_data)} attacks + {len(benign_data)} benign")
        
        # Prepare training data
        X_train, y_binary, y_attack_type, y_phase = analyzer.prepare_multi_attack_data(
            attack_datasets, max_samples_per_attack=2000
        )
        
        # Train the model
        print(f"\\n🚀 TRAINING ENHANCED DEEP TDA")
        print("-" * 60)
        
        best_f1 = analyzer.train(
            X_train, y_binary, y_attack_type, y_phase,
            epochs=20, batch_size=32, learning_rate=0.001
        )
        
        # Generate predictions for validation
        print(f"\\n📊 GENERATING PREDICTIONS")
        print("-" * 60)
        
        # Use a subset for validation
        val_size = min(1000, len(X))
        indices = np.random.choice(len(X), val_size, replace=False)
        X_val = X[indices]
        y_val = np.array(y[indices], dtype=int)
        
        # Get predictions (simplified binary classification)
        analyzer.model.eval()
        import torch
        X_scaled = analyzer.scaler.transform(X_val)
        X_tensor = torch.FloatTensor(X_scaled).to(analyzer.device)
        
        with torch.no_grad():
            outputs = analyzer.model(X_tensor)
            y_pred_proba = torch.softmax(outputs['binary_logits'], dim=1)[:, 1].cpu().numpy()
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        print(f"✅ Predictions generated")
        print(f"   Validation samples: {len(y_val)}")
        print(f"   Predicted attacks: {y_pred.sum()}")
        print(f"   Actual attacks: {y_val.sum()}")
        
        # Validate using framework
        results = validator.validate_classification_results(
            y_true=y_val,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            class_names=['Benign', 'Attack']
        )
        
        # Save additional metadata
        validator.raw_data['model_info'] = {
            'model_type': 'Enhanced Deep TDA',
            'architecture': 'Multi-Attack Transformer',
            'parameters': int(sum(p.numel() for p in analyzer.model.parameters())),
            'input_features': X.shape[1],
            'training_samples': len(X_train),
            'validation_samples': len(y_val),
            'attack_types_trained': list(attack_datasets.keys()),
            'best_training_f1': float(best_f1)
        }
        
        # Complete validation - save results manually since finalize_validation might not exist
        import json
        import os
        
        # Save results to JSON
        results_file = os.path.join(validator.results_dir, "validation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': results,
                'model_info': validator.raw_data['model_info']
            }, f, indent=2)
        
        print(f"\\n🎯 VALIDATION COMPLETE")
        print(f"   Results saved to: {validator.results_dir}")
        print(f"   Plots saved to: {validator.plots_dir}")
        print(f"   F1-Score: {results['f1']:.3f}")
        print(f"   Accuracy: {results['accuracy']:.3f}")
        print(f"   Precision: {results['precision']:.3f}")
        print(f"   Recall: {results['recall']:.3f}")

if __name__ == "__main__":
    validate_enhanced_deep_tda_netflow()
