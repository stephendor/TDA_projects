#!/usr/bin/env python3
"""
Validate Enhanced Deep TDA on NF-CICIDS2018-v3 NetFlow dataset
Using a simplified approach that works with NetFlow feature columns
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from validation.validation_framework import ValidationFramework

# Simple Deep TDA model for NetFlow data
class SimpleDeepTDAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

def load_netflow_data(max_samples=20000):
    """
    Load balanced NetFlow data with multiple attack types
    """
    print("🔧 LOADING NF-CICIDS2018-v3 NETFLOW DATA")
    print("-" * 60)
    
    data_path = "data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    # Load data in chunks to find attacks
    chunk_size = 10000
    attacks_found = []
    benign_found = []
    chunks_processed = 0
    
    print(f"Searching for attacks in chunks of {chunk_size:,}...")
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        chunks_processed += 1
        
        # Check for attacks
        attacks = chunk[chunk['Attack'] != 'Benign']
        benign = chunk[chunk['Attack'] == 'Benign']
        
        if len(attacks) > 0:
            attacks_found.append(attacks)
            print(f"  Chunk {chunks_processed}: Found {len(attacks)} attacks ({attacks['Attack'].iloc[0]})")
        
        if len(benign) > 0:
            benign_found.append(benign)
        
        # Stop when we have enough
        total_attacks = sum(len(df) for df in attacks_found)
        total_benign = sum(len(df) for df in benign_found)
        
        if total_attacks >= max_samples//2 and total_benign >= max_samples//2:
            break
        
        if chunks_processed >= 100:  # Safety limit
            break
    
    # Combine results
    if attacks_found:
        attacks_df = pd.concat(attacks_found, ignore_index=True)
    else:
        raise ValueError("No attacks found in the dataset")
    
    benign_df = pd.concat(benign_found, ignore_index=True)
    
    print(f"\\n📊 DATA SUMMARY")
    print(f"   Total attacks found: {len(attacks_df):,}")
    print(f"   Total benign found: {len(benign_df):,}")
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
    
    # Select numeric features only
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns 
                      if col not in ['Label']]
    
    print(f"\\n🔢 FEATURE SELECTION")
    print(f"   Available numeric features: {len(feature_columns)}")
    print(f"   Feature names: {feature_columns[:10]}...")  # Show first 10
    
    X = combined_df[feature_columns].fillna(0)
    y = combined_df['Label'].values.astype(int)  # 0=benign, 1=attack
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"\\n📊 FINAL DATASET")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Labels: {len(y)} (Attacks: {np.sum(y)}, Benign: {len(y)-np.sum(y)})")
    
    return X.values, y, feature_columns, combined_df['Attack'].values

def validate_deep_tda_netflow():
    """
    Main validation function using proper ValidationFramework
    """
    # Initialize validation framework
    validator = ValidationFramework(
        experiment_name="deep_tda_netflow",
        random_seed=42
    )
    
    # Capture console output
    with validator.capture_console_output():
        print("🔬 DEEP TDA VALIDATION - NF-CICIDS2018-v3 NETFLOW")
        print("=" * 80)
        print("Real multi-attack validation with ValidationFramework")
        print("Using simplified Deep TDA on actual NetFlow features")
        print("=" * 80)
        
        # Load data
        X, y, feature_names, attack_types = load_netflow_data(max_samples=20000)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\\n🏋️ TRAINING DATA")
        print(f"   Training samples: {len(X_train)} (Attacks: {np.sum(y_train)})")
        print(f"   Test samples: {len(X_test)} (Attacks: {np.sum(y_test)})")
        print(f"   Features: {X.shape[1]}")
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleDeepTDAModel(input_dim=X.shape[1], hidden_dim=128)
        model.to(device)
        
        print(f"\\n🧠 MODEL INITIALIZATION")
        print(f"   Device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Architecture: Simple Deep Neural Network")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        # Training loop
        print(f"\\n🚀 TRAINING MODEL")
        print("-" * 60)
        
        model.train()
        epochs = 50
        batch_size = 256
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:2d}: Loss={epoch_loss/total:.4f}, Accuracy={accuracy:.1f}%")
        
        # Generate predictions
        print(f"\\n📊 GENERATING PREDICTIONS")
        print("-" * 60)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_proba = torch.softmax(test_outputs, dim=1)
            y_pred_proba = test_proba[:, 1].cpu().numpy()  # Probability of attack class
            y_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()
        
        print(f"✅ Predictions generated")
        print(f"   Test samples: {len(y_test)}")
        print(f"   Predicted attacks: {np.sum(y_pred)}")
        print(f"   Actual attacks: {np.sum(y_test)}")
        
        # Validate using framework
        results = validator.validate_classification_results(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            class_names=['Benign', 'Attack']
        )
        
        # Save additional metadata
        validator.raw_data['model_info'] = {
            'model_type': 'Simple Deep TDA',
            'architecture': 'Feed-Forward Neural Network',
            'parameters': int(sum(p.numel() for p in model.parameters())),
            'input_features': X.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'dataset': 'NF-CICIDS2018-v3',
            'attack_types': list(set(attack_types[attack_types != 'Benign']))
        }
        
        # Save results
        import json
        import os
        
        results_file = os.path.join(validator.results_dir, "validation_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': results,
                'model_info': validator.raw_data['model_info']
            }, f, indent=2)
        
        print(f"\\n🎯 VALIDATION COMPLETE")
        print(f"   Results directory: {validator.results_dir}")
        print(f"   Plots directory: {validator.plots_dir}")
        print(f"   F1-Score: {results['f1']:.3f}")
        print(f"   Accuracy: {results['accuracy']:.3f}")
        print(f"   Precision: {results['precision']:.3f}")
        print(f"   Recall: {results['recall']:.3f}")
        
        # Display directory structure
        print(f"\\n📁 VALIDATION ARTIFACTS")
        print(f"   📊 Plots: {validator.plots_dir}")
        print(f"   📋 Console output: {validator.results_dir}/console_output.txt")
        print(f"   📄 Metrics: {validator.results_dir}/validation_results.json")

if __name__ == "__main__":
    validate_deep_tda_netflow()
