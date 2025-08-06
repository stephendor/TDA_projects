#!/usr/bin/env python3
"""
Test script for Enhanced Deep TDA with Kill Chain Awareness
Quick validation of model initialization and forward pass
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.algorithms.deep.enhanced_deep_tda import EnhancedDeepTDAAnalyzer

def test_enhanced_deep_tda_initialization():
    """Test model initialization"""
    print("🧪 TESTING ENHANCED DEEP TDA INITIALIZATION")
    print("-" * 60)
    
    try:
        # Initialize analyzer
        analyzer = EnhancedDeepTDAAnalyzer(
            input_dim=80,
            embed_dim=256,
            num_layers=6,
            num_heads=8
        )
        
        print("✅ Model initialized successfully")
        print(f"   Parameters: {sum(p.numel() for p in analyzer.model.parameters()):,}")
        print(f"   Device: {analyzer.device}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(analyzer):
    """Test forward pass with dummy data"""
    print("\n🚀 TESTING FORWARD PASS")
    print("-" * 60)
    
    try:
        # Create dummy batch
        batch_size = 4
        input_dim = 80
        
        # Generate realistic-looking network flow features
        dummy_data = torch.randn(batch_size, input_dim)
        
        # Add some realistic patterns
        dummy_data[:, 0] = torch.abs(dummy_data[:, 0]) * 1000  # Flow Duration
        dummy_data[:, 1] = torch.abs(dummy_data[:, 1]) * 100   # Packet count
        dummy_data[:, 2] = torch.abs(dummy_data[:, 2]) * 50    # Byte count
        
        dummy_data = dummy_data.to(analyzer.device)
        
        print(f"   Input shape: {dummy_data.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = analyzer.model(dummy_data)
            
            print(f"   Binary logits: {outputs['binary_logits'].shape}")
            print(f"   Attack type logits: {outputs['attack_type_logits'].shape}")
            print(f"   Phase logits: {outputs['phase_logits'].shape}")
            print(f"   Features: {outputs['features'].shape}")
            
            # Check for NaN/inf values
            for key, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    if torch.isnan(tensor).any():
                        print(f"   ⚠️ NaN values found in {key}")
                    if torch.isinf(tensor).any():
                        print(f"   ⚠️ Inf values found in {key}")
            
            print("✅ Forward pass completed successfully")
            return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_attack_data_prep():
    """Test multi-attack data preparation with dummy data"""
    print("\n📊 TESTING MULTI-ATTACK DATA PREPARATION")
    print("-" * 60)
    
    try:
        analyzer = EnhancedDeepTDAAnalyzer()
        
        # Create dummy attack datasets
        np.random.seed(42)
        
        attack_datasets = {}
        for attack_type in ['ddos', 'portscan', 'webattacks', 'infiltration']:
            # Generate realistic dummy data
            n_samples = 500
            n_features = 22  # Common features available
            
            # Feature data
            data = np.random.randn(n_samples, n_features)
            data = np.abs(data) * np.random.uniform(1, 100, (1, n_features))  # Make positive
            
            # Create DataFrame
            feature_names = [
                'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
                'Packet Length Mean', 'Packet Length Std', 'FIN Flag Count', 'SYN Flag Count',
                'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
                'Average Packet Size', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
                'Fwd IAT Mean', 'Bwd IAT Mean', 'Active Mean', 'Idle Mean'
            ]
            
            import pandas as pd
            df = pd.DataFrame(data, columns=feature_names)
            
            # Add labels (70% attacks, 30% benign)
            labels = ['BENIGN'] * int(n_samples * 0.3) + [attack_type.upper()] * int(n_samples * 0.7)
            np.random.shuffle(labels)
            df['Label'] = labels
            
            attack_datasets[attack_type] = df
            print(f"   Created {attack_type}: {len(df)} samples, {(df['Label'] != 'BENIGN').sum()} attacks")
        
        # Test data preparation
        X, y_binary, y_attack_type, y_phase = analyzer.prepare_multi_attack_data(
            attack_datasets, max_samples_per_attack=1000
        )
        
        print(f"   Final dataset: {X.shape}")
        print(f"   Binary labels: {y_binary.shape}, attack rate: {np.mean(y_binary)*100:.1f}%")
        print(f"   Attack type labels: {y_attack_type.shape}")
        print(f"   Phase labels: {y_phase.shape}")
        
        print("✅ Multi-attack data preparation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Multi-attack data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    print("🔬 ENHANCED DEEP TDA MODEL TESTING")
    print("=" * 70)
    print("Testing kill chain awareness and multi-attack capabilities")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Model initialization
    analyzer = test_enhanced_deep_tda_initialization()
    if analyzer:
        success_count += 1
        
        # Test 2: Forward pass
        if test_forward_pass(analyzer):
            success_count += 1
    
    # Test 3: Multi-attack data preparation (independent of model)
    if test_multi_attack_data_prep():
        success_count += 1
    
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED - Enhanced Deep TDA ready for training")
        print("   Next: Load real CIC-IDS2017 data and begin multi-attack training")
    elif success_count >= 2:
        print("🟡 MOSTLY SUCCESSFUL - Minor issues to resolve")
        print("   Enhanced Deep TDA core functionality working")
    else:
        print("❌ SIGNIFICANT ISSUES - Enhanced Deep TDA needs debugging")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)