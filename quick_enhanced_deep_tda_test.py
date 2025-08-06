#!/usr/bin/env python3
"""
Quick Enhanced Deep TDA Test
Fast validation of enhanced model on real data subset
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.algorithms.deep.enhanced_deep_tda import EnhancedDeepTDAAnalyzer

def quick_real_data_test():
    """Quick test with real CIC-IDS2017 infiltration data"""
    print("🚀 QUICK ENHANCED DEEP TDA TEST - REAL DATA")
    print("=" * 60)
    
    # Load real infiltration data (small subset)
    file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        print("Using dummy data for testing...")
        
        # Create realistic dummy data
        np.random.seed(42)
        n_samples = 2000
        
        # Create attack datasets
        attack_datasets = {}
        for attack_type in ['ddos', 'portscan']:
            data = np.random.randn(n_samples//2, 22)
            data = np.abs(data) * np.random.uniform(1, 100, (1, 22))
            
            feature_names = [
                'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
                'Packet Length Mean', 'Packet Length Std', 'FIN Flag Count', 'SYN Flag Count',
                'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
                'Average Packet Size', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
                'Fwd IAT Mean', 'Bwd IAT Mean', 'Active Mean', 'Idle Mean'
            ]
            
            df = pd.DataFrame(data, columns=feature_names)
            labels = ['BENIGN'] * (n_samples//4) + [attack_type.upper()] * (n_samples//4)
            np.random.shuffle(labels)
            df['Label'] = labels
            
            attack_datasets[attack_type] = df
            print(f"   Created {attack_type}: {len(df)} samples")
    
    else:
        print("✅ Loading real CIC-IDS2017 data...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Ensure we get both attacks and benign samples
        attacks = df[df['Label'] == 'Infiltration']
        benign = df[df['Label'] == 'BENIGN']
        
        print(f"   Total attacks found: {len(attacks)}")
        print(f"   Total benign found: {len(benign):,}")
        
        # Take all attacks + reasonable benign sample
        benign_sample = benign.sample(n=min(1500, len(benign)), random_state=42)
        df_sample = pd.concat([attacks, benign_sample])
        
        attack_datasets = {'infiltration': df_sample}
        print(f"   Final sample: {len(df_sample)} samples ({len(attacks)} attacks + {len(benign_sample)} benign)")
    
    # Initialize analyzer
    analyzer = EnhancedDeepTDAAnalyzer(
        input_dim=80,
        embed_dim=128,  # Smaller for speed
        num_layers=3,   # Fewer layers
        num_heads=4     # Fewer heads
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in analyzer.model.parameters()):,}")
    
    # Prepare data
    print("\n📊 Preparing training data...")
    X, y_binary, y_attack_type, y_phase = analyzer.prepare_multi_attack_data(
        attack_datasets, max_samples_per_attack=1000
    )
    
    # Quick training (few epochs)
    print("\n🎯 Quick training (5 epochs)...")
    start_time = time.time()
    
    best_f1 = analyzer.train(
        X=X,
        y_binary=y_binary,
        y_attack_type=y_attack_type, 
        y_phase=y_phase,
        epochs=5,  # Very few epochs for speed
        batch_size=32,
        learning_rate=1e-3
    )
    
    training_time = time.time() - start_time
    print(f"\n⏱️ Training completed in {training_time:.1f}s")
    
    # Evaluation
    print("\n🔬 Quick evaluation...")
    results = analyzer.evaluate_comprehensive(X, y_binary, y_attack_type, y_phase)
    
    print(f"\n📈 RESULTS:")
    print(f"   Binary F1: {results['binary_f1']:.3f}")
    print(f"   Attack Type F1: {results['attack_type_f1']:.3f}")
    print(f"   Kill Chain F1: {results['phase_f1']:.3f}")
    
    # Assessment
    baseline_f1 = 0.706
    if results['binary_f1'] >= 0.85:
        status = "🎉 BREAKTHROUGH ACHIEVED"
    elif results['binary_f1'] >= baseline_f1:
        status = "✅ IMPROVEMENT OVER BASELINE"
    elif results['binary_f1'] >= 0.60:
        status = "📈 REASONABLE PERFORMANCE"
    else:
        status = "🔧 NEEDS OPTIMIZATION"
    
    print(f"\n{status}")
    print(f"Baseline (70.6%) → Enhanced ({results['binary_f1']*100:.1f}%)")
    print(f"Improvement: {results['binary_f1'] - baseline_f1:+.3f}")
    
    return results

if __name__ == "__main__":
    try:
        results = quick_real_data_test()
        print("\n✅ Quick test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()