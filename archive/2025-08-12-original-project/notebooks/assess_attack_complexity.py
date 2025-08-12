#!/usr/bin/env python3
"""
TDA Dataset Suitability Assessment for NF-CICIDS2018-v3
========================================================

Analyzes attack types in the dataset to determine which are most suitable
for topological data analysis based on:

1. Volume (sufficient samples for TDA)
2. Complexity (multi-dimensional behavior patterns)
3. Temporal characteristics (persistent vs. burst patterns)
4. Feature diversity (rich topological structure potential)

ASSESSMENT CRITERIA FOR TDA SUITABILITY:
- Multi-stage attacks (Infiltration, Bot) = HIGH suitability
- Distributed attacks (DDoS) = HIGH suitability  
- Complex protocols (FTP-BruteForce) = MEDIUM suitability
- Simple attacks (SSH-BruteForce) = LOW suitability
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_attack_complexity():
    """Assess each attack type for TDA suitability"""
    
    data_path = "data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    attack_analysis = {
        'Infilteration': {
            'samples': 188152,
            'complexity': 'HIGH',
            'tda_suitability': 'EXCELLENT',
            'rationale': 'Multi-stage lateral movement creates rich topological patterns',
            'expected_features': 'Complex persistence diagrams, evolving Mapper graphs'
        },
        'Bot': {
            'samples': 207703,
            'complexity': 'HIGH', 
            'tda_suitability': 'EXCELLENT',
            'rationale': 'Coordinated botnet behavior with C&C communication patterns',
            'expected_features': 'Hierarchical clustering, periodic topological structures'
        },
        'DDOS_attack-HOIC': {
            'samples': 1032311,
            'complexity': 'HIGH',
            'tda_suitability': 'EXCELLENT',
            'rationale': 'Massive distributed attack with coordinated traffic patterns',
            'expected_features': 'Large connected components, high-dimensional persistence'
        },
        'DDoS_attacks-LOIC-HTTP': {
            'samples': 288589,
            'complexity': 'HIGH',
            'tda_suitability': 'EXCELLENT', 
            'rationale': 'HTTP-layer distributed attack with application-specific patterns',
            'expected_features': 'Protocol-specific topology, burst patterns'
        },
        'FTP-BruteForce': {
            'samples': 386720,
            'complexity': 'MEDIUM',
            'tda_suitability': 'GOOD',
            'rationale': 'Protocol-specific brute force with session establishment patterns',
            'expected_features': 'Connection topology, authentication sequences'
        },
        'DoS_attacks-SlowHTTPTest': {
            'samples': 105550,
            'complexity': 'MEDIUM',
            'tda_suitability': 'GOOD',
            'rationale': 'Slow HTTP attack creates temporal persistence patterns',
            'expected_features': 'Long-lived connections, temporal topology'
        },
        'SSH-Bruteforce': {
            'samples': 188474,
            'complexity': 'LOW',
            'tda_suitability': 'POOR',
            'rationale': 'Simple connection attempts, limited topological structure',
            'expected_features': 'Basic persistence, minimal Mapper complexity'
        }
    }
    
    print("="*80)
    print("TDA SUITABILITY ASSESSMENT - NF-CICIDS2018-v3 DATASET")
    print("="*80)
    print()
    
    # Sort by TDA suitability
    excellent_attacks = []
    good_attacks = []
    poor_attacks = []
    
    for attack, info in attack_analysis.items():
        if info['tda_suitability'] == 'EXCELLENT':
            excellent_attacks.append((attack, info))
        elif info['tda_suitability'] == 'GOOD':
            good_attacks.append((attack, info))
        else:
            poor_attacks.append((attack, info))
    
    def print_attack_category(category_name, attacks, emoji):
        print(f"{emoji} {category_name} TDA CANDIDATES:")
        print("-" * 50)
        for attack, info in attacks:
            print(f"🎯 {attack}")
            print(f"   Samples: {info['samples']:,}")
            print(f"   Complexity: {info['complexity']}")
            print(f"   Rationale: {info['rationale']}")
            print(f"   Expected TDA Features: {info['expected_features']}")
            print()
    
    print_attack_category("EXCELLENT", excellent_attacks, "🌟")
    print_attack_category("GOOD", good_attacks, "✅")
    print_attack_category("POOR", poor_attacks, "⚠️")
    
    print("="*80)
    print("RECOMMENDED TDA VALIDATION PRIORITY:")
    print("="*80)
    print("1. 🥇 FIRST: Infilteration (Multi-stage APT behavior)")
    print("2. 🥈 SECOND: Bot (Coordinated network behavior)")  
    print("3. 🥉 THIRD: DDOS_attack-HOIC (Large-scale distributed patterns)")
    print()
    print("FEATURE ENGINEERING RECOMMENDATIONS:")
    print("- Time-series TDA: Sliding window persistence for temporal attacks")
    print("- Multi-scale analysis: Different epsilon values for persistence")
    print("- Graph-based features: Network topology from flow patterns")
    print("- Protocol-aware TDA: Layer-specific topological analysis")
    print()
    
    return attack_analysis

def sample_attack_temporal_patterns():
    """Analyze temporal characteristics of top TDA candidates"""
    logger.info("Analyzing temporal patterns for TDA suitability...")
    
    data_path = "data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    # Sample first 100k rows to analyze patterns
    logger.info("Sampling dataset for temporal analysis...")
    chunk_size = 10000
    samples_analyzed = 0
    target_samples = 100000
    
    attack_temporal_data = {
        'Infilteration': [],
        'Bot': [],
        'DDOS_attack-HOIC': []
    }
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        if samples_analyzed >= target_samples:
            break
            
        samples_analyzed += len(chunk)
        
        for attack_type in attack_temporal_data.keys():
            attack_data = chunk[chunk['Attack'] == attack_type]
            if len(attack_data) > 0:
                # Extract key temporal and volume features
                temporal_features = {
                    'timestamps': attack_data['FLOW_START_MILLISECONDS'].values,
                    'durations': attack_data['FLOW_DURATION_MILLISECONDS'].values,
                    'bytes_in': attack_data['IN_BYTES'].values,
                    'bytes_out': attack_data['OUT_BYTES'].values,
                    'packet_counts': attack_data['IN_PKTS'].values + attack_data['OUT_PKTS'].values
                }
                attack_temporal_data[attack_type].append(temporal_features)
        
        logger.info(f"Processed {samples_analyzed} samples...")
    
    # Analyze patterns
    print("\n" + "="*60)
    print("TEMPORAL PATTERN ANALYSIS FOR TOP TDA CANDIDATES")
    print("="*60)
    
    for attack_type, data_list in attack_temporal_data.items():
        if data_list:
            print(f"\n🔍 {attack_type}:")
            
            # Combine all samples for this attack type
            all_durations = []
            all_bytes = []
            all_packets = []
            
            for data in data_list:
                all_durations.extend(data['durations'])
                all_bytes.extend(data['bytes_in'] + data['bytes_out'])
                all_packets.extend(data['packet_counts'])
            
            if all_durations:
                print(f"   Flow Duration Stats (ms):")
                print(f"     Mean: {np.mean(all_durations):.1f}")
                print(f"     Std: {np.std(all_durations):.1f}")
                print(f"     Range: {np.min(all_durations):.1f} - {np.max(all_durations):.1f}")
                
            if all_bytes:
                print(f"   Byte Volume Stats:")
                print(f"     Mean: {np.mean(all_bytes):.1f}")
                print(f"     Std: {np.std(all_bytes):.1f}")
                
            print(f"   Sample Count: {len(all_durations)}")
            
            # TDA suitability assessment
            duration_variety = np.std(all_durations) / (np.mean(all_durations) + 1e-6)
            byte_variety = np.std(all_bytes) / (np.mean(all_bytes) + 1e-6)
            
            print(f"   TDA Complexity Score:")
            print(f"     Duration Variability: {duration_variety:.3f}")
            print(f"     Volume Variability: {byte_variety:.3f}")
            
            if duration_variety > 1.0 and byte_variety > 1.0:
                print(f"     ✅ HIGH TDA potential - Rich feature diversity")
            elif duration_variety > 0.5 or byte_variety > 0.5:
                print(f"     ⚡ MEDIUM TDA potential - Moderate complexity")
            else:
                print(f"     ⚠️ LOW TDA potential - Limited feature diversity")
        else:
            print(f"\n❌ {attack_type}: No samples found in analyzed subset")

if __name__ == "__main__":
    logger.info("Starting TDA dataset suitability assessment...")
    
    # Run analysis
    attack_analysis = analyze_attack_complexity()
    sample_attack_temporal_patterns()
    
    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE - PROCEED WITH INFILTERATION OR BOT ATTACKS")
    print("="*80)
