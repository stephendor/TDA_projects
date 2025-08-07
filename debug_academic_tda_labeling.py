#!/usr/bin/env python3
"""
Debug Academic TDA Window Labeling
=================================

Investigating why all windows are being labeled as benign despite containing attacks.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_window_labeling():
    """Debug window labeling to understand attack detection"""
    
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    print("🔍 Investigating window labeling logic...")
    
    chunk_size = 10000
    window_size_ms = 5 * 60 * 1000  # 5 minutes
    
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size)):
        print(f"\nChunk {i+1}:")
        
        # Convert numeric columns
        numeric_cols = [col for col in chunk.columns 
                       if col not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack']]
        
        for col in numeric_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk.fillna(0)
        
        # Check attack distribution
        print(f"Total rows: {len(chunk)}")
        print(f"Attack distribution:")
        attack_counts = chunk['Attack'].value_counts()
        for attack, count in attack_counts.items():
            print(f"  {attack}: {count} ({count/len(chunk)*100:.1f}%)")
        
        # Focus on SSH-Bruteforce
        ssh_data = chunk[chunk['Attack'] == 'SSH-Bruteforce']
        benign_data = chunk[chunk['Attack'] == 'Benign']
        
        if len(ssh_data) > 0:
            print(f"\nSSH-Bruteforce analysis:")
            print(f"SSH-Bruteforce flows: {len(ssh_data)}")
            print(f"Benign flows: {len(benign_data)}")
            
            # Time analysis
            if 'FLOW_START_MILLISECONDS' in chunk.columns:
                combined_data = pd.concat([ssh_data, benign_data[:len(ssh_data)*2]])  # Include some benign
                combined_data = combined_data.sort_values('FLOW_START_MILLISECONDS')
                
                print(f"Combined data (SSH + some benign): {len(combined_data)}")
                
                min_time = combined_data['FLOW_START_MILLISECONDS'].min()
                max_time = combined_data['FLOW_START_MILLISECONDS'].max()
                print(f"Time range: {min_time} to {max_time} ({max_time - min_time} ms total)")
                
                # Create sample windows
                current_time = min_time
                window_count = 0
                for j in range(5):  # Test first 5 windows
                    window_end = current_time + window_size_ms
                    window_data = combined_data[
                        (combined_data['FLOW_START_MILLISECONDS'] >= current_time) &
                        (combined_data['FLOW_START_MILLISECONDS'] < window_end)
                    ]
                    
                    if len(window_data) > 0:
                        window_count += 1
                        n_attack = len(window_data[window_data['Attack'] == 'SSH-Bruteforce'])
                        n_benign = len(window_data[window_data['Attack'] == 'Benign'])
                        attack_ratio = n_attack / len(window_data)
                        
                        print(f"  Window {j+1}: {len(window_data)} flows ({n_attack} attack, {n_benign} benign)")
                        print(f"    Attack ratio: {attack_ratio:.3f}")
                        print(f"    Would be labeled as: {'attack' if attack_ratio > 0.1 else 'benign'}")
                        
                        if attack_ratio > 0:
                            print(f"    ✓ Found attack window with ratio {attack_ratio:.3f}")
                    
                    current_time += window_size_ms // 2  # 50% overlap
                
                print(f"Total windows created: {window_count}")
                
            break  # Only process first chunk with SSH-Bruteforce
        
        if i >= 3:  # Limit search
            break
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("- Need to examine why attack_ratio is always 0")
    print("- May need to adjust attack_ratio threshold (currently 0.1)")
    print("- May need to ensure temporal overlap between attacks and benign")
    print("="*60)

if __name__ == "__main__":
    debug_window_labeling()
