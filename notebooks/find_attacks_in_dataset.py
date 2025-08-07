#!/usr/bin/env python3
"""
Find Attacks in CICIDS Dataset
==============================

Scanning the full dataset to locate actual attack samples.
"""

import pandas as pd
import numpy as np

def find_attacks_in_dataset():
    """Scan dataset to find where attacks are located"""
    
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    print("🔍 Scanning full dataset for attacks...")
    
    chunk_size = 50000
    total_rows = 0
    attack_found = False
    chunk_with_attacks = []
    
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size)):
        total_rows += len(chunk)
        
        # Check attack distribution
        attack_counts = chunk['Attack'].value_counts()
        
        if len(attack_counts) > 1 or 'Benign' not in attack_counts:  # Found non-benign
            attack_found = True
            print(f"\n✓ FOUND ATTACKS IN CHUNK {i+1} (rows {i*chunk_size+1}-{(i+1)*chunk_size}):")
            for attack, count in attack_counts.items():
                print(f"  {attack}: {count} ({count/len(chunk)*100:.1f}%)")
            
            chunk_with_attacks.append(i+1)
            
            # Save this chunk for detailed analysis
            if not attack_found:
                chunk.to_csv('/home/stephen-dorman/dev/TDA_projects/attack_chunk_sample.csv', index=False)
                print(f"  Saved sample to attack_chunk_sample.csv")
        else:
            print(f"Chunk {i+1}: {len(chunk)} rows - All Benign")
        
        # Stop after finding a few chunks with attacks
        if len(chunk_with_attacks) >= 3:
            break
        
        # Limit total scan
        if i >= 50:  # Don't scan forever
            break
    
    print(f"\nDataset scan complete:")
    print(f"Total rows scanned: {total_rows:,}")
    print(f"Attack chunks found: {chunk_with_attacks}")
    
    if not chunk_with_attacks:
        print("❌ No attacks found in scanned portion")
        print("Recommendation: Check dataset structure or scan more chunks")
    else:
        print(f"✓ Attacks found in {len(chunk_with_attacks)} chunks")
        print("Recommendation: Use these chunks for TDA analysis")
    
    return chunk_with_attacks

if __name__ == "__main__":
    find_attacks_in_dataset()
