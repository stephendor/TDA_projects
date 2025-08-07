#!/usr/bin/env python3
"""
Complete Dataset Attack Analysis
===============================

Efficiently scan entire dataset to map all attack types and their locations.
This is the RIGHT way to approach this problem.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def complete_dataset_analysis():
    """Scan entire dataset efficiently to map all attacks"""
    
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    
    print("🔍 COMPLETE DATASET ATTACK ANALYSIS")
    print("=" * 60)
    
    # Efficient scan - just read Attack column and row indices
    chunk_size = 100000
    total_rows = 0
    attack_distribution = Counter()
    attack_locations = defaultdict(list)  # attack_type -> list of (chunk_num, rows_in_chunk)
    
    print("Scanning entire dataset for attack distribution...")
    
    for chunk_num, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size, usecols=['Attack'])):
        chunk_attacks = chunk['Attack'].value_counts()
        
        # Update global distribution
        for attack, count in chunk_attacks.items():
            attack_distribution[attack] += count
            attack_locations[attack].append((chunk_num, count))
        
        total_rows += len(chunk)
        
        # Progress update
        if chunk_num % 10 == 0:
            print(f"  Processed chunk {chunk_num+1}: {total_rows:,} rows")
    
    print(f"\n✅ COMPLETE DATASET ANALYSIS:")
    print(f"Total rows: {total_rows:,}")
    print(f"Unique attack types: {len(attack_distribution)}")
    
    print(f"\n📊 ATTACK DISTRIBUTION:")
    for attack, count in attack_distribution.most_common():
        percentage = (count / total_rows) * 100
        print(f"  {attack:20s}: {count:8,} ({percentage:5.2f}%)")
    
    print(f"\n📍 ATTACK LOCATIONS BY CHUNK:")
    attack_types = [attack for attack in attack_distribution.keys() if attack != 'Benign']
    
    for attack in sorted(attack_types):
        chunks_with_attack = attack_locations[attack]
        total_attack_count = sum(count for _, count in chunks_with_attack)
        chunk_numbers = [chunk_num for chunk_num, _ in chunks_with_attack]
        
        print(f"\n  {attack}:")
        print(f"    Total samples: {total_attack_count:,}")
        print(f"    Found in chunks: {min(chunk_numbers)+1}-{max(chunk_numbers)+1} ({len(chunks_with_attack)} chunks)")
        print(f"    Chunk range: {chunk_numbers[:5]}{'...' if len(chunk_numbers) > 5 else ''}")
    
    # Create efficient loading strategy
    print(f"\n💡 EFFICIENT LOADING STRATEGY:")
    print("For TDA analysis, recommend:")
    
    # Find chunks with good attack/benign mix
    mixed_chunks = []
    max_chunk = max(max(chunk_num for chunk_num, _ in locations) for locations in attack_locations.values() if locations)
    for chunk_num in range(max_chunk + 1):
        chunk_attacks = []
        chunk_benign = 0
        
        for attack in attack_types:
            for loc_chunk, count in attack_locations[attack]:
                if loc_chunk == chunk_num:
                    chunk_attacks.append((attack, count))
        
        for loc_chunk, count in attack_locations['Benign']:
            if loc_chunk == chunk_num:
                chunk_benign = count
        
        if chunk_attacks and chunk_benign > 0:
            total_attacks = sum(count for _, count in chunk_attacks)
            attack_ratio = total_attacks / (total_attacks + chunk_benign)
            if 0.01 <= attack_ratio <= 0.5:  # Good mix
                mixed_chunks.append((chunk_num, attack_ratio, chunk_attacks))
    
    mixed_chunks.sort(key=lambda x: x[1], reverse=True)  # Sort by attack ratio
    
    print(f"  Found {len(mixed_chunks)} chunks with good attack/benign mix:")
    for i, (chunk_num, ratio, attacks) in enumerate(mixed_chunks[:10]):
        attack_names = [name for name, _ in attacks]
        print(f"    Chunk {chunk_num+1}: {ratio:.1%} attacks ({', '.join(attack_names)})")
    
    # Save attack location map for efficient loading
    attack_map = {
        'total_rows': total_rows,
        'attack_distribution': dict(attack_distribution),
        'attack_locations': {k: v for k, v in attack_locations.items()},
        'recommended_chunks': mixed_chunks[:20],  # Top 20 mixed chunks
        'chunk_size': chunk_size
    }
    
    # Save as JSON for quick loading
    import json
    with open('/home/stephen-dorman/dev/TDA_projects/attack_location_map.json', 'w') as f:
        json.dump(attack_map, f, indent=2)
    
    print(f"\n💾 Saved attack location map to attack_location_map.json")
    print(f"   Use this for targeted data loading in TDA analysis")
    
    return attack_map

def load_targeted_attack_data(attack_types=None, max_samples_per_type=5000):
    """Load specific attacks efficiently using the attack map"""
    
    if attack_types is None:
        attack_types = ['SSH-Bruteforce', 'Bot', 'FTP-BruteForce']
    
    # Load attack map
    import json
    try:
        with open('/home/stephen-dorman/dev/TDA_projects/attack_location_map.json', 'r') as f:
            attack_map = json.load(f)
    except FileNotFoundError:
        print("❌ Attack map not found. Run complete_dataset_analysis() first.")
        return None
    
    data_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/NF-CICIDS2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
    chunk_size = attack_map['chunk_size']
    
    print(f"🎯 TARGETED ATTACK DATA LOADING")
    print(f"Target attacks: {attack_types}")
    print(f"Max samples per type: {max_samples_per_type}")
    
    collected_data = {attack: [] for attack in attack_types}
    collected_data['Benign'] = []
    
    # Get chunks that contain our target attacks
    target_chunks = set()
    for attack in attack_types:
        if attack in attack_map['attack_locations']:
            for chunk_num, _ in attack_map['attack_locations'][attack]:
                target_chunks.add(chunk_num)
    
    target_chunks = sorted(target_chunks)
    print(f"Will scan {len(target_chunks)} chunks containing target attacks")
    
    # Load only the chunks we need
    for chunk_num in target_chunks:
        # Calculate skip rows and nrows for this chunk
        skip_rows = chunk_num * chunk_size + 1  # +1 for header
        
        try:
            chunk = pd.read_csv(data_path, skiprows=skip_rows, nrows=chunk_size, 
                              names=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'PROTOCOL', 'L4_SRC_PORT', 'L4_DST_PORT',
                                   'FLOW_DURATION_MILLISECONDS', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
                                   'IN_PKTS', 'IN_BYTES', 'OUT_PKTS', 'OUT_BYTES', 'Label', 'Attack'])
            
            # Process each target attack type
            for attack in attack_types:
                attack_data = chunk[chunk['Attack'] == attack]
                if len(attack_data) > 0:
                    # Limit samples per attack type
                    remaining = max_samples_per_type - len(collected_data[attack])
                    if remaining > 0:
                        collected_data[attack].append(attack_data.head(remaining))
            
            # Also collect some benign data
            benign_data = chunk[chunk['Attack'] == 'Benign']
            if len(benign_data) > 0:
                remaining = max_samples_per_type - len(collected_data['Benign'])
                if remaining > 0:
                    collected_data['Benign'].append(benign_data.head(remaining))
            
        except Exception as e:
            print(f"⚠️ Error reading chunk {chunk_num}: {e}")
            continue
        
        # Check if we have enough data
        all_sufficient = all(
            len(collected_data[attack]) >= max_samples_per_type 
            for attack in attack_types + ['Benign']
            if collected_data[attack]
        )
        if all_sufficient:
            break
    
    # Combine collected data
    final_data = {}
    for attack in attack_types + ['Benign']:
        if collected_data[attack]:
            final_data[attack] = pd.concat(collected_data[attack], ignore_index=True)
            print(f"✅ {attack}: {len(final_data[attack]):,} samples")
        else:
            print(f"❌ {attack}: No samples found")
    
    return final_data

if __name__ == "__main__":
    # First, analyze the complete dataset
    attack_map = complete_dataset_analysis()
    
    print(f"\n" + "="*60)
    print("TESTING TARGETED LOADING:")
    
    # Then test targeted loading
    target_data = load_targeted_attack_data(['SSH-Bruteforce', 'Bot', 'FTP-BruteForce'], max_samples_per_type=1000)
    
    if target_data:
        print(f"\n✅ Successfully loaded targeted attack data:")
        for attack, data in target_data.items():
            print(f"   {attack}: {len(data)} samples")
    else:
        print(f"❌ Failed to load targeted data")
