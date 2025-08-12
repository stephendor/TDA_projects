#!/usr/bin/env python3
import pandas as pd

# Load the infiltration file
file_path = 'data/apt_datasets/cicids2017/GeneratedLabelledFlows/TrafficLabelling /Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
df = pd.read_csv(file_path)

print('Total rows:', len(df))
print('\nLabel distribution:')
print(df[' Label'].value_counts())

attack_mask = df[' Label'] != 'BENIGN'
attack_pct = attack_mask.mean() * 100
print(f'\nAttack percentage: {attack_pct:.2f}%')

if attack_mask.any():
    attacks = df[attack_mask]
    print('\nAttack types found:')
    print(attacks[' Label'].value_counts())