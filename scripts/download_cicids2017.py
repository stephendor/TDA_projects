#!/usr/bin/env python3
"""
Download and prepare CIC-IDS2017 dataset for TDA analysis
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

def create_data_directory():
    """Create directory structure for datasets."""
    data_dir = Path("data/apt_datasets/cicids2017")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def download_with_progress(url, filename):
    """Download file with progress indication."""
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='')
        
        print(f"\n✅ Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def get_cicids2017_info():
    """Get information about CIC-IDS2017 dataset files."""
    
    # NOTE: These are the official dataset URLs and file information
    # The actual dataset needs to be downloaded from the official source
    
    dataset_info = {
        "base_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "description": "CIC-IDS2017 contains benign and the most up-to-date common attacks",
        "files": {
            "Monday-WorkingHours.pcap_ISCX.csv": {
                "description": "Monday normal traffic",
                "attacks": ["None - Benign traffic"],
                "size_mb": "~500MB"
            },
            "Tuesday-WorkingHours.pcap_ISCX.csv": {
                "description": "Tuesday traffic with attacks", 
                "attacks": ["SSH-Patator", "FTP-Patator"],
                "size_mb": "~450MB"
            },
            "Wednesday-workingHours.pcap_ISCX.csv": {
                "description": "Wednesday traffic with DoS attacks",
                "attacks": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"],
                "size_mb": "~440MB"  
            },
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": {
                "description": "Thursday morning web attacks",
                "attacks": ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"],
                "size_mb": "~170MB"
            },
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": {
                "description": "Thursday afternoon infiltration",
                "attacks": ["Infiltration"],
                "size_mb": "~520MB"
            },
            "Friday-WorkingHours-Morning.pcap_ISCX.csv": {
                "description": "Friday morning botnet attacks",
                "attacks": ["Bot"],
                "size_mb": "~190MB"
            },
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": {
                "description": "Friday afternoon port scanning",
                "attacks": ["PortScan"],
                "size_mb": "~50MB"
            },
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": {
                "description": "Friday afternoon DDoS attacks",
                "attacks": ["DDoS"],
                "size_mb": "~130MB"
            }
        },
        "features": [
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
            "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
            "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
            "Fwd IAT Total", "Fwd IAT Mean", "Bwd IAT Total", "Bwd IAT Mean",
            "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
            "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
            "Min Packet Length", "Max Packet Length", "Packet Length Mean",
            "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
            "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
            "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
            "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
            "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
            "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
            "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
            "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
            "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
            "Label"
        ],
        "total_flows": "~2.8 million flows",
        "attack_percentage": "~20% attack traffic"
    }
    
    return dataset_info

def print_dataset_info():
    """Print information about the CIC-IDS2017 dataset."""
    
    print("=" * 80)
    print("CIC-IDS2017 DATASET INFORMATION")
    print("=" * 80)
    
    info = get_cicids2017_info()
    
    print(f"\n📊 Dataset: CIC-IDS2017")
    print(f"📝 Description: {info['description']}")
    print(f"🌐 Source: {info['base_url']}")
    print(f"📈 Total Flows: {info['total_flows']}")
    print(f"🎯 Attack Ratio: {info['attack_percentage']}")
    
    print(f"\n📁 DATASET FILES:")
    print("-" * 60)
    
    for filename, file_info in info['files'].items():
        print(f"\n📄 {filename}")
        print(f"   Description: {file_info['description']}")
        print(f"   Size: {file_info['size_mb']}")
        print(f"   Attacks: {', '.join(file_info['attacks'])}")
    
    print(f"\n🔍 FEATURES ({len(info['features'])} total):")
    print("-" * 60)
    
    # Print features in columns for readability
    features = info['features']
    for i in range(0, len(features), 3):
        row_features = features[i:i+3]
        print("   ".join(f"{feat:<25}" for feat in row_features))
    
    print(f"\n💡 TDA APPLICATION OPPORTUNITIES:")
    print("-" * 60)
    print("✅ Temporal flow analysis - Flow Duration, IAT patterns")
    print("✅ Network topology evolution - Connection patterns over time") 
    print("✅ Multi-scale analysis - Different attack phases")
    print("✅ Persistent homology - Long-term attack campaigns")
    print("✅ Mapper analysis - Attack progression visualization")

def create_download_script():
    """Create script instructions for downloading the dataset."""
    
    data_dir = create_data_directory()
    
    script_content = """#!/bin/bash
# CIC-IDS2017 Dataset Download Script
# 
# NOTE: The CIC-IDS2017 dataset requires registration and manual download
# from the official source: https://www.unb.ca/cic/datasets/ids-2017.html
#
# Steps:
# 1. Visit https://www.unb.ca/cic/datasets/ids-2017.html  
# 2. Register and request access to the dataset
# 3. Download the CSV files to this directory: data/apt_datasets/cicids2017/
# 4. Run the preprocessing script: python scripts/preprocess_cicids2017.py

echo "CIC-IDS2017 Dataset Download Instructions"
echo "========================================"
echo ""
echo "The CIC-IDS2017 dataset requires manual download:"
echo "1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html"
echo "2. Register for access"
echo "3. Download CSV files to: data/apt_datasets/cicids2017/"
echo "4. Total download size: ~2.5GB"
echo ""
echo "Files to download:"
echo "- Monday-WorkingHours.pcap_ISCX.csv"
echo "- Tuesday-WorkingHours.pcap_ISCX.csv" 
echo "- Wednesday-workingHours.pcap_ISCX.csv"
echo "- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
echo "- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
echo "- Friday-WorkingHours-Morning.pcap_ISCX.csv"
echo "- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
echo "- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
echo ""
echo "After download, run: python scripts/preprocess_cicids2017.py"
"""
    
    script_path = data_dir / "download_instructions.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"\n📝 Download instructions saved to: {script_path}")
    return script_path

def main():
    """Main function to set up CIC-IDS2017 dataset acquisition."""
    
    print_dataset_info()
    
    # Create directory structure
    data_dir = create_data_directory()
    print(f"\n📁 Created data directory: {data_dir}")
    
    # Create download script
    script_path = create_download_script()
    
    # Create README
    readme_content = """# CIC-IDS2017 Dataset

## Overview
This directory contains the CIC-IDS2017 dataset for TDA-based APT detection research.

## Dataset Information
- **Source**: University of New Brunswick Canadian Institute for Cybersecurity
- **URL**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Total Size**: ~2.5GB
- **Format**: CSV files with network flow features
- **Flows**: ~2.8 million labeled network flows

## Files Structure
```
data/apt_datasets/cicids2017/
├── Monday-WorkingHours.pcap_ISCX.csv          # Benign traffic
├── Tuesday-WorkingHours.pcap_ISCX.csv         # SSH/FTP attacks  
├── Wednesday-workingHours.pcap_ISCX.csv       # DoS attacks
├── Thursday-*-WebAttacks.pcap_ISCX.csv        # Web attacks
├── Thursday-*-Infilteration.pcap_ISCX.csv     # Infiltration
├── Friday-*-Morning.pcap_ISCX.csv             # Botnet
├── Friday-*-PortScan.pcap_ISCX.csv            # Port scanning
└── Friday-*-DDos.pcap_ISCX.csv                # DDoS attacks
```

## Usage
1. Download dataset files manually (registration required)
2. Run preprocessing: `python scripts/preprocess_cicids2017.py`
3. Apply TDA analysis: `python scripts/tda_analysis_cicids2017.py`

## TDA Application
This dataset is ideal for TDA analysis because:
- Temporal flow patterns reveal attack progression
- Network topology changes during attacks
- Persistent patterns in long-term campaigns
- Multi-scale analysis opportunities (packet → flow → session → campaign)
"""
    
    readme_path = data_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📝 README created: {readme_path}")
    
    print(f"\n" + "=" * 80)
    print("DATASET SETUP COMPLETE")
    print("=" * 80)
    print(f"✅ Directory created: {data_dir}")
    print(f"✅ Download instructions: {script_path}")
    print(f"✅ Documentation: {readme_path}")
    print(f"\n🚀 Next step: Run the download script to get dataset access instructions")
    print(f"   Command: bash {script_path}")

if __name__ == "__main__":
    main()