#!/usr/bin/env python3
"""
Research publicly available APT datasets for TDA analysis
"""

print("=" * 80)
print("APT DATASET RESEARCH FOR TDA ANALYSIS")
print("=" * 80)

print("\n1. PUBLICLY AVAILABLE APT DATASETS:")
print("-" * 60)

datasets = {
    "DARPA Intrusion Detection": {
        "description": "Classic network intrusion dataset with labeled attacks",
        "url": "https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset",
        "pros": ["Well-established", "Labeled data", "Network flows"],
        "cons": ["Old (1998)", "Synthetic", "Not modern APTs"],
        "tda_applicability": "Medium - network flows can be analyzed temporally"
    },
    
    "CTU-13": {
        "description": "Botnet traffic dataset with normal and infected traffic",
        "url": "https://www.stratosphereips.org/datasets-ctu13",
        "pros": ["Real botnet traffic", "Multiple scenarios", "Network flows"],
        "cons": ["Botnets not APTs", "Limited APT-specific patterns"],
        "tda_applicability": "High - continuous traffic patterns ideal for TDA"
    },
    
    "UNSW-NB15": {
        "description": "Modern network dataset with contemporary attack vectors",
        "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "pros": ["Modern attacks", "Diverse attack types", "Well-documented"],
        "cons": ["Not specifically APT-focused"],
        "tda_applicability": "High - temporal patterns in network flows"
    },
    
    "CIC-IDS2017/2018": {
        "description": "Canadian Institute for Cybersecurity intrusion detection datasets",
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "pros": ["Recent", "Realistic traffic", "Multiple attack types"],
        "cons": ["Large files", "Processing intensive"],
        "tda_applicability": "Very High - detailed flow data perfect for TDA"
    },
    
    "CICIDS2019": {
        "description": "Updated IDS dataset with more contemporary attacks",
        "url": "https://www.unb.ca/cic/datasets/ids-2019.html", 
        "pros": ["Most recent", "Contemporary attacks", "Good documentation"],
        "cons": ["Very large", "Complex preprocessing needed"],
        "tda_applicability": "Very High - best temporal granularity"
    },
    
    "TON_IoT": {
        "description": "IoT and industrial network dataset with APT-like behaviors",
        "url": "https://research.unsw.edu.au/projects/toniot-datasets",
        "pros": ["IoT focus", "Multi-stage attacks", "Recent"],
        "cons": ["IoT-specific", "May not generalize"],
        "tda_applicability": "High - IoT patterns show persistence over time"
    },
    
    "EMBER": {
        "description": "Malware detection dataset (PE files)",
        "url": "https://github.com/elastic/ember",
        "pros": ["Large scale", "Real malware", "Good features"],
        "cons": ["Static analysis", "Not network-based"],
        "tda_applicability": "Low - not temporal network data"
    }
}

for name, info in datasets.items():
    print(f"\n📊 {name}")
    print(f"   Description: {info['description']}")
    print(f"   URL: {info['url']}")
    print(f"   Pros: {', '.join(info['pros'])}")
    print(f"   Cons: {', '.join(info['cons'])}")
    print(f"   TDA Applicability: {info['tda_applicability']}")

print("\n\n2. RECOMMENDED PRIORITY ORDER:")
print("-" * 60)

recommendations = [
    {
        "rank": 1,
        "dataset": "CIC-IDS2017/2018",
        "reason": "Best balance of realism, temporal data, and TDA applicability",
        "action": "Download and preprocess flow data for temporal TDA analysis"
    },
    {
        "rank": 2, 
        "dataset": "CTU-13",
        "reason": "Persistent botnet traffic patterns ideal for persistent homology",
        "action": "Focus on long-duration scenarios (1, 8, 13) with C&C patterns"
    },
    {
        "rank": 3,
        "dataset": "UNSW-NB15", 
        "reason": "Good variety of attack types, well-processed",
        "action": "Use for validating TDA approach across different attack types"
    }
]

for rec in recommendations:
    print(f"\n{rec['rank']}. {rec['dataset']}")
    print(f"   Reason: {rec['reason']}")
    print(f"   Action: {rec['action']}")

print("\n\n3. TDA APPLICATION STRATEGY:")
print("-" * 60)

print("Based on dataset characteristics, we should apply TDA to:")
print("\n📈 TEMPORAL ANALYSIS:")
print("   - Time series of connection counts, byte flows, packet rates")
print("   - Sliding window embeddings to capture attack phases")
print("   - Persistent homology to find long-term patterns")

print("\n🔗 NETWORK TOPOLOGY:")
print("   - Connection graphs between hosts")
print("   - Mapper algorithm on network structure evolution")
print("   - Topological changes during attack progression")

print("\n🎯 MULTI-SCALE PATTERNS:")
print("   - Short-term: Individual session patterns")
print("   - Medium-term: Daily/weekly behavioral changes") 
print("   - Long-term: Persistent infrastructure patterns")

print("\n\n4. DATA PREPROCESSING STRATEGY:")
print("-" * 60)

preprocessing_steps = [
    "1. Extract temporal flow sequences (src, dst, bytes, packets, duration)",
    "2. Create time-windowed feature vectors (1min, 5min, 1hr windows)",
    "3. Build connection matrices for graph topology analysis",
    "4. Generate sliding window embeddings for persistent homology",
    "5. Label sequences based on ground truth attack timings"
]

for step in preprocessing_steps:
    print(step)

print("\n\n5. EXPECTED TDA INSIGHTS:")
print("-" * 60)

insights = [
    "🔍 Reconnaissance patterns: Topology of scanning behaviors",
    "🚪 Initial compromise: Persistent connections to C&C servers", 
    "📊 Lateral movement: Evolution of internal network topology",
    "📤 Data exfiltration: Persistent large-volume flows",
    "⏱️ Temporal persistence: Attack campaigns spanning days/weeks"
]

for insight in insights:
    print(insight)

print("\n\n6. IMPLEMENTATION PLAN:")
print("-" * 60)

plan = [
    "Phase 1: Download and explore CIC-IDS2017 dataset",
    "Phase 2: Implement temporal feature extraction pipeline",
    "Phase 3: Apply persistent homology to time series data",
    "Phase 4: Test Mapper algorithm on network topology evolution",
    "Phase 5: Validate against known APT attack timelines",
    "Phase 6: Compare with baseline non-TDA methods"
]

for i, phase in enumerate(plan, 1):
    print(f"{i}. {phase}")

print("\n" + "=" * 80)
print("RESEARCH COMPLETE - READY TO ACQUIRE REAL DATA")
print("=" * 80)