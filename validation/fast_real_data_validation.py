#!/usr/bin/env python3
"""
FAST REAL DATA TDA VALIDATION
==============================
Purpose: Test TDA methods on REAL CIC-IDS2017 data with manageable sample sizes
Data: Real infiltration attacks but sampled for computational efficiency
"""

#!/usr/bin/env python3
"""
FAST REAL DATA TDA VALIDATION
==============================
Purpose: Test TDA methods on REAL CIC-IDS2017 data with manageable sample sizes
Data: Real infiltration attacks but sampled for computational efficiency
"""

import sys
import os
sys.path.append('/home/stephen-dorman/dev/TDA_projects')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

from src.utils.data_loader import load_sampled_real_data
from src.utils.model_testing import run_detector_test
from src.utils.results_saver import save_validation_results

def main():
    """Run fast real data validation"""
    print("⚡ FAST REAL DATA TDA VALIDATION")
    print("=" * 50)
    print("Strategy: Use ALL real attacks, sample benign for speed")
    print("=" * 50)
    
    # Load sampled real data
    X, y = load_sampled_real_data(max_benign=5000)
    
    results = []
    
    # Test 1: Baseline APT Detector
    try:
        from src.cybersecurity.apt_detection import APTDetector
        result = run_detector_test(APTDetector, "Baseline APT Detector", X, y, verbose=False)
        results.append(result)
    except ImportError as e:
        print(f"❌ Cannot import baseline: {e}")
        results.append({'method': 'Baseline APT Detector', 'status': 'IMPORT_FAILED'})
    
    # Test 2: Improved APT Detector
    try:
        from src.cybersecurity.apt_detection_improved import ImprovedAPTDetector
        result = run_detector_test(
            ImprovedAPTDetector, "Improved APT Detector", X, y, 
            ensemble_size=2, verbose=False
        )
        results.append(result)
    except ImportError as e:
        print(f"❌ Cannot import improved: {e}")
        results.append({'method': 'Improved APT Detector', 'status': 'IMPORT_FAILED'})
    
    # Test 3: Enhanced APT Detector
    try:
        from src.cybersecurity.apt_detection_optimized import EnhancedAPTDetector
        result = run_detector_test(
            EnhancedAPTDetector, "Enhanced APT Detector", X, y, verbose=False
        )
        results.append(result)
    except ImportError as e:
        print(f"❌ Cannot import enhanced: {e}")
        results.append({'method': 'Enhanced APT Detector', 'status': 'IMPORT_FAILED'})
    
    # Results summary
    print("\n" + "=" * 60)
    print("🏆 FAST REAL DATA RESULTS")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']
    
    if successful:
        successful.sort(key=lambda x: x['f1_score'], reverse=True)
        
        print("\n✅ REAL DATA PERFORMANCE:")
        print(f"{'Method':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Attacks':<10} {'Time'}")
        print("-" * 70)
        
        for result in successful:
            method = result['method'][:24]  # Truncate if needed
            f1 = result['f1_score']
            precision = result['precision']
            recall = result['recall']
            attacks = f"{result['attacks_detected']}/{result['total_attacks']}"
            time_val = result['training_time']
            
            print(f"{method:<25} {f1:.3f}   {precision:.3f}     {recall:.3f}   {attacks:<10} {time_val:.1f}s")
        
        # Best performer
        best = successful[0]
        print(f"\n🏆 BEST ON REAL DATA: {best['method']}")
        print(f"   F1-Score: {best['f1_score']:.3f} ({best['f1_score']*100:.1f}%)")
        print(f"   Real attacks detected: {best['attacks_detected']}/{best['total_attacks']}")
        
        # Target check
        if best['f1_score'] >= 0.75:
            print("✅ Target achieved (F1 >= 0.75)!")
        else:
            gap = 0.75 - best['f1_score']
            print(f"⚠️ Gap to target: {gap:.3f} F1-score")
    
    if failed:
        print(f"\n❌ FAILED/UNAVAILABLE:")
        for result in failed:
            print(f"   - {result['method']}: {result['status']}")
    
    # Save results
    dataset_info = {
        'total_samples': len(X),
        'attack_samples': int(np.sum(y)),
        'attack_rate': float(np.mean(y)),
        'features': int(X.shape[1])
    }
    save_validation_results(results, "Fast Real Data Validation", dataset_info, 'fast_real_data_results.json')
    
    print("⚡ Fast validation complete!")
    
    return results

if __name__ == "__main__":
    results = main()


