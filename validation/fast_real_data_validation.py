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

def load_sampled_real_data(max_benign=5000):
    """Load real CIC-IDS2017 data but sample benign for speed"""
    print("🎯 LOADING SAMPLED REAL CIC-IDS2017 DATA")
    print("-" * 50)
    
    real_data_path = "data/apt_datasets/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    
    df = pd.read_csv(real_data_path)
    df.columns = df.columns.str.strip()
    
    print(f"Original dataset: {df.shape}")
    
    # Separate attacks and benign
    attacks = df[df['Label'] == 'Infiltration']
    benign = df[df['Label'] == 'BENIGN']
    
    print(f"Original attacks: {len(attacks)}")
    print(f"Original benign: {len(benign)}")
    
    # Keep ALL attacks, sample benign for speed
    benign_sampled = benign.sample(n=min(max_benign, len(benign)), random_state=42)
    
    # Combine
    df_sampled = pd.concat([attacks, benign_sampled])
    
    print(f"Sampled dataset: {df_sampled.shape}")
    print(f"Attack rate: {len(attacks)}/{len(df_sampled)} = {len(attacks)/len(df_sampled):.3%}")
    
    # Prepare features
    feature_cols = [col for col in df_sampled.columns if col != 'Label']
    X = df_sampled[feature_cols].select_dtypes(include=[np.number])
    
    # Clean data
    X = X.fillna(X.median())
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
    
    # Binary labels
    y = (df_sampled['Label'] == 'Infiltration').astype(int)
    
    print(f"Final features: {X.shape[1]} dimensions")
    print(f"Final samples: {len(X)} ({y.sum()} attacks)")
    
    return X.values, y.values

def test_detector_fast(detector_class, detector_name, X, y, **kwargs):
    """Fast detector test"""
    print(f"\n🧪 TESTING {detector_name}")
    print("-" * 40)
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)} samples ({y_train.sum()} attacks)")
        print(f"Test: {len(X_test)} samples ({y_test.sum()} attacks)")
        
        start_time = time.time()
        detector = detector_class(**kwargs)
        
        # Handle different training approaches
        if "baseline" in detector_name.lower():
            # Unsupervised - normal data only
            X_train_normal = X_train[y_train == 0]
            detector.fit(X_train_normal)
        else:
            # Supervised
            detector.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred = detector.predict(X_test)
        
        # Metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"✅ Results:")
        print(f"   F1: {f1:.3f} ({f1*100:.1f}%)")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   Time: {train_time:.1f}s")
        print(f"   Attacks found: {cm[1,1]}/{y_test.sum()}")
        print(f"   False positives: {cm[0,1]}")
        
        return {
            'method': detector_name,
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'training_time': train_time,
            'attacks_detected': int(cm[1,1]),
            'total_attacks': int(y_test.sum()),
            'false_positives': int(cm[0,1]),
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {'method': detector_name, 'status': 'FAILED', 'error': str(e)}

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
        result = test_detector_fast(APTDetector, "Baseline APT Detector", X, y, verbose=False)
        results.append(result)
    except ImportError as e:
        print(f"❌ Cannot import baseline: {e}")
        results.append({'method': 'Baseline APT Detector', 'status': 'IMPORT_FAILED'})
    
    # Test 2: Improved APT Detector
    try:
        from src.cybersecurity.apt_detection_improved import ImprovedAPTDetector
        result = test_detector_fast(
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
        result = test_detector_fast(
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
    import json
    output_file = 'fast_real_data_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'validation_type': 'Fast Real Data Validation',
            'dataset': 'CIC-IDS2017 Infiltration (sampled)',
            'results': results,
            'best_performer': successful[0] if successful else None
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("⚡ Fast validation complete!")
    
    return results

if __name__ == "__main__":
    results = main()
