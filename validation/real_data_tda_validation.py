#!/usr/bin/env python3
"""
REAL DATA TDA VALIDATION - NO SYNTHETIC FALLBACKS
===================================================
Purpose: Test TDA methods ONLY on real CIC-IDS2017 infiltration attacks
Data: 36 real infiltration attacks vs 288,566 benign samples
Focus: Rigorous validation on challenging real-world APT detection
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

from src.utils.data_loader import load_real_cicids_infiltration

def test_baseline_apt_detector(X, y):
    """Test the baseline APT detector on real data"""
    print("\\n🔬 TESTING BASELINE APT DETECTOR")
    print("-" * 50)
    
    try:
        from src.cybersecurity.apt_detection import APTDetector
        
        # Split data stratified to ensure attacks in both train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples ({y_train.sum()} attacks)")
        print(f"Test set: {len(X_test)} samples ({y_test.sum()} attacks)")
        
        # Baseline is unsupervised - train only on normal data
        X_train_normal = X_train[y_train == 0]
        print(f"Training on {len(X_train_normal)} normal samples...")
        
        start_time = time.time()
        detector = APTDetector(verbose=False)
        detector.fit(X_train_normal)
        train_time = time.time() - start_time
        
        # Predict on test set
        y_pred = detector.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\\n✅ BASELINE APT DETECTOR RESULTS:")
        print(f"   F1-Score: {f1:.4f} ({f1*100:.1f}%)")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f"   Recall: {recall:.4f} ({recall*100:.1f}%)")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Confusion Matrix:")
        print(f"     [[TN={cm[0,0]:4d}, FP={cm[0,1]:2d}]]")
        print(f"     [[FN={cm[1,0]:4d}, TP={cm[1,1]:2d}]]")
        
        return {
            'method': 'Baseline APT Detector',
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'training_time': train_time,
            'attacks_detected': cm[1,1],
            'total_attacks': y_test.sum(),
            'false_positives': cm[0,1],
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"❌ Baseline APT Detector FAILED: {e}")
        return {'method': 'Baseline APT Detector', 'status': 'FAILED', 'error': str(e)}

def test_improved_apt_detector(X, y):
    """Test the improved APT detector on real data"""
    print("\\n🔬 TESTING IMPROVED APT DETECTOR")
    print("-" * 50)
    
    try:
        from src.cybersecurity.apt_detection_improved import ImprovedAPTDetector
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples ({y_train.sum()} attacks)")
        print(f"Test set: {len(X_test)} samples ({y_test.sum()} attacks)")
        
        start_time = time.time()
        # Use supervised approach with reduced ensemble for speed
        detector = ImprovedAPTDetector(ensemble_size=2, verbose=False)
        detector.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = detector.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\\n✅ IMPROVED APT DETECTOR RESULTS:")
        print(f"   F1-Score: {f1:.4f} ({f1*100:.1f}%)")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f"   Recall: {recall:.4f} ({recall*100:.1f}%)")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Confusion Matrix:")
        print(f"     [[TN={cm[0,0]:4d}, FP={cm[0,1]:2d}]]")
        print(f"     [[FN={cm[1,0]:4d}, TP={cm[1,1]:2d}]]")
        
        return {
            'method': 'Improved APT Detector',
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'training_time': train_time,
            'attacks_detected': cm[1,1],
            'total_attacks': y_test.sum(),
            'false_positives': cm[0,1],
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"❌ Improved APT Detector FAILED: {e}")
        return {'method': 'Improved APT Detector', 'status': 'FAILED', 'error': str(e)}

def test_enhanced_apt_detector(X, y):
    """Test the enhanced APT detector on real data"""
    print("\\n🔬 TESTING ENHANCED APT DETECTOR")
    print("-" * 50)
    
    try:
        from src.cybersecurity.apt_detection_optimized import EnhancedAPTDetector
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples ({y_train.sum()} attacks)")
        print(f"Test set: {len(X_test)} samples ({y_test.sum()} attacks)")
        
        start_time = time.time()
        detector = EnhancedAPTDetector(verbose=False)
        detector.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = detector.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\\n✅ ENHANCED APT DETECTOR RESULTS:")
        print(f"   F1-Score: {f1:.4f} ({f1*100:.1f}%)")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f"   Recall: {recall:.4f} ({recall*100:.1f}%)")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Confusion Matrix:")
        print(f"     [[TN={cm[0,0]:4d}, FP={cm[0,1]:2d}]]")
        print(f"     [[FN={cm[1,0]:4d}, TP={cm[1,1]:2d}]]")
        
        return {
            'method': 'Enhanced APT Detector',
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'training_time': train_time,
            'attacks_detected': cm[1,1],
            'total_attacks': y_test.sum(),
            'false_positives': cm[0,1],
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        print(f"❌ Enhanced APT Detector FAILED: {e}")
        return {'method': 'Enhanced APT Detector', 'status': 'FAILED', 'error': str(e)}

def main():
    """Run comprehensive real data validation"""
    print("🎯 REAL DATA TDA VALIDATION - CIC-IDS2017 INFILTRATION")
    print("=" * 70)
    print("Dataset: 36 real infiltration attacks vs 288,566 benign samples")
    print("Challenge: Detect rare APT attacks in realistic network traffic")
    print("=" * 70)
    
    # Load ONLY real data
    X, y = load_real_cicids_infiltration()
    
    # Test all available detectors
    results = []
    
    # Test 1: Baseline
    result1 = test_baseline_apt_detector(X, y)
    results.append(result1)
    
    # Test 2: Improved
    result2 = test_improved_apt_detector(X, y)
    results.append(result2)
    
    # Test 3: Enhanced
    result3 = test_enhanced_apt_detector(X, y)
    results.append(result3)
    
    # Display final results
    print("\\n" + "=" * 70)
    print("🏆 REAL DATA VALIDATION RESULTS - CIC-IDS2017 INFILTRATION")
    print("=" * 70)
    
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    failed_results = [r for r in results if r['status'] == 'FAILED']
    
    if successful_results:
        # Sort by F1-score
        successful_results.sort(key=lambda x: x['f1_score'], reverse=True)
        
        print(f"\\n✅ SUCCESSFUL TESTS ON REAL INFILTRATION DATA:")
        print(f"{'Rank':<4} {'Method':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Attacks Found':<13} {'Time':<6}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results, 1):
            method = result['method']
            f1 = result['f1_score']
            precision = result['precision']
            recall = result['recall']
            attacks_found = f"{result['attacks_detected']}/{result['total_attacks']}"
            time_val = result['training_time']
            
            print(f"{i:<4} {method:<25} {f1:.3f}   {precision:.3f}     {recall:.3f}   {attacks_found:<13} {time_val:.1f}s")
        
        # Show best performer
        best = successful_results[0]
        print(f"\\n🏆 BEST REAL DATA PERFORMANCE:")
        print(f"   Method: {best['method']}")
        print(f"   F1-Score: {best['f1_score']:.3f} ({best['f1_score']*100:.1f}%)")
        print(f"   Infiltration Attacks Detected: {best['attacks_detected']}/{best['total_attacks']}")
        print(f"   False Positives: {best['false_positives']}")
        print(f"   Training Time: {best['training_time']:.2f}s")
        
        # Target assessment
        target_f1 = 0.75
        best_f1 = best['f1_score']
        if best_f1 >= target_f1:
            print(f"\\n🎯 TARGET ACHIEVED! ({best_f1:.3f} >= {target_f1:.3f})")
        else:
            gap = target_f1 - best_f1
            print(f"\\n⚠️  Gap to target: {gap:.3f} F1-score ({gap/target_f1*100:.1f}%)")
    
    if failed_results:
        print(f"\\n❌ FAILED TESTS:")
        for result in failed_results:
            print(f"   - {result['method']}: {result.get('error', 'Unknown error')}")
    
    # Save results
    import json
    timestamp = pd.Timestamp.now().isoformat()
    
    validation_summary = {
        'validation_date': timestamp,
        'validation_type': 'Real CIC-IDS2017 Infiltration Data',
        'dataset_info': {
            'total_samples': len(X),
            'attack_samples': int(np.sum(y)),
            'attack_rate': float(np.mean(y)),
            'features': int(X.shape[1])
        },
        'results': results,
        'best_performer': successful_results[0] if successful_results else None,
        'target_f1': 0.75,
        'target_achieved': successful_results[0]['f1_score'] >= 0.75 if successful_results else False
    }
    
    output_file = 'real_data_tda_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to: {output_file}")
    print("🎉 Real data TDA validation complete!")
    
    return results

if __name__ == "__main__":
    results = main()
