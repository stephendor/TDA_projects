#!/usr/bin/env python3
"""
Debug APT Detectors - Find out what's actually happening
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from src.cybersecurity.apt_detection import APTDetector
from src.cybersecurity.apt_detection_improved import ImprovedAPTDetector


def debug_detectors():
    """Debug both detectors to understand the actual performance."""
    
    print("=" * 80)
    print("DEBUGGING APT DETECTORS - FINDING THE ACTUAL TRUTH")
    print("=" * 80)
    
    # Create clearly separable test data
    np.random.seed(42)
    
    # Make the problem easier to see if detectors work
    print("\n1. CREATING CLEARLY SEPARABLE DATA:")
    print("-" * 50)
    
    # Normal traffic: centered around 0
    normal_data = np.random.normal(0, 0.5, (80, 10))
    
    # APT traffic: clearly different, centered around 3 
    apt_data = np.random.normal(3, 0.5, (20, 10))
    
    X_train_normal = normal_data[:60]  # Use first 60 for training
    X_test = np.vstack([normal_data[60:], apt_data])  # Test on remaining 20 normal + 20 APT
    y_test = np.array([0] * 20 + [1] * 20)
    
    print(f"Training data (normal only): {X_train_normal.shape}")
    print(f"Test data: {X_test.shape} (20 normal + 20 APT)")
    print(f"Normal mean: {np.mean(normal_data):.3f}, std: {np.std(normal_data):.3f}")
    print(f"APT mean: {np.mean(apt_data):.3f}, std: {np.std(apt_data):.3f}")
    print(f"Separation: {np.mean(apt_data) - np.mean(normal_data):.3f} standard deviations")
    
    # Test baseline detector
    print("\n2. TESTING BASELINE DETECTOR:")
    print("-" * 50)
    
    try:
        baseline_detector = APTDetector(verbose=False)
        baseline_detector.fit(X_train_normal)
        
        baseline_pred = baseline_detector.predict(X_test)
        baseline_proba = baseline_detector.predict_proba(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        
        print(f"✅ Baseline detector trained successfully")
        print(f"Baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
        print(f"Baseline predictions: {np.sum(baseline_pred == 0)} normal, {np.sum(baseline_pred == 1)} APT")
        print(f"Actual labels:       {np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} APT")
        
        # Show some sample predictions
        print("Sample predictions (first 10):")
        print(f"True:      {y_test[:10]}")
        print(f"Baseline:  {baseline_pred[:10]}")
        print(f"Prob:      {baseline_proba[:10].round(3)}")
        
    except Exception as e:
        print(f"❌ Baseline detector failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test improved detector
    print("\n3. TESTING IMPROVED DETECTOR:")
    print("-" * 50)
    
    try:
        improved_detector = ImprovedAPTDetector(
            ensemble_size=2,  # Smaller for debugging
            verbose=False
        )
        
        print("Fitting improved detector...")
        improved_detector.fit(X_train_normal, np.zeros(len(X_train_normal)))
        
        print("Making predictions...")
        improved_pred = improved_detector.predict(X_test)
        improved_proba = improved_detector.predict_proba(X_test)
        improved_accuracy = accuracy_score(y_test, improved_pred)
        
        print(f"✅ Improved detector trained successfully")
        print(f"Improved accuracy: {improved_accuracy:.4f} ({improved_accuracy*100:.1f}%)")
        print(f"Improved predictions: {np.sum(improved_pred == 0)} normal, {np.sum(improved_pred == 1)} APT")
        print(f"Actual labels:        {np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} APT")
        
        # Show some sample predictions
        print("Sample predictions (first 10):")
        print(f"True:      {y_test[:10]}")
        print(f"Improved:  {improved_pred[:10]}")
        print(f"Prob:      {improved_proba[:10].round(3)}")
        
        # Debug the adaptive threshold
        print(f"\nDEBUG INFO:")
        print(f"Probability range: {np.min(improved_proba):.3f} to {np.max(improved_proba):.3f}")
        print(f"Probability mean: {np.mean(improved_proba):.3f}")
        
        # Test what happens with different thresholds
        print(f"\nTHRESHOLD ANALYSIS:")
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            thresh_pred = (improved_proba >= thresh).astype(int)
            thresh_acc = accuracy_score(y_test, thresh_pred)
            print(f"Threshold {thresh}: {thresh_acc:.3f} accuracy, {np.sum(thresh_pred)} APT predictions")
        
    except Exception as e:
        print(f"❌ Improved detector failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Comparison
    print("\n4. DIRECT COMPARISON:")
    print("-" * 50)
    
    print(f"Baseline:  {baseline_accuracy:.4f} accuracy")
    print(f"Improved:  {improved_accuracy:.4f} accuracy") 
    
    if improved_accuracy > baseline_accuracy:
        improvement = ((improved_accuracy - baseline_accuracy) / baseline_accuracy) * 100
        print(f"✅ Improvement: +{improvement:.1f}%")
    else:
        degradation = ((baseline_accuracy - improved_accuracy) / baseline_accuracy) * 100
        print(f"❌ Degradation: -{degradation:.1f}%")
    
    # Show classification reports
    print(f"\nBASELINE CLASSIFICATION REPORT:")
    print(classification_report(y_test, baseline_pred, target_names=['Normal', 'APT']))
    
    print(f"\nIMPROVED CLASSIFICATION REPORT:")
    print(classification_report(y_test, improved_pred, target_names=['Normal', 'APT']))
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'improved_accuracy': improved_accuracy,
        'baseline_predictions': baseline_pred,
        'improved_predictions': improved_pred,
        'true_labels': y_test
    }


if __name__ == "__main__":
    print("Starting APT detector debugging...")
    results = debug_detectors()
    
    if results:
        print("\n" + "=" * 80)
        print("DEBUGGING COMPLETE")
        print("=" * 80)
        
        if results['improved_accuracy'] > results['baseline_accuracy']:
            print("✅ Improved detector is actually working better")
        else:
            print("❌ Improved detector is NOT working - needs major fixes")
            print("🔧 The 'improved' detector claims are INCORRECT")
    else:
        print("❌ Debugging failed - detectors have serious issues")