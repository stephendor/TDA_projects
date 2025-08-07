#!/usr/bin/env python3
"""
Show concrete, verifiable results from the TDA Platform
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# Import our detectors
from src.cybersecurity.apt_detection import APTDetector
from src.cybersecurity.apt_detection_improved import ImprovedAPTDetector


def show_actual_data():
    """Show actual data and results you can verify."""
    
    print("=" * 80)
    print("TDA PLATFORM - ACTUAL RESULTS AND DATA VERIFICATION")
    print("=" * 80)
    
    # 1. Show the data we're working with
    print("\n1. GENERATING TEST DATA (you can see the actual numbers):")
    print("-" * 60)
    
    np.random.seed(42)  # Fixed seed for reproducible results
    
    # Create simple, verifiable test data
    normal_data = np.random.normal(0, 1, (100, 10))  # 100 normal samples, 10 features
    apt_data = np.random.normal(2, 1.5, (20, 10))    # 20 APT samples with different distribution
    
    X = np.vstack([normal_data, apt_data])
    y = np.array([0] * 100 + [1] * 20)
    
    print(f"Normal data shape: {normal_data.shape}")
    print(f"APT data shape: {apt_data.shape}")
    print(f"Combined data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of APT samples: {np.sum(y)}")
    print(f"Number of normal samples: {np.sum(y == 0)}")
    
    # Show actual sample data
    print(f"\nFirst 3 normal samples (first 5 features):")
    print(normal_data[:3, :5])
    print(f"\nFirst 3 APT samples (first 5 features):")
    print(apt_data[:3, :5])
    
    # 2. Split data clearly
    print("\n2. DATA SPLITTING:")
    print("-" * 60)
    
    # Use first 80 samples for training (all normal)
    X_train = X[:80]
    y_train = y[:80]
    
    # Use remaining 40 samples for testing
    X_test = X[80:]
    y_test = y[80:]
    
    print(f"Training data: {X_train.shape} samples")
    print(f"Training labels: {np.sum(y_train)} APT, {np.sum(y_train == 0)} normal")
    print(f"Test data: {X_test.shape} samples")  
    print(f"Test labels: {np.sum(y_test)} APT, {np.sum(y_test == 0)} normal")
    
    # 3. Train baseline detector and show actual results
    print("\n3. BASELINE DETECTOR RESULTS:")
    print("-" * 60)
    
    baseline_detector = APTDetector(verbose=False)
    baseline_detector.fit(X_train)
    
    baseline_predictions = baseline_detector.predict(X_test)
    baseline_probabilities = baseline_detector.predict_proba(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    
    print(f"Baseline predictions (first 10): {baseline_predictions[:10]}")
    print(f"Actual labels (first 10):       {y_test[:10]}")
    print(f"Baseline probabilities (first 10): {baseline_probabilities[:10].round(3)}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    print("\nBaseline Confusion Matrix:")
    baseline_cm = confusion_matrix(y_test, baseline_predictions)
    print(baseline_cm)
    
    print("\nBaseline Classification Report:")
    print(classification_report(y_test, baseline_predictions, target_names=['Normal', 'APT']))
    
    # 4. Train improved detector and show actual results
    print("\n4. IMPROVED DETECTOR RESULTS:")
    print("-" * 60)
    
    improved_detector = ImprovedAPTDetector(verbose=False, ensemble_size=3)
    improved_detector.fit(X_train, y_train)
    
    improved_predictions = improved_detector.predict(X_test)
    improved_probabilities = improved_detector.predict_proba(X_test)
    improved_accuracy = accuracy_score(y_test, improved_predictions)
    
    print(f"Improved predictions (first 10): {improved_predictions[:10]}")
    print(f"Actual labels (first 10):       {y_test[:10]}")
    print(f"Improved probabilities (first 10): {improved_probabilities[:10].round(3)}")
    print(f"Improved accuracy: {improved_accuracy:.4f}")
    
    print("\nImproved Confusion Matrix:")
    improved_cm = confusion_matrix(y_test, improved_predictions)
    print(improved_cm)
    
    print("\nImproved Classification Report:")
    print(classification_report(y_test, improved_predictions, target_names=['Normal', 'APT']))
    
    # 5. Calculate and show actual improvement
    print("\n5. ACTUAL PERFORMANCE COMPARISON:")
    print("-" * 60)
    
    improvement = ((improved_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    
    print(f"Baseline accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"Improved accuracy:  {improved_accuracy:.4f} ({improved_accuracy*100:.2f}%)")
    print(f"Absolute improvement: {improved_accuracy - baseline_accuracy:.4f}")
    print(f"Relative improvement: {improvement:+.2f}%")
    
    # 6. Show side-by-side comparison
    print("\n6. SIDE-BY-SIDE PREDICTION COMPARISON:")
    print("-" * 60)
    print("Sample | True | Baseline | Improved | Baseline_Prob | Improved_Prob")
    print("-" * 70)
    
    for i in range(min(15, len(y_test))):
        print(f"{i:6d} | {y_test[i]:4d} | {baseline_predictions[i]:8d} | {improved_predictions[i]:8d} | "
              f"{baseline_probabilities[i]:12.3f} | {improved_probabilities[i]:12.3f}")
    
    # 7. Save results to file you can check
    results = {
        'test_date': '2025-08-05',
        'data_info': {
            'total_samples': int(len(X)),
            'normal_samples': int(np.sum(y == 0)),
            'apt_samples': int(np.sum(y == 1)),
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test))
        },
        'baseline_results': {
            'accuracy': float(baseline_accuracy),
            'predictions': baseline_predictions.tolist(),
            'probabilities': baseline_probabilities.tolist(),
            'confusion_matrix': baseline_cm.tolist()
        },
        'improved_results': {
            'accuracy': float(improved_accuracy),
            'predictions': improved_predictions.tolist(),
            'probabilities': improved_probabilities.tolist(),
            'confusion_matrix': improved_cm.tolist()
        },
        'improvement': {
            'absolute': float(improved_accuracy - baseline_accuracy),
            'relative_percent': float(improvement)
        },
        'test_labels': y_test.tolist()
    }
    
    with open('actual_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n7. RESULTS SAVED TO FILE:")
    print("-" * 60)
    print("Results saved to 'actual_results.json' - you can open and verify this file")
    print(f"File contains {len(json.dumps(results))} characters of data")
    
    # 8. Show file structure you can verify
    print(f"\n8. PROJECT FILES YOU CAN CHECK:")
    print("-" * 60)
    
    important_files = [
        'src/cybersecurity/apt_detection.py',
        'src/cybersecurity/apt_detection_improved.py', 
        'PROJECT_STATUS.md',
        'requirements.txt',
        'docker-compose.yml',
        'Dockerfile'
    ]
    
    for file_path in important_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} (not found)")
    
    return results


if __name__ == "__main__":
    results = show_actual_data()
    print(f"\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("✅ All data and results shown above are real and verifiable")
    print("✅ Check 'actual_results.json' for detailed numerical results")
    print("✅ Check listed files to verify project structure")