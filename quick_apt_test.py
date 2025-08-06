#!/usr/bin/env python3
"""
Quick test for Enhanced APT Detector performance validation.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import warnings

# Import both detectors
from src.cybersecurity.apt_detection import APTDetector
from src.cybersecurity.apt_detection_optimized import EnhancedAPTDetector

warnings.filterwarnings('ignore')


def generate_simple_network_data(n_normal=200, n_apt=50, seed=42):
    """Generate simple synthetic network data for quick testing."""
    np.random.seed(seed)
    
    # Normal traffic patterns - simple gaussian
    normal_data = np.random.normal(0, 1, (n_normal, 20))
    
    # APT patterns - slightly different distribution
    apt_data = np.random.normal(1.5, 0.8, (n_apt, 20))
    
    # Combine data
    X = np.vstack([normal_data, apt_data])
    y = np.array([0] * n_normal + [1] * n_apt)
    
    return X, y


def quick_test():
    """Quick performance test."""
    print("Quick Enhanced APT Detector Test")
    print("=" * 40)
    
    # Generate test data
    X, y = generate_simple_network_data(n_normal=150, n_apt=30)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Test baseline detector
    print("\nTesting Baseline Detector...")
    baseline = APTDetector(verbose=False)
    
    start_time = time.time()
    baseline.fit(X_train)
    baseline_train_time = time.time() - start_time
    
    start_time = time.time()
    baseline_pred = baseline.predict(X_test)
    baseline_predict_time = time.time() - start_time
    
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(
        y_test, baseline_pred, average='binary'
    )
    
    print(f"Baseline Results:")
    print(f"  Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"  Precision: {baseline_precision:.4f}")
    print(f"  Recall: {baseline_recall:.4f}")
    print(f"  F1: {baseline_f1:.4f}")
    print(f"  Train time: {baseline_train_time:.2f}s")
    print(f"  Predict time: {baseline_predict_time:.4f}s")
    
    # Test enhanced detector with simplified parameters
    print("\nTesting Enhanced Detector...")
    enhanced = EnhancedAPTDetector(
        multiscale_windows=[5, 10],  # Reduced complexity
        ensemble_size=3,              # Smaller ensemble
        verbose=False
    )
    
    start_time = time.time()
    enhanced.fit(X_train, y_train)
    enhanced_train_time = time.time() - start_time
    
    start_time = time.time()
    enhanced_pred = enhanced.predict(X_test)
    enhanced_predict_time = time.time() - start_time
    
    enhanced_accuracy = accuracy_score(y_test, enhanced_pred)
    enhanced_precision, enhanced_recall, enhanced_f1, _ = precision_recall_fscore_support(
        y_test, enhanced_pred, average='binary'
    )
    
    print(f"Enhanced Results:")
    print(f"  Accuracy: {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
    print(f"  Precision: {enhanced_precision:.4f}")
    print(f"  Recall: {enhanced_recall:.4f}")
    print(f"  F1: {enhanced_f1:.4f}")
    print(f"  Train time: {enhanced_train_time:.2f}s")
    print(f"  Predict time: {enhanced_predict_time:.4f}s")
    
    # Comparison
    print(f"\n{'='*40}")
    print("COMPARISON")
    print(f"{'='*40}")
    
    acc_improvement = ((enhanced_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    precision_improvement = ((enhanced_precision - baseline_precision) / baseline_precision) * 100 if baseline_precision > 0 else 0
    recall_improvement = ((enhanced_recall - baseline_recall) / baseline_recall) * 100 if baseline_recall > 0 else 0
    
    print(f"Accuracy improvement: {acc_improvement:+.2f}%")
    print(f"Precision improvement: {precision_improvement:+.2f}%")
    print(f"Recall improvement: {recall_improvement:+.2f}%")
    
    # Check if target achieved
    target = 0.95
    print(f"\nTarget Achievement:")
    print(f"Target accuracy: {target*100:.1f}%")
    print(f"Enhanced accuracy: {enhanced_accuracy*100:.2f}%")
    
    if enhanced_accuracy >= target:
        print("✅ TARGET ACHIEVED!")
    else:
        gap = (target - enhanced_accuracy) * 100
        print(f"❌ Gap to target: {gap:.2f} percentage points")
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'enhanced_accuracy': enhanced_accuracy,
        'improvement': acc_improvement,
        'target_achieved': enhanced_accuracy >= target,
        'baseline_f1': baseline_f1,
        'enhanced_f1': enhanced_f1
    }


if __name__ == "__main__":
    results = quick_test()
    print(f"\nTest completed. Enhanced detector accuracy: {results['enhanced_accuracy']*100:.2f}%")