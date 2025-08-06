#!/usr/bin/env python3
"""
Test script for Enhanced APT Detector performance validation.

This script tests the new enhanced APT detection algorithm against the baseline
and validates the accuracy improvements from 82% to 95%+ target.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import time
import warnings

# Import both detectors
from src.cybersecurity.apt_detection import APTDetector
from src.cybersecurity.apt_detection_optimized import EnhancedAPTDetector

warnings.filterwarnings('ignore')


def generate_synthetic_network_data(n_normal=1000, n_apt=200, seed=42):
    """Generate synthetic network data for testing."""
    np.random.seed(seed)
    
    # Normal traffic patterns
    normal_data = []
    for i in range(n_normal):
        # Simulate normal network patterns
        base_pattern = np.random.normal(0, 1, 50)
        # Add some periodic patterns
        time_pattern = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.3
        noise = np.random.normal(0, 0.1, 50)
        pattern = base_pattern + time_pattern + noise
        normal_data.append(pattern)
    
    # APT patterns (more subtle, persistent)
    apt_data = []
    for i in range(n_apt):
        # Simulate APT patterns with subtle anomalies
        base_pattern = np.random.normal(0.5, 0.8, 50)  # Slightly different mean/std
        # Add persistent low-level anomalies
        persistent_anomaly = np.random.exponential(0.2, 50) * 0.4
        # Add temporal clustering
        cluster_mask = np.random.choice([0, 1], 50, p=[0.7, 0.3])
        cluster_anomaly = cluster_mask * np.random.exponential(0.5, 50)
        noise = np.random.normal(0, 0.05, 50)  # Less noise
        pattern = base_pattern + persistent_anomaly + cluster_anomaly + noise
        apt_data.append(pattern)
    
    # Combine data
    X = np.array(normal_data + apt_data)
    y = np.array([0] * n_normal + [1] * n_apt)
    
    return X, y


def evaluate_detector(detector, X_train, y_train, X_test, y_test, detector_name):
    """Evaluate a detector and return performance metrics."""
    print(f"\n{'='*50}")
    print(f"Evaluating {detector_name}")
    print(f"{'='*50}")
    
    # Training
    start_time = time.time()
    detector.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)
    predict_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    
    # Handle different probability output formats
    if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        y_proba_binary = y_proba[:, 1]  # APT class probability
    else:
        y_proba_binary = y_proba.flatten()
    
    try:
        auc_score = roc_auc_score(y_test, y_proba_binary)
    except:
        auc_score = None
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Normal', 'APT'], output_dict=True)
    
    results = {
        'detector_name': detector_name,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'precision_apt': report['APT']['precision'],
        'recall_apt': report['APT']['recall'],
        'f1_apt': report['APT']['f1-score'],
        'precision_normal': report['Normal']['precision'],
        'recall_normal': report['Normal']['recall'],
        'f1_normal': report['Normal']['f1-score'],
        'train_time': train_time,
        'predict_time': predict_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print results
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    if auc_score:
        print(f"AUC Score: {auc_score:.4f}")
    print(f"APT Detection - Precision: {results['precision_apt']:.4f}, Recall: {results['recall_apt']:.4f}, F1: {results['f1_apt']:.4f}")
    print(f"Training Time: {train_time:.2f}s, Prediction Time: {predict_time:.2f}s")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                Normal  APT")
    print(f"Actual Normal  {results['confusion_matrix'][0,0]:6d} {results['confusion_matrix'][0,1]:4d}")
    print(f"       APT     {results['confusion_matrix'][1,0]:6d} {results['confusion_matrix'][1,1]:4d}")
    
    return results


def test_enhanced_apt_detector():
    """Main test function for enhanced APT detector validation."""
    print("Enhanced APT Detector Performance Validation")
    print("=" * 60)
    
    # Generate test data
    print("Generating synthetic network data...")
    X, y = generate_synthetic_network_data(n_normal=1500, n_apt=300, seed=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples ({np.sum(y_train)} APT)")
    print(f"Test set: {len(X_test)} samples ({np.sum(y_test)} APT)")
    
    # Test baseline detector
    baseline_detector = APTDetector(verbose=False)
    baseline_results = evaluate_detector(
        baseline_detector, X_train, y_train, X_test, y_test, "Baseline APT Detector"
    )
    
    # Test enhanced detector
    enhanced_detector = EnhancedAPTDetector(verbose=False)
    enhanced_results = evaluate_detector(
        enhanced_detector, X_train, y_train, X_test, y_test, "Enhanced APT Detector"
    )
    
    # Comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    metrics = ['accuracy', 'precision_apt', 'recall_apt', 'f1_apt', 'auc_score']
    
    print(f"{'Metric':<15} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<12}")
    print("-" * 55)
    
    for metric in metrics:
        baseline_val = baseline_results.get(metric, 0)
        enhanced_val = enhanced_results.get(metric, 0)
        
        if baseline_val and enhanced_val:
            improvement = ((enhanced_val - baseline_val) / baseline_val) * 100
            print(f"{metric:<15} {baseline_val:<12.4f} {enhanced_val:<12.4f} {improvement:+8.2f}%")
        else:
            print(f"{metric:<15} {baseline_val:<12.4f} {enhanced_val:<12.4f} {'N/A':<12}")
    
    # Performance timing comparison
    print(f"\nTiming Comparison:")
    print(f"Training Time   - Baseline: {baseline_results['train_time']:.2f}s, Enhanced: {enhanced_results['train_time']:.2f}s")
    print(f"Prediction Time - Baseline: {baseline_results['predict_time']:.4f}s, Enhanced: {enhanced_results['predict_time']:.4f}s")
    
    # Target achievement check
    target_accuracy = 0.95
    enhanced_accuracy = enhanced_results['accuracy']
    
    print(f"\n{'='*60}")
    print("TARGET ACHIEVEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"Target Accuracy: {target_accuracy*100:.1f}%")
    print(f"Enhanced Detector Accuracy: {enhanced_accuracy*100:.2f}%")
    
    if enhanced_accuracy >= target_accuracy:
        print("✅ TARGET ACHIEVED! Enhanced detector meets 95%+ accuracy requirement")
    else:
        gap = (target_accuracy - enhanced_accuracy) * 100
        print(f"❌ Target not met. Gap: {gap:.2f} percentage points")
    
    # Advanced analysis for enhanced detector
    if hasattr(enhanced_detector, 'analyze_apt_patterns'):
        print(f"\n{'='*60}")
        print("ENHANCED DETECTOR ANALYSIS")
        print(f"{'='*60}")
        
        try:
            analysis = enhanced_detector.analyze_apt_patterns(X_test)
            print(f"Threat Assessment: {analysis.get('threat_assessment', 'N/A')}")
            print(f"APT Percentage: {analysis.get('apt_percentage', 0):.2f}%")
            print(f"High Risk Samples: {len(analysis.get('high_risk_samples', []))}")
            print(f"Confidence Score: {analysis.get('confidence_score', 0):.4f}")
            
            # Feature importance
            if hasattr(enhanced_detector, '_get_feature_importance'):
                importance = enhanced_detector._get_feature_importance()
                if importance:
                    print(f"\nFeature Importance:")
                    for feature, score in importance.items():
                        print(f"  {feature}: {score:.4f}")
            
        except Exception as e:
            print(f"Enhanced analysis failed: {e}")
    
    return baseline_results, enhanced_results


if __name__ == "__main__":
    # Run the performance validation
    baseline_results, enhanced_results = test_enhanced_apt_detector()
    
    # Save results
    results_summary = {
        'baseline': baseline_results,
        'enhanced': enhanced_results,
        'test_date': pd.Timestamp.now().isoformat(),
        'target_achieved': enhanced_results['accuracy'] >= 0.95
    }
    
    # Export to JSON for documentation
    import json
    with open('apt_detector_performance_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Results saved to apt_detector_performance_results.json")
    print("Performance validation complete.")