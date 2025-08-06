#!/usr/bin/env python3
"""
Comprehensive test comparing all APT detector versions.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import time
import warnings

# Import all detector versions
from src.cybersecurity.apt_detection import APTDetector
from src.cybersecurity.apt_detection_improved import ImprovedAPTDetector

warnings.filterwarnings('ignore')


def generate_realistic_network_data(n_normal=300, n_apt=75, seed=42):
    """Generate more realistic network data for testing."""
    np.random.seed(seed)
    
    # Normal traffic patterns - multiple types
    normal_data = []
    
    # Regular web traffic
    for i in range(n_normal // 3):
        pattern = np.random.exponential(0.5, 30) + np.random.normal(0, 0.1, 30)
        normal_data.append(pattern)
    
    # Email/messaging traffic
    for i in range(n_normal // 3):
        pattern = np.random.poisson(2, 30) + np.random.normal(0, 0.2, 30)
        normal_data.append(pattern)
    
    # File transfer traffic
    for i in range(n_normal - 2 * (n_normal // 3)):
        bursts = np.random.choice([0, 1], 30, p=[0.8, 0.2])
        pattern = bursts * np.random.exponential(2, 30) + np.random.normal(0.5, 0.1, 30)
        normal_data.append(pattern)
    
    # APT patterns - subtle, persistent anomalies
    apt_data = []
    for i in range(n_apt):
        # Base pattern similar to normal but with subtle differences
        base_pattern = np.random.exponential(0.6, 30) + np.random.normal(0.1, 0.05, 30)
        
        # Add persistent low-level data exfiltration
        persistent_exfil = np.random.exponential(0.1, 30) * 0.3
        
        # Add command & control communication patterns
        c2_pattern = np.zeros(30)
        c2_indices = np.random.choice(30, size=np.random.randint(2, 6), replace=False)
        c2_pattern[c2_indices] = np.random.exponential(0.2, len(c2_indices))
        
        # Add reconnaissance scanning
        scan_pattern = np.random.choice([0, 1], 30, p=[0.9, 0.1])
        scan_pattern = scan_pattern * np.random.exponential(0.15, 30) * 0.2
        
        pattern = base_pattern + persistent_exfil + c2_pattern + scan_pattern
        apt_data.append(pattern)
    
    # Combine data
    X = np.array(normal_data + apt_data)
    y = np.array([0] * n_normal + [1] * n_apt)
    
    return X, y


def evaluate_detector(detector, X_train, y_train, X_test, y_test, detector_name):
    """Evaluate a detector and return comprehensive metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {detector_name}")
    print(f"{'='*60}")
    
    # Training
    start_time = time.time()
    detector.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)
    predict_time = time.time() - start_time
    
    # Handle different probability formats
    if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
        y_proba_binary = y_proba[:, 1]
    else:
        y_proba_binary = y_proba.flatten() if hasattr(y_proba, 'flatten') else y_proba
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_test, y_proba_binary)
    except:
        auc_score = None
    
    # Print results
    print(f"Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if auc_score:
        print(f"  AUC Score: {auc_score:.4f}")
    
    print(f"  Training Time:   {train_time:.2f}s")
    print(f"  Prediction Time: {predict_time:.4f}s")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Normal   APT")
    print(f"    Actual Normal  {tn:4d}   {fp:3d}")
    print(f"           APT     {fn:4d}   {tp:3d}")
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"  Specificity: {specificity:.4f}")
    print(f"  False Positive Rate: {false_positive_rate:.4f}")
    
    return {
        'detector_name': detector_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'train_time': train_time,
        'predict_time': predict_time,
        'confusion_matrix': cm,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def comprehensive_comparison():
    """Run comprehensive comparison of all APT detectors."""
    print("Comprehensive APT Detector Performance Comparison")
    print("=" * 70)
    
    # Generate realistic test data
    print("Generating realistic network traffic data...")
    X, y = generate_realistic_network_data(n_normal=400, n_apt=100, seed=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset Summary:")
    print(f"  Training set: {len(X_train)} samples ({np.sum(y_train)} APT, {len(y_train) - np.sum(y_train)} Normal)")
    print(f"  Test set:     {len(X_test)} samples ({np.sum(y_test)} APT, {len(y_test) - np.sum(y_test)} Normal)")
    print(f"  APT ratio:    {np.mean(y)*100:.1f}%")
    
    # Test all detectors
    detectors = [
        (APTDetector(verbose=False), "Baseline APT Detector"),
        (ImprovedAPTDetector(verbose=False), "Improved APT Detector")
    ]
    
    results = []
    
    for detector, name in detectors:
        try:
            result = evaluate_detector(detector, X_train, y_train, X_test, y_test, name)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {name}: {e}")
            continue
    
    # Comparative analysis
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*70}")
        
        baseline_result = results[0]
        improved_result = results[1]
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        
        print(f"{'Metric':<12} {'Baseline':<10} {'Improved':<10} {'Improvement':<12} {'Status':<10}")
        print("-" * 70)
        
        improvements = {}
        for metric in metrics_to_compare:
            baseline_val = baseline_result.get(metric, 0)
            improved_val = improved_result.get(metric, 0)
            
            if baseline_val and improved_val and baseline_val > 0:
                improvement = ((improved_val - baseline_val) / baseline_val) * 100
                status = "✅ Better" if improvement > 0 else "❌ Worse"
                improvements[metric] = improvement
            else:
                improvement = 0
                status = "N/A"
                improvements[metric] = 0
            
            print(f"{metric:<12} {baseline_val:<10.4f} {improved_val:<10.4f} {improvement:+8.2f}% {status:<10}")
        
        # Overall assessment
        print(f"\n{'='*70}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*70}")
        
        # Target achievement
        target_accuracy = 0.95
        improved_accuracy = improved_result['accuracy']
        
        print(f"Target Accuracy: {target_accuracy*100:.1f}%")
        print(f"Improved Detector Accuracy: {improved_accuracy*100:.2f}%")
        
        if improved_accuracy >= target_accuracy:
            print("🎯 TARGET ACHIEVED! Improved detector meets 95%+ accuracy requirement")
        else:
            gap = (target_accuracy - improved_accuracy) * 100
            print(f"📊 Target gap: {gap:.2f} percentage points")
            
            # Suggest next steps based on performance
            if improved_accuracy >= 0.90:
                print("💡 Very close to target - minor tuning needed")
            elif improved_accuracy >= 0.85:
                print("💡 Good progress - focus on precision/recall balance")
            else:
                print("💡 Significant improvement needed - review feature engineering")
        
        # Performance efficiency
        print(f"\nPerformance Efficiency:")
        baseline_total_time = baseline_result['train_time'] + baseline_result['predict_time']
        improved_total_time = improved_result['train_time'] + improved_result['predict_time']
        
        time_overhead = ((improved_total_time - baseline_total_time) / baseline_total_time) * 100
        print(f"Time Overhead: {time_overhead:+.1f}%")
        
        if time_overhead < 50:
            print("⚡ Acceptable computational overhead")
        else:
            print("⚠️ High computational overhead - consider optimization")
        
        # Best performer summary
        best_accuracy = max(r['accuracy'] for r in results)
        best_detector = next(r['detector_name'] for r in results if r['accuracy'] == best_accuracy)
        
        print(f"\n🏆 Best Performing Detector: {best_detector}")
        print(f"🎯 Best Accuracy: {best_accuracy*100:.2f}%")
        
        return results
    
    return results


if __name__ == "__main__":
    results = comprehensive_comparison()
    
    print(f"\n{'='*70}")
    print("Test completed. Results summary saved to console.")
    
    # Save summary
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\nBest detector: {best_result['detector_name']} ({best_result['accuracy']*100:.2f}% accuracy)")
    
    print("Performance comparison complete.")