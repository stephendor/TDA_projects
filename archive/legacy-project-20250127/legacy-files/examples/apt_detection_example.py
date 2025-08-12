"""
Example: APT Detection with TDA

This example demonstrates how to use the TDA platform for detecting
Advanced Persistent Threats in network traffic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import TDA platform modules
from src.cybersecurity import APTDetector
from src.cybersecurity.apt_detection_improved import ImprovedAPTDetector
from src.core import TopologyUtils


def generate_synthetic_network_data(n_samples=2000, n_features=20, apt_ratio=0.05):
    """
    Generate synthetic network traffic data with embedded APT patterns.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_features : int  
        Number of network features
    apt_ratio : float
        Fraction of samples containing APT patterns
        
    Returns:
    --------
    data : np.ndarray
        Network feature data
    labels : np.ndarray
        True labels (1 for APT, 0 for normal)
    timestamps : np.ndarray
        Timestamps for each sample
    """
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(hours=i/10) for i in range(n_samples)]
    
    # Normal traffic (baseline)
    normal_samples = int(n_samples * (1 - apt_ratio))
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=normal_samples
    )
    
    # APT traffic (anomalous patterns)
    apt_samples = n_samples - normal_samples
    # APTs often show subtle, persistent patterns
    apt_mean = np.random.normal(0, 0.5, n_features)  # Slight shift
    apt_cov = np.eye(n_features) * 1.5  # Slightly higher variance
    apt_data = np.random.multivariate_normal(
        mean=apt_mean,
        cov=apt_cov,
        size=apt_samples
    )
    
    # Add some persistent patterns to APT data
    for i in range(apt_samples):
        # Gradual drift pattern
        drift = np.sin(np.arange(n_features) * i * 0.01) * 0.3
        apt_data[i] += drift
        
        # Some features correlate (coordinated attack)
        if n_features >= 5:
            correlation_strength = 0.4
            apt_data[i][1] = apt_data[i][0] * correlation_strength + \
                           np.random.normal(0, 0.1)
            apt_data[i][3] = apt_data[i][2] * correlation_strength + \
                           np.random.normal(0, 0.1)
    
    # Combine data
    data = np.vstack([normal_data, apt_data])
    labels = np.hstack([np.zeros(normal_samples), np.ones(apt_samples)])
    
    # Shuffle to mix normal and APT patterns
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    timestamps = np.array(timestamps)[indices]
    
    return data, labels, timestamps


def main():
    """Main example execution."""
    print("TDA-based APT Detection Example")
    print("=" * 40)
    
    # Generate synthetic data
    print("Generating synthetic network data...")
    network_data, true_labels, timestamps = generate_synthetic_network_data(
        n_samples=1500,
        n_features=25,
        apt_ratio=0.08
    )
    
    print(f"Generated {len(network_data)} samples with {network_data.shape[1]} features")
    print(f"APT samples: {np.sum(true_labels)} ({np.mean(true_labels)*100:.1f}%)")
    
    # Split data into training (baseline) and testing
    split_idx = int(0.7 * len(network_data))
    
    # Use mostly normal traffic for training
    train_mask = true_labels[:split_idx] == 0  # Only normal traffic
    train_data = network_data[:split_idx][train_mask]
    
    test_data = network_data[split_idx:]
    test_labels = true_labels[split_idx:]
    
    print(f"Training on {len(train_data)} normal samples")
    print(f"Testing on {len(test_data)} samples")
    
    # Initialize and train both APT detectors for comparison
    print("\\nInitializing APT detectors...")
    
    # Baseline detector
    baseline_detector = APTDetector(
        time_window=3600,  # 1 hour windows
        ph_maxdim=2,
        mapper_intervals=12,
        anomaly_threshold=0.15,
        verbose=False
    )
    
    # Improved detector
    improved_detector = ImprovedAPTDetector(
        time_window=3600,
        ph_maxdim=2,
        mapper_intervals=20,
        anomaly_threshold=0.05,
        verbose=False
    )
    
    print("Training baseline detector...")
    baseline_detector.fit(train_data)
    
    print("Training improved detector...")
    improved_detector.fit(train_data, np.zeros(len(train_data)))
    
    # Predict on test data with both detectors
    print("\\nDetecting APTs in test data...")
    
    print("Running baseline detector...")
    baseline_predictions = baseline_detector.predict(test_data)
    baseline_probabilities = baseline_detector.predict_proba(test_data)
    baseline_analysis = baseline_detector.analyze_apt_patterns(test_data)
    
    print("Running improved detector...")
    improved_predictions = improved_detector.predict(test_data)
    improved_probabilities = improved_detector.predict_proba(test_data)
    improved_analysis = improved_detector.analyze_apt_patterns(test_data)
    
    # Compare detection results
    print("\\nDetection Results Comparison:")
    print("=" * 50)
    
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    # Baseline results
    print("BASELINE DETECTOR:")
    print("-" * 20)
    baseline_accuracy = accuracy_score(test_labels, baseline_predictions)
    print(f"Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"APTs detected: {baseline_analysis.get('n_apt_detected', np.sum(baseline_predictions))}")
    print(f"Detection rate: {baseline_analysis.get('apt_percentage', np.mean(baseline_predictions)*100):.2f}%")
    print(f"High-risk samples: {baseline_analysis.get('high_risk_samples', np.sum(baseline_probabilities > 0.8))}")
    
    print("\\nClassification Report (Baseline):")
    print(classification_report(test_labels, baseline_predictions, 
                              target_names=['Normal', 'APT']))
    
    # Improved results
    print("\\n" + "="*50)
    print("IMPROVED DETECTOR:")
    print("-" * 20)
    improved_accuracy = accuracy_score(test_labels, improved_predictions)
    print(f"Accuracy: {improved_accuracy:.4f} ({improved_accuracy*100:.2f}%)")
    print(f"Threat Assessment: {improved_analysis.get('threat_assessment', 'N/A')}")
    print(f"APT Percentage: {improved_analysis.get('apt_percentage', 0):.2f}%")
    print(f"High-risk samples: {len(improved_analysis.get('high_risk_samples', []))}")
    print(f"Confidence Score: {improved_analysis.get('confidence_score', 0):.4f}")
    
    print("\\nClassification Report (Improved):")
    print(classification_report(test_labels, improved_predictions, 
                              target_names=['Normal', 'APT']))
    
    # Performance improvement
    accuracy_improvement = ((improved_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    print(f"\\n🎯 ACCURACY IMPROVEMENT: {accuracy_improvement:+.2f}%")
    
    if improved_accuracy >= 0.95:
        print("✅ TARGET ACHIEVED: 95%+ accuracy reached!")
    else:
        print(f"📊 Gap to 95% target: {(0.95 - improved_accuracy)*100:.2f} percentage points")
    
    # Feature importance analysis for baseline
    print("\\nFeature Importance Analysis (Baseline):")
    print("-" * 40)
    baseline_importance = baseline_detector.get_feature_importance()
    
    if baseline_importance:
        # Sort by importance
        sorted_features = sorted(baseline_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        print("Top 10 most important features (Baseline):")
        for i, (feature_name, importance) in enumerate(sorted_features[:10]):
            print(f"{i+1:2d}. {feature_name:<25}: {importance:.4f}")
    
    # Visualization comparing both detectors
    plt.figure(figsize=(16, 12))
    
    # Plot 1: APT probability scores comparison
    plt.subplot(2, 3, 1)
    plt.plot(baseline_probabilities, alpha=0.7, label='Baseline', color='blue')
    plt.plot(improved_probabilities, alpha=0.7, label='Improved', color='green')
    apt_indices = np.where(test_labels == 1)[0]
    plt.scatter(apt_indices, baseline_probabilities[apt_indices], 
               color='red', alpha=0.8, label='True APTs (Baseline)', s=30)
    plt.scatter(apt_indices, improved_probabilities[apt_indices], 
               color='darkred', alpha=0.8, label='True APTs (Improved)', s=20, marker='s')
    plt.xlabel('Sample Index')
    plt.ylabel('APT Probability')
    plt.title('APT Detection Scores Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy comparison
    plt.subplot(2, 3, 2)
    detectors = ['Baseline', 'Improved']
    accuracies = [baseline_accuracy, improved_accuracy]
    colors = ['blue', 'green']
    
    bars = plt.bar(detectors, accuracies, color=colors, alpha=0.7)
    plt.ylim(0, 1)
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Target')
    plt.ylabel('Accuracy')
    plt.title('Detector Accuracy Comparison')
    plt.legend()
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 0.01, 
                f'{acc*100:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Precision-Recall comparison
    plt.subplot(2, 3, 3)
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calculate precision and recall for both detectors
    baseline_precision, baseline_recall, _, _ = precision_recall_fscore_support(
        test_labels, baseline_predictions, average='binary', zero_division=0)
    improved_precision, improved_recall, _, _ = precision_recall_fscore_support(
        test_labels, improved_predictions, average='binary', zero_division=0)
    
    # Bar plot
    metrics = ['Precision', 'Recall']
    baseline_values = [baseline_precision, baseline_recall]
    improved_values = [improved_precision, improved_recall]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline', color='blue', alpha=0.7)
    plt.bar(x + width/2, improved_values, width, label='Improved', color='green', alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Precision & Recall Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Confusion matrices
    plt.subplot(2, 3, 4)
    baseline_cm = confusion_matrix(test_labels, baseline_predictions)
    plt.imshow(baseline_cm, interpolation='nearest', cmap='Blues')
    plt.title('Baseline Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, baseline_cm[i, j], ha='center', va='center', 
                    color='white' if baseline_cm[i, j] > baseline_cm.max()/2 else 'black')
    
    plt.subplot(2, 3, 5)
    improved_cm = confusion_matrix(test_labels, improved_predictions)
    plt.imshow(improved_cm, interpolation='nearest', cmap='Greens')
    plt.title('Improved Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, improved_cm[i, j], ha='center', va='center', 
                    color='white' if improved_cm[i, j] > improved_cm.max()/2 else 'black')
    
    # Plot 6: Performance summary
    plt.subplot(2, 3, 6)
    from sklearn.metrics import f1_score
    
    baseline_f1 = f1_score(test_labels, baseline_predictions, zero_division=0)
    improved_f1 = f1_score(test_labels, improved_predictions, zero_division=0)
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    baseline_scores = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1]
    improved_scores = [improved_accuracy, improved_precision, improved_recall, improved_f1]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, baseline_scores, width, label='Baseline', color='blue', alpha=0.7)
    plt.bar(x + width/2, improved_scores, width, label='Improved', color='green', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Overall Performance Comparison')
    plt.xticks(x, metrics_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/apt_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Long-term analysis example
    print("\\n" + "="*50)
    print("Long-term APT Detection Analysis")
    print("="*50)
    
    from src.cybersecurity.apt_detection import LongTermAPTDetector
    
    # Create temporal data with timestamps
    temporal_timestamps = np.array([t.timestamp() for t in timestamps[split_idx:]])
    
    long_detector = LongTermAPTDetector(
        window_size=24,  # 24 hour windows
        overlap_size=12,
        min_pattern_duration=3,
        verbose=True
    )
    
    # Fit on historical data (using earlier portion)
    hist_split = len(test_data) // 2
    long_detector.fit(test_data[:hist_split], temporal_timestamps[:hist_split])
    
    # Detect persistent threats
    results = long_detector.detect_persistent_threats(
        test_data[hist_split:], temporal_timestamps[hist_split:]
    )
    
    print(f"\\nTemporal Analysis Results:")
    print(f"Windows analyzed: {len(results['window_scores'])}")
    print(f"Persistent threats detected: {len(results['persistent_threats'])}")
    
    if results['persistent_threats']:
        print("\\nPersistent Threat Details:")
        for i, threat in enumerate(results['persistent_threats']):
            duration_hours = (threat['end_time'] - threat['start_time']) / 3600
            print(f"  Threat {i+1}: {duration_hours:.1f} hours duration, "
                  f"avg score: {threat['avg_apt_score']:.3f}")
    
    print("\\nExample completed successfully!")
    print("Results saved to 'examples/apt_detection_results.png'")


if __name__ == "__main__":
    main()
