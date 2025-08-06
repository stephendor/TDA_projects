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
    
    # Initialize and train APT detector
    print("\\nInitializing APT detector...")
    detector = APTDetector(
        time_window=3600,  # 1 hour windows
        ph_maxdim=2,
        mapper_intervals=12,
        anomaly_threshold=0.15,
        verbose=True
    )
    
    print("Training detector on baseline traffic...")
    detector.fit(train_data)
    
    # Predict on test data
    print("\\nDetecting APTs in test data...")
    predictions = detector.predict(test_data)
    probabilities = detector.predict_proba(test_data)
    
    # Analyze results
    analysis = detector.analyze_apt_patterns(test_data)
    
    print("\\nDetection Results:")
    print("-" * 20)
    print(f"Samples analyzed: {analysis['n_samples']}")
    print(f"APTs detected: {analysis['n_apt_detected']}")
    print(f"Detection rate: {analysis['apt_percentage']:.2f}%")
    print(f"Mean APT score: {analysis['mean_apt_score']:.3f}")
    print(f"High-risk samples: {analysis['high_risk_samples']}")
    
    # Calculate performance metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\\nPerformance Metrics:")
    print("-" * 20)
    print(classification_report(test_labels, predictions, 
                              target_names=['Normal', 'APT']))
    
    print("\\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, predictions)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Feature importance analysis
    print("\\nFeature Importance Analysis:")
    print("-" * 30)
    feature_importance = detector.get_feature_importance()
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)
    
    print("Top 10 most important features:")
    for i, (feature_name, importance) in enumerate(sorted_features[:10]):
        print(f"{i+1:2d}. {feature_name:<25}: {importance:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: APT probability scores over time
    plt.subplot(2, 2, 1)
    plt.plot(probabilities, alpha=0.7, label='APT Probability')
    apt_indices = np.where(test_labels == 1)[0]
    plt.scatter(apt_indices, probabilities[apt_indices], 
               color='red', alpha=0.8, label='True APTs')
    plt.xlabel('Sample Index')
    plt.ylabel('APT Probability')
    plt.title('APT Detection Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Feature importance
    plt.subplot(2, 2, 2)
    top_features = sorted_features[:15]
    feature_names = [f.split('_')[-1] for f, _ in top_features]
    importances = [imp for _, imp in top_features]
    
    plt.barh(range(len(feature_names)), importances)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Importance Score')
    plt.title('Top Feature Importance')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Detection threshold analysis
    plt.subplot(2, 2, 3)
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions, recalls, f1_scores = [], [], []
    
    for threshold in thresholds:
        pred_thresh = (probabilities > threshold).astype(int)
        tp = np.sum((pred_thresh == 1) & (test_labels == 1))
        fp = np.sum((pred_thresh == 1) & (test_labels == 0))
        fn = np.sum((pred_thresh == 0) & (test_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Sample data distribution
    plt.subplot(2, 2, 4)
    # Use first two principal components for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(test_data)
    
    normal_mask = test_labels == 0
    apt_mask = test_labels == 1
    
    plt.scatter(data_2d[normal_mask, 0], data_2d[normal_mask, 1], 
               alpha=0.6, label='Normal', color='blue', s=20)
    plt.scatter(data_2d[apt_mask, 0], data_2d[apt_mask, 1], 
               alpha=0.8, label='APT', color='red', s=30)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Data Distribution (PCA)')
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
