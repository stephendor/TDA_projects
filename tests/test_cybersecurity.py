"""
Test suite for TDA Platform cybersecurity functionality.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.cybersecurity.apt_detection import APTDetector
from src.cybersecurity.iot_classification import IoTClassifier
from src.cybersecurity.network_analysis import NetworkAnalyzer


class TestAPTDetector:
    """Test cases for APT detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = APTDetector(
            time_window=3600,
            ph_maxdim=2,
            verbose=False
        )
        
        # Create synthetic network data
        np.random.seed(42)
        self.normal_data = np.random.randn(100, 20)
        self.apt_data = np.random.randn(50, 20) + 2  # Shifted distribution for APT
        self.combined_data = np.vstack([self.normal_data, self.apt_data])
    
    def test_initialization(self):
        """Test APT detector initialization."""
        assert self.detector.time_window == 3600
        assert self.detector.ph_maxdim == 2
        assert self.detector.verbose == False
        assert not self.detector.is_fitted
    
    def test_fit_normal_data(self):
        """Test fitting on normal traffic data."""
        self.detector.fit(self.normal_data)
        
        assert self.detector.is_fitted
        assert hasattr(self.detector, 'baseline_features_')
        assert hasattr(self.detector, 'anomaly_threshold_')
    
    def test_predict_functionality(self):
        """Test APT prediction on new data."""
        # Fit on normal data
        self.detector.fit(self.normal_data)
        
        # Predict on combined data (should detect APT patterns)
        predictions = self.detector.predict(self.combined_data)
        
        assert len(predictions) == len(self.combined_data)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should detect some APT patterns
        apt_detected = sum(predictions)
        assert apt_detected > 0
    
    def test_analyze_apt_patterns(self):
        """Test detailed APT pattern analysis."""
        self.detector.fit(self.normal_data)
        
        analysis = self.detector.analyze_apt_patterns(self.apt_data)
        
        assert 'apt_percentage' in analysis
        assert 'high_risk_samples' in analysis
        assert 'threat_assessment' in analysis
        assert 'temporal_analysis' in analysis
        
        assert 0 <= analysis['apt_percentage'] <= 100
        assert isinstance(analysis['high_risk_samples'], list)
    
    def test_empty_data_handling(self):
        """Test handling of empty input data."""
        with pytest.raises(ValueError):
            self.detector.fit(np.array([]))
        
        with pytest.raises(ValueError):
            self.detector.predict(np.array([]))
    
    def test_insufficient_data(self):
        """Test handling of insufficient data points."""
        # Very small dataset
        small_data = np.random.randn(2, 5)
        
        # Should handle gracefully without crashing
        self.detector.fit(small_data)
        predictions = self.detector.predict(small_data)
        
        assert len(predictions) == len(small_data)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        self.detector.fit(self.normal_data)
        
        # Should have feature importance after fitting
        if hasattr(self.detector, 'feature_importance_'):
            importance = self.detector.feature_importance_
            assert len(importance) > 0
            assert all(score >= 0 for score in importance)


class TestIoTClassifier:
    """Test cases for IoT device classification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = IoTClassifier(verbose=False)
        
        # Synthetic IoT device data (different device types)
        np.random.seed(42)
        self.camera_data = np.random.normal(0, 1, (50, 15))  # Camera traffic patterns
        self.sensor_data = np.random.normal(2, 0.5, (50, 15))  # Sensor traffic patterns
        self.router_data = np.random.normal(-1, 1.5, (50, 15))  # Router traffic patterns
        
        # Combine with labels
        self.X = np.vstack([self.camera_data, self.sensor_data, self.router_data])
        self.y = np.array([0] * 50 + [1] * 50 + [2] * 50)  # 3 device types
    
    def test_initialization(self):
        """Test IoT classifier initialization."""
        assert self.classifier.verbose == False
        assert not self.classifier.is_fitted
    
    def test_fit_predict_cycle(self):
        """Test complete fit-predict cycle."""
        # Fit classifier
        self.classifier.fit(self.X, self.y)
        assert self.classifier.is_fitted
        
        # Predict on new data
        predictions = self.classifier.predict(self.X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_classify_device_type(self):
        """Test device type classification."""
        self.classifier.fit(self.X, self.y)
        
        # Test single device classification
        device_features = self.camera_data[0:1]  # Single camera sample
        device_type = self.classifier.classify_device_type(device_features)
        
        assert 'device_type' in device_type
        assert 'confidence' in device_type
        assert 'topological_features' in device_type
        assert 0 <= device_type['confidence'] <= 1
    
    def test_detect_spoofing(self):
        """Test device spoofing detection."""
        self.classifier.fit(self.X, self.y)
        
        # Create spoofed device data (anomalous patterns)
        spoofed_data = np.random.normal(10, 2, (5, 15))  # Very different pattern
        
        spoofing_results = self.classifier.detect_spoofing(spoofed_data)
        
        assert 'spoofing_detected' in spoofing_results
        assert 'anomaly_scores' in spoofing_results
        assert 'suspicious_samples' in spoofing_results
        assert isinstance(spoofing_results['spoofing_detected'], bool)
    
    def test_empty_data_handling(self):
        """Test handling of empty training data."""
        with pytest.raises(ValueError):
            self.classifier.fit(np.array([]), np.array([]))


class TestNetworkAnalyzer:
    """Test cases for network analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = NetworkAnalyzer(verbose=False)
        
        # Synthetic network traffic data
        np.random.seed(42)
        self.normal_traffic = np.random.randn(200, 25)
        self.anomalous_traffic = np.random.randn(50, 25) + 3  # Shifted for anomalies
    
    def test_initialization(self):
        """Test network analyzer initialization."""
        assert self.analyzer.verbose == False
        assert not self.analyzer.is_fitted
    
    def test_fit_normal_traffic(self):
        """Test fitting on normal network traffic."""
        self.analyzer.fit(self.normal_traffic)
        
        assert self.analyzer.is_fitted
        assert hasattr(self.analyzer, 'baseline_topology_')
    
    def test_detect_anomalies(self):
        """Test network anomaly detection."""
        self.analyzer.fit(self.normal_traffic)
        
        # Detect anomalies in mixed traffic
        combined_traffic = np.vstack([self.normal_traffic[:50], self.anomalous_traffic])
        anomalies = self.analyzer.detect_anomalies(combined_traffic)
        
        assert 'anomaly_flags' in anomalies
        assert 'anomaly_scores' in anomalies
        assert 'topology_analysis' in anomalies
        
        assert len(anomalies['anomaly_flags']) == len(combined_traffic)
        assert all(flag in [0, 1] for flag in anomalies['anomaly_flags'])
    
    def test_analyze_traffic_patterns(self):
        """Test traffic pattern analysis."""
        self.analyzer.fit(self.normal_traffic)
        
        patterns = self.analyzer.analyze_traffic_patterns(self.normal_traffic)
        
        assert 'pattern_summary' in patterns
        assert 'topological_features' in patterns
        assert 'temporal_analysis' in patterns
        
        # Should contain meaningful pattern information
        assert isinstance(patterns['pattern_summary'], dict)
    
    def test_performance_metrics(self):
        """Test performance metrics computation."""
        self.analyzer.fit(self.normal_traffic)
        
        # Create ground truth labels (first 50 normal, rest anomalous)
        y_true = np.array([0] * 50 + [1] * 50)
        combined_traffic = np.vstack([self.normal_traffic[:50], self.anomalous_traffic])
        
        anomalies = self.analyzer.detect_anomalies(combined_traffic)
        y_pred = anomalies['anomaly_flags']
        
        metrics = self.analyzer.compute_performance_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Metrics should be reasonable values
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1


class TestCybersecurityIntegration:
    """Integration tests for cybersecurity modules."""
    
    def test_apt_iot_integration(self):
        """Test integration between APT detection and IoT classification."""
        # Create synthetic data
        np.random.seed(42)
        network_data = np.random.randn(100, 20)
        device_data = np.random.randn(50, 15)
        device_labels = np.random.randint(0, 3, 50)
        
        # Initialize components
        apt_detector = APTDetector(verbose=False)
        iot_classifier = IoTClassifier(verbose=False)
        
        # Fit both components
        apt_detector.fit(network_data[:70])
        iot_classifier.fit(device_data, device_labels)
        
        # Both should be functional
        apt_predictions = apt_detector.predict(network_data[70:])
        iot_predictions = iot_classifier.predict(device_data[:10])
        
        assert len(apt_predictions) == 30
        assert len(iot_predictions) == 10
    
    def test_end_to_end_threat_detection(self):
        """Test complete threat detection pipeline."""
        np.random.seed(42)
        
        # Simulate complete threat detection workflow
        network_traffic = np.random.randn(150, 25)
        
        # Initialize all components
        apt_detector = APTDetector(verbose=False)
        network_analyzer = NetworkAnalyzer(verbose=False)
        
        # Fit on baseline data
        baseline_data = network_traffic[:100]
        apt_detector.fit(baseline_data)
        network_analyzer.fit(baseline_data)
        
        # Analyze new traffic
        new_traffic = network_traffic[100:]
        
        apt_results = apt_detector.analyze_apt_patterns(new_traffic)
        network_results = network_analyzer.detect_anomalies(new_traffic)
        
        # Results should be consistent and meaningful
        assert apt_results['apt_percentage'] >= 0
        assert len(network_results['anomaly_flags']) == len(new_traffic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])