"""
Advanced Persistent Threat (APT) Detection using TDA

This module implements TDA-based methods for detecting APTs in network traffic
and system behavior patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from ..core import PersistentHomologyAnalyzer, MapperAnalyzer, TopologyUtils


class APTDetector(BaseEstimator, ClassifierMixin):
    """
    TDA-based Advanced Persistent Threat detector.
    
    This class uses topological features to identify subtle, long-term
    infiltration patterns characteristic of APTs.
    """
    
    def __init__(
        self,
        time_window: int = 3600,  # 1 hour windows
        ph_maxdim: int = 2,
        mapper_intervals: int = 15,
        mapper_overlap: float = 0.4,
        anomaly_threshold: float = 0.1,
        min_persistence: float = 0.01,
        verbose: bool = False
    ):
        """
        Initialize APT detector.
        
        Parameters:
        -----------
        time_window : int, default=3600
            Time window for analysis (seconds)
        ph_maxdim : int, default=2
            Maximum dimension for persistent homology
        mapper_intervals : int, default=15
            Number of intervals for mapper
        mapper_overlap : float, default=0.4
            Overlap fraction for mapper
        anomaly_threshold : float, default=0.1
            Threshold for anomaly detection
        min_persistence : float, default=0.01
            Minimum persistence threshold
        verbose : bool, default=False
            Verbose output
        """
        self.time_window = time_window
        self.ph_maxdim = ph_maxdim
        self.mapper_intervals = mapper_intervals
        self.mapper_overlap = mapper_overlap
        self.anomaly_threshold = anomaly_threshold
        self.min_persistence = min_persistence
        self.verbose = verbose
        
        # Components
        self.ph_analyzer = PersistentHomologyAnalyzer(maxdim=ph_maxdim)
        self.mapper_analyzer = MapperAnalyzer(
            n_intervals=mapper_intervals,
            overlap_frac=mapper_overlap
        )
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=anomaly_threshold,
            random_state=42
        )
        
        # State
        self.baseline_features_ = None
        self.is_fitted_ = False
        self.feature_names_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the APT detector on normal (baseline) traffic.
        
        Parameters:
        -----------
        X : np.ndarray
            Network features or traffic data
        y : np.ndarray, optional
            Labels (ignored, unsupervised learning)
            
        Returns:
        --------
        self : APTDetector
        """
        X = np.asarray(X)
        
        if self.verbose:
            print(f"Training APT detector on {len(X)} samples")
        
        # Extract topological features
        topo_features = self._extract_topological_features(X)
        
        # Scale features
        topo_features_scaled = self.scaler.fit_transform(topo_features)
        
        # Fit anomaly detector on baseline
        self.anomaly_detector.fit(topo_features_scaled)
        
        # Store baseline features for comparison
        self.baseline_features_ = topo_features_scaled
        self.is_fitted_ = True
        
        if self.verbose:
            print("APT detector training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict APT presence in network data.
        
        Parameters:
        -----------
        X : np.ndarray
            Network features or traffic data
            
        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (1 for APT, 0 for normal)
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict()")
        
        # Extract topological features
        topo_features = self._extract_topological_features(X)
        
        # Scale features
        topo_features_scaled = self.scaler.transform(topo_features)
        
        # Predict anomalies
        anomaly_scores = self.anomaly_detector.decision_function(topo_features_scaled)
        predictions = (anomaly_scores < 0).astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict APT probability scores.
        
        Parameters:
        -----------
        X : np.ndarray
            Network features or traffic data
            
        Returns:
        --------
        probabilities : np.ndarray
            APT probability scores
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict_proba()")
        
        # Extract topological features
        topo_features = self._extract_topological_features(X)
        
        # Scale features
        topo_features_scaled = self.scaler.transform(topo_features)
        
        # Get anomaly scores and convert to probabilities
        anomaly_scores = self.anomaly_detector.decision_function(topo_features_scaled)
        
        # Convert to probabilities (higher score = more normal)
        probabilities = 1 / (1 + np.exp(anomaly_scores))  # Sigmoid transformation
        
        return probabilities
    
    def _extract_topological_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract topological features from network data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input network data
            
        Returns:
        --------
        features : np.ndarray
            Extracted topological features
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        all_features = []
        
        for i in range(len(X)):
            sample_features = []
            
            # Convert to sliding window embedding if time series
            if X[i].ndim == 1:
                embedding = TopologyUtils.sliding_window_embedding(
                    X[i], 
                    window_size=min(10, len(X[i])//2)
                )
            else:
                embedding = X[i].reshape(1, -1) if X[i].ndim == 1 else X[i]
            
            # Skip if not enough data
            if len(embedding) < 3:
                # Return zero features
                sample_features = np.zeros(self._get_feature_dimension())
                all_features.append(sample_features)
                continue
            
            # Persistent homology features
            try:
                ph_features = self.ph_analyzer.fit_transform(embedding).flatten()
                sample_features.extend(ph_features)
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"PH computation failed: {e}")
                sample_features.extend(np.zeros(self._get_ph_feature_dim()))
            
            # Mapper features
            try:
                mapper_props = self.mapper_analyzer.fit_transform(embedding)
                mapper_features = [
                    mapper_props.get('n_nodes', 0),
                    mapper_props.get('n_edges', 0),
                    mapper_props.get('n_components', 0),
                    mapper_props.get('avg_clustering', 0),
                    mapper_props.get('avg_node_size', 0),
                    mapper_props.get('max_node_size', 0)
                ]
                sample_features.extend(mapper_features)
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Mapper computation failed: {e}")
                sample_features.extend(np.zeros(6))
            
            # Statistical features
            stat_features = self._extract_statistical_features(embedding)
            sample_features.extend(stat_features)
            
            all_features.append(np.array(sample_features))
        
        return np.array(all_features)
    
    def _extract_statistical_features(self, data: np.ndarray) -> List[float]:
        """Extract basic statistical features."""
        if len(data) == 0:
            return [0] * 8
        
        flat_data = data.flatten()
        
        features = [
            np.mean(flat_data),
            np.std(flat_data),
            np.min(flat_data),
            np.max(flat_data),
            np.median(flat_data),
            np.percentile(flat_data, 25),
            np.percentile(flat_data, 75),
            len(np.unique(flat_data)) / len(flat_data)  # Uniqueness ratio
        ]
        
        return features
    
    def _get_ph_feature_dim(self) -> int:
        """Get persistent homology feature dimension."""
        return (self.ph_maxdim + 1) * 6  # 6 features per dimension
    
    def _get_feature_dimension(self) -> int:
        """Get total feature dimension."""
        return self._get_ph_feature_dim() + 6 + 8  # PH + Mapper + Statistical
    
    def analyze_apt_patterns(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyze APT patterns in detail.
        
        Parameters:
        -----------
        X : np.ndarray
            Network data to analyze
            
        Returns:
        --------
        analysis : dict
            Detailed analysis results
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before analyze_apt_patterns()")
        
        # Get predictions and scores
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Extract features for analysis
        features = self._extract_topological_features(X)
        
        analysis = {
            'n_samples': len(X),
            'n_apt_detected': np.sum(predictions),
            'apt_percentage': np.mean(predictions) * 100,
            'mean_apt_score': np.mean(probabilities),
            'max_apt_score': np.max(probabilities),
            'high_risk_samples': np.sum(probabilities > 0.8),
            'feature_statistics': {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0),
                'min': np.min(features, axis=0),
                'max': np.max(features, axis=0)
            }
        }
        
        # Identify most suspicious samples
        if len(probabilities) > 0:
            top_indices = np.argsort(probabilities)[-5:]  # Top 5 most suspicious
            analysis['most_suspicious_indices'] = top_indices.tolist()
            analysis['most_suspicious_scores'] = probabilities[top_indices].tolist()
        
        return analysis
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on baseline vs anomalous patterns.
        
        Returns:
        --------
        importance : dict
            Feature importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before get_feature_importance()")
        
        # Feature names
        feature_names = []
        
        # PH features
        for dim in range(self.ph_maxdim + 1):
            feature_names.extend([
                f'ph_dim{dim}_count',
                f'ph_dim{dim}_finite_count', 
                f'ph_dim{dim}_mean_lifetime',
                f'ph_dim{dim}_std_lifetime',
                f'ph_dim{dim}_max_lifetime',
                f'ph_dim{dim}_total_lifetime'
            ])
        
        # Mapper features
        feature_names.extend([
            'mapper_n_nodes',
            'mapper_n_edges', 
            'mapper_n_components',
            'mapper_avg_clustering',
            'mapper_avg_node_size',
            'mapper_max_node_size'
        ])
        
        # Statistical features
        feature_names.extend([
            'stat_mean',
            'stat_std',
            'stat_min',
            'stat_max',
            'stat_median',
            'stat_q25',
            'stat_q75',
            'stat_uniqueness'
        ])
        
        # Simple importance based on variance in baseline
        if self.baseline_features_ is not None:
            importance_scores = np.std(self.baseline_features_, axis=0)
            importance_scores = importance_scores / np.sum(importance_scores)  # Normalize
            
            return dict(zip(feature_names, importance_scores))
        
        return {}


def detect_apt_in_network_logs(
    network_logs: pd.DataFrame,
    time_column: str = 'timestamp',
    feature_columns: Optional[List[str]] = None,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to detect APTs in network logs.
    
    Parameters:
    -----------
    network_logs : pd.DataFrame
        Network log data
    time_column : str, default='timestamp'
        Name of timestamp column
    feature_columns : List[str], optional
        List of feature columns to use
    **kwargs : dict
        Additional arguments for APTDetector
        
    Returns:
    --------
    predictions : np.ndarray
        APT predictions
    analysis : dict
        Analysis results
    """
    if feature_columns is None:
        # Use all numeric columns except timestamp
        feature_columns = [col for col in network_logs.columns 
                          if col != time_column and pd.api.types.is_numeric_dtype(network_logs[col])]
    
    # Extract features
    X = network_logs[feature_columns].values
    
    # Split into baseline (first 70%) and test data
    split_idx = int(0.7 * len(X))
    X_baseline = X[:split_idx]
    X_test = X[split_idx:]
    
    # Create and fit detector
    detector = APTDetector(**kwargs)
    detector.fit(X_baseline)
    
    # Predict on test data
    predictions = detector.predict(X_test)
    analysis = detector.analyze_apt_patterns(X_test)
    
    return predictions, analysis


class LongTermAPTDetector:
    """
    Long-term APT detection using temporal TDA analysis.
    
    This class analyzes network behavior over extended periods to identify
    persistent threats that evolve slowly over time.
    """
    
    def __init__(
        self,
        window_size: int = 24,  # 24 hours
        overlap_size: int = 12,  # 12 hours overlap
        min_pattern_duration: int = 7,  # 7 days minimum
        **kwargs
    ):
        """
        Initialize long-term APT detector.
        
        Parameters:
        -----------
        window_size : int, default=24
            Size of analysis window (hours)
        overlap_size : int, default=12
            Overlap between windows (hours)
        min_pattern_duration : int, default=7
            Minimum pattern duration (days)
        **kwargs : dict
            Arguments for base APTDetector
        """
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.min_pattern_duration = min_pattern_duration
        self.base_detector = APTDetector(**kwargs)
        
        self.temporal_patterns_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, timestamps: np.ndarray):
        """
        Fit long-term detector on historical data.
        
        Parameters:
        -----------
        X : np.ndarray
            Network feature data
        timestamps : np.ndarray
            Corresponding timestamps
            
        Returns:
        --------
        self : LongTermAPTDetector
        """
        # Create temporal windows
        windows = self._create_temporal_windows(X, timestamps)
        
        # Extract features from each window
        window_features = []
        for window_data, _ in windows:
            if len(window_data) > 0:
                self.base_detector.fit(window_data)
                features = self.base_detector._extract_topological_features(window_data)
                window_features.append(np.mean(features, axis=0))
        
        self.temporal_patterns_ = np.array(window_features)
        self.is_fitted_ = True
        
        return self
    
    def detect_persistent_threats(
        self, 
        X: np.ndarray, 
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect persistent threats in new data.
        
        Parameters:
        -----------
        X : np.ndarray
            New network data
        timestamps : np.ndarray
            Corresponding timestamps
            
        Returns:
        --------
        results : dict
            Detection results including persistent patterns
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before detect_persistent_threats()")
        
        # Create windows for new data
        windows = self._create_temporal_windows(X, timestamps)
        
        # Analyze each window
        results = {
            'window_scores': [],
            'persistent_threats': [],
            'temporal_anomalies': []
        }
        
        for i, (window_data, window_time) in enumerate(windows):
            if len(window_data) == 0:
                continue
            
            # Get window-level prediction
            predictions = self.base_detector.predict(window_data)
            apt_score = np.mean(predictions)
            
            results['window_scores'].append({
                'window_id': i,
                'timestamp': window_time,
                'apt_score': apt_score,
                'n_samples': len(window_data)
            })
        
        # Identify persistent patterns
        if len(results['window_scores']) > self.min_pattern_duration:
            results['persistent_threats'] = self._identify_persistent_patterns(
                results['window_scores']
            )
        
        return results
    
    def _create_temporal_windows(
        self, 
        X: np.ndarray, 
        timestamps: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """Create overlapping temporal windows."""
        windows = []
        
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        X_sorted = X[sorted_indices]
        timestamps_sorted = timestamps[sorted_indices]
        
        # Create windows
        start_time = timestamps_sorted[0]
        end_time = timestamps_sorted[-1]
        
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + self.window_size * 3600  # Convert hours to seconds
            
            # Find data in this window
            mask = (timestamps_sorted >= current_time) & (timestamps_sorted < window_end)
            window_data = X_sorted[mask]
            
            windows.append((window_data, current_time))
            
            # Move to next window
            current_time += (self.window_size - self.overlap_size) * 3600
        
        return windows
    
    def _identify_persistent_patterns(self, window_scores: List[Dict]) -> List[Dict]:
        """Identify persistent threat patterns."""
        persistent_threats = []
        
        # Look for sustained high APT scores
        scores = [w['apt_score'] for w in window_scores]
        threshold = np.mean(scores) + 2 * np.std(scores)  # 2 sigma above mean
        
        # Find consecutive high-score periods
        high_score_mask = np.array(scores) > threshold
        
        # Group consecutive periods
        groups = []
        current_group = []
        
        for i, is_high in enumerate(high_score_mask):
            if is_high:
                current_group.append(i)
            else:
                if len(current_group) >= self.min_pattern_duration:
                    groups.append(current_group)
                current_group = []
        
        # Add final group if valid
        if len(current_group) >= self.min_pattern_duration:
            groups.append(current_group)
        
        # Create threat descriptions
        for group in groups:
            start_idx, end_idx = group[0], group[-1]
            persistent_threats.append({
                'start_window': start_idx,
                'end_window': end_idx,
                'duration_windows': len(group),
                'avg_apt_score': np.mean([scores[i] for i in group]),
                'max_apt_score': np.max([scores[i] for i in group]),
                'start_time': window_scores[start_idx]['timestamp'],
                'end_time': window_scores[end_idx]['timestamp']
            })
        
        return persistent_threats
