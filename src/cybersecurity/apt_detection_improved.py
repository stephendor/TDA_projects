"""
Improved Advanced Persistent Threat (APT) Detection using Enhanced TDA

This module implements a practical enhanced TDA-based APT detector that focuses
on key improvements over the baseline while maintaining computational efficiency.
Target: 95%+ accuracy improvement from baseline 82%.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from ..core import PersistentHomologyAnalyzer, MapperAnalyzer, TopologyUtils


class ImprovedAPTDetector(BaseEstimator, ClassifierMixin):
    """
    Improved TDA-based APT detector with practical enhancements.
    
    Key improvements over baseline:
    - Enhanced topological feature extraction
    - Ensemble detection approach
    - Robust statistical preprocessing
    - Temporal pattern analysis
    """
    
    def __init__(
        self,
        time_window: int = 3600,
        ph_maxdim: int = 2,
        mapper_intervals: int = 20,  # Increased from 15
        mapper_overlap: float = 0.5,  # Increased from 0.4
        anomaly_threshold: float = 0.05,  # Decreased for sensitivity
        min_persistence: float = 0.001,  # Decreased for detail
        ensemble_size: int = 3,  # Practical ensemble size
        stability_threshold: float = 0.8,
        verbose: bool = False
    ):
        """
        Initialize improved APT detector.
        
        Parameters:
        -----------
        time_window : int, default=3600
            Time window for analysis (seconds)
        ph_maxdim : int, default=2
            Maximum dimension for persistent homology
        mapper_intervals : int, default=20
            Number of intervals for mapper (increased from baseline)
        mapper_overlap : float, default=0.5
            Overlap fraction for mapper (increased from baseline)
        anomaly_threshold : float, default=0.05
            Threshold for anomaly detection (decreased for sensitivity)
        min_persistence : float, default=0.001
            Minimum persistence threshold (decreased for detail)
        ensemble_size : int, default=3
            Number of ensemble detectors (practical size)
        stability_threshold : float, default=0.8
            Threshold for topological stability
        verbose : bool, default=False
            Verbose output
        """
        self.time_window = time_window
        self.ph_maxdim = ph_maxdim
        self.mapper_intervals = mapper_intervals
        self.mapper_overlap = mapper_overlap
        self.anomaly_threshold = anomaly_threshold
        self.min_persistence = min_persistence
        self.ensemble_size = ensemble_size
        self.stability_threshold = stability_threshold
        self.verbose = verbose
        
        # Enhanced components
        self.ph_analyzer = PersistentHomologyAnalyzer(maxdim=ph_maxdim)
        self.mapper_analyzer = MapperAnalyzer(
            n_intervals=mapper_intervals,
            overlap_frac=mapper_overlap
        )
        
        # Use RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        
        # Ensemble of anomaly detectors with different parameters
        self.ensemble_detectors = [
            IsolationForest(
                contamination=anomaly_threshold,
                random_state=42 + i,
                n_estimators=100,
                max_samples=0.8,
                bootstrap=True
            ) for i in range(ensemble_size)
        ]
        
        # Supervised classifier for known APT patterns
        self.supervised_classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        # State variables
        self.baseline_features_ = None
        self.is_fitted_ = False
        self.feature_names_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the improved APT detector.
        
        Parameters:
        -----------
        X : np.ndarray
            Network features or traffic data
        y : np.ndarray, optional
            Labels (0=normal, 1=APT) for supervised learning
            
        Returns:
        --------
        self : ImprovedAPTDetector
        """
        X = np.asarray(X)
        
        if self.verbose:
            print(f"Training improved APT detector on {len(X)} samples")
        
        # Extract enhanced topological features
        topo_features = self._extract_improved_features(X)
        
        if self.verbose:
            print(f"Extracted {topo_features.shape[1]} enhanced features")
        
        # Scale features using robust scaler
        topo_features_scaled = self.scaler.fit_transform(topo_features)
        
        # Fit ensemble of anomaly detectors with slight variations
        for i, detector in enumerate(self.ensemble_detectors):
            if self.verbose and i == 0:
                print("Training ensemble detectors...")
            
            # Add small amount of noise for diversity
            noise = np.random.normal(0, 0.01, topo_features_scaled.shape)
            detector.fit(topo_features_scaled + noise)
        
        # If labels provided, train supervised classifier
        if y is not None:
            if self.verbose:
                print("Training supervised classifier with labels...")
            self.supervised_classifier.fit(topo_features_scaled, y)
        
        # Store baseline features
        self.baseline_features_ = topo_features_scaled
        self.is_fitted_ = True
        
        if self.verbose:
            print("Improved APT detector training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict APT presence with improved detection.
        
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
        
        # Get prediction probabilities
        probabilities = self.predict_proba(X)
        
        # Use adaptive threshold
        threshold = self._compute_adaptive_threshold(probabilities)
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict APT probability scores with ensemble voting.
        
        Parameters:
        -----------
        X : np.ndarray
            Network features or traffic data
            
        Returns:
        --------
        probabilities : np.ndarray
            APT probability scores (0-1)
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict_proba()")
        
        # Extract enhanced features
        topo_features = self._extract_improved_features(X)
        topo_features_scaled = self.scaler.transform(topo_features)
        
        # Ensemble voting
        ensemble_scores = []
        for detector in self.ensemble_detectors:
            scores = detector.decision_function(topo_features_scaled)
            # Convert to probabilities (lower score = more anomalous)
            probs = 1 / (1 + np.exp(scores))
            ensemble_scores.append(probs)
        
        # Average ensemble predictions
        ensemble_probs = np.mean(ensemble_scores, axis=0)
        
        # If supervised classifier available, combine predictions
        if hasattr(self.supervised_classifier, 'predict_proba'):
            try:
                supervised_probs = self.supervised_classifier.predict_proba(topo_features_scaled)
                if supervised_probs.shape[1] > 1:  # Multi-class
                    supervised_probs = supervised_probs[:, 1]  # APT class
                else:
                    supervised_probs = supervised_probs.flatten()
                
                # Weighted combination (60% ensemble, 40% supervised)
                final_probs = 0.6 * ensemble_probs + 0.4 * supervised_probs
            except:
                final_probs = ensemble_probs
        else:
            final_probs = ensemble_probs
        
        return final_probs
    
    def _extract_improved_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract improved topological and statistical features efficiently.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        all_features = []
        
        for i in range(len(X)):
            sample_features = []
            sample = X[i]
            
            # Enhanced persistent homology features
            ph_features = self._extract_enhanced_ph_features(sample)
            sample_features.extend(ph_features)
            
            # Enhanced mapper features
            mapper_features = self._extract_enhanced_mapper_features(sample)
            sample_features.extend(mapper_features)
            
            # Enhanced statistical features
            stat_features = self._extract_enhanced_statistical_features(sample)
            sample_features.extend(stat_features)
            
            # Topological stability features
            stability_features = self._extract_stability_features(sample)
            sample_features.extend(stability_features)
            
            all_features.append(np.array(sample_features))
        
        # Convert to array and handle numerical issues
        features_array = np.array(all_features)
        
        # Replace infinite values with large finite values
        features_array = np.where(np.isinf(features_array), 
                                 np.sign(features_array) * 1e10, 
                                 features_array)
        
        # Replace NaN values with zeros
        features_array = np.where(np.isnan(features_array), 0, features_array)
        
        return features_array
    
    def _extract_enhanced_ph_features(self, sample: np.ndarray) -> List[float]:
        """Extract enhanced persistent homology features."""
        features = []
        
        try:
            # Create point cloud from time series
            if len(sample) < 10:
                embedding = sample.reshape(-1, 1)
            else:
                # Use sliding window embedding for better topology
                window_size = min(5, len(sample) // 2)
                embedding = TopologyUtils.sliding_window_embedding(sample, window_size)
            
            # Compute persistent homology
            self.ph_analyzer.fit(embedding)
            diagrams = self.ph_analyzer.get_persistence_diagram()
            
            # Extract enhanced features from each dimension
            for dim in range(min(len(diagrams), self.ph_maxdim + 1)):
                if len(diagrams[dim]) == 0:
                    # No features in this dimension
                    features.extend([0] * 8)
                else:
                    pairs = diagrams[dim]
                    births = pairs[:, 0]
                    deaths = pairs[:, 1]
                    
                    # Handle infinite persistence
                    finite_mask = deaths != np.inf
                    if np.sum(finite_mask) > 0:
                        finite_births = births[finite_mask]
                        finite_deaths = deaths[finite_mask]
                        persistence_values = finite_deaths - finite_births
                        
                        # Enhanced statistical features
                        features.extend([
                            len(pairs),  # Total features
                            np.sum(finite_mask),  # Finite features
                            np.mean(persistence_values) if len(persistence_values) > 0 else 0,
                            np.std(persistence_values) if len(persistence_values) > 0 else 0,
                            np.max(persistence_values) if len(persistence_values) > 0 else 0,
                            np.sum(persistence_values) if len(persistence_values) > 0 else 0,
                            stats.skew(persistence_values) if len(persistence_values) > 1 else 0,
                            stats.kurtosis(persistence_values) if len(persistence_values) > 1 else 0
                        ])
                    else:
                        features.extend([0] * 8)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Enhanced PH feature extraction failed: {e}")
            # Return zero features if computation fails
            features = [0] * (8 * (self.ph_maxdim + 1))
        
        return features
    
    def _extract_enhanced_mapper_features(self, sample: np.ndarray) -> List[float]:
        """Extract enhanced Mapper algorithm features."""
        try:
            # Create appropriate embedding
            if len(sample) < 10:
                embedding = sample.reshape(-1, 1)
            else:
                window_size = min(5, len(sample) // 2)
                embedding = TopologyUtils.sliding_window_embedding(sample, window_size)
            
            # Apply enhanced Mapper algorithm
            mapper_result = self.mapper_analyzer.fit_transform(embedding)
            
            # Extract enhanced graph features
            num_nodes = len(mapper_result.get('nodes', []))
            num_edges = len(mapper_result.get('edges', []))
            
            features = [
                num_nodes,
                num_edges,
                num_edges / max(num_nodes, 1),  # Connectivity ratio
                mapper_result.get('n_components', 0),
                mapper_result.get('avg_clustering', 0),
                mapper_result.get('avg_node_size', 0),
                mapper_result.get('max_node_size', 0),
                mapper_result.get('diameter', 0) if mapper_result.get('diameter') is not None else 0
            ]
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Enhanced Mapper feature extraction failed: {e}")
            features = [0] * 8
        
        return features
    
    def _extract_enhanced_statistical_features(self, sample: np.ndarray) -> List[float]:
        """Extract enhanced statistical and geometric features."""
        # Helper function to safely compute statistics
        def safe_stat(func, data, default=0):
            try:
                result = func(data)
                return result if np.isfinite(result) else default
            except:
                return default
        
        # Basic statistics
        features = [
            safe_stat(np.mean, sample),
            safe_stat(np.std, sample),
            safe_stat(np.min, sample),
            safe_stat(np.max, sample),
            safe_stat(np.median, sample),
            safe_stat(lambda x: stats.skew(x), sample),
            safe_stat(lambda x: stats.kurtosis(x), sample),
            safe_stat(lambda x: len(np.unique(x)) / len(x), sample)
        ]
        
        # Distribution analysis
        if len(sample) > 5:
            # Quantiles
            features.extend([
                safe_stat(lambda x: np.percentile(x, 25), sample),
                safe_stat(lambda x: np.percentile(x, 75), sample),
                safe_stat(lambda x: np.percentile(x, 90), sample)
            ])
            
            # Trend analysis
            x = np.arange(len(sample))
            try:
                slope, _, r_value, _, _ = stats.linregress(x, sample)
                features.extend([
                    slope if np.isfinite(slope) else 0,
                    r_value**2 if np.isfinite(r_value) else 0
                ])
            except:
                features.extend([0, 0])
        else:
            features.extend([0] * 5)
        
        # Autocorrelation
        if len(sample) > 2:
            try:
                autocorr = np.corrcoef(sample[:-1], sample[1:])[0, 1]
                features.append(autocorr if np.isfinite(autocorr) else 0)
            except:
                features.append(0)
        else:
            features.append(0)
        
        return features
    
    def _extract_stability_features(self, sample: np.ndarray) -> List[float]:
        """Extract topological stability features."""
        # Simple stability measures
        features = []
        
        # Variance-based stability
        features.append(np.var(sample))
        
        # Local stability (difference between consecutive elements)
        if len(sample) > 1:
            local_diffs = np.diff(sample)
            features.extend([
                np.mean(np.abs(local_diffs)),
                np.std(local_diffs),
                np.max(np.abs(local_diffs))
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _compute_adaptive_threshold(self, probabilities: np.ndarray) -> float:
        """Compute adaptive threshold based on probability distribution."""
        if len(probabilities) == 0:
            return 0.5
        
        # Use percentile-based adaptive threshold
        # This helps handle different data distributions
        base_threshold = 0.5
        
        # Adjust based on distribution
        prob_std = np.std(probabilities)
        prob_mean = np.mean(probabilities)
        
        # If probabilities are very spread out, use higher threshold
        if prob_std > 0.2:
            adaptive_threshold = prob_mean + prob_std
        else:
            adaptive_threshold = base_threshold
        
        # Ensure reasonable bounds
        return np.clip(adaptive_threshold, 0.3, 0.8)
    
    def analyze_apt_patterns(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyze APT patterns with improved detection.
        
        Parameters:
        -----------
        X : np.ndarray
            Network data to analyze
            
        Returns:
        --------
        analysis : Dict[str, Any]
            Comprehensive analysis results
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before analyze_apt_patterns()")
        
        # Get predictions and probabilities
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)
        
        # Enhanced analysis
        apt_percentage = float(np.mean(probabilities) * 100)
        high_risk_samples = np.where(probabilities > 0.7)[0].tolist()
        
        # Determine threat assessment
        if apt_percentage >= 80:
            threat_level = "CRITICAL"
        elif apt_percentage >= 60:
            threat_level = "HIGH"
        elif apt_percentage >= 40:
            threat_level = "MEDIUM"
        elif apt_percentage >= 20:
            threat_level = "LOW"
        else:
            threat_level = "MINIMAL"
        
        return {
            'apt_percentage': apt_percentage,
            'threat_assessment': threat_level,
            'high_risk_samples': high_risk_samples,
            'confidence_score': float(1.0 / (1.0 + np.std(probabilities))),  # Higher for more consistent predictions
            'detection_summary': {
                'total_samples': len(X),
                'apt_detected': int(np.sum(predictions)),
                'mean_risk_score': float(np.mean(probabilities)),
                'max_risk_score': float(np.max(probabilities))
            }
        }
    
    @property
    def is_fitted(self) -> bool:
        """Check if detector is fitted."""
        return self.is_fitted_