"""
Optimized Advanced Persistent Threat (APT) Detection using Enhanced TDA

This module implements improved TDA-based methods for detecting APTs with
enhanced feature extraction, ensemble methods, and temporal analysis.
Target: 95%+ accuracy improvement from baseline 82%.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
import networkx as nx

from ..core import PersistentHomologyAnalyzer, MapperAnalyzer, TopologyUtils


class EnhancedAPTDetector(BaseEstimator, ClassifierMixin):
    """
    Enhanced TDA-based APT detector with improved accuracy and robustness.
    
    Key improvements:
    - Multi-scale topological analysis
    - Temporal pattern recognition
    - Ensemble learning approach
    - Advanced feature engineering
    - Statistical stability measures
    """
    
    def __init__(
        self,
        time_window: int = 3600,
        ph_maxdim: int = 2,
        multiscale_windows: List[int] = [10, 30, 100],
        mapper_intervals: int = 20,
        mapper_overlap: float = 0.5,
        anomaly_threshold: float = 0.05,
        min_persistence: float = 0.001,
        ensemble_size: int = 5,
        stability_threshold: float = 0.8,
        temporal_context: int = 5,
        verbose: bool = False
    ):
        """
        Initialize enhanced APT detector.
        
        Parameters:
        -----------
        time_window : int, default=3600
            Primary time window for analysis (seconds)
        ph_maxdim : int, default=2
            Maximum dimension for persistent homology
        multiscale_windows : List[int]
            Multiple time scales for analysis
        mapper_intervals : int, default=20
            Number of intervals for mapper (increased from 15)
        mapper_overlap : float, default=0.5
            Overlap fraction for mapper (increased from 0.4)
        anomaly_threshold : float, default=0.05
            Threshold for anomaly detection (decreased for sensitivity)
        min_persistence : float, default=0.001
            Minimum persistence threshold (decreased for detail)
        ensemble_size : int, default=5
            Number of ensemble detectors
        stability_threshold : float, default=0.8
            Threshold for topological stability
        temporal_context : int, default=5
            Number of previous windows for temporal analysis
        verbose : bool, default=False
            Verbose output
        """
        self.time_window = time_window
        self.ph_maxdim = ph_maxdim
        self.multiscale_windows = multiscale_windows
        self.mapper_intervals = mapper_intervals
        self.mapper_overlap = mapper_overlap
        self.anomaly_threshold = anomaly_threshold
        self.min_persistence = min_persistence
        self.ensemble_size = ensemble_size
        self.stability_threshold = stability_threshold
        self.temporal_context = temporal_context
        self.verbose = verbose
        
        # Enhanced components
        self.ph_analyzers = [
            PersistentHomologyAnalyzer(maxdim=ph_maxdim) 
            for _ in range(len(multiscale_windows))
        ]
        self.mapper_analyzer = MapperAnalyzer(
            n_intervals=mapper_intervals,
            overlap_frac=mapper_overlap
        )
        
        # Use RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        
        # Ensemble of anomaly detectors
        self.ensemble_detectors = [
            IsolationForest(
                contamination=anomaly_threshold,
                random_state=42 + i,
                n_estimators=200,  # Increased for stability
                max_samples='auto',
                bootstrap=True
            ) for i in range(ensemble_size)
        ]
        
        # Supervised classifier for known APT patterns
        self.supervised_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # State variables
        self.baseline_features_ = None
        self.is_fitted_ = False
        self.feature_names_ = None
        self.temporal_history_ = []
        self.baseline_statistics_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the enhanced APT detector.
        
        Parameters:
        -----------
        X : np.ndarray
            Network features or traffic data
        y : np.ndarray, optional
            Labels (0=normal, 1=APT) for supervised learning
            
        Returns:
        --------
        self : EnhancedAPTDetector
        """
        X = np.asarray(X)
        
        if self.verbose:
            print(f"Training enhanced APT detector on {len(X)} samples")
        
        # Extract comprehensive topological features
        topo_features = self._extract_enhanced_features(X)
        
        if self.verbose:
            print(f"Extracted {topo_features.shape[1]} topological features")
        
        # Scale features using robust scaler
        topo_features_scaled = self.scaler.fit_transform(topo_features)
        
        # Fit ensemble of anomaly detectors
        for i, detector in enumerate(self.ensemble_detectors):
            if self.verbose and i == 0:
                print("Training ensemble detectors...")
            
            # Add noise for diversity
            noise = np.random.normal(0, 0.01, topo_features_scaled.shape)
            detector.fit(topo_features_scaled + noise)
        
        # If labels provided, train supervised classifier
        if y is not None:
            if self.verbose:
                print("Training supervised classifier with labels...")
            self.supervised_classifier.fit(topo_features_scaled, y)
        
        # Compute baseline statistics for stability analysis
        self.baseline_statistics_ = self._compute_baseline_statistics(topo_features_scaled)
        
        # Store baseline features
        self.baseline_features_ = topo_features_scaled
        self.is_fitted_ = True
        
        if self.verbose:
            print("Enhanced APT detector training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict APT presence with enhanced detection.
        
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
        
        # Enhanced threshold with adaptive adjustment
        adaptive_threshold = self._compute_adaptive_threshold(X)
        predictions = (probabilities >= adaptive_threshold).astype(int)
        
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
        topo_features = self._extract_enhanced_features(X)
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
            supervised_probs = self.supervised_classifier.predict_proba(topo_features_scaled)
            if supervised_probs.shape[1] > 1:  # Multi-class
                supervised_probs = supervised_probs[:, 1]  # APT class
            else:
                supervised_probs = supervised_probs.flatten()
            
            # Weighted combination (70% ensemble, 30% supervised)
            final_probs = 0.7 * ensemble_probs + 0.3 * supervised_probs
        else:
            final_probs = ensemble_probs
        
        # Apply temporal smoothing
        final_probs = self._apply_temporal_smoothing(final_probs)
        
        return final_probs
    
    def analyze_apt_patterns(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive APT pattern analysis with enhanced metrics.
        
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
        
        # Determine threat assessment with more granular levels
        if apt_percentage >= 90:
            threat_level = "CRITICAL"
        elif apt_percentage >= 70:
            threat_level = "HIGH"
        elif apt_percentage >= 40:
            threat_level = "MEDIUM"
        elif apt_percentage >= 20:
            threat_level = "LOW"
        else:
            threat_level = "MINIMAL"
        
        # Advanced pattern analysis
        pattern_analysis = self._analyze_patterns(X, probabilities)
        temporal_analysis = self._analyze_temporal_patterns(X, probabilities)
        topological_analysis = self._analyze_topological_signatures(X)
        stability_analysis = self._analyze_stability(X)
        
        return {
            'apt_percentage': apt_percentage,
            'threat_assessment': threat_level,
            'high_risk_samples': high_risk_samples,
            'confidence_score': float(np.std(probabilities)),  # Lower std = higher confidence
            'pattern_analysis': pattern_analysis,
            'temporal_analysis': temporal_analysis,
            'topological_signatures': topological_analysis,
            'stability_metrics': stability_analysis,
            'detection_details': {
                'ensemble_agreement': self._compute_ensemble_agreement(X),
                'feature_importance': self._get_feature_importance(),
                'anomaly_clusters': self._identify_anomaly_clusters(X, probabilities)
            }
        }
    
    def _extract_enhanced_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive topological and statistical features.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Process each sample individually
        batch_features = []
        
        for sample_idx in range(len(X)):
            sample = X[sample_idx]
            all_features = []
            
            # Multi-scale persistent homology features
            for i, window_size in enumerate(self.multiscale_windows):
                if len(sample) >= window_size:
                    # Subsample or window the data
                    if len(sample) > window_size:
                        indices = np.linspace(0, len(sample)-1, window_size, dtype=int)
                        sample_windowed = sample[indices].reshape(1, -1)
                    else:
                        sample_windowed = sample.reshape(1, -1)
                else:
                    sample_windowed = sample.reshape(1, -1)
                
                # Extract PH features at this scale
                ph_features = self._extract_persistent_homology_features(sample_windowed, i)
                all_features.append(ph_features.flatten())
            
            # Process single sample for other features
            sample_2d = sample.reshape(1, -1)
            
            # Mapper-based features
            mapper_features = self._extract_mapper_features(sample_2d)
            all_features.append(mapper_features.flatten())
            
            # Statistical and geometric features
            statistical_features = self._extract_statistical_features(sample_2d)
            all_features.append(statistical_features.flatten())
            
            # Network topology features
            network_features = self._extract_network_features(sample_2d)
            all_features.append(network_features.flatten())
            
            # Time series features
            temporal_features = self._extract_temporal_features(sample_2d)
            all_features.append(temporal_features.flatten())
            
            # Concatenate all features for this sample
            sample_features = np.concatenate(all_features)
            batch_features.append(sample_features)
        
        return np.array(batch_features)
    
    def _extract_persistent_homology_features(self, X: np.ndarray, scale_idx: int) -> np.ndarray:
        """Extract persistent homology features at specific scale."""
        try:
            # Use appropriate PH analyzer for this scale
            ph_analyzer = self.ph_analyzers[scale_idx]
            
            # Compute persistence diagram
            ph_analyzer.fit(X)
            diagrams = ph_analyzer.get_persistence_diagram()
            
            features = []
            
            # Extract features from each dimension
            for dim in range(min(len(diagrams), self.ph_maxdim + 1)):
                if len(diagrams[dim]) == 0:
                    # No features in this dimension
                    dim_features = np.zeros(10)  # Fixed size
                else:
                    pairs = diagrams[dim]
                    births = pairs[:, 0]
                    deaths = pairs[:, 1]
                    
                    # Handle infinite persistence
                    finite_mask = deaths != np.inf
                    finite_pairs = pairs[finite_mask]
                    
                    if len(finite_pairs) > 0:
                        finite_births = finite_pairs[:, 0]
                        finite_deaths = finite_pairs[:, 1]
                        persistence_values = finite_deaths - finite_births
                        
                        # Statistical features of persistence
                        dim_features = np.array([
                            len(pairs),  # Total features
                            len(finite_pairs),  # Finite features
                            np.sum(~finite_mask),  # Infinite features
                            np.mean(persistence_values) if len(persistence_values) > 0 else 0,
                            np.std(persistence_values) if len(persistence_values) > 0 else 0,
                            np.max(persistence_values) if len(persistence_values) > 0 else 0,
                            np.sum(persistence_values) if len(persistence_values) > 0 else 0,
                            np.median(persistence_values) if len(persistence_values) > 0 else 0,
                            stats.skew(persistence_values) if len(persistence_values) > 1 else 0,
                            stats.kurtosis(persistence_values) if len(persistence_values) > 1 else 0
                        ])
                    else:
                        dim_features = np.zeros(10)
                
                features.extend(dim_features)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: PH feature extraction failed: {e}")
            # Return zero features if computation fails
            return np.zeros((1, 30))  # 3 dimensions × 10 features each
    
    def _extract_mapper_features(self, X: np.ndarray) -> np.ndarray:
        """Extract Mapper algorithm features."""
        try:
            # Apply Mapper algorithm
            mapper_result = self.mapper_analyzer.fit_transform(X)
            
            # Extract graph features
            num_nodes = len(mapper_result.get('nodes', []))
            num_edges = len(mapper_result.get('edges', []))
            
            # Basic graph statistics
            if num_nodes > 0:
                avg_node_size = np.mean([len(node.get('points', [])) for node in mapper_result.get('nodes', [])])
                connectivity = num_edges / max(num_nodes, 1)
            else:
                avg_node_size = 0
                connectivity = 0
            
            # Create graph for advanced analysis
            G = nx.Graph()
            for edge in mapper_result.get('edges', []):
                G.add_edge(edge['source'], edge['target'])
            
            # Advanced graph features
            if len(G.nodes()) > 0:
                try:
                    clustering_coeff = nx.average_clustering(G)
                    num_components = nx.number_connected_components(G)
                    
                    if nx.is_connected(G):
                        diameter = nx.diameter(G)
                        avg_path_length = nx.average_shortest_path_length(G)
                    else:
                        diameter = 0
                        avg_path_length = 0
                        
                except:
                    clustering_coeff = 0
                    num_components = 0
                    diameter = 0
                    avg_path_length = 0
            else:
                clustering_coeff = 0
                num_components = 0
                diameter = 0
                avg_path_length = 0
            
            features = np.array([
                num_nodes,
                num_edges, 
                avg_node_size,
                connectivity,
                clustering_coeff,
                num_components,
                diameter,
                avg_path_length
            ])
            
            return features.reshape(1, -1)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Mapper feature extraction failed: {e}")
            return np.zeros((1, 8))
    
    def _extract_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """Extract statistical and geometric features."""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(X),
            np.std(X),
            np.min(X),
            np.max(X),
            np.median(X),
            stats.skew(X.flatten()),
            stats.kurtosis(X.flatten())
        ])
        
        # Distance-based features
        if len(X) > 1:
            distances = pdist(X)
            features.extend([
                np.mean(distances),
                np.std(distances),
                np.min(distances),
                np.max(distances),
                np.median(distances)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Correlation features (if multiple dimensions)
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            features.extend([
                np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
                np.std(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
                np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
                np.min(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _extract_network_features(self, X: np.ndarray) -> np.ndarray:
        """Extract network topology features."""
        features = []
        
        # Create adjacency matrix based on similarity
        if len(X) > 1:
            distances = squareform(pdist(X))
            # Threshold for connectivity (adaptive)
            threshold = np.percentile(distances, 20)  # Connect nearest 20%
            adjacency = (distances <= threshold).astype(int)
            np.fill_diagonal(adjacency, 0)
            
            # Network metrics
            G = nx.from_numpy_array(adjacency)
            
            features.extend([
                G.number_of_nodes(),
                G.number_of_edges(),
                nx.density(G),
                nx.number_connected_components(G),
                nx.average_clustering(G) if len(G.nodes()) > 0 else 0
            ])
            
            # Degree statistics
            degrees = [d for n, d in G.degree()]
            if degrees:
                features.extend([
                    np.mean(degrees),
                    np.std(degrees),
                    np.max(degrees)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _extract_temporal_features(self, X: np.ndarray) -> np.ndarray:
        """Extract temporal pattern features."""
        features = []
        
        # Trend analysis
        if len(X) > 2:
            # Linear trend for each dimension
            for dim in range(X.shape[1]):
                time_points = np.arange(len(X))
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, X[:, dim])
                features.extend([slope, r_value**2])  # Slope and R²
        else:
            features.extend([0, 0] * X.shape[1])
        
        # Autocorrelation features
        if len(X) > 5:
            mean_autocorr = []
            for dim in range(min(X.shape[1], 5)):  # Limit to 5 dimensions
                signal = X[:, dim]
                # Compute autocorrelation at lag 1
                if len(signal) > 1:
                    autocorr = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                    mean_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
            
            if mean_autocorr:
                features.extend([
                    np.mean(mean_autocorr),
                    np.std(mean_autocorr)
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _compute_adaptive_threshold(self, X: np.ndarray) -> float:
        """Compute adaptive threshold based on data characteristics."""
        if self.baseline_features_ is None:
            return 0.5  # Default threshold
        
        # Extract features from current data
        current_features = self._extract_enhanced_features(X)
        current_features_scaled = self.scaler.transform(current_features)
        
        # Compute distance to baseline
        baseline_centroid = np.mean(self.baseline_features_, axis=0)
        distance_to_baseline = np.linalg.norm(current_features_scaled - baseline_centroid, axis=1)
        
        # Adaptive threshold based on distance
        base_threshold = 0.5
        distance_factor = np.mean(distance_to_baseline)
        
        # Increase threshold if far from baseline
        adaptive_threshold = base_threshold + 0.1 * np.tanh(distance_factor)
        
        return min(adaptive_threshold, 0.9)  # Cap at 0.9
    
    def _apply_temporal_smoothing(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce false positives."""
        # Store in temporal history
        self.temporal_history_.append(probabilities)
        
        # Keep only recent history
        if len(self.temporal_history_) > self.temporal_context:
            self.temporal_history_ = self.temporal_history_[-self.temporal_context:]
        
        # Apply exponential smoothing
        if len(self.temporal_history_) > 1:
            weights = np.exp(np.linspace(-2, 0, len(self.temporal_history_)))
            weights /= np.sum(weights)
            
            smoothed = np.zeros_like(probabilities)
            for i, hist_probs in enumerate(self.temporal_history_):
                if len(hist_probs) == len(probabilities):
                    smoothed += weights[i] * hist_probs
            
            return smoothed
        
        return probabilities
    
    def _compute_baseline_statistics(self, features: np.ndarray) -> Dict[str, Any]:
        """Compute baseline statistics for stability analysis."""
        return {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'quantiles': np.percentile(features, [25, 50, 75], axis=0)
        }
    
    def _analyze_patterns(self, X: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze APT patterns in the data."""
        high_risk_indices = probabilities > 0.7
        
        if not np.any(high_risk_indices):
            return {'pattern_type': 'none', 'characteristics': {}}
        
        high_risk_data = X[high_risk_indices]
        
        # Analyze characteristics of high-risk samples
        pattern_analysis = {
            'pattern_type': 'suspected_apt',
            'characteristics': {
                'num_high_risk_samples': int(np.sum(high_risk_indices)),
                'avg_risk_score': float(np.mean(probabilities[high_risk_indices])),
                'risk_distribution': {
                    'critical': int(np.sum(probabilities > 0.9)),
                    'high': int(np.sum((probabilities > 0.7) & (probabilities <= 0.9))),
                    'medium': int(np.sum((probabilities > 0.5) & (probabilities <= 0.7)))
                }
            }
        }
        
        return pattern_analysis
    
    def _analyze_temporal_patterns(self, X: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal patterns in APT activity."""
        # Look for temporal clustering
        high_risk_indices = np.where(probabilities > 0.7)[0]
        
        if len(high_risk_indices) < 2:
            return {'temporal_clustering': False, 'pattern_strength': 0.0}
        
        # Analyze gaps between high-risk samples
        gaps = np.diff(high_risk_indices)
        
        temporal_analysis = {
            'temporal_clustering': len(gaps) > 0 and np.std(gaps) < np.mean(gaps),
            'pattern_strength': float(1.0 / (1.0 + np.std(gaps))) if len(gaps) > 0 else 0.0,
            'avg_gap': float(np.mean(gaps)) if len(gaps) > 0 else 0.0,
            'burst_patterns': self._detect_burst_patterns(high_risk_indices)
        }
        
        return temporal_analysis
    
    def _detect_burst_patterns(self, indices: np.ndarray) -> Dict[str, Any]:
        """Detect burst patterns in APT activity."""
        if len(indices) < 5:
            return {'bursts_detected': False, 'num_bursts': 0}
        
        # Use peak detection to find bursts
        gaps = np.diff(indices)
        
        # Find short gaps (potential bursts)
        short_gaps = gaps < np.percentile(gaps, 25)
        
        # Count burst regions
        burst_starts = np.where(np.diff(np.concatenate(([False], short_gaps, [False]))) > 0)[0]
        burst_ends = np.where(np.diff(np.concatenate(([False], short_gaps, [False]))) < 0)[0]
        
        num_bursts = len(burst_starts)
        
        return {
            'bursts_detected': num_bursts > 0,
            'num_bursts': int(num_bursts),
            'avg_burst_length': float(np.mean(burst_ends - burst_starts)) if num_bursts > 0 else 0.0
        }
    
    def _analyze_topological_signatures(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze topological signatures for APT characterization."""
        try:
            # Extract persistence features
            ph_features = self._extract_persistent_homology_features(X, 0)
            
            # Mapper features
            mapper_features = self._extract_mapper_features(X)
            
            return {
                'ph_complexity': float(np.sum(ph_features)),
                'mapper_connectivity': float(mapper_features[0, 3]) if mapper_features.size > 3 else 0.0,
                'topological_stability': self._compute_topological_stability(X)
            }
            
        except Exception as e:
            return {'error': str(e), 'ph_complexity': 0.0, 'mapper_connectivity': 0.0}
    
    def _compute_topological_stability(self, X: np.ndarray) -> float:
        """Compute topological stability measure."""
        if len(X) < 10:
            return 0.0
        
        # Add noise and recompute features
        noise_levels = [0.01, 0.05, 0.1]
        stability_scores = []
        
        original_features = self._extract_persistent_homology_features(X, 0)
        
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, X.shape)
            noisy_X = X + noise
            
            try:
                noisy_features = self._extract_persistent_homology_features(noisy_X, 0)
                
                # Compute similarity
                if original_features.size > 0 and noisy_features.size > 0:
                    similarity = np.corrcoef(original_features.flatten(), noisy_features.flatten())[0, 1]
                    stability_scores.append(similarity if not np.isnan(similarity) else 0.0)
            except:
                stability_scores.append(0.0)
        
        return float(np.mean(stability_scores))
    
    def _analyze_stability(self, X: np.ndarray) -> Dict[str, Any]:
        """Analyze overall stability metrics."""
        return {
            'topological_stability': self._compute_topological_stability(X),
            'statistical_stability': self._compute_statistical_stability(X),
            'temporal_stability': self._compute_temporal_stability(X)
        }
    
    def _compute_statistical_stability(self, X: np.ndarray) -> float:
        """Compute statistical stability."""
        if self.baseline_statistics_ is None:
            return 0.0
        
        current_stats = self._compute_baseline_statistics(self._extract_enhanced_features(X))
        
        # Compare with baseline
        mean_diff = np.mean(np.abs(current_stats['mean'] - self.baseline_statistics_['mean']))
        std_ratio = np.mean(current_stats['std'] / (self.baseline_statistics_['std'] + 1e-8))
        
        # Stability score (lower difference = higher stability)
        stability = 1.0 / (1.0 + mean_diff + np.abs(std_ratio - 1.0))
        
        return float(stability)
    
    def _compute_temporal_stability(self, X: np.ndarray) -> float:
        """Compute temporal stability."""
        if len(self.temporal_history_) < 2:
            return 1.0
        
        # Compare current with recent history
        current_probs = self.predict_proba(X)
        recent_probs = self.temporal_history_[-2] if len(self.temporal_history_) >= 2 else current_probs
        
        if len(current_probs) == len(recent_probs):
            stability = np.corrcoef(current_probs, recent_probs)[0, 1]
            return float(stability if not np.isnan(stability) else 0.0)
        
        return 0.0
    
    def _compute_ensemble_agreement(self, X: np.ndarray) -> float:
        """Compute agreement between ensemble detectors."""
        if not self.is_fitted_:
            return 0.0
        
        features = self._extract_enhanced_features(X)
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        for detector in self.ensemble_detectors:
            scores = detector.decision_function(features_scaled)
            preds = (scores < 0).astype(int)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        
        # Compute agreement as variance in predictions
        agreement = 1.0 - np.mean(np.var(predictions, axis=0))
        
        return float(np.clip(agreement, 0.0, 1.0))
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from supervised classifier."""
        if hasattr(self.supervised_classifier, 'feature_importances_'):
            importances = self.supervised_classifier.feature_importances_
            
            # Create feature names (simplified)
            feature_names = [
                'ph_features', 'mapper_features', 'statistical_features',
                'network_features', 'temporal_features'
            ]
            
            # Group importance by feature type
            feature_groups = {
                'persistent_homology': np.mean(importances[:30]) if len(importances) > 30 else 0,
                'mapper_topology': np.mean(importances[30:38]) if len(importances) > 38 else 0,
                'statistical': np.mean(importances[38:54]) if len(importances) > 54 else 0,
                'network_topology': np.mean(importances[54:62]) if len(importances) > 62 else 0,
                'temporal_patterns': np.mean(importances[62:]) if len(importances) > 62 else 0
            }
            
            return feature_groups
        
        return {}
    
    def _identify_anomaly_clusters(self, X: np.ndarray, probabilities: np.ndarray) -> List[Dict[str, Any]]:
        """Identify clusters of anomalous samples."""
        high_risk_indices = np.where(probabilities > 0.7)[0]
        
        if len(high_risk_indices) < 2:
            return []
        
        # Simple clustering based on temporal proximity
        clusters = []
        current_cluster = [high_risk_indices[0]]
        
        for i in range(1, len(high_risk_indices)):
            if high_risk_indices[i] - high_risk_indices[i-1] <= 5:  # Close in time
                current_cluster.append(high_risk_indices[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append({
                        'start_index': int(current_cluster[0]),
                        'end_index': int(current_cluster[-1]),
                        'size': len(current_cluster),
                        'avg_risk_score': float(np.mean(probabilities[current_cluster]))
                    })
                current_cluster = [high_risk_indices[i]]
        
        # Add final cluster
        if len(current_cluster) >= 2:
            clusters.append({
                'start_index': int(current_cluster[0]),
                'end_index': int(current_cluster[-1]),
                'size': len(current_cluster),
                'avg_risk_score': float(np.mean(probabilities[current_cluster]))
            })
        
        return clusters
    
    @property
    def is_fitted(self) -> bool:
        """Check if detector is fitted."""
        return self.is_fitted_