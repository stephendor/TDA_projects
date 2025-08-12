#!/usr/bin/env python3
"""
Persistence Feature Enhancement
Based on TDA Review and ML_Ideas recommendations
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import logging

try:
    import gudhi as gd
    import giotto_tda as gt
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage, PersistenceLandscape, BettiCurve
    from gtda.plotting import plot_diagram
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
    print("⚠️ giotto-tda not available. Install with: pip install giotto-tda")

logger = logging.getLogger(__name__)

class PersistenceFeatureEnhancer:
    """
    Enhanced persistence feature extraction using multiple vectorization methods
    Implements strategies from TDA review for better feature representation
    """
    
    def __init__(self):
        self.homology_dimensions = [0, 1, 2]  # Track components, loops, voids
        self.vectorizers = {}
        
        if GTDA_AVAILABLE:
            self._setup_vectorizers()
    
    def _setup_vectorizers(self):
        """Setup multiple vectorization methods"""
        # Persistence Images - most popular method
        self.vectorizers['images'] = PersistenceImage(
            sigma=0.1,
            n_bins=20,
            homology_dimensions=self.homology_dimensions
        )
        
        # Persistence Landscapes - theoretical guarantees
        self.vectorizers['landscapes'] = PersistenceLandscape(
            n_layers=5,
            n_bins=100,
            homology_dimensions=self.homology_dimensions
        )
        
        # Betti Curves - interpretable
        self.vectorizers['betti'] = BettiCurve(
            n_bins=100,
            homology_dimensions=self.homology_dimensions
        )
    
    def extract_enhanced_features(self, point_clouds: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract multiple types of persistence features
        Following TDA ML_Ideas multi-modal approach
        """
        if not GTDA_AVAILABLE:
            return self._fallback_features(point_clouds)
            
        logger.info("Extracting enhanced persistence features...")
        
        # Compute persistence diagrams
        vr = VietorisRipsPersistence(
            homology_dimensions=self.homology_dimensions,
            n_jobs=-1
        )
        
        diagrams = vr.fit_transform(point_clouds)
        logger.info(f"Computed persistence diagrams: {diagrams.shape}")
        
        # Extract features using multiple vectorization methods
        features = {}
        
        for method_name, vectorizer in self.vectorizers.items():
            try:
                feature_matrix = vectorizer.fit_transform(diagrams)
                features[method_name] = feature_matrix
                logger.info(f"{method_name} features shape: {feature_matrix.shape}")
            except Exception as e:
                logger.warning(f"Failed to extract {method_name} features: {e}")
        
        # Add statistical features
        features['statistics'] = self._extract_statistical_features(diagrams)
        
        return features
    
    def _extract_statistical_features(self, diagrams: np.ndarray) -> np.ndarray:
        """Extract statistical features from persistence diagrams"""
        statistical_features = []
        
        for diagram_set in diagrams:
            features = []
            
            # For each homology dimension
            for dim in self.homology_dimensions:
                dim_points = diagram_set[diagram_set[:, 2] == dim][:, :2]  # Birth, death pairs
                
                if len(dim_points) == 0:
                    # No features in this dimension
                    features.extend([0] * 8)
                    continue
                
                # Persistence values
                persistence = dim_points[:, 1] - dim_points[:, 0]
                
                # Statistical summaries
                features.extend([
                    len(dim_points),  # Number of features
                    np.sum(persistence),  # Total persistence
                    np.mean(persistence),  # Average persistence
                    np.std(persistence),   # Persistence variance
                    np.max(persistence),   # Maximum persistence
                    np.min(persistence),   # Minimum persistence
                    np.median(persistence), # Median persistence
                    np.sum(persistence**2)  # Persistence energy
                ])
            
            statistical_features.append(features)
        
        return np.array(statistical_features)
    
    def _fallback_features(self, point_clouds: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Fallback feature extraction without giotto-tda"""
        logger.warning("Using fallback feature extraction")
        
        features = []
        for cloud in point_clouds:
            # Basic geometric features
            cloud_features = [
                len(cloud),  # Number of points
                np.mean(cloud, axis=0).mean(),  # Global centroid
                np.std(cloud, axis=0).mean(),   # Global spread
                np.min(cloud),  # Global minimum
                np.max(cloud),  # Global maximum
            ]
            
            # Distance-based features
            if len(cloud) > 1:
                distances = np.pdist(cloud)
                cloud_features.extend([
                    np.mean(distances),  # Average distance
                    np.std(distances),   # Distance variance
                    np.max(distances),   # Maximum distance
                ])
            else:
                cloud_features.extend([0, 0, 0])
            
            features.append(cloud_features)
        
        return {'fallback': np.array(features)}
    
    def fuse_features(self, feature_dict: Dict[str, np.ndarray], 
                     fusion_method: str = 'concatenate') -> np.ndarray:
        """
        Fuse multiple feature representations
        Based on TDA ML_Ideas multi-modal fusion approach
        """
        if not feature_dict:
            raise ValueError("No features to fuse")
        
        if fusion_method == 'concatenate':
            # Simple concatenation
            feature_matrices = list(feature_dict.values())
            return np.concatenate(feature_matrices, axis=1)
        
        elif fusion_method == 'weighted':
            # Weighted fusion based on feature importance
            weights = self._compute_feature_weights(feature_dict)
            weighted_features = []
            
            for name, features in feature_dict.items():
                weight = weights.get(name, 1.0)
                weighted_features.append(features * weight)
            
            return np.concatenate(weighted_features, axis=1)
        
        elif fusion_method == 'pca':
            # PCA-based fusion
            from sklearn.decomposition import PCA
            
            concatenated = np.concatenate(list(feature_dict.values()), axis=1)
            pca = PCA(n_components=min(100, concatenated.shape[1]))
            return pca.fit_transform(concatenated)
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def _compute_feature_weights(self, feature_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute feature importance weights"""
        weights = {}
        
        # Weight by feature variance (higher variance = more informative)
        for name, features in feature_dict.items():
            variance = np.mean(np.var(features, axis=0))
            weights[name] = max(0.1, variance / (1 + variance))  # Normalized weight
        
        return weights
    
    def create_enhanced_classifier(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Create enhanced classifier using multi-modal persistence features"""
        # Use Random Forest for robustness with high-dimensional features
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf.fit(X_scaled, y)
        return clf, scaler

class MultiScaleTDAAnalyzer:
    """
    Multi-scale TDA analysis for improved performance
    Based on TDA review recommendations
    """
    
    def __init__(self):
        self.scales = [0.1, 0.2, 0.5, 1.0, 2.0]  # Multiple filtration scales
        self.feature_enhancer = PersistenceFeatureEnhancer()
    
    def analyze_multiscale(self, point_clouds: List[np.ndarray]) -> np.ndarray:
        """Analyze topology at multiple scales"""
        logger.info("Performing multi-scale TDA analysis...")
        
        all_features = []
        
        for scale in self.scales:
            # Scale point clouds
            scaled_clouds = [cloud * scale for cloud in point_clouds]
            
            # Extract features at this scale
            scale_features = self.feature_enhancer.extract_enhanced_features(scaled_clouds)
            fused_features = self.feature_enhancer.fuse_features(scale_features)
            
            all_features.append(fused_features)
            logger.info(f"Scale {scale}: {fused_features.shape}")
        
        # Combine multi-scale features
        multiscale_features = np.concatenate(all_features, axis=1)
        logger.info(f"Multi-scale features shape: {multiscale_features.shape}")
        
        return multiscale_features

def demonstrate_enhanced_persistence():
    """Demonstrate enhanced persistence feature extraction"""
    print("🔬 ENHANCED PERSISTENCE FEATURES DEMO")
    print("=" * 60)
    
    # Create sample point clouds with different topologies
    np.random.seed(42)
    
    point_clouds = []
    labels = []
    
    # Normal data - random cluster
    for i in range(50):
        cloud = np.random.normal(0, 1, (30, 3))
        point_clouds.append(cloud)
        labels.append(0)
    
    # Anomalous data - ring structure (has 1-dimensional hole)
    for i in range(50):
        angles = np.random.uniform(0, 2*np.pi, 30)
        radius = 2 + np.random.normal(0, 0.1, 30)
        x = radius * np.cos(angles) + np.random.normal(0, 0.1, 30)
        y = radius * np.sin(angles) + np.random.normal(0, 0.1, 30)
        z = np.random.normal(0, 0.1, 30)
        cloud = np.column_stack([x, y, z])
        point_clouds.append(cloud)
        labels.append(1)
    
    labels = np.array(labels)
    
    # Test enhanced feature extraction
    enhancer = PersistenceFeatureEnhancer()
    features = enhancer.extract_enhanced_features(point_clouds)
    
    print(f"\n✅ Feature extraction results:")
    for method, feature_matrix in features.items():
        print(f"   {method}: {feature_matrix.shape}")
    
    # Test multi-scale analysis
    multiscale_analyzer = MultiScaleTDAAnalyzer()
    multiscale_features = multiscale_analyzer.analyze_multiscale(point_clouds)
    
    print(f"\n✅ Multi-scale features: {multiscale_features.shape}")
    
    # Test classification performance
    fused_features = enhancer.fuse_features(features)
    print(f"✅ Fused features: {fused_features.shape}")
    
    # Simple train/test split
    n_train = len(point_clouds) // 2
    X_train, X_test = fused_features[:n_train], fused_features[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]
    
    clf, scaler = enhancer.create_enhanced_classifier(X_train, y_train)
    
    # Evaluate
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    
    f1 = f1_score(y_test, y_pred)
    print(f"\n🎯 Classification Results:")
    print(f"   F1-Score: {f1:.3f}")
    print(f"   Feature Importance (top 5):")
    
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[-5:]
    for i, idx in enumerate(reversed(top_indices)):
        print(f"     Feature {idx}: {importances[idx]:.3f}")
    
    return fused_features, multiscale_features, f1

if __name__ == "__main__":
    demonstrate_enhanced_persistence()