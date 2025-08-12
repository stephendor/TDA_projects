"""
Topology Utilities Module

This module provides utility functions for topological data analysis,
including distance computations, data preprocessing, and common operations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import warnings


class TopologyUtils:
    """
    Utility class for common topological data analysis operations.
    """
    
    @staticmethod
    def compute_distance_matrix(
        X: np.ndarray,
        metric: str = 'euclidean',
        **kwargs
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        metric : str, default='euclidean'
            Distance metric
        **kwargs : dict
            Additional arguments for distance computation
            
        Returns:
        --------
        distances : np.ndarray
            Pairwise distance matrix
        """
        return pairwise_distances(X, metric=metric, **kwargs)
    
    @staticmethod
    def subsample_data(
        X: np.ndarray,
        n_samples: int,
        method: str = 'random',
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample data for computational efficiency.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        n_samples : int
            Number of samples to select
        method : str, default='random'
            Subsampling method ('random', 'farthest_point')
        random_state : int, optional
            Random seed
            
        Returns:
        --------
        X_sub : np.ndarray
            Subsampled data
        indices : np.ndarray
            Indices of selected samples
        """
        if n_samples >= len(X):
            return X, np.arange(len(X))
        
        if method == 'random':
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(X), size=n_samples, replace=False)
            return X[indices], indices
        
        elif method == 'farthest_point':
            return TopologyUtils._farthest_point_sampling(X, n_samples, random_state)
        
        else:
            raise ValueError(f"Unknown subsampling method: {method}")
    
    @staticmethod
    def _farthest_point_sampling(
        X: np.ndarray,
        n_samples: int,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Farthest point sampling for diverse subset selection.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        n_samples : int
            Number of samples
        random_state : int, optional
            Random seed
            
        Returns:
        --------
        X_sub : np.ndarray
            Subsampled data
        indices : np.ndarray
            Selected indices
        """
        n_points = len(X)
        if n_samples >= n_points:
            return X, np.arange(n_points)
        
        rng = np.random.RandomState(random_state)
        
        # Start with random point
        selected_indices = [rng.randint(0, n_points)]
        
        for _ in range(n_samples - 1):
            # Compute distances to all selected points
            distances = np.min([
                np.linalg.norm(X - X[idx], axis=1) 
                for idx in selected_indices
            ], axis=0)
            
            # Select point farthest from all selected points
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)
        
        selected_indices = np.array(selected_indices)
        return X[selected_indices], selected_indices
    
    @staticmethod
    def add_noise(
        X: np.ndarray,
        noise_level: float = 0.1,
        noise_type: str = 'gaussian',
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Add noise to data for robustness testing.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        noise_level : float, default=0.1
            Noise intensity
        noise_type : str, default='gaussian'
            Type of noise ('gaussian', 'uniform')
        random_state : int, optional
            Random seed
            
        Returns:
        --------
        X_noisy : np.ndarray
            Data with added noise
        """
        rng = np.random.RandomState(random_state)
        
        if noise_type == 'gaussian':
            noise = rng.normal(0, noise_level, X.shape)
        elif noise_type == 'uniform':
            noise = rng.uniform(-noise_level, noise_level, X.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return X + noise
    
    @staticmethod
    def estimate_intrinsic_dimension(
        X: np.ndarray,
        k: int = 10,
        method: str = 'mle'
    ) -> float:
        """
        Estimate intrinsic dimension of data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        k : int, default=10
            Number of nearest neighbors
        method : str, default='mle'
            Estimation method ('mle', 'correlation')
            
        Returns:
        --------
        dimension : float
            Estimated intrinsic dimension
        """
        if method == 'mle':
            return TopologyUtils._mle_dimension(X, k)
        elif method == 'correlation':
            return TopologyUtils._correlation_dimension(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _mle_dimension(X: np.ndarray, k: int) -> float:
        """
        Maximum likelihood estimation of intrinsic dimension.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        k : int
            Number of nearest neighbors
            
        Returns:
        --------
        dimension : float
            Estimated dimension
        """
        n_samples, _ = X.shape
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Remove self-distance (first column)
        distances = distances[:, 1:]
        
        # MLE estimate
        r_k = distances[:, -1]  # Distance to k-th neighbor
        r_1 = distances[:, 0]   # Distance to nearest neighbor
        
        # Avoid log(0)
        ratio = r_k / (r_1 + 1e-10)
        log_ratio = np.log(ratio + 1e-10)
        
        # Estimate dimension
        dimension = (k - 1) / np.mean(log_ratio)
        
        return max(1.0, dimension)  # Ensure positive dimension
    
    @staticmethod
    def _correlation_dimension(X: np.ndarray, r_min: float = 0.01, r_max: float = 1.0, n_points: int = 20) -> float:
        """
        Correlation dimension estimation.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        r_min : float, default=0.01
            Minimum radius
        r_max : float, default=1.0
            Maximum radius
        n_points : int, default=20
            Number of radius points
            
        Returns:
        --------
        dimension : float
            Estimated correlation dimension
        """
        # Compute distance matrix
        distances = pairwise_distances(X)
        n_points_data = len(X)
        
        # Radius values
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        correlations = []
        
        for r in radii:
            # Count pairs within radius r
            count = np.sum(distances < r) - n_points_data  # Exclude self-pairs
            correlation = count / (n_points_data * (n_points_data - 1))
            correlations.append(correlation + 1e-10)  # Avoid log(0)
        
        # Fit line to log-log plot
        log_r = np.log(radii)
        log_c = np.log(correlations)
        
        # Linear regression to find slope
        coeffs = np.polyfit(log_r, log_c, 1)
        dimension = coeffs[0]
        
        return max(1.0, dimension)
    
    @staticmethod
    def sliding_window_embedding(
        time_series: np.ndarray,
        window_size: int,
        step_size: int = 1
    ) -> np.ndarray:
        """
        Create sliding window embedding of time series.
        
        Parameters:
        -----------
        time_series : np.ndarray
            1D time series data
        window_size : int
            Size of sliding window
        step_size : int, default=1
            Step size for sliding window
            
        Returns:
        --------
        embedding : np.ndarray
            Sliding window embedding
        """
        time_series = np.asarray(time_series).flatten()
        n_points = len(time_series)
        
        if window_size > n_points:
            raise ValueError("Window size cannot be larger than time series length")
        
        # Calculate number of windows
        n_windows = (n_points - window_size) // step_size + 1
        
        # Create embedding
        embedding = np.zeros((n_windows, window_size))
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            embedding[i] = time_series[start_idx:end_idx]
        
        return embedding
    
    @staticmethod
    def normalize_data(
        X: np.ndarray,
        method: str = 'standard',
        **kwargs
    ) -> Tuple[np.ndarray, Any]:
        """
        Normalize data using various methods.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        method : str, default='standard'
            Normalization method ('standard', 'minmax', 'robust')
        **kwargs : dict
            Additional arguments for scaler
            
        Returns:
        --------
        X_normalized : np.ndarray
            Normalized data
        scaler : sklearn scaler
            Fitted scaler object
        """
        if method == 'standard':
            scaler = StandardScaler(**kwargs)
        elif method == 'minmax':
            scaler = MinMaxScaler(**kwargs)
        elif method == 'robust':
            scaler = RobustScaler(**kwargs)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        X_normalized = scaler.fit_transform(X)
        return X_normalized, scaler
    
    @staticmethod
    def reduce_dimension(
        X: np.ndarray,
        method: str = 'pca',
        n_components: int = 2,
        **kwargs
    ) -> Tuple[np.ndarray, Any]:
        """
        Reduce dimensionality of data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        method : str, default='pca'
            Dimensionality reduction method ('pca', 'tsne')
        n_components : int, default=2
            Number of components
        **kwargs : dict
            Additional arguments for reducer
            
        Returns:
        --------
        X_reduced : np.ndarray
            Reduced data
        reducer : sklearn reducer
            Fitted reducer object
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
            X_reduced = reducer.fit_transform(X)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, **kwargs)
            X_reduced = reducer.fit_transform(X)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return X_reduced, reducer
    
    @staticmethod
    def compute_persistence_statistics(diagrams: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute statistics from persistence diagrams.
        
        Parameters:
        -----------
        diagrams : List[np.ndarray]
            List of persistence diagrams (one per dimension)
            
        Returns:
        --------
        stats : dict
            Dictionary of persistence statistics
        """
        stats = {}
        
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                stats[f'dim_{dim}_count'] = 0
                stats[f'dim_{dim}_total_persistence'] = 0
                stats[f'dim_{dim}_max_persistence'] = 0
                stats[f'dim_{dim}_mean_persistence'] = 0
                continue
            
            # Remove infinite bars
            finite_mask = np.isfinite(dgm[:, 1])
            finite_dgm = dgm[finite_mask]
            
            if len(finite_dgm) > 0:
                lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                stats[f'dim_{dim}_count'] = len(dgm)
                stats[f'dim_{dim}_finite_count'] = len(finite_dgm)
                stats[f'dim_{dim}_total_persistence'] = np.sum(lifetimes)
                stats[f'dim_{dim}_max_persistence'] = np.max(lifetimes)
                stats[f'dim_{dim}_mean_persistence'] = np.mean(lifetimes)
                stats[f'dim_{dim}_std_persistence'] = np.std(lifetimes)
            else:
                stats[f'dim_{dim}_count'] = len(dgm)
                stats[f'dim_{dim}_finite_count'] = 0
                stats[f'dim_{dim}_total_persistence'] = 0
                stats[f'dim_{dim}_max_persistence'] = 0
                stats[f'dim_{dim}_mean_persistence'] = 0
                stats[f'dim_{dim}_std_persistence'] = 0
        
        return stats
    
    @staticmethod
    def filter_short_lived_features(
        diagram: np.ndarray,
        min_lifetime: float = 0.01
    ) -> np.ndarray:
        """
        Filter out short-lived topological features.
        
        Parameters:
        -----------
        diagram : np.ndarray
            Persistence diagram
        min_lifetime : float, default=0.01
            Minimum lifetime threshold
            
        Returns:
        --------
        filtered_diagram : np.ndarray
            Filtered persistence diagram
        """
        if len(diagram) == 0:
            return diagram
        
        # Calculate lifetimes
        finite_mask = np.isfinite(diagram[:, 1])
        lifetimes = np.where(
            finite_mask,
            diagram[:, 1] - diagram[:, 0],
            np.inf
        )
        
        # Filter by minimum lifetime
        significant_mask = lifetimes >= min_lifetime
        
        return diagram[significant_mask]


# Convenience functions
def create_distance_matrix(X: np.ndarray, metric: str = 'euclidean', **kwargs) -> np.ndarray:
    """
    Create distance matrix from point cloud.
    
    Parameters:
    -----------
    X : np.ndarray
        Input point cloud
    metric : str
        Distance metric
    **kwargs
        Additional arguments
        
    Returns:
    --------
    np.ndarray
        Distance matrix
    """
    return TopologyUtils.compute_distance_matrix(X, metric=metric, **kwargs)


def create_point_cloud_circle(n_points: int = 100, radius: float = 1.0, noise: float = 0.1) -> np.ndarray:
    """Create noisy circle point cloud for testing."""
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    points = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
    
    return points


def create_point_cloud_torus(
    n_points: int = 1000,
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    noise: float = 0.1
) -> np.ndarray:
    """Create noisy torus point cloud for testing."""
    # Random angles
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(0, 2*np.pi, n_points)
    
    # Torus parametrization
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    
    points = np.column_stack([x, y, z])
    
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
    
    return points


def create_point_cloud_sphere(n_points: int = 500, radius: float = 1.0, noise: float = 0.1) -> np.ndarray:
    """Create noisy sphere point cloud for testing."""
    # Random points on unit sphere
    points = np.random.randn(n_points, 3)
    points = radius * points / np.linalg.norm(points, axis=1, keepdims=True)
    
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
    
    return points
