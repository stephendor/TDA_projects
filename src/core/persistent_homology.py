"""
Persistent Homology Analysis Module

This module provides core functionality for computing and analyzing 
persistent homology across different applications.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
import ripser
import gudhi
from persim import plot_diagrams, bottleneck, wasserstein


class PersistentHomologyAnalyzer(BaseEstimator, TransformerMixin):
    """
    Core class for persistent homology analysis across applications.
    
    This class provides a unified interface for computing persistence diagrams
    using different backends (ripser, gudhi) and extracting topological features.
    """
    
    def __init__(
        self,
        backend: str = 'ripser',
        maxdim: int = 2,
        thresh: float = np.inf,
        coeff: int = 2,
        distance_matrix: bool = False,
        metric: str = 'euclidean'
    ):
        """
        Initialize the persistent homology analyzer.
        
        Parameters:
        -----------
        backend : str, default='ripser'
            Backend to use ('ripser' or 'gudhi')
        maxdim : int, default=2
            Maximum homology dimension to compute
        thresh : float, default=np.inf
            Maximum edge length for filtration
        coeff : int, default=2
            Coefficient field for homology computation
        distance_matrix : bool, default=False
            Whether input is a distance matrix
        metric : str, default='euclidean'
            Distance metric to use
        """
        self.backend = backend
        self.maxdim = maxdim
        self.thresh = thresh
        self.coeff = coeff
        self.distance_matrix = distance_matrix
        self.metric = metric
        self.persistence_diagrams_ = None
        self.features_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Compute persistence diagrams for input data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input point cloud or distance matrix
        y : np.ndarray, optional
            Target values (ignored)
            
        Returns:
        --------
        self : PersistentHomologyAnalyzer
        """
        if self.backend == 'ripser':
            result = ripser.ripser(
                X,
                maxdim=self.maxdim,
                thresh=self.thresh,
                coeff=self.coeff,
                distance_matrix=self.distance_matrix,
                metric=self.metric
            )
            self.persistence_diagrams_ = result['dgms']
        
        elif self.backend == 'gudhi':
            if self.distance_matrix:
                # Use distance matrix for Rips complex
                rips_complex = gudhi.RipsComplex(
                    distance_matrix=X,
                    max_edge_length=self.thresh
                )
            else:
                # Use point cloud for Rips complex
                rips_complex = gudhi.RipsComplex(
                    points=X,
                    max_edge_length=self.thresh
                )
            
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.maxdim)
            persistence = simplex_tree.persistence(
                homology_coeff_field=self.coeff,
                min_persistence=0
            )
            
            # Convert to ripser format for consistency
            self.persistence_diagrams_ = self._gudhi_to_ripser_format(persistence)
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract topological features from persistence diagrams.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (not used in transform, uses fitted diagrams)
            
        Returns:
        --------
        features : np.ndarray
            Extracted topological features
        """
        if self.persistence_diagrams_ is None:
            raise ValueError("Must call fit() before transform()")
        
        features = []
        
        for dim, dgm in enumerate(self.persistence_diagrams_):
            if len(dgm) == 0:
                # No features in this dimension
                dim_features = np.zeros(6)
            else:
                # Basic persistence statistics
                births = dgm[:, 0]
                deaths = dgm[:, 1]
                lifetimes = deaths - births
                
                # Remove infinite bars for statistics
                finite_mask = np.isfinite(deaths)
                finite_lifetimes = lifetimes[finite_mask]
                
                if len(finite_lifetimes) > 0:
                    dim_features = np.array([
                        len(dgm),  # Total number of features
                        np.sum(finite_mask),  # Number of finite features
                        np.mean(finite_lifetimes),  # Mean lifetime
                        np.std(finite_lifetimes),  # Std of lifetimes
                        np.max(finite_lifetimes),  # Max lifetime
                        np.sum(finite_lifetimes)  # Sum of lifetimes
                    ])
                else:
                    dim_features = np.array([len(dgm), 0, 0, 0, 0, 0])
            
            features.extend(dim_features)
        
        self.features_ = np.array(features)
        return self.features_.reshape(1, -1)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : np.ndarray
            Input point cloud or distance matrix
        y : np.ndarray, optional
            Target values (ignored)
            
        Returns:
        --------
        features : np.ndarray
            Extracted topological features
        """
        return self.fit(X, y).transform(X)
    
    def plot_diagrams(self, title: Optional[str] = None, **kwargs):
        """
        Plot persistence diagrams.
        
        Parameters:
        -----------
        title : str, optional
            Title for the plot
        **kwargs : dict
            Additional arguments for plot_diagrams
        """
        if self.persistence_diagrams_ is None:
            raise ValueError("Must call fit() before plotting")
        
        plot_diagrams(self.persistence_diagrams_, title=title, **kwargs)
    
    def _gudhi_to_ripser_format(self, persistence: List[Tuple]) -> List[np.ndarray]:
        """Convert GUDHI persistence format to ripser format."""
        max_dim = max([p[0] for p in persistence])
        diagrams = [[] for _ in range(max_dim + 1)]
        
        for dim, (birth, death) in persistence:
            diagrams[dim].append([birth, death])
        
        return [np.array(dgm) if dgm else np.empty((0, 2)) for dgm in diagrams]


def compute_persistence_diagram(
    X: np.ndarray,
    backend: str = 'ripser',
    maxdim: int = 2,
    **kwargs
) -> List[np.ndarray]:
    """
    Convenience function to compute persistence diagrams.
    
    Parameters:
    -----------
    X : np.ndarray
        Input point cloud or distance matrix
    backend : str, default='ripser'
        Backend to use
    maxdim : int, default=2
        Maximum dimension
    **kwargs : dict
        Additional arguments for analyzer
        
    Returns:
    --------
    diagrams : List[np.ndarray]
        Persistence diagrams for each dimension
    """
    analyzer = PersistentHomologyAnalyzer(backend=backend, maxdim=maxdim, **kwargs)
    analyzer.fit(X)
    return analyzer.persistence_diagrams_


def persistence_landscape(
    diagram: np.ndarray,
    resolution: int = 100,
    k: int = 1
) -> np.ndarray:
    """
    Compute persistence landscape for a given diagram.
    
    Parameters:
    -----------
    diagram : np.ndarray
        Persistence diagram
    resolution : int, default=100
        Number of points in landscape
    k : int, default=1
        Landscape level
        
    Returns:
    --------
    landscape : np.ndarray
        Persistence landscape values
    """
    if len(diagram) == 0:
        return np.zeros(resolution)
    
    # Remove infinite points
    finite_mask = np.isfinite(diagram[:, 1])
    finite_diagram = diagram[finite_mask]
    
    if len(finite_diagram) == 0:
        return np.zeros(resolution)
    
    # Create grid
    min_birth = np.min(finite_diagram[:, 0])
    max_death = np.max(finite_diagram[:, 1])
    t_values = np.linspace(min_birth, max_death, resolution)
    
    # Compute landscape function
    landscape = np.zeros(resolution)
    
    for i, t in enumerate(t_values):
        values = []
        for birth, death in finite_diagram:
            if birth <= t <= death:
                values.append(min(t - birth, death - t))
        
        if len(values) >= k:
            landscape[i] = sorted(values, reverse=True)[k-1]
    
    return landscape


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """
    Compute bottleneck distance between two persistence diagrams.
    
    Parameters:
    -----------
    dgm1, dgm2 : np.ndarray
        Persistence diagrams
        
    Returns:
    --------
    distance : float
        Bottleneck distance
    """
    return bottleneck(dgm1, dgm2)


def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray, q: int = 2) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams.
    
    Parameters:
    -----------
    dgm1, dgm2 : np.ndarray
        Persistence diagrams
    q : int, default=2
        Wasserstein order
        
    Returns:
    --------
    distance : float
        Wasserstein distance
    """
    return wasserstein(dgm1, dgm2, order=q)
