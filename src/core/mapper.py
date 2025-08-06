"""
Mapper Analysis Module

This module provides TDA Mapper functionality for dimensionality reduction
and topological network analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

try:
    import kmapper as km
    KMAPPER_AVAILABLE = True
except ImportError:
    KMAPPER_AVAILABLE = False


class MapperAnalyzer(BaseEstimator, TransformerMixin):
    """
    TDA Mapper implementation for topological network analysis.
    
    This class provides functionality to create Mapper graphs from high-dimensional
    data, enabling visualization and analysis of data topology.
    """
    
    def __init__(
        self,
        filter_func: Optional[Callable] = None,
        filter_params: Optional[Dict] = None,
        n_intervals: int = 10,
        overlap_frac: float = 0.3,
        clusterer: Optional[Any] = None,
        scaler: Optional[Any] = None,
        verbose: bool = False
    ):
        """
        Initialize Mapper analyzer.
        
        Parameters:
        -----------
        filter_func : callable, optional
            Filter function to apply (default: PCA with 2 components)
        filter_params : dict, optional
            Parameters for filter function
        n_intervals : int, default=10
            Number of intervals for cover
        overlap_frac : float, default=0.3
            Fraction of overlap between intervals
        clusterer : sklearn clusterer, optional
            Clustering algorithm (default: DBSCAN)
        scaler : sklearn scaler, optional
            Data scaler (default: StandardScaler)
        verbose : bool, default=False
            Whether to print progress
        """
        self.filter_func = filter_func
        self.filter_params = filter_params or {}
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.clusterer = clusterer
        self.scaler = scaler
        self.verbose = verbose
        
        # Initialize components
        self._setup_components()
        
        # Results
        self.mapper_graph_ = None
        self.filter_values_ = None
        self.node_data_ = None
        
    def _setup_components(self):
        """Setup default components if not provided."""
        if self.filter_func is None:
            self.filter_func = PCA
            if 'n_components' not in self.filter_params:
                self.filter_params['n_components'] = 2
        
        if self.clusterer is None:
            self.clusterer = DBSCAN(eps=0.5, min_samples=3)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the Mapper to data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target values (used for coloring if provided)
            
        Returns:
        --------
        self : MapperAnalyzer
        """
        X = np.asarray(X)
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply filter function
        if callable(self.filter_func):
            filter_obj = self.filter_func(**self.filter_params)
            if hasattr(filter_obj, 'fit_transform'):
                self.filter_values_ = filter_obj.fit_transform(X_scaled)
            else:
                self.filter_values_ = filter_obj(X_scaled)
        else:
            self.filter_values_ = X_scaled
        
        # Ensure filter values are 2D
        if self.filter_values_.ndim == 1:
            self.filter_values_ = self.filter_values_.reshape(-1, 1)
        
        # Create Mapper graph
        self.mapper_graph_ = self._create_mapper_graph(X_scaled, y)
        
        return self
    
    def _create_mapper_graph(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> nx.Graph:
        """
        Create Mapper graph from data.
        
        Parameters:
        -----------
        X : np.ndarray
            Scaled input data
        y : np.ndarray, optional
            Target values
            
        Returns:
        --------
        graph : nx.Graph
            Mapper graph
        """
        n_points, n_dims = self.filter_values_.shape
        
        # Create cover intervals for each dimension
        covers = []
        for dim in range(n_dims):
            min_val = np.min(self.filter_values_[:, dim])
            max_val = np.max(self.filter_values_[:, dim])
            
            # Create overlapping intervals
            interval_length = (max_val - min_val) / (self.n_intervals - self.overlap_frac * (self.n_intervals - 1))
            overlap_length = interval_length * self.overlap_frac
            
            intervals = []
            for i in range(self.n_intervals):
                start = min_val + i * (interval_length - overlap_length)
                end = start + interval_length
                intervals.append((start, end))
            
            covers.append(intervals)
        
        # Generate all combinations of intervals (hypercubes)
        from itertools import product
        hypercubes = list(product(*covers))
        
        # Create graph
        graph = nx.Graph()
        node_id = 0
        self.node_data_ = {}
        
        for cube_idx, hypercube in enumerate(hypercubes):
            # Find points in this hypercube
            mask = np.ones(n_points, dtype=bool)
            for dim, (start, end) in enumerate(hypercube):
                mask &= (self.filter_values_[:, dim] >= start) & (self.filter_values_[:, dim] <= end)
            
            if np.sum(mask) == 0:
                continue
            
            # Extract points in hypercube
            cube_points = X[mask]
            cube_indices = np.where(mask)[0]
            
            if len(cube_points) < 2:
                # Single point or empty - create single node
                if len(cube_points) == 1:
                    graph.add_node(node_id)
                    self.node_data_[node_id] = {
                        'indices': cube_indices.tolist(),
                        'size': len(cube_indices),
                        'hypercube': hypercube,
                        'centroid': np.mean(cube_points, axis=0),
                        'target_mean': np.mean(y[cube_indices]) if y is not None else None
                    }
                    node_id += 1
                continue
            
            # Cluster points in hypercube
            cluster_labels = self.clusterer.fit_predict(cube_points)
            unique_labels = np.unique(cluster_labels)
            
            # Create nodes for each cluster (excluding noise points labeled -1)
            cube_nodes = []
            for label in unique_labels:
                if label == -1:  # Skip noise points in DBSCAN
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_indices = cube_indices[cluster_mask]
                cluster_points = cube_points[cluster_mask]
                
                graph.add_node(node_id)
                self.node_data_[node_id] = {
                    'indices': cluster_indices.tolist(),
                    'size': len(cluster_indices),
                    'hypercube': hypercube,
                    'centroid': np.mean(cluster_points, axis=0),
                    'target_mean': np.mean(y[cluster_indices]) if y is not None else None,
                    'cluster_label': label
                }
                cube_nodes.append(node_id)
                node_id += 1
        
        # Add edges between overlapping nodes
        nodes = list(graph.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Check if nodes share any data points
                indices1 = set(self.node_data_[node1]['indices'])
                indices2 = set(self.node_data_[node2]['indices'])
                
                if indices1.intersection(indices2):
                    graph.add_edge(node1, node2)
        
        return graph
    
    def transform(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Transform data and return graph properties.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        properties : dict
            Graph properties and statistics
        """
        if self.mapper_graph_ is None:
            raise ValueError("Must call fit() before transform()")
        
        # Compute graph properties
        properties = {
            'n_nodes': self.mapper_graph_.number_of_nodes(),
            'n_edges': self.mapper_graph_.number_of_edges(),
            'n_components': nx.number_connected_components(self.mapper_graph_),
            'avg_clustering': nx.average_clustering(self.mapper_graph_),
            'diameter': nx.diameter(self.mapper_graph_) if nx.is_connected(self.mapper_graph_) else np.inf
        }
        
        # Add node size statistics
        node_sizes = [self.node_data_[node]['size'] for node in self.mapper_graph_.nodes()]
        properties.update({
            'avg_node_size': np.mean(node_sizes),
            'max_node_size': np.max(node_sizes),
            'min_node_size': np.min(node_sizes)
        })
        
        return properties
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target values
            
        Returns:
        --------
        properties : dict
            Graph properties and statistics
        """
        return self.fit(X, y).transform(X)
    
    def visualize_2d(
        self,
        color_by: str = 'size',
        figsize: Tuple[int, int] = (12, 8),
        node_size_factor: float = 100,
        title: Optional[str] = None
    ):
        """
        Visualize Mapper graph in 2D using matplotlib.
        
        Parameters:
        -----------
        color_by : str, default='size'
            Node coloring scheme ('size', 'target_mean', or node attribute)
        figsize : tuple, default=(12, 8)
            Figure size
        node_size_factor : float, default=100
            Factor for node sizes
        title : str, optional
            Plot title
        """
        if self.mapper_graph_ is None:
            raise ValueError("Must call fit() before visualization")
        
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(self.mapper_graph_, k=1, iterations=50)
        
        # Get node colors and sizes
        node_colors = []
        node_sizes = []
        
        for node in self.mapper_graph_.nodes():
            node_data = self.node_data_[node]
            
            # Size based on number of points
            node_sizes.append(node_data['size'] * node_size_factor)
            
            # Color based on specified attribute
            if color_by == 'size':
                node_colors.append(node_data['size'])
            elif color_by == 'target_mean' and node_data.get('target_mean') is not None:
                node_colors.append(node_data['target_mean'])
            else:
                node_colors.append(1)  # Default color
        
        # Draw graph
        nx.draw_networkx_edges(self.mapper_graph_, pos, alpha=0.6, edge_color='gray')
        scatter = plt.scatter(
            [pos[node][0] for node in self.mapper_graph_.nodes()],
            [pos[node][1] for node in self.mapper_graph_.nodes()],
            c=node_colors,
            s=node_sizes,
            alpha=0.8,
            cmap='viridis'
        )
        
        plt.colorbar(scatter, label=color_by)
        plt.title(title or f'Mapper Graph (colored by {color_by})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_interactive(
        self,
        color_by: str = 'size',
        node_size_factor: float = 10,
        title: Optional[str] = None
    ):
        """
        Create interactive visualization using Plotly.
        
        Parameters:
        -----------
        color_by : str, default='size'
            Node coloring scheme
        node_size_factor : float, default=10
            Factor for node sizes
        title : str, optional
            Plot title
        """
        if self.mapper_graph_ is None:
            raise ValueError("Must call fit() before visualization")
        
        # Create layout
        pos = nx.spring_layout(self.mapper_graph_, k=1, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        for edge in self.mapper_graph_.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        for node in self.mapper_graph_.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.node_data_[node]
            node_info.append(f'Node {node}<br>Size: {node_data["size"]}<br>Indices: {node_data["indices"][:10]}...')
            
            # Size and color
            node_sizes.append(node_data['size'] * node_size_factor)
            
            if color_by == 'size':
                node_colors.append(node_data['size'])
            elif color_by == 'target_mean' and node_data.get('target_mean') is not None:
                node_colors.append(node_data['target_mean'])
            else:
                node_colors.append(1)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_info,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.05,
                    title=color_by
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title or f'Interactive Mapper Graph (colored by {color_by})',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Interactive Mapper visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#888", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        fig.show()


def compute_mapper_graph(
    X: np.ndarray,
    filter_func: Optional[Callable] = None,
    n_intervals: int = 10,
    overlap_frac: float = 0.3,
    **kwargs
) -> Tuple[nx.Graph, Dict]:
    """
    Convenience function to compute Mapper graph.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    filter_func : callable, optional
        Filter function
    n_intervals : int, default=10
        Number of intervals
    overlap_frac : float, default=0.3
        Overlap fraction
    **kwargs : dict
        Additional arguments
        
    Returns:
    --------
    graph : nx.Graph
        Mapper graph
    node_data : dict
        Node information
    """
    analyzer = MapperAnalyzer(
        filter_func=filter_func,
        n_intervals=n_intervals,
        overlap_frac=overlap_frac,
        **kwargs
    )
    analyzer.fit(X)
    return analyzer.mapper_graph_, analyzer.node_data_
