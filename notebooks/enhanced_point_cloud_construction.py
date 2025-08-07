#!/usr/bin/env python3
"""
Enhanced Point Cloud Construction for Network Data
Based on TDA_ML_Ideas insights and academic research
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class EnhancedPointCloudConstructor:
    """
    Advanced point cloud construction methods for network flow data
    Implements strategies from TDA review and ML_Ideas document
    """
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
    def multi_dimensional_embedding(self, 
                                  df: pd.DataFrame,
                                  temporal_cols: List[str],
                                  spatial_cols: List[str],
                                  behavioral_cols: List[str],
                                  embedding_dim: int = 64) -> np.ndarray:
        """
        Create multi-dimensional point cloud grouping related features
        Based on TDA_ML_Ideas Project 3: Multi-Modal Feature Fusion
        """
        logger.info("Creating multi-dimensional point cloud embedding...")
        
        point_clouds = []
        
        # 1. Temporal embedding (time-based features)
        if temporal_cols:
            temporal_data = df[temporal_cols].fillna(0)
            temporal_cloud = self._create_temporal_embedding(temporal_data)
            point_clouds.append(('temporal', temporal_cloud))
            
        # 2. Spatial embedding (IP/port relationships)  
        if spatial_cols:
            spatial_data = df[spatial_cols].fillna(0)
            spatial_cloud = self._create_spatial_embedding(spatial_data)
            point_clouds.append(('spatial', spatial_cloud))
            
        # 3. Behavioral embedding (statistical features)
        if behavioral_cols:
            behavioral_data = df[behavioral_cols].fillna(0)
            behavioral_cloud = self._create_behavioral_embedding(behavioral_data)
            point_clouds.append(('behavioral', behavioral_cloud))
        
        # Fuse embeddings
        return self._fuse_point_clouds(point_clouds, target_dim=embedding_dim)
    
    def _create_temporal_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """Create time-delay embedding with enhanced features"""
        # Normalize temporal features
        scaled_data = self.scalers['standard'].fit_transform(data)
        
        # Create sliding window embeddings
        window_size = min(10, len(data) // 4)
        embeddings = []
        
        for i in range(len(scaled_data) - window_size + 1):
            window = scaled_data[i:i + window_size]
            # Flatten window and add statistical moments
            flat_window = window.flatten()
            stats = [
                np.mean(window, axis=0),
                np.std(window, axis=0), 
                np.min(window, axis=0),
                np.max(window, axis=0)
            ]
            combined = np.concatenate([flat_window] + [s.flatten() for s in stats])
            embeddings.append(combined)
            
        return np.array(embeddings) if embeddings else scaled_data
    
    def _create_spatial_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """Create spatial embedding preserving network topology"""
        # Focus on connection patterns
        scaled_data = self.scalers['robust'].fit_transform(data)
        
        # Use PCA to preserve spatial relationships
        pca = PCA(n_components=min(32, scaled_data.shape[1]))
        spatial_features = pca.fit_transform(scaled_data)
        
        # Add distance-based features
        distances = []
        for i in range(len(spatial_features)):
            # Distance to centroid
            centroid_dist = np.linalg.norm(spatial_features[i] - np.mean(spatial_features, axis=0))
            # Distance to nearest neighbor
            other_points = np.delete(spatial_features, i, axis=0)
            if len(other_points) > 0:
                nn_dist = np.min([np.linalg.norm(spatial_features[i] - p) for p in other_points])
            else:
                nn_dist = 0
            distances.append([centroid_dist, nn_dist])
            
        return np.concatenate([spatial_features, np.array(distances)], axis=1)
    
    def _create_behavioral_embedding(self, data: pd.DataFrame) -> np.ndarray:
        """Create behavioral embedding capturing statistical patterns"""
        scaled_data = self.scalers['standard'].fit_transform(data)
        
        # Calculate rolling statistics for behavioral patterns
        behavioral_features = []
        for i in range(len(scaled_data)):
            # Local neighborhood analysis
            start_idx = max(0, i - 5)
            end_idx = min(len(scaled_data), i + 6)
            neighborhood = scaled_data[start_idx:end_idx]
            
            # Statistical features
            features = np.concatenate([
                scaled_data[i],  # Current point
                np.mean(neighborhood, axis=0),  # Local mean
                np.std(neighborhood, axis=0),   # Local variance
                np.median(neighborhood, axis=0) # Local median
            ])
            behavioral_features.append(features)
            
        return np.array(behavioral_features)
    
    def _fuse_point_clouds(self, point_clouds: List[Tuple[str, np.ndarray]], 
                          target_dim: int) -> np.ndarray:
        """Fuse multiple point clouds into unified representation"""
        if not point_clouds:
            raise ValueError("No point clouds provided")
            
        # Find common length
        min_length = min(cloud.shape[0] for _, cloud in point_clouds)
        
        fused_features = []
        for name, cloud in point_clouds:
            # Truncate to common length
            cloud_truncated = cloud[:min_length]
            
            # Reduce dimensionality if needed
            if cloud_truncated.shape[1] > target_dim // len(point_clouds):
                pca = PCA(n_components=target_dim // len(point_clouds))
                cloud_reduced = pca.fit_transform(cloud_truncated)
            else:
                cloud_reduced = cloud_truncated
                
            fused_features.append(cloud_reduced)
            logger.info(f"{name} cloud shape: {cloud_reduced.shape}")
        
        return np.concatenate(fused_features, axis=1)

class GraphBasedTDAConstructor:
    """
    Graph-based TDA following Collins et al. (2020) approach
    Uses network structure directly instead of point clouds
    """
    
    def construct_network_graphs(self, df: pd.DataFrame) -> List[nx.Graph]:
        """
        Construct time-windowed network graphs from flow data
        Following TDA review recommendations for graph-based analysis
        """
        logger.info("Constructing network graphs for TDA...")
        
        # Group flows by time windows
        time_windows = self._create_time_windows(df, window_size='5T')  # 5-minute windows
        
        graphs = []
        for window_data in time_windows:
            G = nx.Graph()
            
            # Add nodes and edges based on network flows
            for _, flow in window_data.iterrows():
                src_ip = flow.get('Source IP', f"src_{hash(str(flow)) % 1000}")
                dst_ip = flow.get('Destination IP', f"dst_{hash(str(flow)) % 1000}")
                
                # Add nodes
                G.add_node(src_ip)
                G.add_node(dst_ip)
                
                # Add weighted edge
                weight = flow.get('Total Length of Fwd Packets', 1)
                if G.has_edge(src_ip, dst_ip):
                    G[src_ip][dst_ip]['weight'] += weight
                else:
                    G.add_edge(src_ip, dst_ip, weight=weight)
            
            graphs.append(G)
            
        logger.info(f"Created {len(graphs)} network graphs")
        return graphs
    
    def create_graph_filtrations(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """
        Create filtrations using network structure
        Implementation of Collins et al. (2020) approach
        """
        logger.info("Creating graph-based filtrations...")
        
        filtrations = []
        for G in graphs:
            if len(G.nodes()) == 0:
                continue
                
            # Create adjacency matrix
            nodes = list(G.nodes())
            n_nodes = len(nodes)
            adj_matrix = np.zeros((n_nodes, n_nodes))
            
            # Fill adjacency matrix with weights
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if G.has_edge(u, v):
                        adj_matrix[i, j] = G[u][v]['weight']
            
            # Create filtration based on edge weights
            filtration = self._weight_based_filtration(adj_matrix)
            filtrations.append(filtration)
            
        return filtrations
    
    def _create_time_windows(self, df: pd.DataFrame, window_size: str) -> List[pd.DataFrame]:
        """Create time-based windows for graph construction"""
        # If timestamp column exists, use it
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        
        if not time_cols:
            # Split data into equal chunks
            chunk_size = len(df) // 10  # 10 windows
            windows = []
            for i in range(0, len(df), chunk_size):
                windows.append(df.iloc[i:i + chunk_size])
            return windows
        
        # Use actual timestamps
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy[time_cols[0]], errors='coerce')
        df_copy = df_copy.dropna(subset=['timestamp'])
        
        # Group by time windows
        windows = []
        grouped = df_copy.groupby(pd.Grouper(key='timestamp', freq=window_size))
        for _, group in grouped:
            if len(group) > 0:
                windows.append(group)
                
        return windows if windows else [df]
    
    def _weight_based_filtration(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Create filtration based on edge weights"""
        # Normalize weights
        max_weight = np.max(adj_matrix)
        if max_weight > 0:
            adj_matrix = adj_matrix / max_weight
        
        # Create filtration levels
        filtration_levels = np.linspace(0, 1, 50)
        filtration = np.zeros_like(adj_matrix)
        
        for level in filtration_levels:
            mask = adj_matrix >= level
            filtration[mask] = level
            
        return filtration

def demonstrate_enhanced_construction():
    """Demonstrate enhanced point cloud construction methods"""
    print("🔬 ENHANCED POINT CLOUD CONSTRUCTION DEMO")
    print("=" * 60)
    
    # Create sample network data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        # Temporal features
        'Flow Duration': np.random.exponential(1000, n_samples),
        'Total Fwd Packets': np.random.poisson(10, n_samples),
        'Total Backward Packets': np.random.poisson(5, n_samples),
        
        # Spatial features  
        'Source IP': [f"192.168.1.{i%254+1}" for i in range(n_samples)],
        'Destination IP': [f"10.0.0.{i%100+1}" for i in range(n_samples)],
        'Source Port': np.random.randint(1024, 65535, n_samples),
        'Destination Port': np.random.choice([80, 443, 22, 21, 25], n_samples),
        
        # Behavioral features
        'Total Length of Fwd Packets': np.random.lognormal(8, 2, n_samples),
        'Fwd Packet Length Mean': np.random.normal(500, 200, n_samples),
        'Flow Bytes/s': np.random.lognormal(10, 3, n_samples),
        'Flow Packets/s': np.random.gamma(2, 2, n_samples),
    })
    
    # Test enhanced point cloud construction
    constructor = EnhancedPointCloudConstructor()
    
    temporal_cols = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']
    spatial_cols = ['Source Port', 'Destination Port']  
    behavioral_cols = ['Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Flow Bytes/s', 'Flow Packets/s']
    
    enhanced_cloud = constructor.multi_dimensional_embedding(
        sample_data, temporal_cols, spatial_cols, behavioral_cols
    )
    
    print(f"✅ Enhanced point cloud shape: {enhanced_cloud.shape}")
    print(f"   Features per dimension: {enhanced_cloud.shape[1]}")
    print(f"   Sample points: {enhanced_cloud.shape[0]}")
    
    # Test graph-based construction
    graph_constructor = GraphBasedTDAConstructor()
    graphs = graph_constructor.construct_network_graphs(sample_data)
    filtrations = graph_constructor.create_graph_filtrations(graphs)
    
    print(f"\n✅ Graph-based construction:")
    print(f"   Network graphs: {len(graphs)}")
    print(f"   Filtrations: {len(filtrations)}")
    print(f"   Avg nodes per graph: {np.mean([len(g.nodes()) for g in graphs]):.1f}")
    
    return enhanced_cloud, graphs, filtrations

if __name__ == "__main__":
    demonstrate_enhanced_construction()