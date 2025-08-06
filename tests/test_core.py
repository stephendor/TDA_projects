"""
Test suite for TDA Platform core functionality.
"""

import numpy as np
import pytest
from src.core import PersistentHomologyAnalyzer, TopologyUtils
from src.core.topology_utils import (
    create_point_cloud_circle, 
    create_point_cloud_sphere, 
    create_point_cloud_torus
)


class TestPersistentHomologyAnalyzer:
    """Test cases for persistent homology analysis."""
    
    def test_circle_data(self):
        """Test PH analysis on circle data."""
        # Create circle point cloud
        circle_data = create_point_cloud_circle(n_points=100, noise=0.1)
        
        analyzer = PersistentHomologyAnalyzer(maxdim=1)
        features = analyzer.fit_transform(circle_data)
        
        # Should detect 1D cycle (circle)
        assert features is not None
        assert len(features.shape) == 2
        assert features.shape[1] > 0  # Should have features
    
    def test_empty_data(self):
        """Test handling of empty data."""
        analyzer = PersistentHomologyAnalyzer()
        
        with pytest.raises(ValueError):
            analyzer.fit(np.array([]))
    
    def test_single_point(self):
        """Test handling of single point."""
        analyzer = PersistentHomologyAnalyzer()
        
        # Single point should not crash
        single_point = np.array([[1, 2, 3]])
        features = analyzer.fit_transform(single_point)
        
        assert features is not None


class TestTopologyUtils:
    """Test cases for topology utilities."""
    
    def test_distance_matrix(self):
        """Test distance matrix computation."""
        points = np.random.randn(10, 3)
        distances = TopologyUtils.compute_distance_matrix(points)
        
        assert distances.shape == (10, 10)
        assert np.allclose(distances, distances.T)  # Should be symmetric
        assert np.allclose(np.diag(distances), 0)  # Diagonal should be zero
    
    def test_subsampling(self):
        """Test data subsampling."""
        data = np.random.randn(100, 5)
        
        # Random subsampling
        sub_data, indices = TopologyUtils.subsample_data(data, 50, method='random')
        assert len(sub_data) == 50
        assert len(indices) == 50
        
        # Farthest point sampling
        sub_data, indices = TopologyUtils.subsample_data(data, 20, method='farthest_point')
        assert len(sub_data) == 20
    
    def test_synthetic_shapes(self):
        """Test synthetic shape generation."""
        # Circle
        circle = create_point_cloud_circle(n_points=50)
        assert circle.shape == (50, 2)
        
        # Sphere
        sphere = create_point_cloud_sphere(n_points=100)
        assert sphere.shape == (100, 3)
        
        # Torus
        torus = create_point_cloud_torus(n_points=200)
        assert torus.shape == (200, 3)


if __name__ == "__main__":
    pytest.main([__file__])
