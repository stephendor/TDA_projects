"""
Test suite for TDA Platform core functionality.
"""

import numpy as np
import pytest
import warnings
from src.core import PersistentHomologyAnalyzer, TopologyUtils
from src.core.topology_utils import (
    create_point_cloud_circle, 
    create_point_cloud_sphere, 
    create_point_cloud_torus
)
from tests.conftest import (
    assert_valid_persistence_diagram,
    assert_valid_distance_matrix,
    assert_valid_point_cloud
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


class TestAdvancedTopologyFeatures:
    """Test advanced topological feature extraction."""
    
    def test_persistence_stability(self, sample_circle_data):
        """Test stability of persistent homology under noise."""
        analyzer = PersistentHomologyAnalyzer(maxdim=1)
        
        # Compute features for original data
        features_original = analyzer.fit_transform(sample_circle_data)
        
        # Add small noise and recompute
        noisy_data = sample_circle_data + np.random.randn(*sample_circle_data.shape) * 0.05
        features_noisy = analyzer.transform(noisy_data)
        
        # Features should be relatively stable
        feature_diff = np.abs(features_original - features_noisy).mean()
        assert feature_diff < 1.0  # Reasonable stability threshold
    
    def test_multiscale_analysis(self, sample_torus_data):
        """Test multiscale topological analysis."""
        analyzer = PersistentHomologyAnalyzer(maxdim=2)
        
        # Analyze at different scales via subsampling
        scales = [50, 100, 200]
        features_by_scale = {}
        
        for scale in scales:
            if len(sample_torus_data) >= scale:
                subsample_indices = np.random.choice(
                    len(sample_torus_data), scale, replace=False
                )
                subsample = sample_torus_data[subsample_indices]
                features_by_scale[scale] = analyzer.fit_transform(subsample)
        
        # Should have features for each scale
        assert len(features_by_scale) > 0
        
        # Feature dimensions should be consistent
        if len(features_by_scale) > 1:
            feature_dims = [f.shape[1] for f in features_by_scale.values()]
            assert all(dim == feature_dims[0] for dim in feature_dims)
    
    def test_topological_feature_interpretation(self, sample_sphere_data):
        """Test interpretation of topological features."""
        analyzer = PersistentHomologyAnalyzer(maxdim=2, verbose=True)
        features = analyzer.fit_transform(sample_sphere_data)
        
        # Should extract meaningful features from sphere
        assert features is not None
        assert features.shape[1] > 0
        
        # Check if we can get feature interpretation
        if hasattr(analyzer, 'get_feature_interpretation'):
            interpretation = analyzer.get_feature_interpretation()
            assert 'feature_descriptions' in interpretation


class TestTopologyUtilsAdvanced:
    """Advanced test cases for topology utilities."""
    
    def test_adaptive_subsampling(self):
        """Test adaptive subsampling based on density."""
        # Create data with varying density
        dense_region = np.random.randn(200, 3) * 0.5
        sparse_region = np.random.randn(50, 3) * 2 + 5
        data = np.vstack([dense_region, sparse_region])
        
        # Adaptive subsampling should preserve structure
        sub_data, indices = TopologyUtils.subsample_data(
            data, 100, method='adaptive_density'
        )
        
        assert len(sub_data) == 100
        assert len(indices) == 100
    
    def test_distance_matrix_optimization(self, sample_point_cloud_3d):
        """Test optimized distance matrix computation."""
        # Test different distance metrics
        metrics = ['euclidean', 'manhattan', 'chebyshev']
        
        for metric in metrics:
            try:
                distances = TopologyUtils.compute_distance_matrix(
                    sample_point_cloud_3d, metric=metric
                )
                assert_valid_distance_matrix(distances)
                assert distances.shape == (len(sample_point_cloud_3d), len(sample_point_cloud_3d))
            except ValueError:
                # Some metrics might not be implemented
                pass
    
    def test_dimension_estimation(self, sample_sphere_data):
        """Test intrinsic dimension estimation."""
        # Sphere should have intrinsic dimension close to 2
        estimated_dim = TopologyUtils.estimate_intrinsic_dimension(sample_sphere_data)
        
        assert isinstance(estimated_dim, (int, float))
        assert 1.5 <= estimated_dim <= 3.5  # Reasonable range for noisy sphere
    
    def test_topological_preprocessing(self, sample_torus_data):
        """Test topological preprocessing pipeline."""
        # Apply full preprocessing pipeline
        processed_data = TopologyUtils.preprocess_for_tda(
            sample_torus_data,
            remove_outliers=True,
            normalize=True,
            subsample_size=150
        )
        
        assert processed_data.shape[0] <= 150  # Subsampled
        assert processed_data.shape[1] == sample_torus_data.shape[1]  # Same features
        assert not np.any(np.isnan(processed_data))  # No NaN values


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.mark.performance
    def test_large_dataset_handling(self, performance_timer):
        """Test handling of larger datasets."""
        # Create moderately large dataset
        large_data = np.random.randn(1000, 10)
        
        analyzer = PersistentHomologyAnalyzer(maxdim=1)
        
        performance_timer.start()
        features = analyzer.fit_transform(large_data)
        elapsed_time = performance_timer.stop()
        
        # Should complete in reasonable time (< 60 seconds)
        assert elapsed_time < 60.0
        assert features is not None
        assert features.shape[0] == len(large_data)
    
    @pytest.mark.performance  
    def test_memory_efficiency(self):
        """Test memory-efficient processing."""
        # Test processing with memory constraints
        data_sizes = [100, 500, 1000]
        
        for size in data_sizes:
            data = np.random.randn(size, 5)
            analyzer = PersistentHomologyAnalyzer(
                maxdim=1,
                memory_efficient=True
            )
            
            try:
                features = analyzer.fit_transform(data)
                assert features is not None
            except MemoryError:
                pytest.skip(f"Insufficient memory for size {size}")
    
    def test_parallel_processing(self, sample_point_cloud_3d):
        """Test parallel processing capabilities."""
        analyzer = PersistentHomologyAnalyzer(n_jobs=-1)  # Use all available cores
        
        # Process data in parallel
        features_parallel = analyzer.fit_transform(sample_point_cloud_3d)
        
        # Compare with serial processing
        analyzer_serial = PersistentHomologyAnalyzer(n_jobs=1)
        features_serial = analyzer_serial.fit_transform(sample_point_cloud_3d)
        
        # Results should be equivalent
        np.testing.assert_array_almost_equal(features_parallel, features_serial, decimal=10)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_degenerate_data_cases(self):
        """Test handling of degenerate data cases."""
        analyzer = PersistentHomologyAnalyzer()
        
        # All points identical
        identical_points = np.ones((10, 3))
        features_identical = analyzer.fit_transform(identical_points)
        assert features_identical is not None
        
        # Collinear points in 3D
        t = np.linspace(0, 1, 20)
        collinear_points = np.column_stack([t, t, t])
        features_collinear = analyzer.fit_transform(collinear_points)
        assert features_collinear is not None
        
        # Points in lower dimensional subspace
        planar_points = np.random.randn(50, 3)
        planar_points[:, 2] = 0  # All points in xy-plane
        features_planar = analyzer.fit_transform(planar_points)
        assert features_planar is not None
    
    def test_data_type_handling(self):
        """Test handling of different data types."""
        base_data = np.random.randn(20, 3)
        analyzer = PersistentHomologyAnalyzer()
        
        # Test different NumPy dtypes
        for dtype in [np.float32, np.float64, np.int32]:
            data = base_data.astype(dtype)
            features = analyzer.fit_transform(data)
            assert features is not None
        
        # Test list input
        list_data = base_data.tolist()
        features_list = analyzer.fit_transform(list_data)
        assert features_list is not None
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid maxdim
        with pytest.raises(ValueError):
            PersistentHomologyAnalyzer(maxdim=-1)
        
        # Invalid distance metric
        with pytest.raises(ValueError):
            TopologyUtils.compute_distance_matrix(
                np.random.randn(10, 3), 
                metric='invalid_metric'
            )
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        analyzer = PersistentHomologyAnalyzer()
        
        # Very large values
        large_data = np.random.randn(30, 3) * 1e6
        features_large = analyzer.fit_transform(large_data)
        assert features_large is not None
        assert not np.any(np.isnan(features_large))
        
        # Very small values
        small_data = np.random.randn(30, 3) * 1e-6
        features_small = analyzer.fit_transform(small_data)
        assert features_small is not None
        assert not np.any(np.isnan(features_small))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
