"""
Pytest configuration and shared fixtures for TDA Platform tests.
"""

import numpy as np
import pytest
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing


@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*ripser.*")
    warnings.filterwarnings("ignore", message=".*matplotlib.*")


@pytest.fixture(scope="session")
def random_seed():
    """Set consistent random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_point_cloud_2d():
    """Generate 2D point cloud for testing."""
    np.random.seed(42)
    return np.random.randn(50, 2)


@pytest.fixture
def sample_point_cloud_3d():
    """Generate 3D point cloud for testing."""
    np.random.seed(42)
    return np.random.randn(75, 3)


@pytest.fixture
def sample_circle_data():
    """Generate noisy circle data for topology testing."""
    np.random.seed(42)
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    radius = 1.0
    noise_level = 0.1
    
    x = radius * np.cos(theta) + np.random.randn(n_points) * noise_level
    y = radius * np.sin(theta) + np.random.randn(n_points) * noise_level
    
    return np.column_stack([x, y])


@pytest.fixture
def sample_sphere_data():
    """Generate noisy sphere data for topology testing."""
    np.random.seed(42)
    n_points = 200
    
    # Generate points on unit sphere with noise
    phi = np.random.uniform(0, 2*np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Add noise
    noise = np.random.randn(n_points, 3) * 0.1
    points = np.column_stack([x, y, z]) + noise
    
    return points


@pytest.fixture
def sample_torus_data():
    """Generate noisy torus data for topology testing."""
    np.random.seed(42)
    n_points = 300
    R = 2.0  # Major radius
    r = 1.0  # Minor radius
    noise_level = 0.1
    
    # Generate angles
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    # Torus parametrization
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    # Add noise
    noise = np.random.randn(n_points, 3) * noise_level
    points = np.column_stack([x, y, z]) + noise
    
    return points


@pytest.fixture
def sample_network_data():
    """Generate synthetic network traffic data for cybersecurity testing."""
    np.random.seed(42)
    
    # Normal network traffic
    normal_traffic = np.random.randn(200, 25)
    
    # Anomalous traffic (shifted distribution)
    anomalous_traffic = np.random.randn(50, 25) + 2
    
    # Combined data
    all_traffic = np.vstack([normal_traffic, anomalous_traffic])
    labels = np.array([0] * 200 + [1] * 50)  # 0 = normal, 1 = anomalous
    
    return {
        'normal_traffic': normal_traffic,
        'anomalous_traffic': anomalous_traffic,
        'all_traffic': all_traffic,
        'labels': labels
    }


@pytest.fixture
def sample_financial_data():
    """Generate synthetic financial data for finance testing."""
    np.random.seed(42)
    
    # Price time series
    n_days = 252  # One year
    returns = np.random.randn(n_days) * 0.02  # 2% daily volatility
    prices = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion
    
    # Multi-asset portfolio
    n_assets = 5
    correlation_matrix = np.random.rand(n_assets, n_assets)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    portfolio_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=correlation_matrix * 0.015,  # 1.5% volatility
        size=n_days
    )
    
    # Portfolio weights
    weights = np.array([0.2, 0.3, 0.15, 0.25, 0.1])
    
    return {
        'prices': prices,
        'returns': returns,
        'portfolio_returns': portfolio_returns,
        'weights': weights,
        'correlation_matrix': correlation_matrix
    }


@pytest.fixture
def sample_persistence_diagram():
    """Generate sample persistence diagram for testing."""
    np.random.seed(42)
    
    # 0-dimensional persistence (components)
    births_0d = np.random.uniform(0, 0.5, 10)
    deaths_0d = births_0d + np.random.uniform(0.1, 1.0, 10)
    persistence_0d = np.column_stack([births_0d, deaths_0d])
    
    # 1-dimensional persistence (loops)
    births_1d = np.random.uniform(0.2, 0.8, 5)
    deaths_1d = births_1d + np.random.uniform(0.1, 0.5, 5)
    # Add one infinite bar
    deaths_1d[0] = np.inf
    persistence_1d = np.column_stack([births_1d, deaths_1d])
    
    return [persistence_0d, persistence_1d]


@pytest.fixture
def sample_distance_matrix():
    """Generate sample distance matrix for testing."""
    np.random.seed(42)
    n = 20
    
    # Generate random distance matrix
    distances = np.random.rand(n, n)
    distances = (distances + distances.T) / 2  # Make symmetric
    np.fill_diagonal(distances, 0)  # Zero diagonal
    
    # Ensure triangle inequality (approximate)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if distances[i, j] + distances[j, k] < distances[i, k]:
                    distances[i, k] = distances[i, j] + distances[j, k]
                    distances[k, i] = distances[i, k]
    
    return distances


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


# Pytest markers for categorizing tests
pytest_markers = [
    "unit: Unit tests for individual components",
    "integration: Integration tests across modules", 
    "performance: Performance and benchmarking tests",
    "visualization: Tests that generate plots/visualizations",
    "slow: Tests that take longer to run",
    "requires_gpu: Tests that require GPU acceleration",
    "requires_internet: Tests that require internet connection"
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


@pytest.fixture
def mock_model_results():
    """Generate mock model results for evaluation testing."""
    np.random.seed(42)
    
    return {
        'binary_classification': {
            'y_true': np.random.randint(0, 2, 100),
            'y_pred': np.random.randint(0, 2, 100),
            'y_scores': np.random.rand(100)
        },
        'multiclass_classification': {
            'y_true': np.random.randint(0, 4, 100),
            'y_pred': np.random.randint(0, 4, 100),
            'y_scores': np.random.rand(100, 4)
        },
        'regression': {
            'y_true': np.random.randn(100),
            'y_pred': np.random.randn(100)
        }
    }


# Test data validation helpers
def assert_valid_persistence_diagram(persistence_diagram):
    """Assert that persistence diagram has valid format."""
    assert isinstance(persistence_diagram, list)
    for dim, pairs in enumerate(persistence_diagram):
        assert isinstance(pairs, np.ndarray)
        assert pairs.shape[1] == 2  # Birth-death pairs
        assert np.all(pairs[:, 0] <= pairs[:, 1])  # Birth <= death


def assert_valid_distance_matrix(distance_matrix):
    """Assert that distance matrix is valid."""
    assert isinstance(distance_matrix, np.ndarray)
    assert distance_matrix.ndim == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    assert np.allclose(distance_matrix, distance_matrix.T)  # Symmetric
    assert np.allclose(np.diag(distance_matrix), 0)  # Zero diagonal
    assert np.all(distance_matrix >= 0)  # Non-negative


def assert_valid_point_cloud(point_cloud, min_points=3, max_dim=10):
    """Assert that point cloud data is valid for TDA."""
    assert isinstance(point_cloud, np.ndarray)
    assert point_cloud.ndim == 2
    assert point_cloud.shape[0] >= min_points
    assert point_cloud.shape[1] <= max_dim
    assert not np.any(np.isnan(point_cloud))
    assert not np.any(np.isinf(point_cloud))


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed_time()
        
        def elapsed_time(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# Skip conditions for optional dependencies
sklearn_available = pytest.mark.skipif(
    not pytest.importorskip("sklearn"),
    reason="scikit-learn not available"
)

gudhi_available = pytest.mark.skipif(
    not pytest.importorskip("gudhi"),
    reason="GUDHI not available"
)

ripser_available = pytest.mark.skipif(
    not pytest.importorskip("ripser"),
    reason="Ripser not available"
)