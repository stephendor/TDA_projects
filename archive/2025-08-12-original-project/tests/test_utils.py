"""
Test suite for TDA Platform utilities functionality.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from src.utils.data_preprocessing import DataPreprocessor
from src.utils.visualization import TDAVisualizer
from src.utils.evaluation import ModelEvaluator


class TestDataPreprocessor:
    """Test cases for data preprocessing utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create various types of test data
        np.random.seed(42)
        
        # Standard tabular data
        self.tabular_data = np.random.randn(100, 10)
        
        # Time series data
        self.time_series = np.cumsum(np.random.randn(200)) + 100
        
        # Data with missing values
        self.data_with_nans = self.tabular_data.copy()
        self.data_with_nans[10:15, 2:5] = np.nan
        
        # Data with outliers
        self.data_with_outliers = self.tabular_data.copy()
        self.data_with_outliers[5, :] = 10  # Strong outlier
        self.data_with_outliers[25, :] = -8  # Another outlier
    
    def test_initialization(self):
        """Test data preprocessor initialization."""
        assert self.preprocessor is not None
        assert hasattr(self.preprocessor, 'scaler_')
        assert not hasattr(self.preprocessor, 'fitted_')
    
    def test_validate_input_data(self):
        """Test input data validation."""
        # Valid data should pass
        validation_result = self.preprocessor.validate_input_data(self.tabular_data)
        assert validation_result['is_valid'] == True
        assert validation_result['n_samples'] == 100
        assert validation_result['n_features'] == 10
        
        # Empty data should fail
        empty_validation = self.preprocessor.validate_input_data(np.array([]))
        assert empty_validation['is_valid'] == False
        
        # Single point should be flagged
        single_point = np.array([[1, 2, 3]])
        single_validation = self.preprocessor.validate_input_data(single_point)
        assert single_validation['warnings'] is not None
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Test mean imputation
        cleaned_data_mean = self.preprocessor.handle_missing_values(
            self.data_with_nans, method='mean'
        )
        assert not np.any(np.isnan(cleaned_data_mean))
        assert cleaned_data_mean.shape == self.data_with_nans.shape
        
        # Test median imputation
        cleaned_data_median = self.preprocessor.handle_missing_values(
            self.data_with_nans, method='median'
        )
        assert not np.any(np.isnan(cleaned_data_median))
        
        # Test forward fill for time series
        ts_with_nans = self.time_series.copy()
        ts_with_nans[50:55] = np.nan
        cleaned_ts = self.preprocessor.handle_missing_values(
            ts_with_nans.reshape(-1, 1), method='forward_fill'
        )
        assert not np.any(np.isnan(cleaned_ts))
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        outlier_info = self.preprocessor.detect_outliers(
            self.data_with_outliers, method='iqr'
        )
        
        assert 'outlier_indices' in outlier_info
        assert 'outlier_scores' in outlier_info
        assert 'n_outliers' in outlier_info
        
        # Should detect the outliers we inserted
        assert outlier_info['n_outliers'] > 0
        assert 5 in outlier_info['outlier_indices'] or 25 in outlier_info['outlier_indices']
    
    def test_normalize_data(self):
        """Test data normalization."""
        # Standard scaling
        normalized_standard = self.preprocessor.normalize_data(
            self.tabular_data, method='standard'
        )
        assert np.allclose(np.mean(normalized_standard, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(normalized_standard, axis=0), 1, atol=1e-10)
        
        # Min-max scaling
        normalized_minmax = self.preprocessor.normalize_data(
            self.tabular_data, method='minmax'
        )
        assert np.all(normalized_minmax >= 0)
        assert np.all(normalized_minmax <= 1)
        
        # Robust scaling
        normalized_robust = self.preprocessor.normalize_data(
            self.tabular_data, method='robust'
        )
        assert normalized_robust.shape == self.tabular_data.shape
    
    def test_create_time_embeddings(self):
        """Test time series embedding creation."""
        # Test basic embedding
        embeddings = self.preprocessor.create_time_embeddings(
            self.time_series, window_size=10, step_size=1
        )
        
        expected_n_embeddings = len(self.time_series) - 10 + 1
        assert embeddings.shape == (expected_n_embeddings, 10)
        
        # Test with different step size
        embeddings_step2 = self.preprocessor.create_time_embeddings(
            self.time_series, window_size=5, step_size=2
        )
        assert embeddings_step2.shape[1] == 5
        assert embeddings_step2.shape[0] < embeddings.shape[0]  # Fewer embeddings due to step size
    
    def test_dimensionality_reduction(self):
        """Test dimensionality reduction techniques."""
        # PCA reduction
        reduced_pca = self.preprocessor.reduce_dimensionality(
            self.tabular_data, method='pca', n_components=5
        )
        assert reduced_pca.shape == (100, 5)
        
        # UMAP reduction (if available)
        try:
            reduced_umap = self.preprocessor.reduce_dimensionality(
                self.tabular_data, method='umap', n_components=3
            )
            assert reduced_umap.shape == (100, 3)
        except ImportError:
            # UMAP not available, skip this test
            pass
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Test full pipeline
        processed_data = self.preprocessor.preprocess_pipeline(
            self.data_with_nans,
            handle_missing=True,
            detect_outliers=True,
            normalize=True,
            reduce_dims=True,
            target_dims=5
        )
        
        assert processed_data['processed_data'].shape[1] == 5  # Reduced dimensions
        assert not np.any(np.isnan(processed_data['processed_data']))  # No missing values
        assert 'preprocessing_summary' in processed_data


class TestTDAVisualizer:
    """Test cases for TDA visualization utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = TDAVisualizer()
        
        # Create synthetic data for visualization
        np.random.seed(42)
        
        # Point cloud data
        self.point_cloud = np.random.randn(50, 3)
        
        # Synthetic persistence diagram data
        self.persistence_diagram = [
            np.array([[0.0, 0.5], [0.1, 0.8], [0.2, np.inf]]),  # 0-dimensional features
            np.array([[0.3, 0.7], [0.4, 0.9]])                   # 1-dimensional features
        ]
        
        # Distance matrix
        self.distance_matrix = np.random.rand(20, 20)
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        np.fill_diagonal(self.distance_matrix, 0)
    
    def test_initialization(self):
        """Test TDA visualizer initialization."""
        assert self.visualizer is not None
        assert hasattr(self.visualizer, 'figure_size')
        assert hasattr(self.visualizer, 'color_palette')
    
    def test_plot_point_cloud(self):
        """Test point cloud visualization."""
        # 2D point cloud
        point_cloud_2d = self.point_cloud[:, :2]
        fig, ax = self.visualizer.plot_point_cloud(point_cloud_2d, title="Test 2D Point Cloud")
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
        
        # 3D point cloud
        fig, ax = self.visualizer.plot_point_cloud(self.point_cloud, title="Test 3D Point Cloud")
        assert fig is not None
        plt.close(fig)
    
    def test_plot_persistence_diagram(self):
        """Test persistence diagram visualization."""
        fig, ax = self.visualizer.plot_persistence_diagram(
            self.persistence_diagram, 
            title="Test Persistence Diagram"
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_persistence_landscape(self):
        """Test persistence landscape visualization."""
        # Create simple persistence data for landscape
        persistence_data = self.persistence_diagram[1]  # 1D features
        
        fig, ax = self.visualizer.plot_persistence_landscape(
            persistence_data,
            title="Test Persistence Landscape"
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_distance_matrix(self):
        """Test distance matrix heatmap visualization."""
        fig, ax = self.visualizer.plot_distance_matrix(
            self.distance_matrix,
            title="Test Distance Matrix"
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_mapper_graph(self):
        """Test Mapper graph visualization."""
        # Create simple graph structure
        nodes = list(range(10))
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 7)]
        node_colors = np.random.rand(10)
        
        fig, ax = self.visualizer.plot_mapper_graph(
            nodes, edges, node_colors,
            title="Test Mapper Graph"
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_create_dashboard(self):
        """Test comprehensive TDA dashboard creation."""
        # Create mock analysis results
        analysis_results = {
            'point_cloud': self.point_cloud[:, :2],
            'persistence_diagram': self.persistence_diagram,
            'distance_matrix': self.distance_matrix[:10, :10],  # Smaller for visualization
            'summary_stats': {
                'n_points': len(self.point_cloud),
                'n_components_0d': len(self.persistence_diagram[0]),
                'n_components_1d': len(self.persistence_diagram[1])
            }
        }
        
        fig = self.visualizer.create_tda_dashboard(
            analysis_results,
            title="Test TDA Dashboard"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_save_visualization(self):
        """Test saving visualizations to file."""
        # Create a simple plot
        fig, ax = self.visualizer.plot_point_cloud(
            self.point_cloud[:, :2], 
            title="Test Save"
        )
        
        # Test saving (without actually writing to disk in tests)
        with patch('matplotlib.pyplot.savefig') as mock_save:
            self.visualizer.save_figure(fig, "test_plot.png", dpi=150)
            mock_save.assert_called_once()
        
        plt.close(fig)


class TestModelEvaluator:
    """Test cases for model evaluation utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Create synthetic evaluation data
        np.random.seed(42)
        
        # Binary classification data
        self.y_true_binary = np.random.randint(0, 2, 100)
        self.y_pred_binary = np.random.randint(0, 2, 100)
        self.y_scores_binary = np.random.rand(100)
        
        # Multi-class classification data
        self.y_true_multiclass = np.random.randint(0, 3, 100)
        self.y_pred_multiclass = np.random.randint(0, 3, 100)
        
        # Regression data
        self.y_true_regression = np.random.randn(100)
        self.y_pred_regression = self.y_true_regression + np.random.randn(100) * 0.1
    
    def test_initialization(self):
        """Test model evaluator initialization."""
        assert self.evaluator is not None
    
    def test_classification_metrics(self):
        """Test classification performance metrics."""
        # Binary classification
        binary_metrics = self.evaluator.compute_classification_metrics(
            self.y_true_binary, self.y_pred_binary
        )
        
        assert 'accuracy' in binary_metrics
        assert 'precision' in binary_metrics
        assert 'recall' in binary_metrics
        assert 'f1_score' in binary_metrics
        assert 'confusion_matrix' in binary_metrics
        
        # All metrics should be between 0 and 1
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            assert 0 <= binary_metrics[metric] <= 1
        
        # Multi-class classification
        multiclass_metrics = self.evaluator.compute_classification_metrics(
            self.y_true_multiclass, self.y_pred_multiclass
        )
        
        assert 'accuracy' in multiclass_metrics
        assert 'macro_avg' in multiclass_metrics
        assert 'weighted_avg' in multiclass_metrics
    
    def test_regression_metrics(self):
        """Test regression performance metrics."""
        regression_metrics = self.evaluator.compute_regression_metrics(
            self.y_true_regression, self.y_pred_regression
        )
        
        assert 'mae' in regression_metrics  # Mean Absolute Error
        assert 'mse' in regression_metrics  # Mean Squared Error
        assert 'rmse' in regression_metrics  # Root Mean Squared Error
        assert 'r2_score' in regression_metrics  # R-squared
        assert 'explained_variance' in regression_metrics
        
        # MAE and RMSE should be positive
        assert regression_metrics['mae'] >= 0
        assert regression_metrics['rmse'] >= 0
        
        # R-squared should be reasonable (can be negative for bad models)
        assert -1 <= regression_metrics['r2_score'] <= 1
    
    def test_cross_validation(self):
        """Test cross-validation evaluation."""
        # Create a simple mock model
        class MockModel:
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
        
        # Generate synthetic features
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = MockModel()
        
        cv_results = self.evaluator.cross_validate_model(
            model, X, y, cv_folds=3, scoring='accuracy'
        )
        
        assert 'cv_scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'fold_details' in cv_results
        
        # Should have 3 CV scores
        assert len(cv_results['cv_scores']) == 3
        
        # Mean and std should be reasonable
        assert 0 <= cv_results['mean_score'] <= 1
        assert cv_results['std_score'] >= 0
    
    def test_roc_analysis(self):
        """Test ROC curve analysis."""
        roc_results = self.evaluator.compute_roc_analysis(
            self.y_true_binary, self.y_scores_binary
        )
        
        assert 'auc_score' in roc_results
        assert 'fpr' in roc_results
        assert 'tpr' in roc_results
        assert 'thresholds' in roc_results
        
        # AUC should be between 0 and 1
        assert 0 <= roc_results['auc_score'] <= 1
        
        # FPR and TPR arrays should have same length
        assert len(roc_results['fpr']) == len(roc_results['tpr'])
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Create multiple sets of predictions
        model_predictions = {
            'model_a': self.y_pred_binary,
            'model_b': np.random.randint(0, 2, 100),
            'model_c': np.random.randint(0, 2, 100)
        }
        
        comparison_results = self.evaluator.compare_models(
            self.y_true_binary, model_predictions
        )
        
        assert 'model_rankings' in comparison_results
        assert 'detailed_metrics' in comparison_results
        assert 'statistical_tests' in comparison_results
        
        # Should have metrics for all three models
        assert len(comparison_results['detailed_metrics']) == 3
        
        # Rankings should contain all model names
        rankings = comparison_results['model_rankings']
        model_names = set(model_predictions.keys())
        ranked_names = set(ranking['model'] for ranking in rankings)
        assert model_names == ranked_names
    
    def test_learning_curve_analysis(self):
        """Test learning curve analysis."""
        # Create simple mock model for learning curves
        class MockLearningModel:
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                # Simple model that improves with more training data
                return np.random.binomial(1, 0.6, len(X))
        
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, 200)
        model = MockLearningModel()
        
        learning_curve_results = self.evaluator.compute_learning_curve(
            model, X, y, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9]
        )
        
        assert 'train_sizes' in learning_curve_results
        assert 'train_scores' in learning_curve_results
        assert 'validation_scores' in learning_curve_results
        assert 'fit_times' in learning_curve_results
        
        # Should have scores for each training size
        assert len(learning_curve_results['train_scores']) == 5
        assert len(learning_curve_results['validation_scores']) == 5


class TestUtilsIntegration:
    """Integration tests for utils modules."""
    
    def test_preprocessing_visualization_integration(self):
        """Test integration between preprocessing and visualization."""
        np.random.seed(42)
        
        # Create test data
        raw_data = np.random.randn(100, 8)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_result = preprocessor.preprocess_pipeline(
            raw_data,
            normalize=True,
            reduce_dims=True,
            target_dims=3
        )
        
        processed_data = processed_result['processed_data']
        
        # Visualize processed data
        visualizer = TDAVisualizer()
        fig, ax = visualizer.plot_point_cloud(processed_data, title="Processed Data")
        
        assert fig is not None
        assert processed_data.shape == (100, 3)
        plt.close(fig)
    
    def test_evaluation_visualization_integration(self):
        """Test integration between evaluation and visualization."""
        np.random.seed(42)
        
        # Create evaluation data
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        
        # Compute evaluation metrics
        evaluator = ModelEvaluator()
        classification_metrics = evaluator.compute_classification_metrics(y_true, y_pred)
        roc_results = evaluator.compute_roc_analysis(y_true, y_scores)
        
        # Create visualizations
        visualizer = TDAVisualizer()
        
        # Test that we can create evaluation visualizations
        # (Mock the actual plotting since we're testing integration)
        assert classification_metrics['accuracy'] is not None
        assert roc_results['auc_score'] is not None
    
    def test_end_to_end_utils_workflow(self):
        """Test complete utilities workflow."""
        np.random.seed(42)
        
        # Start with raw data
        raw_data = np.random.randn(150, 10)
        raw_data[20:25, :] = np.nan  # Add missing values
        
        # Preprocess
        preprocessor = DataPreprocessor()
        processed_result = preprocessor.preprocess_pipeline(
            raw_data,
            handle_missing=True,
            normalize=True,
            reduce_dims=True,
            target_dims=4
        )
        
        processed_data = processed_result['processed_data']
        
        # Create synthetic labels for evaluation
        labels = np.random.randint(0, 2, len(processed_data))
        predictions = np.random.randint(0, 2, len(processed_data))
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.compute_classification_metrics(labels, predictions)
        
        # Visualize
        visualizer = TDAVisualizer()
        fig, ax = visualizer.plot_point_cloud(
            processed_data[:, :2], 
            title="End-to-End Workflow Result"
        )
        
        # All components should work together
        assert processed_data.shape == (150, 4)
        assert 'accuracy' in metrics
        assert fig is not None
        
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])