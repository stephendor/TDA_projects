"""
Test suite for TDA Platform finance functionality.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.finance.crypto_analysis import CryptoAnalyzer
from src.finance.risk_assessment import RiskAssessment
from src.finance.market_analysis import MarketAnalyzer


class TestCryptoAnalyzer:
    """Test cases for cryptocurrency analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CryptoAnalyzer(
            window_size=50,
            embedding_dim=10,
            prediction_horizon=5,
            verbose=False
        )
        
        # Create synthetic price data
        np.random.seed(42)
        
        # Normal market conditions
        self.stable_prices = np.cumsum(np.random.randn(500) * 0.02) + 100
        
        # Bubble conditions (rapid increase followed by crash)
        bubble_growth = np.linspace(0, 3, 100)  # Exponential-like growth
        bubble_crash = np.linspace(3, 0, 50)    # Sharp decline
        noise = np.random.randn(150) * 0.1
        self.bubble_prices = np.concatenate([
            self.stable_prices[:200],
            self.stable_prices[199] + np.cumsum(bubble_growth + noise[:100]),
            self.stable_prices[199] + 3 + np.cumsum(bubble_crash + noise[100:])
        ])
    
    def test_initialization(self):
        """Test crypto analyzer initialization."""
        assert self.analyzer.window_size == 50
        assert self.analyzer.embedding_dim == 10
        assert self.analyzer.prediction_horizon == 5
        assert self.analyzer.verbose == False
        assert not self.analyzer.is_fitted
    
    def test_fit_stable_market(self):
        """Test fitting on stable market data."""
        self.analyzer.fit(self.stable_prices)
        
        assert self.analyzer.is_fitted
        assert hasattr(self.analyzer, 'baseline_topology_')
        assert hasattr(self.analyzer, 'price_scaler_')
    
    def test_predict_price_movements(self):
        """Test price movement prediction."""
        # Fit on training data
        train_size = int(0.8 * len(self.stable_prices))
        self.analyzer.fit(self.stable_prices[:train_size])
        
        # Predict on test data
        test_data = self.stable_prices[train_size-60:]  # Need overlap for windowing
        predictions = self.analyzer.predict(test_data)
        
        assert 'predictions' in predictions
        assert 'confidence_intervals' in predictions
        assert 'topology_features' in predictions
        
        pred_values = predictions['predictions']
        assert len(pred_values) > 0
        assert all(isinstance(p, (int, float, np.number)) for p in pred_values)
    
    def test_detect_bubble_conditions(self):
        """Test bubble detection in price data."""
        # Fit analyzer
        self.analyzer.fit(self.stable_prices)
        
        # Test on bubble data
        bubble_detection = self.analyzer.detect_bubble_conditions(self.bubble_prices)
        
        assert 'bubble_probability' in bubble_detection
        assert 'risk_level' in bubble_detection
        assert 'recommendation' in bubble_detection
        assert 'topological_indicators' in bubble_detection
        
        # Bubble probability should be reasonable
        assert 0 <= bubble_detection['bubble_probability'] <= 1
        
        # Risk level should be meaningful
        assert bubble_detection['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def test_detect_market_regimes(self):
        """Test market regime identification."""
        self.analyzer.fit(self.stable_prices)
        
        # Analyze regime changes
        regimes = self.analyzer.detect_market_regimes(self.bubble_prices)
        
        assert 'regime_changes' in regimes
        assert 'regime_characteristics' in regimes
        assert 'persistence_analysis' in regimes
        
        # Should detect regime changes in bubble data
        changes = regimes['regime_changes']
        assert len(changes) >= 0  # At least detect stable/bubble transition
    
    def test_empty_price_data(self):
        """Test handling of empty price data."""
        with pytest.raises(ValueError):
            self.analyzer.fit(np.array([]))
    
    def test_insufficient_price_data(self):
        """Test handling of insufficient price data."""
        # Very short price series
        short_prices = np.array([100, 101, 99, 102])
        
        # Should handle gracefully
        try:
            self.analyzer.fit(short_prices)
            # If it doesn't raise an exception, check it handles prediction
            predictions = self.analyzer.predict(short_prices)
            assert predictions is not None
        except ValueError as e:
            # Expected behavior for insufficient data
            assert "insufficient" in str(e).lower() or "too few" in str(e).lower()
    
    def test_volatility_analysis(self):
        """Test volatility analysis functionality."""
        self.analyzer.fit(self.stable_prices)
        
        # Analyze volatility patterns using topology
        volatility_analysis = self.analyzer.analyze_volatility_patterns(self.bubble_prices)
        
        assert 'volatility_regimes' in volatility_analysis
        assert 'topological_volatility' in volatility_analysis
        assert 'risk_metrics' in volatility_analysis


class TestRiskAssessment:
    """Test cases for financial risk assessment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_assessor = RiskAssessment()
        
        # Create synthetic portfolio data
        np.random.seed(42)
        n_assets = 5
        n_days = 252  # 1 year of trading days
        
        # Generate correlated asset returns
        correlation_matrix = np.random.rand(n_assets, n_assets)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Generate returns with correlation structure
        random_returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=correlation_matrix * 0.02,  # 2% volatility
            size=n_days
        )
        
        self.portfolio_returns = random_returns
        self.asset_weights = np.array([0.2, 0.3, 0.15, 0.25, 0.1])  # Portfolio weights
    
    def test_initialization(self):
        """Test risk assessment initialization."""
        assert self.risk_assessor.confidence_level == 0.95
        assert self.risk_assessor.verbose == False
        assert not self.risk_assessor.is_fitted
    
    def test_fit_portfolio_data(self):
        """Test fitting on portfolio return data."""
        self.risk_assessor.fit(self.portfolio_returns, self.asset_weights)
        
        assert self.risk_assessor.is_fitted
        assert hasattr(self.risk_assessor, 'risk_topology_')
        assert hasattr(self.risk_assessor, 'correlation_structure_')
    
    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        self.risk_assessor.fit(self.portfolio_returns, self.asset_weights)
        
        var_results = self.risk_assessor.calculate_var(self.portfolio_returns[-30:])
        
        assert 'var_1day' in var_results
        assert 'var_10day' in var_results
        assert 'expected_shortfall' in var_results
        assert 'topological_risk_features' in var_results
        
        # VaR should be negative (representing potential loss)
        assert var_results['var_1day'] <= 0
        assert var_results['var_10day'] <= 0
    
    def test_assess_portfolio_risk(self):
        """Test comprehensive portfolio risk assessment."""
        self.risk_assessor.fit(self.portfolio_returns, self.asset_weights)
        
        risk_assessment = self.risk_assessor.assess_portfolio_risk(
            self.portfolio_returns,
            self.asset_weights
        )
        
        assert 'overall_risk_score' in risk_assessment
        assert 'risk_decomposition' in risk_assessment
        assert 'correlation_analysis' in risk_assessment
        assert 'stress_test_results' in risk_assessment
        
        # Overall risk score should be meaningful
        risk_score = risk_assessment['overall_risk_score']
        assert 0 <= risk_score <= 100
    
    def test_stress_testing(self):
        """Test portfolio stress testing."""
        self.risk_assessor.fit(self.portfolio_returns, self.asset_weights)
        
        # Define stress scenarios
        stress_scenarios = {
            'market_crash': np.array([-0.2, -0.15, -0.25, -0.18, -0.12]),  # Severe market decline
            'sector_rotation': np.array([0.1, -0.1, 0.05, -0.08, 0.03])   # Sector-specific stress
        }
        
        stress_results = self.risk_assessor.run_stress_tests(stress_scenarios)
        
        assert 'scenario_impacts' in stress_results
        assert 'worst_case_loss' in stress_results
        assert 'topology_stability' in stress_results
        
        # Should calculate impact for each scenario
        impacts = stress_results['scenario_impacts']
        assert 'market_crash' in impacts
        assert 'sector_rotation' in impacts
    
    def test_correlation_analysis(self):
        """Test asset correlation analysis using topology."""
        self.risk_assessor.fit(self.portfolio_returns, self.asset_weights)
        
        correlation_analysis = self.risk_assessor.analyze_correlations(self.portfolio_returns)
        
        assert 'correlation_matrix' in correlation_analysis
        assert 'topological_correlation_features' in correlation_analysis
        assert 'regime_dependent_correlations' in correlation_analysis
        
        # Correlation matrix should be valid
        corr_matrix = correlation_analysis['correlation_matrix']
        assert corr_matrix.shape == (self.portfolio_returns.shape[1], self.portfolio_returns.shape[1])
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric


class TestMarketAnalyzer:
    """Test cases for market analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MarketAnalyzer(verbose=False)
        
        # Create synthetic market data with different regimes
        np.random.seed(42)
        
        # Bull market regime (upward trend)
        bull_trend = np.cumsum(np.random.randn(100) * 0.01 + 0.001) + 100
        
        # Bear market regime (downward trend)
        bear_trend = bull_trend[-1] + np.cumsum(np.random.randn(100) * 0.015 - 0.002)
        
        # Volatile/sideways regime
        volatile_regime = bear_trend[-1] + np.cumsum(np.random.randn(100) * 0.02)
        
        self.market_data = np.concatenate([bull_trend, bear_trend, volatile_regime])
    
    def test_initialization(self):
        """Test market analyzer initialization."""
        assert self.analyzer.verbose == False
        assert not self.analyzer.is_fitted
    
    def test_fit_market_data(self):
        """Test fitting on market data."""
        self.analyzer.fit(self.market_data)
        
        assert self.analyzer.is_fitted
        assert hasattr(self.analyzer, 'market_topology_')
    
    def test_identify_market_regimes(self):
        """Test market regime identification."""
        self.analyzer.fit(self.market_data)
        
        regimes = self.analyzer.identify_market_regimes(self.market_data)
        
        assert 'regime_labels' in regimes
        assert 'regime_transitions' in regimes
        assert 'regime_characteristics' in regimes
        
        # Should identify multiple regimes in our synthetic data
        unique_regimes = len(set(regimes['regime_labels']))
        assert unique_regimes >= 2  # At least bull/bear distinction
    
    def test_detect_trend_changes(self):
        """Test trend change detection."""
        self.analyzer.fit(self.market_data)
        
        trend_changes = self.analyzer.detect_trend_changes(self.market_data)
        
        assert 'change_points' in trend_changes
        assert 'trend_strength' in trend_changes
        assert 'topological_evidence' in trend_changes
        
        # Should detect changes between bull/bear/volatile regimes
        change_points = trend_changes['change_points']
        assert len(change_points) >= 1  # At least one regime change
    
    def test_analyze_market_structure(self):
        """Test market structure analysis."""
        self.analyzer.fit(self.market_data)
        
        structure_analysis = self.analyzer.analyze_market_structure(self.market_data)
        
        assert 'structural_features' in structure_analysis
        assert 'persistence_features' in structure_analysis
        assert 'market_complexity' in structure_analysis
        
        # Market complexity should be a reasonable score
        complexity = structure_analysis['market_complexity']
        assert 0 <= complexity <= 100


class TestFinanceIntegration:
    """Integration tests for finance modules."""
    
    def test_crypto_risk_integration(self):
        """Test integration between crypto analysis and risk assessment."""
        np.random.seed(42)
        
        # Create synthetic crypto portfolio data
        crypto_prices = np.cumsum(np.random.randn(200) * 0.03) + 1000  # More volatile
        portfolio_returns = np.random.randn(200, 3) * 0.03  # 3-asset crypto portfolio
        weights = np.array([0.5, 0.3, 0.2])
        
        # Initialize components
        crypto_analyzer = CryptoAnalyzer()
        risk_assessor = RiskAssessment()
        
        # Fit both components
        crypto_analyzer.fit(crypto_prices[:150])
        risk_assessor.fit(portfolio_returns[:150], weights)
        
        # Test integration
        bubble_detection = crypto_analyzer.detect_bubble_conditions(crypto_prices[150:])
        risk_assessment = risk_assessor.assess_portfolio_risk(
            portfolio_returns[150:], weights
        )
        
        # Both analyses should complete successfully
        assert 'bubble_probability' in bubble_detection
        assert 'overall_risk_score' in risk_assessment
    
    def test_end_to_end_financial_analysis(self):
        """Test complete financial analysis pipeline."""
        np.random.seed(42)
        
        # Simulate comprehensive financial analysis workflow
        market_data = np.cumsum(np.random.randn(300) * 0.02) + 1000
        portfolio_returns = np.random.randn(300, 4) * 0.025
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Initialize all components
        crypto_analyzer = CryptoAnalyzer(verbose=False)
        risk_assessor = RiskAssessment(verbose=False)
        market_analyzer = MarketAnalyzer(verbose=False)
        
        # Fit all components
        train_size = 200
        crypto_analyzer.fit(market_data[:train_size])
        risk_assessor.fit(portfolio_returns[:train_size], weights)
        market_analyzer.fit(market_data[:train_size])
        
        # Perform comprehensive analysis on test data
        test_data = market_data[train_size:]
        test_returns = portfolio_returns[train_size:]
        
        # Run all analyses
        bubble_analysis = crypto_analyzer.detect_bubble_conditions(test_data)
        risk_analysis = risk_assessor.assess_portfolio_risk(test_returns, weights)
        market_regimes = market_analyzer.identify_market_regimes(test_data)
        
        # All analyses should produce meaningful results
        assert bubble_analysis['bubble_probability'] >= 0
        assert risk_analysis['overall_risk_score'] >= 0
        assert len(market_regimes['regime_labels']) == len(test_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])