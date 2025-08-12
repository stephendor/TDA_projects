"""
Cryptocurrency Analysis using TDA

This module implements TDA-based methods for analyzing cryptocurrency
markets, detecting bubbles, and predicting market movements.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings

from ..core import PersistentHomologyAnalyzer, MapperAnalyzer, TopologyUtils


class CryptoAnalyzer(BaseEstimator, RegressorMixin):
    """
    TDA-based cryptocurrency market analyzer.
    
    This class uses topological features to analyze cryptocurrency price movements,
    detect market regimes, and predict future trends.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        embedding_dim: int = 10,
        ph_maxdim: int = 2,
        prediction_horizon: int = 1,
        feature_combination: str = 'all',
        volatility_adjustment: bool = True,
        verbose: bool = False
    ):
        """
        Initialize cryptocurrency analyzer.
        
        Parameters:
        -----------
        window_size : int, default=50
            Size of sliding window for analysis
        embedding_dim : int, default=10
            Dimension for time series embedding
        ph_maxdim : int, default=2
            Maximum dimension for persistent homology
        prediction_horizon : int, default=1
            Number of periods ahead to predict
        feature_combination : str, default='all'
            Feature combination strategy ('all', 'tda_only', 'traditional_only')
        volatility_adjustment : bool, default=True
            Whether to adjust for volatility
        verbose : bool, default=False
            Verbose output
        """
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.ph_maxdim = ph_maxdim
        self.prediction_horizon = prediction_horizon
        self.feature_combination = feature_combination
        self.volatility_adjustment = volatility_adjustment
        self.verbose = verbose
        
        # Components
        self.ph_analyzer = PersistentHomologyAnalyzer(maxdim=ph_maxdim)
        self.mapper_analyzer = MapperAnalyzer(n_intervals=12, overlap_frac=0.3)
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # State
        self.is_fitted_ = False
        self.feature_names_ = None
        self.market_regimes_ = None
    
    def fit(self, price_data: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the cryptocurrency analyzer.
        
        Parameters:
        -----------
        price_data : np.ndarray
            Historical price data (time series)
        y : np.ndarray, optional
            Target values (if None, will predict future prices)
            
        Returns:
        --------
        self : CryptoAnalyzer
        """
        price_data = np.asarray(price_data).flatten()
        
        if self.verbose:
            print(f"Fitting crypto analyzer on {len(price_data)} price points")
        
        # Create features and targets
        X_features, y_targets = self._create_features_and_targets(price_data, y)
        
        if len(X_features) == 0:
            raise ValueError("Not enough data to create features")
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X_features)
        
        # Fit regressor
        self.regressor.fit(X_scaled, y_targets)
        
        self.is_fitted_ = True
        
        if self.verbose:
            print("Crypto analyzer fitting completed")
        
        return self
    
    def predict(self, price_data: np.ndarray) -> np.ndarray:
        """
        Predict future cryptocurrency prices.
        
        Parameters:
        -----------
        price_data : np.ndarray
            Recent price data for prediction
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted future prices
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict()")
        
        price_data = np.asarray(price_data).flatten()
        
        # Extract features from recent data
        X_features = self._extract_features(price_data)
        
        if len(X_features) == 0:
            return np.array([])
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X_features)
        
        # Make predictions
        predictions = self.regressor.predict(X_scaled)
        
        return predictions
    
    def _create_features_and_targets(
        self, 
        price_data: np.ndarray, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features and targets from price data.
        
        Parameters:
        -----------
        price_data : np.ndarray
            Price time series
        y : np.ndarray, optional
            Target values
            
        Returns:
        --------
        X_features : np.ndarray
            Feature matrix
        y_targets : np.ndarray
            Target values
        """
        # Extract features using sliding windows
        features_list = []
        targets_list = []
        
        for i in range(self.window_size, len(price_data) - self.prediction_horizon):
            # Extract window
            window_data = price_data[i-self.window_size:i]
            
            # Extract features
            window_features = self._extract_features_from_window(window_data)
            features_list.append(window_features)
            
            # Create target
            if y is not None:
                target = y[i]
            else:
                # Predict future price change
                current_price = price_data[i]
                future_price = price_data[i + self.prediction_horizon]
                target = (future_price - current_price) / current_price  # Relative change
            
            targets_list.append(target)
        
        if not features_list:
            return np.array([]), np.array([])
        
        return np.array(features_list), np.array(targets_list)
    
    def _extract_features(self, price_data: np.ndarray) -> np.ndarray:
        """
        Extract features from price data.
        
        Parameters:
        -----------
        price_data : np.ndarray
            Price time series
            
        Returns:
        --------
        features : np.ndarray
            Extracted features
        """
        if len(price_data) < self.window_size:
            return np.array([])
        
        # Use the most recent window
        window_data = price_data[-self.window_size:]
        features = self._extract_features_from_window(window_data)
        
        return features.reshape(1, -1)
    
    def _extract_features_from_window(self, window_data: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive features from a price window.
        
        Parameters:
        -----------
        window_data : np.ndarray
            Price window
            
        Returns:
        --------
        features : np.ndarray
            Feature vector
        """
        all_features = []
        
        # Traditional technical features
        if self.feature_combination in ['all', 'traditional_only']:
            traditional_features = self._extract_traditional_features(window_data)
            all_features.extend(traditional_features)
        
        # TDA features
        if self.feature_combination in ['all', 'tda_only']:
            tda_features = self._extract_tda_features(window_data)
            all_features.extend(tda_features)
        
        return np.array(all_features)
    
    def _extract_traditional_features(self, prices: np.ndarray) -> List[float]:
        """
        Extract traditional technical analysis features.
        
        Parameters:
        -----------
        prices : np.ndarray
            Price data
            
        Returns:
        --------
        features : list
            Traditional features
        """
        features = []
        
        # Price statistics
        features.extend([
            np.mean(prices),
            np.std(prices),
            np.min(prices),
            np.max(prices),
            (prices[-1] - prices[0]) / prices[0],  # Total return
        ])
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        if len(returns) > 0:
            features.extend([
                np.mean(returns),
                np.std(returns),
                stats.skew(returns) if len(returns) > 2 else 0,
                stats.kurtosis(returns) if len(returns) > 3 else 0,
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Moving averages
        if len(prices) >= 5:
            ma5 = np.mean(prices[-5:])
            features.append((prices[-1] - ma5) / ma5)
        else:
            features.append(0)
        
        if len(prices) >= 10:
            ma10 = np.mean(prices[-10:])
            features.append((prices[-1] - ma10) / ma10)
        else:
            features.append(0)
        
        # Volatility
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            features.append(volatility)
        else:
            features.append(0)
        
        return features
    
    def _extract_tda_features(self, prices: np.ndarray) -> List[float]:
        """
        Extract TDA-based features.
        
        Parameters:
        -----------
        prices : np.ndarray
            Price data
            
        Returns:
        --------
        features : list
            TDA features
        """
        features = []
        
        # Create time series embedding
        embedding = TopologyUtils.sliding_window_embedding(
            prices, 
            window_size=min(self.embedding_dim, len(prices)//2),
            step_size=1
        )
        
        if len(embedding) < 3:
            # Not enough data for TDA
            return [0] * (self._get_tda_feature_dim())
        
        # Persistent homology features
        try:
            ph_features = self.ph_analyzer.fit_transform(embedding).flatten()
            features.extend(ph_features)
        except Exception as e:
            if self.verbose:
                warnings.warn(f"PH computation failed: {e}")
            features.extend([0] * ((self.ph_maxdim + 1) * 6))
        
        # Mapper features
        try:
            mapper_props = self.mapper_analyzer.fit_transform(embedding)
            mapper_features = [
                mapper_props.get('n_nodes', 0),
                mapper_props.get('n_edges', 0),
                mapper_props.get('n_components', 0),
                mapper_props.get('avg_clustering', 0),
                mapper_props.get('diameter', 0) if np.isfinite(mapper_props.get('diameter', np.inf)) else 0
            ]
            features.extend(mapper_features)
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Mapper computation failed: {e}")
            features.extend([0] * 5)
        
        # Topological complexity measures
        try:
            complexity_features = self._compute_topological_complexity(embedding)
            features.extend(complexity_features)
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Complexity computation failed: {e}")
            features.extend([0] * 3)
        
        return features
    
    def _compute_topological_complexity(self, embedding: np.ndarray) -> List[float]:
        """
        Compute topological complexity measures.
        
        Parameters:
        -----------
        embedding : np.ndarray
            Time series embedding
            
        Returns:
        --------
        complexity : list
            Complexity measures
        """
        if len(embedding) < 3:
            return [0, 0, 0]
        
        # Intrinsic dimension estimate
        try:
            intrinsic_dim = TopologyUtils.estimate_intrinsic_dimension(embedding)
        except:
            intrinsic_dim = 0
        
        # Effective dimension (ratio of variances)
        try:
            _, s, _ = np.linalg.svd(embedding - np.mean(embedding, axis=0))
            s_normalized = s / np.sum(s)
            effective_dim = np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10)))
        except:
            effective_dim = 0
        
        # Trajectory curvature
        try:
            if len(embedding) > 2:
                diffs = np.diff(embedding, axis=0)
                curvatures = []
                for i in range(len(diffs) - 1):
                    v1, v2 = diffs[i], diffs[i+1]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    curvature = 1 - cos_angle
                    curvatures.append(curvature)
                avg_curvature = np.mean(curvatures)
            else:
                avg_curvature = 0
        except:
            avg_curvature = 0
        
        return [intrinsic_dim, effective_dim, avg_curvature]
    
    def _get_tda_feature_dim(self) -> int:
        """Get TDA feature dimension."""
        return (self.ph_maxdim + 1) * 6 + 5 + 3  # PH + Mapper + Complexity
    
    def detect_market_regimes(
        self, 
        price_data: np.ndarray, 
        n_regimes: int = 3
    ) -> Dict[str, Any]:
        """
        Detect market regimes using topological features.
        
        Parameters:
        -----------
        price_data : np.ndarray
            Price time series
        n_regimes : int, default=3
            Number of market regimes to identify
            
        Returns:
        --------
        regimes : dict
            Market regime analysis
        """
        from sklearn.cluster import KMeans
        
        # Extract features for all windows
        features_list = []
        timestamps = []
        
        for i in range(self.window_size, len(price_data)):
            window_data = price_data[i-self.window_size:i]
            features = self._extract_features_from_window(window_data)
            features_list.append(features)
            timestamps.append(i)
        
        if not features_list:
            return {'regimes': [], 'transitions': []}
        
        X_features = np.array(features_list)
        
        # Scale features
        X_scaled = StandardScaler().fit_transform(X_features)
        
        # Cluster into regimes
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(X_scaled)
        
        # Identify regime transitions
        transitions = []
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                transitions.append({
                    'timestamp': timestamps[i],
                    'from_regime': regime_labels[i-1],
                    'to_regime': regime_labels[i],
                    'price': price_data[timestamps[i]]
                })
        
        # Characterize regimes
        regime_characteristics = {}
        for regime_id in range(n_regimes):
            mask = regime_labels == regime_id
            if np.sum(mask) > 0:
                regime_timestamps = np.array(timestamps)[mask]
                regime_prices = price_data[regime_timestamps]
                regime_returns = np.diff(regime_prices) / regime_prices[:-1]
                
                regime_characteristics[regime_id] = {
                    'n_periods': np.sum(mask),
                    'avg_return': np.mean(regime_returns) if len(regime_returns) > 0 else 0,
                    'volatility': np.std(regime_returns) if len(regime_returns) > 0 else 0,
                    'avg_price': np.mean(regime_prices),
                    'duration_avg': np.mean(np.diff(regime_timestamps)) if len(regime_timestamps) > 1 else 0
                }
        
        return {
            'regime_labels': regime_labels,
            'timestamps': timestamps,
            'regime_characteristics': regime_characteristics,
            'transitions': transitions,
            'n_transitions': len(transitions)
        }
    
    def detect_bubble_conditions(
        self, 
        price_data: np.ndarray, 
        lookback_period: int = 100
    ) -> Dict[str, Any]:
        """
        Detect potential bubble conditions using TDA.
        
        Parameters:
        -----------
        price_data : np.ndarray
            Price time series
        lookback_period : int, default=100
            Lookback period for analysis
            
        Returns:
        --------
        bubble_analysis : dict
            Bubble detection results
        """
        if len(price_data) < lookback_period:
            return {'bubble_probability': 0, 'confidence': 0, 'features': {}}
        
        # Analyze recent period
        recent_data = price_data[-lookback_period:]
        
        # Extract features
        features = self._extract_features_from_window(recent_data)
        
        # Compute bubble indicators
        returns = np.diff(recent_data) / recent_data[:-1]
        
        # Traditional bubble indicators
        price_acceleration = np.polyfit(range(len(recent_data)), recent_data, 2)[0]
        volatility_trend = np.polyfit(range(len(returns)), 
                                    pd.Series(returns).rolling(10).std().fillna(0), 1)[0]
        
        # TDA-based bubble indicators
        embedding = TopologyUtils.sliding_window_embedding(recent_data, window_size=10)
        
        if len(embedding) >= 3:
            # Topological instability
            ph_diagrams = self.ph_analyzer.fit(embedding).persistence_diagrams_
            instability = self._compute_topological_instability(ph_diagrams)
            
            # Mapper complexity
            mapper_props = self.mapper_analyzer.fit_transform(embedding)
            complexity = mapper_props.get('n_components', 1) / mapper_props.get('n_nodes', 1)
        else:
            instability = 0
            complexity = 0
        
        # Combine indicators
        bubble_score = (
            0.3 * (price_acceleration > 0) +
            0.2 * (volatility_trend > 0) +
            0.3 * min(instability, 1) +
            0.2 * min(complexity, 1)
        )
        
        return {
            'bubble_probability': bubble_score,
            'confidence': min(len(recent_data) / lookback_period, 1),
            'features': {
                'price_acceleration': price_acceleration,
                'volatility_trend': volatility_trend,
                'topological_instability': instability,
                'mapper_complexity': complexity
            },
            'recommendation': 'HIGH_RISK' if bubble_score > 0.7 else 
                           'MODERATE_RISK' if bubble_score > 0.4 else 'LOW_RISK'
        }
    
    def _compute_topological_instability(self, diagrams: List[np.ndarray]) -> float:
        """
        Compute topological instability measure.
        
        Parameters:
        -----------
        diagrams : List[np.ndarray]
            Persistence diagrams
            
        Returns:
        --------
        instability : float
            Instability measure
        """
        if not diagrams or len(diagrams) == 0:
            return 0
        
        instability = 0
        
        for dgm in diagrams:
            if len(dgm) == 0:
                continue
            
            # Remove infinite points
            finite_mask = np.isfinite(dgm[:, 1])
            finite_dgm = dgm[finite_mask]
            
            if len(finite_dgm) > 0:
                lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                # High instability if many short-lived features
                short_lived_ratio = np.sum(lifetimes < np.median(lifetimes)) / len(lifetimes)
                instability += short_lived_ratio
        
        return instability / len(diagrams) if len(diagrams) > 0 else 0


def analyze_cryptocurrency_derivatives(
    spot_prices: np.ndarray,
    futures_prices: Optional[np.ndarray] = None,
    options_data: Optional[Dict[str, np.ndarray]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze cryptocurrency derivatives using TDA methods.
    
    Parameters:
    -----------
    spot_prices : np.ndarray
        Spot price time series
    futures_prices : np.ndarray, optional
        Futures price time series
    options_data : Dict[str, np.ndarray], optional
        Options data (implied volatility, etc.)
    **kwargs : dict
        Additional arguments for CryptoAnalyzer
        
    Returns:
    --------
    analysis : dict
        Derivatives analysis results
    """
    results = {}
    
    # Spot price analysis
    spot_analyzer = CryptoAnalyzer(**kwargs)
    spot_analyzer.fit(spot_prices)
    
    results['spot_analysis'] = {
        'regimes': spot_analyzer.detect_market_regimes(spot_prices),
        'bubble_risk': spot_analyzer.detect_bubble_conditions(spot_prices),
        'predictions': spot_analyzer.predict(spot_prices[-50:]) if len(spot_prices) >= 50 else []
    }
    
    # Futures analysis
    if futures_prices is not None:
        futures_analyzer = CryptoAnalyzer(**kwargs)
        futures_analyzer.fit(futures_prices)
        
        # Compute basis (futures - spot premium)
        min_len = min(len(spot_prices), len(futures_prices))
        basis = futures_prices[-min_len:] - spot_prices[-min_len:]
        basis_returns = np.diff(basis) / np.abs(basis[:-1] + 1e-10)
        
        results['futures_analysis'] = {
            'regimes': futures_analyzer.detect_market_regimes(futures_prices),
            'basis_statistics': {
                'mean_basis': float(np.mean(basis)),
                'basis_volatility': float(np.std(basis_returns)) if len(basis_returns) > 0 else 0,
                'contango_backwardation': 'contango' if np.mean(basis) > 0 else 'backwardation'
            }
        }
    
    # Options analysis (if available)
    if options_data is not None:
        options_analysis = {}
        
        for key, data in options_data.items():
            if len(data) > 10:  # Minimum data for analysis
                # Analyze implied volatility surface topology
                iv_kwargs = kwargs.copy()
                iv_kwargs['window_size'] = min(20, len(data)//2)
                iv_analyzer = CryptoAnalyzer(**iv_kwargs)
                iv_analyzer.fit(data)
                
                options_analysis[key] = {
                    'regimes': iv_analyzer.detect_market_regimes(data),
                    'volatility_statistics': {
                        'mean_iv': float(np.mean(data)),
                        'iv_volatility': float(np.std(np.diff(data) / data[:-1])) if len(data) > 1 else 0
                    }
                }
        
        results['options_analysis'] = options_analysis
    
    # Cross-asset correlation analysis
    if futures_prices is not None:
        min_len = min(len(spot_prices), len(futures_prices))
        spot_returns = np.diff(spot_prices[-min_len:]) / spot_prices[-min_len:-1]
        futures_returns = np.diff(futures_prices[-min_len:]) / futures_prices[-min_len:-1]
        
        if len(spot_returns) > 1:
            correlation = np.corrcoef(spot_returns, futures_returns)[0, 1]
            results['cross_asset_analysis'] = {
                'spot_futures_correlation': float(correlation) if np.isfinite(correlation) else 0
            }
    
    return results


def analyze_defi_yield_strategies(
    yield_data: Dict[str, np.ndarray],
    risk_free_rate: float = 0.02,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze DeFi yield farming strategies using topological risk measures.
    
    Parameters:
    -----------
    yield_data : Dict[str, np.ndarray]
        Dictionary mapping strategy names to yield time series
    risk_free_rate : float, default=0.02
        Risk-free rate for Sharpe ratio calculation
    **kwargs : dict
        Additional arguments for CryptoAnalyzer
        
    Returns:
    --------
    analysis : dict
        DeFi strategy analysis results
    """
    strategy_analyses = {}
    
    for strategy_name, yields in yield_data.items():
        if len(yields) < 10:
            continue
        
        # Analyze yield time series
        yield_kwargs = kwargs.copy()
        yield_kwargs['window_size'] = min(15, len(yields)//3)
        yield_analyzer = CryptoAnalyzer(**yield_kwargs)
        yield_analyzer.fit(yields)
        
        # Calculate risk metrics
        yield_returns = np.diff(yields) / np.abs(yields[:-1] + 1e-10)
        
        # Traditional metrics
        avg_yield = float(np.mean(yields))
        yield_volatility = float(np.std(yield_returns)) if len(yield_returns) > 0 else 0
        sharpe_ratio = (avg_yield - risk_free_rate) / (yield_volatility + 1e-10)
        
        # Topological risk metrics
        regimes = yield_analyzer.detect_market_regimes(yields)
        bubble_risk = yield_analyzer.detect_bubble_conditions(yields)
        
        # Drawdown analysis
        cumulative_yields = np.cumprod(1 + yield_returns)
        running_max = np.maximum.accumulate(cumulative_yields)
        drawdowns = (cumulative_yields - running_max) / running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        strategy_analyses[strategy_name] = {
            'traditional_metrics': {
                'average_yield': avg_yield,
                'volatility': yield_volatility,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': max_drawdown
            },
            'topological_analysis': {
                'regimes': regimes,
                'bubble_risk': bubble_risk,
                'regime_stability': regimes['n_transitions'] / len(yields) if len(yields) > 0 else 0
            }
        }
    
    # Rank strategies by risk-adjusted returns
    strategy_rankings = []
    for name, analysis in strategy_analyses.items():
        risk_score = (
            0.4 * (1 - analysis['topological_analysis']['bubble_risk']['bubble_probability']) +
            0.3 * analysis['traditional_metrics']['sharpe_ratio'] +
            0.3 * (1 + analysis['traditional_metrics']['max_drawdown'])  # max_drawdown is negative
        )
        
        strategy_rankings.append({
            'strategy': name,
            'risk_adjusted_score': risk_score,
            'recommendation': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.4 else 'LOW'
        })
    
    strategy_rankings.sort(key=lambda x: x['risk_adjusted_score'], reverse=True)
    
    return {
        'strategy_analyses': strategy_analyses,
        'rankings': strategy_rankings,
        'summary': {
            'total_strategies': len(strategy_analyses),
            'high_rated': len([s for s in strategy_rankings if s['recommendation'] == 'HIGH']),
            'best_strategy': strategy_rankings[0]['strategy'] if strategy_rankings else None
        }
    }


def analyze_cryptocurrency_portfolio(
    price_data_dict: Dict[str, np.ndarray],
    weights: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a cryptocurrency portfolio using TDA.
    
    Parameters:
    -----------
    price_data_dict : Dict[str, np.ndarray]
        Dictionary mapping crypto symbols to price arrays
    weights : np.ndarray, optional
        Portfolio weights (equal weights if None)
    **kwargs : dict
        Additional arguments for CryptoAnalyzer
        
    Returns:
    --------
    analysis : dict
        Portfolio analysis results
    """
    symbols = list(price_data_dict.keys())
    price_data = np.array([price_data_dict[symbol] for symbol in symbols])
    
    if weights is None:
        weights = np.ones(len(symbols)) / len(symbols)
    
    # Create portfolio price series
    portfolio_prices = np.sum(price_data * weights.reshape(-1, 1), axis=0)
    
    # Analyze portfolio
    analyzer = CryptoAnalyzer(**kwargs)
    analyzer.fit(portfolio_prices)
    
    # Individual asset analysis
    asset_analyses = {}
    for symbol in symbols:
        asset_analyzer = CryptoAnalyzer(**kwargs)
        asset_analyzer.fit(price_data_dict[symbol])
        
        asset_analyses[symbol] = {
            'regimes': asset_analyzer.detect_market_regimes(price_data_dict[symbol]),
            'bubble_risk': asset_analyzer.detect_bubble_conditions(price_data_dict[symbol])
        }
    
    # Portfolio-level analysis
    portfolio_regimes = analyzer.detect_market_regimes(portfolio_prices)
    portfolio_bubble_risk = analyzer.detect_bubble_conditions(portfolio_prices)
    
    # Risk attribution analysis
    asset_contributions = {}
    for i, symbol in enumerate(symbols):
        asset_weight = weights[i]
        asset_volatility = np.std(np.diff(price_data[i]) / price_data[i][:-1])
        risk_contribution = asset_weight * asset_volatility
        asset_contributions[symbol] = {
            'weight': float(asset_weight),
            'volatility': float(asset_volatility),
            'risk_contribution': float(risk_contribution)
        }
    
    return {
        'portfolio_analysis': {
            'regimes': portfolio_regimes,
            'bubble_risk': portfolio_bubble_risk,
            'risk_attribution': asset_contributions
        },
        'asset_analyses': asset_analyses,
        'weights': weights,
        'symbols': symbols
    }
