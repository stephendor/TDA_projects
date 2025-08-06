"""
Finance API Routes

Provides endpoints for financial risk analysis using TDA methods,
including cryptocurrency bubble detection and portfolio risk assessment.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status, Query
from pydantic import BaseModel, Field, validator

from ...finance import CryptoAnalyzer, RiskAssessment, MarketAnalyzer
from ...utils.database import get_db_manager
from ...utils.cache import get_cache_manager


router = APIRouter()


# Request Models
class PriceDataRequest(BaseModel):
    """Request model for price time series data."""
    prices: List[float] = Field(..., description="Price time series data")
    timestamps: Optional[List[str]] = Field(None, description="Timestamps for price data")
    asset_symbol: str = Field("UNKNOWN", description="Asset symbol (e.g., BTC, ETH)")
    
    @validator('prices')
    def validate_prices(cls, v):
        if len(v) < 50:
            raise ValueError("Minimum 50 price points required for analysis")
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive")
        return v


class CryptoBubbleRequest(PriceDataRequest):
    """Request model for cryptocurrency bubble detection."""
    window_size: int = Field(default=50, ge=20, le=200, description="Analysis window size")
    embedding_dim: int = Field(default=10, ge=3, le=20, description="Time series embedding dimension")
    prediction_horizon: int = Field(default=5, ge=1, le=30, description="Prediction horizon in days")


class PortfolioRiskRequest(BaseModel):
    """Request model for portfolio risk assessment."""
    returns_data: List[List[float]] = Field(..., description="Asset returns matrix (time x assets)")
    asset_weights: List[float] = Field(..., description="Portfolio weights for each asset")
    asset_symbols: Optional[List[str]] = Field(None, description="Asset symbols")
    confidence_level: float = Field(default=0.95, ge=0.90, le=0.99, description="VaR confidence level")
    
    @validator('returns_data')
    def validate_returns(cls, v):
        if len(v) < 100:
            raise ValueError("Minimum 100 time periods required for risk analysis")
        return v
    
    @validator('asset_weights')
    def validate_weights(cls, v):
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError("Asset weights must sum to 1.0")
        return v


class MarketAnalysisRequest(PriceDataRequest):
    """Request model for market regime analysis."""
    include_regimes: bool = Field(default=True, description="Include regime identification")
    include_trends: bool = Field(default=True, description="Include trend analysis")
    include_structure: bool = Field(default=True, description="Include market structure analysis")


# Response Models
class BubbleDetection(BaseModel):
    """Model for bubble detection result."""
    bubble_probability: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendation: str
    confidence_interval: Dict[str, float]
    topological_indicators: Dict[str, Any]
    time_to_potential_burst: Optional[int] = None


class CryptoBubbleResponse(BaseModel):
    """Response model for cryptocurrency bubble detection."""
    analysis_id: str
    status: str
    computation_time_ms: float
    asset_symbol: str
    analysis_period: Dict[str, str]
    bubble_detection: BubbleDetection
    market_regimes: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    price_predictions: List[Dict[str, float]]
    topological_features: Dict[str, Any]


class RiskMeasure(BaseModel):
    """Model for risk measure result."""
    var_1day: float
    var_10day: float
    expected_shortfall: float
    confidence_level: float


class PortfolioRiskResponse(BaseModel):
    """Response model for portfolio risk assessment."""
    analysis_id: str
    status: str
    computation_time_ms: float
    portfolio_summary: Dict[str, Any]
    risk_measures: RiskMeasure
    risk_decomposition: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    stress_test_results: Dict[str, Any]
    topological_risk_features: Dict[str, Any]
    overall_risk_score: int


class MarketRegime(BaseModel):
    """Model for market regime identification."""
    regime_id: int
    regime_type: str
    start_period: int
    end_period: int
    characteristics: Dict[str, Any]


class MarketAnalysisResponse(BaseModel):
    """Response model for market analysis."""
    analysis_id: str
    status: str
    computation_time_ms: float
    asset_symbol: str
    market_regimes: List[MarketRegime]
    trend_analysis: Dict[str, Any]
    structure_analysis: Dict[str, Any]
    volatility_patterns: Dict[str, Any]
    topological_summary: Dict[str, Any]


# Cryptocurrency Analysis Endpoints
@router.post("/crypto/bubble-detection", response_model=CryptoBubbleResponse)
async def detect_crypto_bubble(
    request: CryptoBubbleRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Detect cryptocurrency bubble conditions using topological analysis.
    
    Analyzes price time series data to identify bubble formation patterns
    and predict potential market corrections.
    """
    import time
    start_time = time.time()
    
    analysis_id = _generate_analysis_id("crypto_bubble", request.dict())
    
    try:
        prices = np.array(request.prices)
        
        # Check cache
        cached_result = await cache.get(f"crypto_bubble:{analysis_id}")
        if cached_result:
            return CryptoBubbleResponse.parse_obj(cached_result)
        
        # Initialize crypto analyzer
        analyzer = CryptoAnalyzer(
            window_size=request.window_size,
            embedding_dim=request.embedding_dim,
            prediction_horizon=request.prediction_horizon,
            verbose=False
        )
        
        # Fit analyzer on price data
        train_size = int(0.8 * len(prices))
        analyzer.fit(prices[:train_size])
        
        # Detect bubble conditions
        bubble_result = analyzer.detect_bubble_conditions(prices)
        
        # Market regime detection
        regime_result = analyzer.detect_market_regimes(prices)
        
        # Volatility analysis
        volatility_result = analyzer.analyze_volatility_patterns(prices)
        
        # Generate price predictions
        prediction_result = analyzer.predict(prices[-request.window_size:])
        predictions = []
        for i, pred in enumerate(prediction_result['predictions'][:request.prediction_horizon]):
            predictions.append({
                "day": i + 1,
                "predicted_price": float(pred),
                "confidence_lower": float(prediction_result['confidence_intervals'][i][0]),
                "confidence_upper": float(prediction_result['confidence_intervals'][i][1])
            })
        
        # Create bubble detection object
        bubble_detection = BubbleDetection(
            bubble_probability=bubble_result['bubble_probability'],
            risk_level=bubble_result['risk_level'],
            recommendation=bubble_result['recommendation'],
            confidence_interval={
                "lower": bubble_result['bubble_probability'] - 0.1,
                "upper": min(bubble_result['bubble_probability'] + 0.1, 1.0)
            },
            topological_indicators=bubble_result['topological_indicators'],
            time_to_potential_burst=_estimate_burst_timing(bubble_result) if bubble_result['bubble_probability'] > 0.7 else None
        )
        
        # Analysis period
        analysis_period = {
            "start": request.timestamps[0] if request.timestamps else "unknown",
            "end": request.timestamps[-1] if request.timestamps else "unknown",
            "total_periods": len(prices)
        }
        
        computation_time = (time.time() - start_time) * 1000
        
        response = CryptoBubbleResponse(
            analysis_id=analysis_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            asset_symbol=request.asset_symbol,
            analysis_period=analysis_period,
            bubble_detection=bubble_detection,
            market_regimes=regime_result,
            volatility_analysis=volatility_result,
            price_predictions=predictions,
            topological_features=prediction_result.get('topology_features', {})
        )
        
        # Cache and store
        background_tasks.add_task(cache.set, f"crypto_bubble:{analysis_id}", response.dict(), expire=3600)
        background_tasks.add_task(_store_financial_analysis, db, analysis_id, "crypto_bubble", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crypto bubble detection failed: {str(e)}"
        )


@router.post("/portfolio/risk-assessment", response_model=PortfolioRiskResponse)
async def assess_portfolio_risk(
    request: PortfolioRiskRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Comprehensive portfolio risk assessment using topological methods.
    
    Analyzes portfolio returns to compute VaR, expected shortfall,
    and topological risk measures.
    """
    import time
    start_time = time.time()
    
    analysis_id = _generate_analysis_id("portfolio_risk", request.dict())
    
    try:
        returns_data = np.array(request.returns_data)
        weights = np.array(request.asset_weights)
        
        # Initialize risk assessor
        risk_assessor = RiskAssessment(
            confidence_level=request.confidence_level,
            verbose=False
        )
        
        # Fit risk model
        risk_assessor.fit(returns_data, weights)
        
        # Calculate VaR and Expected Shortfall
        var_results = risk_assessor.calculate_var(returns_data)
        
        # Comprehensive risk assessment
        risk_assessment = risk_assessor.assess_portfolio_risk(returns_data, weights)
        
        # Stress testing
        stress_scenarios = {
            "market_crash": np.array([-0.2] * len(weights)),
            "volatility_spike": np.array([0.05] * len(weights))
        }
        stress_results = risk_assessor.run_stress_tests(stress_scenarios)
        
        # Correlation analysis
        correlation_analysis = risk_assessor.analyze_correlations(returns_data)
        
        # Risk measures
        risk_measures = RiskMeasure(
            var_1day=var_results['var_1day'],
            var_10day=var_results['var_10day'],
            expected_shortfall=var_results['expected_shortfall'],
            confidence_level=request.confidence_level
        )
        
        # Portfolio summary
        portfolio_summary = {
            "num_assets": len(weights),
            "total_periods": len(returns_data),
            "asset_symbols": request.asset_symbols or [f"Asset_{i}" for i in range(len(weights))],
            "weights": weights.tolist(),
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        computation_time = (time.time() - start_time) * 1000
        
        response = PortfolioRiskResponse(
            analysis_id=analysis_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            portfolio_summary=portfolio_summary,
            risk_measures=risk_measures,
            risk_decomposition=risk_assessment['risk_decomposition'],
            correlation_analysis=correlation_analysis,
            stress_test_results=stress_results,
            topological_risk_features=var_results['topological_risk_features'],
            overall_risk_score=risk_assessment['overall_risk_score']
        )
        
        # Cache and store
        background_tasks.add_task(cache.set, f"portfolio_risk:{analysis_id}", response.dict(), expire=3600)
        background_tasks.add_task(_store_financial_analysis, db, analysis_id, "portfolio_risk", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Portfolio risk assessment failed: {str(e)}"
        )


@router.post("/market/analysis", response_model=MarketAnalysisResponse)
async def analyze_market(
    request: MarketAnalysisRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Comprehensive market analysis including regime identification and trend analysis.
    
    Uses topological methods to identify market regimes, trend changes,
    and structural patterns.
    """
    import time
    start_time = time.time()
    
    analysis_id = _generate_analysis_id("market_analysis", request.dict())
    
    try:
        prices = np.array(request.prices)
        
        # Initialize market analyzer
        analyzer = MarketAnalyzer(verbose=False)
        analyzer.fit(prices)
        
        market_regimes = []
        trend_analysis = {}
        structure_analysis = {}
        
        # Market regime identification
        if request.include_regimes:
            regime_result = analyzer.identify_market_regimes(prices)
            
            # Convert to structured format
            for i, (start, end) in enumerate(regime_result['regime_transitions']):
                regime = MarketRegime(
                    regime_id=i,
                    regime_type=regime_result['regime_labels'][start],
                    start_period=int(start),
                    end_period=int(end),
                    characteristics=regime_result['regime_characteristics'].get(str(i), {})
                )
                market_regimes.append(regime)
        
        # Trend analysis
        if request.include_trends:
            trend_result = analyzer.detect_trend_changes(prices)
            trend_analysis = trend_result
        
        # Market structure analysis
        if request.include_structure:
            structure_result = analyzer.analyze_market_structure(prices)
            structure_analysis = structure_result
        
        # Volatility patterns
        volatility_patterns = _analyze_volatility_patterns(prices)
        
        # Topological summary
        topological_summary = analyzer.get_topological_summary() if hasattr(analyzer, 'get_topological_summary') else {}
        
        computation_time = (time.time() - start_time) * 1000
        
        response = MarketAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            asset_symbol=request.asset_symbol,
            market_regimes=market_regimes,
            trend_analysis=trend_analysis,
            structure_analysis=structure_analysis,
            volatility_patterns=volatility_patterns,
            topological_summary=topological_summary
        )
        
        # Cache and store
        background_tasks.add_task(cache.set, f"market_analysis:{analysis_id}", response.dict(), expire=3600)
        background_tasks.add_task(_store_financial_analysis, db, analysis_id, "market_analysis", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Market analysis failed: {str(e)}"
        )


# Query endpoints
@router.get("/risk-assessments")
async def get_recent_risk_assessments(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    portfolio_id: Optional[str] = Query(None, description="Filter by portfolio ID"),
    db=Depends(get_db_manager)
):
    """
    Get recent risk assessments.
    """
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        assessments = await db.get_risk_assessments_since(
            timestamp=since,
            portfolio_id=portfolio_id
        )
        
        return {
            "total_assessments": len(assessments),
            "time_window_hours": hours,
            "query_timestamp": datetime.utcnow().isoformat() + "Z",
            "assessments": assessments
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve risk assessments: {str(e)}"
        )


@router.get("/bubble-alerts")
async def get_bubble_alerts(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    db=Depends(get_db_manager)
):
    """
    Get recent bubble detection alerts.
    """
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        alerts = await db.get_bubble_alerts_since(
            timestamp=since,
            risk_level=risk_level
        )
        
        return {
            "total_alerts": len(alerts),
            "time_window_hours": hours,
            "query_timestamp": datetime.utcnow().isoformat() + "Z",
            "alerts": alerts
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve bubble alerts: {str(e)}"
        )


@router.get("/analysis/{analysis_id}")
async def get_financial_analysis(analysis_id: str, db=Depends(get_db_manager)):
    """
    Retrieve a previous financial analysis result.
    """
    try:
        result = await db.get_financial_analysis(analysis_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis {analysis_id} not found"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analysis: {str(e)}"
        )


# Utility functions
def _generate_analysis_id(analysis_type: str, request_data: dict) -> str:
    """Generate analysis ID."""
    timestamp = datetime.utcnow().isoformat()
    combined_data = f"{analysis_type}_{timestamp}_{json.dumps(request_data, sort_keys=True)}"
    return hashlib.md5(combined_data.encode()).hexdigest()[:16]


def _estimate_burst_timing(bubble_result: Dict[str, Any]) -> int:
    """Estimate potential bubble burst timing in days."""
    bubble_prob = bubble_result['bubble_probability']
    
    if bubble_prob > 0.9:
        return 3  # Very high probability, burst imminent
    elif bubble_prob > 0.8:
        return 7  # High probability, burst within a week
    elif bubble_prob > 0.7:
        return 14  # Moderate probability, burst within 2 weeks
    else:
        return 30  # Lower probability, longer horizon


def _analyze_volatility_patterns(prices: np.ndarray) -> Dict[str, Any]:
    """Analyze volatility patterns in price data."""
    # Compute returns
    returns = np.diff(np.log(prices))
    
    # Rolling volatility
    window = 30
    rolling_vol = []
    for i in range(window, len(returns)):
        vol = np.std(returns[i-window:i]) * np.sqrt(252)  # Annualized
        rolling_vol.append(vol)
    
    return {
        "current_volatility": float(np.std(returns[-30:]) * np.sqrt(252)),
        "avg_volatility": float(np.mean(rolling_vol)) if rolling_vol else 0.0,
        "max_volatility": float(np.max(rolling_vol)) if rolling_vol else 0.0,
        "volatility_trend": "increasing" if len(rolling_vol) > 10 and rolling_vol[-1] > rolling_vol[-10] else "decreasing"
    }


async def _store_financial_analysis(db, analysis_id: str, analysis_type: str, result_data: dict):
    """Store financial analysis result in database."""
    try:
        await db.store_financial_analysis(analysis_id, analysis_type, result_data)
    except Exception as e:
        print(f"Failed to store financial analysis: {e}")  # Use proper logging