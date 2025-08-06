"""
Cybersecurity API Routes

Provides endpoints for cybersecurity threat detection using TDA methods,
including APT detection, IoT classification, and network analysis.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status, Query
from pydantic import BaseModel, Field, validator

from ...cybersecurity import APTDetector, IoTClassifier, NetworkAnalyzer
from ...utils.database import get_db_manager
from ...utils.cache import get_cache_manager


router = APIRouter()


# Request Models
class NetworkTrafficRequest(BaseModel):
    """Request model for network traffic data."""
    traffic_data: List[List[float]] = Field(..., description="Network traffic feature vectors")
    timestamp_start: Optional[str] = Field(None, description="Start timestamp (ISO format)")
    timestamp_end: Optional[str] = Field(None, description="End timestamp (ISO format)")
    source_system: str = Field("unknown", description="Source system identifier")
    
    @validator('traffic_data')
    def validate_traffic_data(cls, v):
        if len(v) < 10:
            raise ValueError("Minimum 10 traffic samples required for analysis")
        return v


class APTDetectionRequest(NetworkTrafficRequest):
    """Request model for APT detection."""
    time_window_hours: int = Field(default=1, ge=1, le=24, description="Analysis time window in hours")
    baseline_data: Optional[List[List[float]]] = Field(None, description="Baseline normal traffic data")
    sensitivity: float = Field(default=0.5, ge=0.1, le=1.0, description="Detection sensitivity (0.1-1.0)")


class IoTClassificationRequest(BaseModel):
    """Request model for IoT device classification."""
    device_data: List[List[float]] = Field(..., description="Device traffic/behavior features")
    device_labels: Optional[List[int]] = Field(None, description="Known device labels for training")
    classification_mode: str = Field(default="predict", description="'train', 'predict', or 'both'")
    
    @validator('device_data')
    def validate_device_data(cls, v):
        if len(v) < 5:
            raise ValueError("Minimum 5 device samples required")
        return v


class NetworkAnalysisRequest(NetworkTrafficRequest):
    """Request model for comprehensive network analysis."""
    include_anomalies: bool = Field(default=True, description="Include anomaly detection")
    include_patterns: bool = Field(default=True, description="Include pattern analysis")
    include_topology: bool = Field(default=True, description="Include topological analysis")


# Response Models
class ThreatDetection(BaseModel):
    """Model for individual threat detection."""
    threat_id: str
    threat_type: str
    confidence_score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: str
    source_indicators: List[str]
    topological_evidence: Dict[str, Any]
    recommended_actions: List[str]


class APTDetectionResponse(BaseModel):
    """Response model for APT detection results."""
    analysis_id: str
    status: str
    computation_time_ms: float
    time_window_analyzed: Dict[str, str]
    apt_percentage: float
    threat_level: str
    high_risk_samples: List[int]
    threat_detections: List[ThreatDetection]
    temporal_analysis: Dict[str, Any]
    topological_features: Dict[str, Any]
    recommendations: List[str]


class DeviceClassification(BaseModel):
    """Model for device classification result."""
    device_id: str
    predicted_type: str
    confidence: float
    topological_signature: List[float]
    behavioral_patterns: Dict[str, Any]


class IoTClassificationResponse(BaseModel):
    """Response model for IoT classification results."""
    analysis_id: str
    status: str
    computation_time_ms: float
    classification_results: List[DeviceClassification]
    spoofing_detections: List[Dict[str, Any]]
    model_performance: Optional[Dict[str, float]] = None
    device_type_distribution: Dict[str, int]


class NetworkAnomalyDetection(BaseModel):
    """Model for network anomaly detection."""
    anomaly_id: str
    anomaly_type: str
    severity_score: float
    affected_samples: List[int]
    anomaly_signature: Dict[str, Any]
    explanation: str


class NetworkAnalysisResponse(BaseModel):
    """Response model for network analysis results."""
    analysis_id: str
    status: str
    computation_time_ms: float
    network_summary: Dict[str, Any]
    anomaly_detections: List[NetworkAnomalyDetection]
    traffic_patterns: Dict[str, Any]
    topological_analysis: Dict[str, Any]
    security_assessment: Dict[str, Any]


# APT Detection Endpoints
@router.post("/apt-detection", response_model=APTDetectionResponse)
async def detect_apt_threats(
    request: APTDetectionRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Detect Advanced Persistent Threats (APTs) in network traffic.
    
    Uses topological data analysis to identify subtle, long-term
    attack patterns that traditional methods might miss.
    """
    import time
    start_time = time.time()
    
    analysis_id = _generate_analysis_id("apt", request.dict())
    
    try:
        traffic_data = np.array(request.traffic_data)
        
        # Check cache for recent analysis
        cached_result = await cache.get(f"apt:{analysis_id}")
        if cached_result:
            return APTDetectionResponse.parse_obj(cached_result)
        
        # Initialize APT detector
        detector = APTDetector(
            time_window=request.time_window_hours * 3600,  # Convert to seconds
            ph_maxdim=2,
            verbose=False
        )
        
        # Use baseline data if provided, otherwise use first portion of traffic
        if request.baseline_data:
            baseline = np.array(request.baseline_data)
            detector.fit(baseline)
        else:
            # Use first 70% as baseline
            split_idx = int(0.7 * len(traffic_data))
            detector.fit(traffic_data[:split_idx])
        
        # Analyze traffic for APT patterns
        apt_analysis = detector.analyze_apt_patterns(traffic_data)
        
        # Generate threat detections
        threat_detections = []
        if apt_analysis['high_risk_samples']:
            for i, sample_idx in enumerate(apt_analysis['high_risk_samples'][:10]):  # Limit to top 10
                threat = ThreatDetection(
                    threat_id=f"apt_{analysis_id}_{i}",
                    threat_type="Advanced Persistent Threat",
                    confidence_score=min(apt_analysis['apt_percentage'] / 100.0 + 0.1, 1.0),
                    severity=_determine_threat_severity(apt_analysis['apt_percentage']),
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    source_indicators=[
                        f"Sample index: {sample_idx}",
                        f"Topological anomaly detected",
                        f"Persistence signature deviation: {apt_analysis.get('deviation_score', 'unknown')}"
                    ],
                    topological_evidence={
                        "persistence_features": apt_analysis.get('topological_features', {}),
                        "temporal_correlation": apt_analysis.get('temporal_analysis', {})
                    },
                    recommended_actions=[
                        "Investigate network traffic at specified timestamp",
                        "Check for lateral movement patterns",
                        "Review authentication logs for suspicious activity"
                    ]
                )
                threat_detections.append(threat)
        
        # Prepare time window info
        time_window = {
            "start": request.timestamp_start or "unknown",
            "end": request.timestamp_end or "unknown",
            "duration_hours": request.time_window_hours
        }
        
        # Generate recommendations
        recommendations = _generate_apt_recommendations(apt_analysis)
        
        computation_time = (time.time() - start_time) * 1000
        
        response = APTDetectionResponse(
            analysis_id=analysis_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            time_window_analyzed=time_window,
            apt_percentage=apt_analysis['apt_percentage'],
            threat_level=apt_analysis['threat_assessment'],
            high_risk_samples=apt_analysis['high_risk_samples'],
            threat_detections=threat_detections,
            temporal_analysis=apt_analysis.get('temporal_analysis', {}),
            topological_features=apt_analysis.get('topological_features', {}),
            recommendations=recommendations
        )
        
        # Cache and store results
        background_tasks.add_task(cache.set, f"apt:{analysis_id}", response.dict(), expire=1800)
        background_tasks.add_task(_store_threat_detection, db, analysis_id, "apt", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"APT detection failed: {str(e)}"
        )


@router.post("/iot-classification", response_model=IoTClassificationResponse)
async def classify_iot_devices(
    request: IoTClassificationRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Classify IoT devices and detect spoofing attempts.
    
    Uses topological signatures to identify device types and
    detect when devices are spoofing other device types.
    """
    import time
    start_time = time.time()
    
    analysis_id = _generate_analysis_id("iot", request.dict())
    
    try:
        device_data = np.array(request.device_data)
        
        # Initialize IoT classifier
        classifier = IoTClassifier(verbose=False)
        
        results = []
        spoofing_detections = []
        model_performance = None
        
        # Handle different classification modes
        if request.classification_mode in ["train", "both"] and request.device_labels:
            # Train mode
            labels = np.array(request.device_labels)
            classifier.fit(device_data, labels)
            
            # Get model performance
            predictions = classifier.predict(device_data)
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(labels, predictions)
            model_performance = {"accuracy": accuracy}
        
        if request.classification_mode in ["predict", "both"]:
            # Prediction mode
            for i, device_sample in enumerate(device_data):
                device_sample = device_sample.reshape(1, -1)
                
                # Classify device type
                device_type_result = classifier.classify_device_type(device_sample)
                
                # Check for spoofing
                spoofing_result = classifier.detect_spoofing(device_sample)
                
                classification = DeviceClassification(
                    device_id=f"device_{i}",
                    predicted_type=device_type_result['device_type'],
                    confidence=device_type_result['confidence'],
                    topological_signature=device_type_result['topological_features'],
                    behavioral_patterns=device_type_result.get('behavioral_patterns', {})
                )
                results.append(classification)
                
                # Check for spoofing
                if spoofing_result['spoofing_detected']:
                    spoofing_detections.append({
                        "device_id": f"device_{i}",
                        "spoofing_confidence": max(spoofing_result['anomaly_scores']),
                        "suspicious_samples": spoofing_result['suspicious_samples'],
                        "evidence": spoofing_result.get('evidence', {})
                    })
        
        # Compute device type distribution
        if results:
            type_counts = {}
            for result in results:
                device_type = result.predicted_type
                type_counts[device_type] = type_counts.get(device_type, 0) + 1
        else:
            type_counts = {}
        
        computation_time = (time.time() - start_time) * 1000
        
        response = IoTClassificationResponse(
            analysis_id=analysis_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            classification_results=results,
            spoofing_detections=spoofing_detections,
            model_performance=model_performance,
            device_type_distribution=type_counts
        )
        
        # Cache and store
        background_tasks.add_task(cache.set, f"iot:{analysis_id}", response.dict(), expire=1800)
        background_tasks.add_task(_store_threat_detection, db, analysis_id, "iot_classification", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"IoT classification failed: {str(e)}"
        )


@router.post("/network-analysis", response_model=NetworkAnalysisResponse)
async def analyze_network_traffic(
    request: NetworkAnalysisRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db_manager),
    cache=Depends(get_cache_manager)
):
    """
    Comprehensive network traffic analysis.
    
    Provides anomaly detection, pattern analysis, and topological
    characterization of network traffic.
    """
    import time
    start_time = time.time()
    
    analysis_id = _generate_analysis_id("network", request.dict())
    
    try:
        traffic_data = np.array(request.traffic_data)
        
        # Initialize network analyzer
        analyzer = NetworkAnalyzer(verbose=False)
        analyzer.fit(traffic_data)
        
        anomaly_detections = []
        traffic_patterns = {}
        topological_analysis = {}
        
        # Anomaly detection
        if request.include_anomalies:
            anomaly_results = analyzer.detect_anomalies(traffic_data)
            
            for i, (anomaly_flag, score) in enumerate(zip(
                anomaly_results['anomaly_flags'], 
                anomaly_results['anomaly_scores']
            )):
                if anomaly_flag == 1:  # Anomaly detected
                    anomaly = NetworkAnomalyDetection(
                        anomaly_id=f"anomaly_{analysis_id}_{i}",
                        anomaly_type="Network Traffic Anomaly",
                        severity_score=float(score),
                        affected_samples=[i],
                        anomaly_signature=anomaly_results['topology_analysis'].get(f'sample_{i}', {}),
                        explanation=f"Topological anomaly detected with score {score:.3f}"
                    )
                    anomaly_detections.append(anomaly)
        
        # Pattern analysis
        if request.include_patterns:
            patterns = analyzer.analyze_traffic_patterns(traffic_data)
            traffic_patterns = patterns
        
        # Topological analysis
        if request.include_topology:
            topological_analysis = analyzer.get_topological_summary()
        
        # Network summary
        network_summary = {
            "total_samples": len(traffic_data),
            "feature_dimension": traffic_data.shape[1],
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "source_system": request.source_system,
            "anomaly_rate": len(anomaly_detections) / len(traffic_data) if anomaly_detections else 0.0
        }
        
        # Security assessment
        security_assessment = _generate_security_assessment(
            anomaly_detections, traffic_patterns, topological_analysis
        )
        
        computation_time = (time.time() - start_time) * 1000
        
        response = NetworkAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            computation_time_ms=round(computation_time, 2),
            network_summary=network_summary,
            anomaly_detections=anomaly_detections,
            traffic_patterns=traffic_patterns,
            topological_analysis=topological_analysis,
            security_assessment=security_assessment
        )
        
        # Cache and store
        background_tasks.add_task(cache.set, f"network:{analysis_id}", response.dict(), expire=1800)
        background_tasks.add_task(_store_threat_detection, db, analysis_id, "network_analysis", response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Network analysis failed: {str(e)}"
        )


# Query endpoints
@router.get("/threats")
async def get_recent_threats(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    threat_type: Optional[str] = Query(None, description="Filter by threat type"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    db=Depends(get_db_manager)
):
    """
    Get recent threat detections.
    
    Query recent cybersecurity threats detected by the platform.
    """
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        threats = await db.get_threats_since(
            timestamp=since,
            threat_type=threat_type,
            severity=severity
        )
        
        return {
            "total_threats": len(threats),
            "time_window_hours": hours,
            "query_timestamp": datetime.utcnow().isoformat() + "Z",
            "threats": threats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve threats: {str(e)}"
        )


@router.get("/analysis/{analysis_id}")
async def get_analysis_result(analysis_id: str, db=Depends(get_db_manager)):
    """
    Retrieve a previous cybersecurity analysis result.
    """
    try:
        result = await db.get_analysis_result(analysis_id)
        
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


def _determine_threat_severity(apt_percentage: float) -> str:
    """Determine threat severity based on APT percentage."""
    if apt_percentage >= 80:
        return "CRITICAL"
    elif apt_percentage >= 60:
        return "HIGH"
    elif apt_percentage >= 30:
        return "MEDIUM"
    else:
        return "LOW"


def _generate_apt_recommendations(apt_analysis: Dict[str, Any]) -> List[str]:
    """Generate APT-specific recommendations."""
    recommendations = []
    
    apt_percentage = apt_analysis.get('apt_percentage', 0)
    
    if apt_percentage > 70:
        recommendations.extend([
            "IMMEDIATE: Isolate affected network segments",
            "IMMEDIATE: Review all authentication logs for the past 30 days",
            "Conduct forensic analysis of high-risk samples",
            "Implement enhanced monitoring on identified patterns"
        ])
    elif apt_percentage > 40:
        recommendations.extend([
            "Increase monitoring frequency for detected patterns",
            "Review firewall and access control policies",
            "Conduct targeted threat hunting activities"
        ])
    else:
        recommendations.extend([
            "Continue routine monitoring",
            "Consider baseline adjustment if patterns persist"
        ])
    
    return recommendations


def _generate_security_assessment(anomalies, patterns, topology) -> Dict[str, Any]:
    """Generate security assessment summary."""
    total_anomalies = len(anomalies)
    
    if total_anomalies == 0:
        risk_level = "LOW"
        assessment = "No significant anomalies detected"
    elif total_anomalies < 5:
        risk_level = "MEDIUM"
        assessment = f"{total_anomalies} anomalies detected - monitor closely"
    else:
        risk_level = "HIGH"
        assessment = f"{total_anomalies} anomalies detected - investigate immediately"
    
    return {
        "overall_risk_level": risk_level,
        "assessment_summary": assessment,
        "anomaly_count": total_anomalies,
        "pattern_complexity": len(patterns.get('pattern_types', [])),
        "topological_complexity": topology.get('complexity_score', 0),
        "recommendations": [
            "Review anomalous traffic patterns",
            "Consider updating security policies",
            "Maintain continuous monitoring"
        ]
    }


async def _store_threat_detection(db, analysis_id: str, analysis_type: str, result_data: dict):
    """Store threat detection result in database."""
    try:
        await db.store_threat_detection(analysis_id, analysis_type, result_data)
    except Exception as e:
        print(f"Failed to store threat detection: {e}")  # Use proper logging