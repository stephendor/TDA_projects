"""
Monitoring API Routes

Provides endpoints for system monitoring, metrics collection,
and operational insights.
"""

import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ...utils.database import get_db_manager


router = APIRouter()


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int


class APIMetrics(BaseModel):
    """API performance metrics."""
    endpoint: str
    method: str
    request_count: int
    avg_response_time_ms: float
    error_rate_percent: float
    p95_response_time_ms: float


class TDAComputationMetrics(BaseModel):
    """TDA computation performance metrics."""
    computation_type: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    avg_computation_time_ms: float
    queue_length: int


@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics():
    """
    Get current system performance metrics.
    """
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_percent = (disk.used / disk.total) * 100
    
    # Network I/O
    network = psutil.net_io_counters()
    network_io = {
        "bytes_sent": network.bytes_sent,
        "bytes_recv": network.bytes_recv,
        "packets_sent": network.packets_sent,
        "packets_recv": network.packets_recv
    }
    
    # Process count
    process_count = len(psutil.pids())
    
    return SystemMetrics(
        timestamp=datetime.utcnow().isoformat() + "Z",
        cpu_usage_percent=cpu_percent,
        memory_usage_percent=memory_percent,
        disk_usage_percent=disk_percent,
        network_io=network_io,
        process_count=process_count
    )


@router.get("/api-performance")
async def get_api_performance_metrics(
    hours: int = Query(default=24, ge=1, le=168),
    db=Depends(get_db_manager)
):
    """
    Get API performance metrics for specified time period.
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    
    try:
        # This would query the database for API request metrics
        metrics = await db.get_api_metrics_since(since)
        
        return {
            "time_period_hours": hours,
            "total_requests": sum(m["request_count"] for m in metrics),
            "unique_endpoints": len(set(m["endpoint"] for m in metrics)),
            "overall_error_rate": sum(m["error_rate_percent"] * m["request_count"] for m in metrics) / 
                                sum(m["request_count"] for m in metrics) if metrics else 0,
            "metrics_by_endpoint": metrics
        }
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve API metrics: {str(e)}",
            "time_period_hours": hours,
            "metrics_by_endpoint": []
        }


@router.get("/tda-computations")
async def get_tda_computation_metrics(
    hours: int = Query(default=24, ge=1, le=168),
    db=Depends(get_db_manager)
):
    """
    Get TDA computation performance metrics.
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    
    try:
        metrics = await db.get_tda_computation_metrics_since(since)
        
        return {
            "time_period_hours": hours,
            "summary": {
                "total_computations": sum(m["total_jobs"] for m in metrics),
                "success_rate": sum(m["completed_jobs"] for m in metrics) / 
                              max(sum(m["total_jobs"] for m in metrics), 1) * 100,
                "avg_computation_time_ms": sum(m["avg_computation_time_ms"] * m["total_jobs"] for m in metrics) /
                                         max(sum(m["total_jobs"] for m in metrics), 1)
            },
            "metrics_by_type": metrics
        }
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve TDA metrics: {str(e)}",
            "time_period_hours": hours,
            "metrics_by_type": []
        }


@router.get("/database")
async def get_database_metrics(db=Depends(get_db_manager)):
    """
    Get database performance and health metrics.
    """
    try:
        metrics = await db.get_database_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "connection_pool": metrics.get("connection_pool", {}),
            "query_performance": metrics.get("query_performance", {}),
            "storage_usage": metrics.get("storage_usage", {}),
            "active_connections": metrics.get("active_connections", 0),
            "slow_queries": metrics.get("slow_queries", [])
        }
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve database metrics: {str(e)}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


@router.get("/cache")
async def get_cache_metrics():
    """
    Get cache performance metrics.
    """
    try:
        # This would connect to Redis and get metrics
        # For now, return simulated metrics
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "memory_usage": {
                "used_memory_mb": 45.2,
                "max_memory_mb": 1024.0,
                "usage_percent": 4.4
            },
            "operations": {
                "hits_per_second": 150.5,
                "misses_per_second": 12.3,
                "hit_rate_percent": 92.4
            },
            "key_statistics": {
                "total_keys": 1247,
                "expired_keys": 34,
                "evicted_keys": 2
            }
        }
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve cache metrics: {str(e)}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


@router.get("/alerts")
async def get_active_alerts(db=Depends(get_db_manager)):
    """
    Get current system alerts and warnings.
    """
    try:
        alerts = []
        
        # Check system resource alerts
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        
        if cpu_percent > 80:
            alerts.append({
                "type": "system",
                "level": "warning" if cpu_percent < 90 else "critical",
                "message": f"High CPU usage: {cpu_percent:.1f}%",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        
        if memory_percent > 85:
            alerts.append({
                "type": "system",
                "level": "warning" if memory_percent < 95 else "critical",
                "message": f"High memory usage: {memory_percent:.1f}%",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        
        if disk_percent > 80:
            alerts.append({
                "type": "system",
                "level": "warning" if disk_percent < 90 else "critical",
                "message": f"High disk usage: {disk_percent:.1f}%",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        
        # Check application-specific alerts
        app_alerts = await db.get_active_alerts() if hasattr(db, 'get_active_alerts') else []
        alerts.extend(app_alerts)
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_alerts": len(alerts),
            "critical_count": len([a for a in alerts if a["level"] == "critical"]),
            "warning_count": len([a for a in alerts if a["level"] == "warning"]),
            "alerts": alerts
        }
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve alerts: {str(e)}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "alerts": []
        }


@router.get("/prometheus-metrics")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format for scraping.
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
        
        # Application metrics would be collected from various sources
        metrics = [
            f"# HELP tda_platform_cpu_usage_percent CPU usage percentage",
            f"# TYPE tda_platform_cpu_usage_percent gauge",
            f"tda_platform_cpu_usage_percent {cpu_percent}",
            "",
            f"# HELP tda_platform_memory_usage_percent Memory usage percentage", 
            f"# TYPE tda_platform_memory_usage_percent gauge",
            f"tda_platform_memory_usage_percent {memory_percent}",
            "",
            f"# HELP tda_platform_disk_usage_percent Disk usage percentage",
            f"# TYPE tda_platform_disk_usage_percent gauge", 
            f"tda_platform_disk_usage_percent {disk_percent}",
            "",
            f"# HELP tda_platform_uptime_seconds Application uptime in seconds",
            f"# TYPE tda_platform_uptime_seconds counter",
            f"tda_platform_uptime_seconds {time.time() - 1234567890}",  # Would use actual start time
            ""
        ]
        
        return "\n".join(metrics)
        
    except Exception as e:
        return f"# Error generating metrics: {str(e)}"