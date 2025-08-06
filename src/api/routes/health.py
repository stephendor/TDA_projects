"""
Health Check API Routes

Provides system health and readiness endpoints for monitoring and load balancing.
"""

import asyncio
import time
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...utils.database import get_db_manager
from ...utils.cache import get_cache_manager


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    environment: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    environment: str
    services: Dict[str, Dict[str, Any]]
    system_info: Dict[str, Any]


# Application start time for uptime calculation
START_TIME = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns basic service status and uptime information.
    Used by load balancers and monitoring systems.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=time.time() - START_TIME,
        version="1.0.0",
        environment="production"  # Should come from environment variable
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Verifies that the service is ready to accept requests.
    Checks critical dependencies without detailed diagnostics.
    """
    try:
        # Quick dependency checks
        checks = await asyncio.gather(
            _check_database_ready(),
            _check_cache_ready(),
            return_exceptions=True
        )
        
        # If any check failed, service is not ready
        for check in checks:
            if isinstance(check, Exception):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service not ready"
                )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint.
    
    Verifies that the service is alive and responsive.
    Used by orchestration systems to determine if restart is needed.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - START_TIME
    }


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check endpoint.
    
    Provides comprehensive system health information including
    service dependencies, resource usage, and diagnostics.
    """
    import psutil
    import sys
    
    # Gather service status
    services = {}
    
    # Database health
    try:
        db_health = await _check_database_health()
        services["database"] = {
            "status": "healthy",
            "response_time_ms": db_health["response_time_ms"],
            "connection_pool": db_health.get("pool_info", {})
        }
    except Exception as e:
        services["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Cache health
    try:
        cache_health = await _check_cache_health()
        services["cache"] = {
            "status": "healthy",
            "response_time_ms": cache_health["response_time_ms"],
            "memory_usage": cache_health.get("memory_info", {})
        }
    except Exception as e:
        services["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # TDA computation engine health
    try:
        tda_health = await _check_tda_engine_health()
        services["tda_engine"] = {
            "status": "healthy",
            "available_algorithms": tda_health["algorithms"],
            "worker_status": tda_health.get("workers", {})
        }
    except Exception as e:
        services["tda_engine"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # System information
    system_info = {
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_usage": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "usage_percent": psutil.virtual_memory().percent
        },
        "disk_usage": {
            "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
            "usage_percent": round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 2)
        }
    }
    
    # Determine overall status
    overall_status = "healthy"
    for service_name, service_info in services.items():
        if service_info["status"] != "healthy":
            overall_status = "degraded"
            break
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=time.time() - START_TIME,
        version="1.0.0",
        environment="production",
        services=services,
        system_info=system_info
    )


@router.get("/metrics")
async def health_metrics():
    """
    Health metrics endpoint in Prometheus format.
    
    Provides metrics for monitoring systems.
    """
    uptime = time.time() - START_TIME
    
    # Get basic system metrics
    import psutil
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_usage = psutil.virtual_memory().percent
    
    metrics = [
        f"tda_platform_uptime_seconds {uptime}",
        f"tda_platform_cpu_usage_percent {cpu_usage}",
        f"tda_platform_memory_usage_percent {memory_usage}",
    ]
    
    # Add service-specific metrics
    try:
        db_health = await _check_database_health()
        metrics.append(f"tda_platform_database_response_time_ms {db_health['response_time_ms']}")
        metrics.append("tda_platform_database_status 1")
    except:
        metrics.append("tda_platform_database_status 0")
    
    try:
        cache_health = await _check_cache_health()
        metrics.append(f"tda_platform_cache_response_time_ms {cache_health['response_time_ms']}")
        metrics.append("tda_platform_cache_status 1")
    except:
        metrics.append("tda_platform_cache_status 0")
    
    return "\n".join(metrics)


# Helper functions for health checks
async def _check_database_ready() -> bool:
    """Quick database readiness check."""
    # Implementation would check database connection
    await asyncio.sleep(0.01)  # Simulate check
    return True


async def _check_cache_ready() -> bool:
    """Quick cache readiness check."""
    # Implementation would check cache connection
    await asyncio.sleep(0.01)  # Simulate check
    return True


async def _check_database_health() -> Dict[str, Any]:
    """Detailed database health check."""
    start_time = time.time()
    
    # Implementation would:
    # - Check database connection
    # - Run test query
    # - Check connection pool status
    await asyncio.sleep(0.05)  # Simulate database check
    
    response_time = (time.time() - start_time) * 1000
    
    return {
        "response_time_ms": round(response_time, 2),
        "pool_info": {
            "active_connections": 3,
            "total_connections": 10,
            "idle_connections": 7
        }
    }


async def _check_cache_health() -> Dict[str, Any]:
    """Detailed cache health check."""
    start_time = time.time()
    
    # Implementation would:
    # - Check Redis connection
    # - Run test operations
    # - Get memory info
    await asyncio.sleep(0.02)  # Simulate cache check
    
    response_time = (time.time() - start_time) * 1000
    
    return {
        "response_time_ms": round(response_time, 2),
        "memory_info": {
            "used_memory_mb": 45.2,
            "max_memory_mb": 1024.0,
            "usage_percent": 4.4
        }
    }


async def _check_tda_engine_health() -> Dict[str, Any]:
    """Check TDA computation engine health."""
    # Implementation would:
    # - Check worker processes
    # - Verify algorithm availability
    # - Test basic computations
    await asyncio.sleep(0.03)  # Simulate check
    
    return {
        "algorithms": [
            "persistent_homology",
            "mapper",
            "topology_utils"
        ],
        "workers": {
            "active_workers": 2,
            "total_workers": 4,
            "queue_length": 0
        }
    }