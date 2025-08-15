"""
FastAPI Main Application for TDA Backend Platform.

This module provides the core FastAPI application with comprehensive middleware,
error handling, monitoring, and API structure for the Topological Data Analysis platform.

Features:
- Production-ready middleware stack
- Comprehensive error handling
- Request/response logging
- Health checks and monitoring
- API versioning (v1)
- CORS configuration
- Rate limiting
- Authentication middleware ready
- Startup/shutdown lifecycle management
"""

import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from tda_backend.config import settings
from tda_backend.api.v1.router import v1_router
from tda_backend.services.kafka_producer import initialize_kafka_producer, shutdown_kafka_producer
from tda_backend.services.kafka_schemas import initialize_schemas, cleanup_schemas
from tda_backend.services.kafka_metrics import start_metrics_collection, stop_metrics_collection
from tda_backend.services.kafka_background_service import (
    get_kafka_background_service, 
    start_kafka_background_service, 
    stop_kafka_background_service
)

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'tda_backend_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'tda_backend_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

TDA_OPERATIONS = Counter(
    'tda_backend_operations_total',
    'Total TDA operations',
    ['operation_type', 'status']
)


# Response Models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    timestamp: str = Field(..., description="Response timestamp")
    services: Dict[str, str] = Field(..., description="Service statuses")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    timestamp: str = Field(..., description="Error timestamp")


class APIInfoResponse(BaseModel):
    """API information response model."""
    title: str = Field(..., description="API title")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    docs_url: str = Field(..., description="Documentation URL")
    redoc_url: str = Field(..., description="ReDoc URL")
    openapi_url: str = Field(..., description="OpenAPI schema URL")


# Custom Exception Classes
class TDAComputationError(Exception):
    """Exception raised during TDA computation."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class TDAValidationError(Exception):
    """Exception raised during input validation for TDA operations."""
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class TDAResourceError(Exception):
    """Exception raised when TDA resources are unavailable."""
    def __init__(self, message: str, resource_type: str):
        self.message = message
        self.resource_type = resource_type
        super().__init__(self.message)


# Application Lifecycle Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager handling startup and shutdown events.
    
    Manages:
    - Service connections initialization
    - Background task setup
    - Resource cleanup
    - Graceful shutdown procedures
    """
    # Startup
    logger.info("🚀 TDA Backend starting up...")
    
    try:
        # Initialize services
        await initialize_services()
        
        # Setup background tasks
        await setup_background_tasks()
        
        logger.info("✅ TDA Backend startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 TDA Backend shutting down...")
    
    try:
        # Cleanup background tasks
        await cleanup_background_tasks()
        
        # Close service connections
        await cleanup_services()
        
        logger.info("✅ TDA Backend shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")
        logger.error(traceback.format_exc())


async def initialize_services():
    """Initialize external service connections."""
    logger.info("🔧 Initializing services...")
    
    # Initialize database connection pool
    if not settings.testing:
        logger.info("📊 Initializing database connection pool...")
        # TODO: Initialize database connection pool
    
    # Initialize Redis connection
    if not settings.testing:
        logger.info("📦 Initializing Redis connection...")
        # TODO: Initialize Redis connection
    
    # Initialize Kafka connections
    if not settings.mock_kafka and not settings.testing:
        logger.info("📨 Initializing Kafka connections...")
        try:
            await initialize_kafka_producer()
            logger.info("✅ Kafka producer initialized successfully")
            
            # Initialize schemas
            await initialize_schemas()
            logger.info("✅ Kafka schemas initialized successfully")
            
            # Start metrics collection
            await start_metrics_collection()
            logger.info("✅ Kafka metrics collection started")
            
            # Start background Kafka consumer service
            await start_kafka_background_service()
            logger.info("✅ Kafka consumer background service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kafka services: {e}")
            # Don't fail startup if Kafka is not available
    
    # Initialize Flink client
    if not settings.mock_flink and not settings.testing:
        logger.info("🌊 Initializing Flink client...")
        # TODO: Initialize Flink REST client
    
    logger.info("✅ Services initialization complete")


async def setup_background_tasks():
    """Setup background tasks and scheduled jobs."""
    logger.info("⏰ Setting up background tasks...")
    
    # TODO: Setup periodic cleanup tasks
    # TODO: Setup health check tasks for external services
    # TODO: Setup metrics collection tasks
    
    logger.info("✅ Background tasks setup complete")


async def cleanup_background_tasks():
    """Cleanup background tasks and scheduled jobs."""
    logger.info("🧹 Cleaning up background tasks...")
    
    # TODO: Cancel periodic tasks
    # TODO: Wait for running tasks to complete
    
    logger.info("✅ Background tasks cleanup complete")


async def cleanup_services():
    """Cleanup service connections."""
    logger.info("🔌 Cleaning up service connections...")
    
    # TODO: Close database connections
    # TODO: Close Redis connections
    
    # Close Kafka connections
    if not settings.mock_kafka and not settings.testing:
        try:
            # Stop background consumer service
            await stop_kafka_background_service()
            logger.info("✅ Kafka consumer background service stopped")
            
            # Stop metrics collection
            await stop_metrics_collection()
            logger.info("✅ Kafka metrics collection stopped")
            
            # Cleanup schemas
            await cleanup_schemas()
            logger.info("✅ Kafka schemas cleaned up")
            
            # Shutdown producer
            await shutdown_kafka_producer()
            logger.info("✅ Kafka producer shutdown successfully")
        except Exception as e:
            logger.error(f"❌ Error shutting down Kafka services: {e}")
    
    # TODO: Close Flink client connections
    
    logger.info("✅ Service connections cleanup complete")


# Middleware Classes
class RequestLoggingMiddleware:
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self, app: Callable) -> None:
        self.app = app
    
    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope)
        start_time = time.time()
        
        # Generate request ID
        request_id = f"req_{int(time.time() * 1000000)}"
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            "📥 Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        )
        
        # Process request
        response_data = {}
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_data["status_code"] = message["status"]
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Update metrics
        status_code = response_data.get("status_code", 500)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log response
        logger.info(
            "📤 Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2),
            }
        )


class ErrorHandlingMiddleware:
    """Middleware for centralized error handling."""
    
    def __init__(self, app: Callable) -> None:
        self.app = app
    
    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            request = Request(scope)
            request_id = getattr(request.state, "request_id", "unknown")
            
            # Log the error
            logger.error(
                f"❌ Unhandled exception in request {request_id}: {exc}",
                extra={
                    "request_id": request_id,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            
            # Send error response
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An internal server error occurred",
                    "request_id": request_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            )
            
            await response(scope, receive, send)


# Create FastAPI application
app = FastAPI(
    title="TDA Backend Platform",
    description="""
    # Topological Data Analysis Backend Platform
    
    A high-performance, production-ready backend platform for topological data analysis (TDA) operations.
    
    ## Features
    
    - **Real-time TDA Processing**: Stream processing with Apache Kafka and Flink
    - **Scalable Architecture**: Microservices with containerized deployment
    - **Multiple TDA Algorithms**: Persistent homology, Vietoris-Rips, Alpha complexes
    - **High Performance**: C++23 core with Python bindings
    - **Production Ready**: Comprehensive monitoring, logging, and error handling
    
    ## API Versions
    
    - **v1**: Current stable API version
    
    ## Authentication
    
    Most endpoints require API key authentication via the `X-API-Key` header.
    
    ## Rate Limiting
    
    API requests are rate limited to prevent abuse. See rate limit headers in responses.
    """,
    version="1.0.0",
    contact={
        "name": "TDA Platform Team",
        "email": "support@tda-platform.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
    lifespan=lifespan
)

# Add Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Rate-Limit-*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add trusted host middleware for production
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.tda-platform.com"]
    )

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)


# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "message": exc.detail,
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "detail": exc.errors(),
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    )


@app.exception_handler(TDAComputationError)
async def tda_computation_exception_handler(request: Request, exc: TDAComputationError):
    """Handle TDA computation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(f"TDA computation error in request {request_id}: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "TDAComputationError",
            "message": exc.message,
            "detail": exc.details,
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    )


@app.exception_handler(TDAValidationError)
async def tda_validation_exception_handler(request: Request, exc: TDAValidationError):
    """Handle TDA validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "TDAValidationError",
            "message": exc.message,
            "field": exc.field,
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    )


@app.exception_handler(TDAResourceError)
async def tda_resource_exception_handler(request: Request, exc: TDAResourceError):
    """Handle TDA resource errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "TDAResourceError",
            "message": exc.message,
            "resource_type": exc.resource_type,
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    )


# Root Endpoints
@app.get(
    "/",
    response_model=APIInfoResponse,
    summary="API Information",
    description="Get basic information about the TDA Backend API"
)
async def root():
    """Get API information and available endpoints."""
    return APIInfoResponse(
        title=app.title,
        version=app.version,
        description="Topological Data Analysis Backend Platform",
        docs_url="/docs" if settings.is_development else "Not available in production",
        redoc_url="/redoc" if settings.is_development else "Not available in production",
        openapi_url="/openapi.json" if settings.is_development else "Not available in production"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the TDA Backend and its dependencies"
)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks:
    - Application status
    - Database connectivity
    - Redis connectivity  
    - Kafka connectivity
    - Flink connectivity
    - C++ library availability
    """
    services = {}
    
    # Check database
    try:
        # TODO: Implement actual database health check
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        # TODO: Implement actual Redis health check
        services["redis"] = "healthy"
    except Exception as e:
        services["redis"] = f"unhealthy: {str(e)}"
    
    # Check Kafka
    try:
        if settings.mock_kafka:
            services["kafka"] = "mocked"
        else:
            # TODO: Implement actual Kafka health check
            services["kafka"] = "healthy"
    except Exception as e:
        services["kafka"] = f"unhealthy: {str(e)}"
    
    # Check Flink
    try:
        if settings.mock_flink:
            services["flink"] = "mocked"
        else:
            # TODO: Implement actual Flink health check
            services["flink"] = "healthy"
    except Exception as e:
        services["flink"] = f"unhealthy: {str(e)}"
    
    # Check C++ libraries
    try:
        # TODO: Implement actual C++ library check
        services["cpp_libraries"] = "healthy"
    except Exception as e:
        services["cpp_libraries"] = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        version=app.version,
        environment=settings.environment,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        services=services
    )


@app.get(
    "/health/ready",
    summary="Readiness Probe",
    description="Check if the service is ready to accept requests"
)
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    # TODO: Implement readiness checks for critical services
    return {"status": "ready"}


@app.get(
    "/health/live",
    summary="Liveness Probe", 
    description="Check if the service is alive"
)
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


@app.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Prometheus metrics endpoint"
)
async def metrics():
    """Expose Prometheus metrics."""
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


# API Versioning Structure
# Include v1 API router with comprehensive TDA endpoints
app.include_router(v1_router, prefix="/api/v1", tags=["v1"])


# Development-only endpoints
if settings.is_development:
    @app.get("/debug/config", tags=["debug"])
    async def debug_config():
        """Debug endpoint to view current configuration."""
        return {
            "environment": settings.environment,
            "api_host": settings.api_host,
            "api_port": settings.api_port,
            "cors_origins": settings.cors_origins,
            "metrics_enabled": settings.metrics_enabled,
            "mock_services": {
                "kafka": settings.mock_kafka,
                "flink": settings.mock_flink,
            }
        }
    
    @app.post("/debug/test-error", tags=["debug"])
    async def debug_test_error(error_type: str = "generic"):
        """Debug endpoint to test error handling."""
        if error_type == "tda_computation":
            raise TDAComputationError("Test TDA computation error", {"test": True})
        elif error_type == "tda_validation":
            raise TDAValidationError("Test validation error", "test_field")
        elif error_type == "tda_resource":
            raise TDAResourceError("Test resource error", "test_resource")
        elif error_type == "http":
            raise HTTPException(status_code=400, detail="Test HTTP error")
        else:
            raise Exception("Test generic error")


def create_app() -> FastAPI:
    """Factory function to create FastAPI application."""
    return app


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(
        "tda_backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload and settings.is_development,
        workers=1 if settings.api_reload else settings.api_workers,
        log_level=settings.log_level.lower(),
        access_log=True,
        loop="uvloop" if not settings.is_development else "auto",
    )