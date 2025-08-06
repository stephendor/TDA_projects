"""
TDA Platform API Server

FastAPI-based REST API server for real-time TDA computations.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

from .routes import (
    health_router,
    tda_router,
    cybersecurity_router,
    finance_router,
    monitoring_router
)
from .middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware
)
from ..utils.database import DatabaseManager
from ..utils.cache import CacheManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("🚀 Starting TDA Platform API Server...")
    
    # Initialize database connection
    db_manager = DatabaseManager()
    await db_manager.initialize()
    app.state.db = db_manager
    
    # Initialize cache
    cache_manager = CacheManager()
    await cache_manager.initialize()
    app.state.cache = cache_manager
    
    # Initialize metrics
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)
    
    logger.info("✅ TDA Platform API Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down TDA Platform API Server...")
    await db_manager.close()
    await cache_manager.close()
    logger.info("✅ Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="TDA Platform API",
        description="""
        **Topological Data Analysis Platform** for cybersecurity threat detection 
        and financial risk analysis.
        
        This API provides real-time TDA computations including:
        - Persistent homology analysis
        - Mapper algorithm visualization
        - APT detection and network analysis
        - Cryptocurrency bubble detection
        - Portfolio risk assessment
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
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Register routes
    register_routes(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)


def register_routes(app: FastAPI) -> None:
    """Register API routes."""
    
    # Health check endpoints
    app.include_router(
        health_router,
        prefix="/health",
        tags=["Health"]
    )
    
    # Core TDA endpoints
    app.include_router(
        tda_router,
        prefix="/api/v1/tda",
        tags=["TDA Core"]
    )
    
    # Cybersecurity endpoints
    app.include_router(
        cybersecurity_router,
        prefix="/api/v1/cybersecurity",
        tags=["Cybersecurity"]
    )
    
    # Finance endpoints
    app.include_router(
        finance_router,
        prefix="/api/v1/finance",
        tags=["Finance"]
    )
    
    # Monitoring endpoints
    app.include_router(
        monitoring_router,
        prefix="/api/v1/monitoring",
        tags=["Monitoring"]
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure exception handlers."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
                "timestamp": "2025-08-05T00:00:00Z"  # Would be actual timestamp
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "details": exc.errors(),
                "path": request.url.path,
                "method": request.method,
                "timestamp": "2025-08-05T00:00:00Z"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "path": request.url.path,
                "method": request.method,
                "timestamp": "2025-08-05T00:00:00Z"
            }
        )


# Create the application instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "TDA Platform API",
        "version": "1.0.0",
        "description": "Topological Data Analysis Platform for cybersecurity and finance",
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "operational"
    }


def main():
    """Run the API server."""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        workers=1 if os.getenv("ENVIRONMENT") == "development" else 4,
        log_level="info"
    )


if __name__ == "__main__":
    main()