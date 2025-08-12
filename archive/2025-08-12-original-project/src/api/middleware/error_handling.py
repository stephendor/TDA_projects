"""
Error Handling Middleware

Provides centralized error handling and standardized error responses.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("tda_platform.errors")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling and logging.
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error handling."""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Let FastAPI handle HTTP exceptions normally
            raise
            
        except Exception as e:
            # Handle unexpected exceptions
            error_id = self._generate_error_id()
            
            # Log the error with full context
            await self._log_error(request, e, error_id)
            
            # Return standardized error response
            return self._create_error_response(e, error_id, request)
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def _log_error(self, request: Request, exception: Exception, error_id: str):
        """Log error with full context."""
        
        # Extract request context
        request_context = {
            "error_id": error_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log the error
        logger.error(
            f"Unhandled exception in {request.method} {request.url.path}",
            exc_info=exception,
            extra={
                "error_id": error_id,
                "request_context": request_context,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception)
            }
        )
    
    def _create_error_response(self, exception: Exception, error_id: str, request: Request) -> JSONResponse:
        """Create standardized error response."""
        
        # Determine if this is a known error type
        error_details = self._classify_error(exception)
        
        response_data = {
            "error": {
                "id": error_id,
                "type": error_details["type"],
                "title": error_details["title"],
                "detail": error_details["detail"],
                "status": error_details["status_code"],
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "path": request.url.path,
                "method": request.method
            }
        }
        
        # Add additional context in development
        import os
        if os.getenv("ENVIRONMENT") == "development":
            response_data["error"]["debug"] = {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": self._get_safe_traceback(exception)
            }
        
        return JSONResponse(
            status_code=error_details["status_code"],
            content=response_data
        )
    
    def _classify_error(self, exception: Exception) -> Dict[str, Any]:
        """Classify error and determine appropriate response."""
        
        exception_type = type(exception).__name__
        exception_message = str(exception)
        
        # Database-related errors
        if "database" in exception_message.lower() or "connection" in exception_message.lower():
            return {
                "type": "database_error",
                "title": "Database Connection Error",
                "detail": "A database connectivity issue occurred. Please try again later.",
                "status_code": 503
            }
        
        # Computation errors
        if any(term in exception_message.lower() for term in ["computation", "algorithm", "topology"]):
            return {
                "type": "computation_error", 
                "title": "Computation Error",
                "detail": "An error occurred during TDA computation. Please check your input data.",
                "status_code": 422
            }
        
        # Validation errors
        if "validation" in exception_message.lower() or exception_type == "ValueError":
            return {
                "type": "validation_error",
                "title": "Input Validation Error", 
                "detail": "The provided input data is invalid or malformed.",
                "status_code": 400
            }
        
        # Memory/resource errors
        if exception_type == "MemoryError" or "memory" in exception_message.lower():
            return {
                "type": "resource_error",
                "title": "Insufficient Resources",
                "detail": "The computation requires more resources than available. Try reducing data size.",
                "status_code": 507
            }
        
        # Timeout errors  
        if "timeout" in exception_message.lower() or exception_type == "TimeoutError":
            return {
                "type": "timeout_error",
                "title": "Request Timeout",
                "detail": "The request took too long to process. Try again or reduce complexity.",
                "status_code": 408
            }
        
        # Generic server error
        return {
            "type": "internal_error",
            "title": "Internal Server Error", 
            "detail": "An unexpected error occurred. The development team has been notified.",
            "status_code": 500
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _get_safe_traceback(self, exception: Exception) -> str:
        """Get safe traceback for debugging."""
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return f"Could not format traceback for {type(exception).__name__}"