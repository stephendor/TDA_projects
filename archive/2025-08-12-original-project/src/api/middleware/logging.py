"""
Request Logging Middleware

Logs API requests for monitoring, debugging, and analytics.
"""

import time
import json
import logging
from typing import Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# Configure logger
logger = logging.getLogger("tda_platform.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log API requests and responses.
    """
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
        
    async def dispatch(self, request: Request, call_next):
        """Process and log request."""
        start_time = time.time()
        
        # Extract request information
        request_info = await self._extract_request_info(request)
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Extract response information
            response_info = self._extract_response_info(response, processing_time)
            
            # Log the request/response
            self._log_request_response(request_info, response_info)
            
            return response
            
        except Exception as e:
            # Log errors
            processing_time = (time.time() - start_time) * 1000
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": round(processing_time, 2)
            }
            
            self._log_error(request_info, error_info)
            raise
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract relevant information from the request."""
        
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Get request details
        method = request.method
        url = str(request.url)
        path = request.url.path
        query_params = dict(request.query_params)
        headers = dict(request.headers)
        
        # Remove sensitive headers
        sensitive_headers = ["authorization", "cookie", "x-api-key"]
        for header in sensitive_headers:
            headers.pop(header, None)
        
        request_info = {
            "timestamp": time.time(),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "method": method,
            "url": url,
            "path": path,
            "query_params": query_params,
            "headers": headers,
            "request_size": request.headers.get("content-length", 0)
        }
        
        return request_info
    
    def _extract_response_info(self, response: Response, processing_time: float) -> Dict[str, Any]:
        """Extract relevant information from the response."""
        
        return {
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time, 2),
            "response_size": response.headers.get("content-length", 0),
            "content_type": response.headers.get("content-type", "unknown")
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get the real client IP address."""
        # Check for forwarded headers (when behind a proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    def _log_request_response(self, request_info: Dict[str, Any], response_info: Dict[str, Any]):
        """Log successful request/response."""
        
        log_data = {
            **request_info,
            **response_info,
            "success": True
        }
        
        # Determine log level based on status code
        status_code = response_info["status_code"]
        if status_code >= 500:
            log_level = logging.ERROR
        elif status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Log structured data
        logger.log(
            log_level,
            f"{request_info['method']} {request_info['path']} - "
            f"{status_code} - {response_info['processing_time_ms']}ms",
            extra={"request_data": log_data}
        )
    
    def _log_error(self, request_info: Dict[str, Any], error_info: Dict[str, Any]):
        """Log request errors."""
        
        log_data = {
            **request_info,
            **error_info,
            "success": False
        }
        
        logger.error(
            f"{request_info['method']} {request_info['path']} - "
            f"ERROR: {error_info['error']} - {error_info['processing_time_ms']}ms",
            extra={"request_data": log_data}
        )