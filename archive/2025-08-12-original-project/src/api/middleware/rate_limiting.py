"""
Rate Limiting Middleware

Implements rate limiting for API endpoints to prevent abuse and ensure fair usage.
"""

import time
import asyncio
from typing import Dict, Tuple
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    """
    
    def __init__(self, app, requests_per_minute: int = 60, burst_requests: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_requests = burst_requests
        
        # Storage for request timestamps per client
        self.client_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.burst_counts: Dict[str, int] = defaultdict(int)
        self.last_cleanup = time.time()
        
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Clean up old entries periodically
        await self._cleanup_old_entries()
        
        # Check rate limits
        if not self._check_rate_limits(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": f"{self.requests_per_minute} requests per minute",
                    "retry_after": 60,
                    "timestamp": time.time()
                }
            )
        
        # Record this request
        self._record_request(client_id)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_id)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if available (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limits(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        requests = self.client_requests[client_id]
        
        # Remove requests older than 1 minute
        while requests and current_time - requests[0] > 60:
            requests.popleft()
        
        # Check requests per minute limit
        if len(requests) >= self.requests_per_minute:
            return False
        
        # Check burst limit (requests in last 10 seconds)
        burst_count = sum(1 for req_time in requests if current_time - req_time <= 10)
        if burst_count >= self.burst_requests:
            return False
        
        return True
    
    def _record_request(self, client_id: str):
        """Record a request for the client."""
        self.client_requests[client_id].append(time.time())
    
    def _add_rate_limit_headers(self, response: Response, client_id: str):
        """Add rate limiting headers to response."""
        current_time = time.time()
        requests = self.client_requests[client_id]
        
        # Count requests in current window
        recent_requests = sum(1 for req_time in requests if current_time - req_time <= 60)
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.requests_per_minute - recent_requests))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
    
    async def _cleanup_old_entries(self):
        """Periodically clean up old request records."""
        current_time = time.time()
        
        # Only cleanup every 5 minutes
        if current_time - self.last_cleanup < 300:
            return
        
        self.last_cleanup = current_time
        
        # Remove old entries
        for client_id in list(self.client_requests.keys()):
            requests = self.client_requests[client_id]
            
            # Remove requests older than 1 hour
            while requests and current_time - requests[0] > 3600:
                requests.popleft()
            
            # Remove empty deques
            if not requests:
                del self.client_requests[client_id]