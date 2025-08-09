"""
Authentication middleware for TDA Platform API.

Provides API key-based authentication to secure endpoints.
"""

import os
import hashlib
import secrets
from typing import Optional
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class APIKeyAuth:
    """API Key authentication handler."""
    
    def __init__(self):
        # Get API secret key from environment
        self.secret_key = os.getenv("API_SECRET_KEY")
        if not self.secret_key:
            raise ValueError("API_SECRET_KEY environment variable must be set")
        
        # For demo purposes, generate a valid API key from secret
        # In production, you'd have a proper key management system
        self.valid_api_key = hashlib.sha256(self.secret_key.encode()).hexdigest()[:32]
        
    def verify_api_key(self, api_key: str) -> bool:
        """Verify if the provided API key is valid."""
        if not api_key:
            return False
        
        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(api_key, self.valid_api_key)


# Global auth instance
auth_handler = APIKeyAuth()
security = HTTPBearer()


async def authenticate_api_key(credentials: HTTPAuthorizationCredentials = security) -> bool:
    """
    FastAPI dependency for API key authentication.
    
    Args:
        credentials: Bearer token from Authorization header
        
    Returns:
        bool: True if authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not auth_handler.verify_api_key(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


async def optional_authentication(request: Request) -> Optional[bool]:
    """
    Optional authentication for endpoints that can work with or without auth.
    
    Args:
        request: FastAPI request object
        
    Returns:
        bool or None: True if authenticated, None if no auth provided
    """
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    
    if not auth_header.startswith("Bearer "):
        return None
    
    api_key = auth_header.split(" ")[1]
    return auth_handler.verify_api_key(api_key)


def get_api_key_for_client() -> str:
    """
    Get the current valid API key for client applications.
    This would normally be managed through a proper key management system.
    
    Returns:
        str: Valid API key
    """
    return auth_handler.valid_api_key