"""
TDA Backend Services Package.

This package contains service layer components that bridge the FastAPI application
with the C++ TDA bindings and provide high-level operations for topological data analysis.

Services:
    - TDAService: Core TDA computation service with C++ binding integration
    - FileUploadService: File upload and processing service for point cloud data
    - StorageService: Point cloud storage, metadata, and caching service
    - Cache service integration for performance optimization
    - Background job processing for long-running computations
    - Error handling and validation specific to TDA operations
"""

from .tda_service import TDAService
from .upload_service import FileUploadService, get_upload_service
from .storage_service import StorageService, get_storage_service

# Dependency injection helpers
def get_tda_service() -> TDAService:
    """Get TDA service instance."""
    return TDAService()

__all__ = [
    "TDAService",
    "FileUploadService", 
    "StorageService",
    "get_tda_service",
    "get_upload_service",
    "get_storage_service",
]