"""
API v1 package for TDA backend.

Version 1 of the TDA computation API, providing endpoints for:
- Point cloud operations
- TDA computations (persistent homology, Betti numbers)
- Job management and status tracking
- Results retrieval and export
"""

from .router import router as v1_router

__all__ = ["v1_router"]