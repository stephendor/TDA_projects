"""
TDA Backend - FastAPI backend for the Topological Data Analysis Platform.

This package provides:
- REST API endpoints for TDA computations
- Integration with C++23 TDA core engine  
- Apache Kafka streaming integration
- Apache Flink stream processing
- Job management and result caching
- Monitoring and observability
"""

__version__ = "1.0.0"
__author__ = "TDA Platform Team"
__email__ = "dev@tda-platform.com"

# Package-level imports for convenience
from .config import Settings, get_settings

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "Settings",
    "get_settings",
]