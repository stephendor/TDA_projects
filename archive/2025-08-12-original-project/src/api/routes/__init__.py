"""
TDA Platform API Routes

Modular API route definitions for different platform components.
"""

from .health import router as health_router
from .tda_core import router as tda_router
from .cybersecurity import router as cybersecurity_router
from .finance import router as finance_router
from .monitoring import router as monitoring_router

__all__ = [
    "health_router",
    "tda_router",
    "cybersecurity_router", 
    "finance_router",
    "monitoring_router"
]