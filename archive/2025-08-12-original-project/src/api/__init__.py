"""
TDA Platform API Module

Provides REST API endpoints for real-time TDA computations,
cybersecurity threat detection, and financial risk analysis.
"""

__version__ = "1.0.0"
__author__ = "TDA Platform Team"

from .server import create_app, app
from .routes import (
    health_bp,
    tda_bp,
    cybersecurity_bp,
    finance_bp,
    monitoring_bp
)

__all__ = [
    "create_app",
    "app",
    "health_bp",
    "tda_bp", 
    "cybersecurity_bp",
    "finance_bp",
    "monitoring_bp"
]