"""
Topological Data Analysis Platform
==================================

A comprehensive platform for applying TDA methods to cybersecurity and financial risk analysis.

This package provides:
- Core TDA utilities for persistent homology and mapper analysis
- Cybersecurity modules for APT detection and IoT device classification  
- Financial risk modules for cryptocurrency and multi-asset analysis
- Shared preprocessing and visualization tools
"""

__version__ = "0.1.0"
__author__ = "TDA Platform Team"

# Core imports
from .core import *
from .utils import *

# Domain-specific imports (optional)
try:
    from .cybersecurity import *
except ImportError:
    pass

try:
    from .finance import *
except ImportError:
    pass
