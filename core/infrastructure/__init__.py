"""
Infrastructure components for the EDGP AI Model.
Contains configuration, authentication, monitoring, and error handling.
"""

from .config import get_settings, get_global_settings

# Create settings instance for backward compatibility
settings = get_global_settings()

# Gracefully handle optional dependencies
try:
    from .auth import AuthManager
except ImportError:
    AuthManager = None

try:
    from .monitoring import monitor, metrics_collector
except ImportError:
    monitor = None
    metrics_collector = None

try:
    from .error_handling import ErrorHandler
except ImportError:
    ErrorHandler = None

__all__ = [
    "settings",
    "AuthManager",
    "monitor", 
    "metrics_collector",
    "ErrorHandler"
]
