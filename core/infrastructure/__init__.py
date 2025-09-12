"""
Infrastructure components for the EDGP AI Model.
Contains configuration, authentication, monitoring, and error handling.
"""

from .config import get_settings, get_global_settings

# Create settings instance for backward compatibility
settings = get_global_settings()

# Gracefully handle optional dependencies
try:
    from . import auth
    from .auth import AuthManager
    # Verify auth module loaded properly
    if hasattr(auth, 'pwd_context'):
        auth_available = True
    else:
        auth = None
        AuthManager = None
        auth_available = False
except (ImportError, AttributeError):
    auth = None
    AuthManager = None
    auth_available = False

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
    "auth",
    "AuthManager",
    "monitor", 
    "metrics_collector",
    "ErrorHandler"
]
