"""
Core configuration module.
Re-exports the main configuration for easy access.
"""

from .infrastructure.config import settings, get_settings

__all__ = ['settings', 'get_settings']
