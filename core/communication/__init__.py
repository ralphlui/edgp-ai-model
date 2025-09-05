"""
Communication protocols for agent and external service interactions.
"""

from .mcp import MCPMessageBus, MCPResourceProvider
from .external import external_comm

__all__ = [
    "MCPMessageBus",
    "MCPResourceProvider", 
    "external_comm_manager"
]
