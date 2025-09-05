"""
Core services for the EDGP AI Model.
"""

# Import services individually to avoid circular dependencies
from .llm_gateway import llm_gateway
from .rag_system import rag_system

__all__ = [
    'llm_gateway',
    'rag_system'
]

from .llm_gateway import llm_gateway, LLMResponse
from .llm_bridge import llm_bridge, cross_agent_bridge
from .rag_system import rag_system

__all__ = [
    "llm_gateway",
    "LLMResponse", 
    "llm_bridge",
    "cross_agent_bridge",
    "rag_system"
]
