"""
Agent system modules.
Contains base agent functionality and agent-specific implementations.
"""

# Core agent exports
from .base import (
    BaseAgent, 
    AgentState, 
    AgentStatus, 
    StandardizedAgentMessage, 
    StandardizedAgentTask, 
    AgentRegistry
)

# Re-export for convenience
agent_registry = AgentRegistry()

# Legacy aliases for backward compatibility
AgentMessage = StandardizedAgentMessage
AgentTask = StandardizedAgentTask

__all__ = [
    'BaseAgent',
    'AgentState', 
    'AgentStatus',
    'StandardizedAgentMessage',
    'StandardizedAgentTask',
    'AgentMessage',  # Legacy alias
    'AgentTask',     # Legacy alias
    'AgentRegistry',
    'agent_registry'
]