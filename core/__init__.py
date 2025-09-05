"""
EDGP AI Model Core Module

Reorganized following industry best practices with domain-driven design:

- agents/       : Agent system (base, enhanced, MCP-enabled)
- communication/: Communication protocols (MCP, External)  
- infrastructure/: Cross-cutting concerns (config, auth, monitoring, errors)
- integrations/ : External service integration patterns
- services/     : Business services (LLM, RAG)
- types/        : Type definitions and data models

Note: Import specific modules directly to avoid circular dependencies.
Example: from core.agents.base import BaseAgent
"""

__version__ = "0.1.0"

# Domain modules are available for direct import:
# - core.agents
# - core.communication  
# - core.infrastructure
# - core.integrations
# - core.services
# - core.types
