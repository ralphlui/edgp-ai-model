# Repository Structure & Organization Guide

## Overview
The EDGP AI Model repository has been reorganized following industry best practices with a domain-driven design approach. This document explains the new structure and how to work with it.

## 🏗️ New Architecture

### Domain-Driven Structure
```
core/
├── agents/        # Agent System Domain
├── communication/ # Communication Domain  
├── infrastructure/# Infrastructure Domain
├── integrations/  # Integration Domain
├── services/      # Services Domain
└── types/         # Type System Domain
```

## 📁 Directory Details

### `core/agents/` - Agent System
Contains all agent-related business logic and base classes.

**Files:**
- `base.py` - BaseAgent, AgentStatus, AgentMessage, AgentTask, AgentRegistry
- `enhanced.py` - EnhancedAgentBase with external service integration
- `mcp_enabled.py` - MCP-enabled agents for internal communication

**Import Examples:**
```python
from core.agents.base import BaseAgent, AgentType, AgentStatus
from core.agents.enhanced import EnhancedAgentBase
from core.agents.mcp_enabled import MCPEnabledAgent
```

### `core/communication/` - Communication Protocols
Handles all internal and external communication protocols.

**Files:**
- `mcp.py` - Model Context Protocol for internal agent communication
- `external.py` - AWS SQS/SNS for external microservice communication

**Import Examples:**
```python
from core.communication.mcp import MCPMessageBus, MCPResourceProvider
from core.communication.external import ExternalCommunicationManager
```

### `core/infrastructure/` - Cross-Cutting Concerns
Contains system-wide infrastructure components.

**Files:**
- `config.py` - Configuration management and settings
- `auth.py` - Authentication and authorization
- `monitoring.py` - System monitoring and metrics
- `error_handling.py` - Error handling and exceptions
- `auth_endpoints.py` - Authentication API endpoints
- `monitoring_endpoints.py` - Monitoring API endpoints

**Import Examples:**
```python
from core.infrastructure.config import settings, get_settings
from core.infrastructure.monitoring import SystemMonitor, Metrics
from core.infrastructure.error_handling import ErrorHandler
```

### `core/integrations/` - External Service Integration
Manages patterns and utilities for external microservice integration.

**Files:**
- `patterns.py` - Integration patterns (sync/async/MQ/webhooks)
- `config.py` - Integration configuration management
- `shared.py` - Shared utilities for agent-external service integration
- `endpoints.py` - API endpoints for external service integration

**Import Examples:**
```python
from core.integrations.patterns import integration_orchestrator
from core.integrations.config import integration_config_manager
from core.integrations.shared import create_agent_integration_helper
```

### `core/services/` - Business Services
Contains core business logic services.

**Files:**
- `llm_gateway.py` - LLM gateway for AI model interactions
- `llm_bridge.py` - Bridge between agents and LLM gateway
- `rag_system.py` - Retrieval-Augmented Generation system

**Import Examples:**
```python
from core.services.llm_gateway import llm_gateway, LLMResponse
from core.services.llm_bridge import llm_bridge, cross_agent_bridge
from core.services.rag_system import rag_system
```

### `core/types/` - Type System
Centralized type definitions and data models.

**Files:**
- `base.py` - Base types and enums
- `agent_types.py` - Agent-specific types and capabilities
- `data.py` - Data models and schemas
- `responses.py` - Response models
- `validation.py` - Validation utilities

**Import Examples:**
```python
from core.types.base import AgentType, Severity, ConfidenceLevel
from core.types.agent_types import AgentCapability
from core.types.data import ProcessingRequest, DataSchema
```

## 🔧 Major Improvements Made

### 1. Removed Duplicates
- **3 different `AgentCapability` enums** → Single source of truth in `core/types/agent_types.py`
- **Duplicate imports/exports** in `agents/__init__.py` → Clean single imports
- **Redundant policy_suggestion/main.py** → Removed (functionality in agent.py)

### 2. Applied Industry Best Practices
- **Domain-Driven Design (DDD)** - Organized by business domain
- **Separation of Concerns** - Clear boundaries between domains
- **Single Responsibility Principle** - Each module has one clear purpose
- **Dependency Inversion** - Infrastructure details isolated
- **Clean Architecture** - Layered approach with clear dependencies

### 3. Import Path Updates
Updated 25+ import statements throughout the codebase to use new structure:

**Before:**
```python
from core.agent_base import BaseAgent
from core.enhanced_agents import EnhancedAgentBase
from core.mcp_communication import MCPMessageBus
```

**After:**
```python
from core.agents.base import BaseAgent
from core.agents.enhanced import EnhancedAgentBase  
from core.communication.mcp import MCPMessageBus
```

## 🚀 Benefits

### 1. **Better Maintainability**
- Clear domain boundaries make code easier to navigate
- Reduced coupling between unrelated components
- Easier to locate and modify specific functionality

### 2. **Improved Testability**
- Isolated domains can be tested independently
- Clear dependencies make mocking easier
- Better separation enables unit testing

### 3. **Enhanced Scalability**
- New agents can be added to `agents/` domain
- New services fit naturally in `services/` domain
- External integrations have dedicated space

### 4. **Developer Experience**
- Predictable file locations following industry standards
- Clear import paths reduce confusion
- Self-documenting directory structure

## 📝 Working with the New Structure

### Adding New Agents
1. Create agent class in `agents/[agent_name]/agent.py`
2. Define agent types in `core/types/agents/[agent_name].py`
3. Import in `agents/__init__.py`

### Adding New Services
1. Create service in `core/services/[service_name].py`
2. Add to `core/services/__init__.py`
3. Import where needed

### Adding External Integrations
1. Add patterns to `core/integrations/patterns.py`
2. Add configuration to `core/integrations/config.py`
3. Add endpoints to `core/integrations/endpoints.py`

## ⚠️ Migration Notes

- All import paths have been updated in `main.py`
- Agent implementations maintained backward compatibility
- Type system consolidated to prevent conflicts
- Documentation updated to reflect new structure

This reorganization positions the codebase for better long-term maintainability and follows established enterprise software development patterns.
