# LangChain/LangGraph Integration Guide

## Overview

The collaborative AI platform now includes comprehensive integration with LangChain and LangGraph frameworks, providing sophisticated workflow orchestration, tool integration, and enhanced agent capabilities while maintaining all existing shared services.

## Architecture

### Integration Layer Components

1. **SharedServicesToolkit** - Converts shared services into LangChain tools
2. **LangGraphWorkflowBuilder** - Creates and manages LangGraph workflows  
3. **SharedServicesCallback** - Monitors and tracks LangChain operations
4. **EnhancedAgentBase** - LangChain-integrated agent base class
5. **LangGraphState** - Enhanced state management for workflows

### Key Benefits

- **Workflow Orchestration**: Sequential, parallel, and conditional agent workflows
- **Tool Integration**: Native LangChain tools for all shared services
- **State Management**: Enhanced state tracking across workflow execution
- **Performance Monitoring**: Comprehensive metrics and callback tracking
- **Backward Compatibility**: Existing agents continue to work unchanged

## Quick Start

### 1. Basic Integration Setup

```python
from core.shared import create_shared_services
from core.integrations.langchain_integration import create_langchain_integration

# Initialize shared services
config = {
    "prompt": {"storage_path": ":memory:"},
    "rag": {"storage_path": ":memory:"},
    "memory": {"storage_path": ":memory:"},
    "knowledge": {"storage_path": ":memory:"},
    "context": {"storage_path": ":memory:"}
}

shared_services = await create_shared_services(config)
langchain_integration = await create_langchain_integration(shared_services, config)
```

### 2. Creating Enhanced Agents

```python
from core.agents.enhanced_base import LangChainAgent

class MyAgent(LangChainAgent):
    def __init__(self, shared_services):
        super().__init__(
            agent_id="my_agent",
            agent_type="custom",
            name="My Agent",
            description="Custom agent with LangChain integration",
            shared_services=shared_services,
            capabilities=["analyze", "process", "respond"],
            system_prompt="You are a helpful assistant."
        )
    
    async def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        # Analyze input to determine processing needs
        return {"requires_analysis": True}
    
    def _determine_required_capability(self, input_analysis: Dict[str, Any]) -> str:
        # Determine which capability to use
        return "analyze"
    
    async def _execute_capability(self, capability: str, state) -> Any:
        # Execute the specific capability
        return {"result": "analysis complete"}
    
    async def _compile_response(self, state) -> str:
        # Compile final response
        return "Processing completed successfully"
```

### 3. Creating LangGraph Workflows

```python
# Create a sequential workflow
workflow_agents = [
    {"agent_id": "data_quality", "type": "data_quality"},
    {"agent_id": "compliance", "type": "compliance"},
    {"agent_id": "analytics", "type": "analytics"}
]

langchain_integration.create_custom_workflow(
    "comprehensive_analysis",
    workflow_agents,
    "sequential"
)

# Execute the workflow
result = await langchain_integration.execute_workflow(
    workflow_name="comprehensive_analysis",
    message="Analyze our customer data",
    user_id="user123",
    session_id="session456"
)
```

## Detailed Components

### LangGraphState

Enhanced state management that tracks:

```python
@dataclass
class LangGraphState:
    # Core message and context
    message: str
    user_id: str
    session_id: str
    
    # Workflow management
    current_agent: Optional[str]
    completed_agents: List[str]
    workflow_type: str
    step_count: int
    
    # Results and outputs
    agent_outputs: Dict[str, Any]
    tool_results: Dict[str, Any]
    shared_context: Dict[str, Any]
    
    # Performance tracking
    start_time: float
    agent_durations: Dict[str, float]
    
    # Error handling
    errors: List[str]
    retry_count: int
```

### SharedServicesToolkit

Provides LangChain tools for all shared services:

- **Memory Tools**: Store, retrieve, and search memories
- **RAG Tools**: Add documents and search knowledge
- **Knowledge Tools**: Manage entities and relationships
- **Prompt Tools**: Create and manage prompt templates
- **Context Tools**: Manage session and conversation context
- **Tool Tools**: Register and execute custom tools

Example usage:

```python
# Get available tools
tools = langchain_integration.get_shared_tools()

# Use specific tools in an agent
from langchain_core.tools import tool

@tool
async def custom_analysis(data: str) -> str:
    """Perform custom analysis on data."""
    # Use shared services through toolkit
    memories = await toolkit.memory_search("analysis patterns", limit=5)
    return f"Analysis result based on {len(memories)} patterns"
```

### Workflow Types

#### Sequential Workflow
Agents execute one after another, passing results between them:

```python
# Sequential workflow execution
agent1 → agent2 → agent3 → final_result
```

#### Parallel Workflow
Multiple agents execute simultaneously:

```python
# Parallel workflow execution
        ↗ agent1 ↘
input →   agent2   → combine_results → final_result
        ↘ agent3 ↗
```

#### Conditional Workflow
Agents execute based on conditions:

```python
# Conditional workflow execution
input → condition_check → agent1 (if condition A)
                       → agent2 (if condition B)
                       → agent3 (if condition C)
```

### Performance Monitoring

The integration includes comprehensive monitoring:

```python
# Agent metrics
metrics = agent.get_metrics()
# {
#     "total_requests": 150,
#     "successful_requests": 145,
#     "failed_requests": 5,
#     "average_response_time": 2.3,
#     "capabilities_used": {"analyze": 80, "process": 70},
#     "tools_used": {"memory_search": 45, "rag_search": 30}
# }

# Workflow metrics
workflow_result = await langchain_integration.execute_workflow(...)
# {
#     "success": True,
#     "executed_agents": ["agent1", "agent2", "agent3"],
#     "step_count": 8,
#     "duration": 15.7,
#     "agent_outputs": {...},
#     "performance_metrics": {...}
# }
```

## Advanced Features

### Custom Tool Creation

Create tools that integrate with shared services:

```python
from langchain_core.tools import tool

@tool
async def analyze_customer_sentiment(customer_id: str) -> str:
    """Analyze customer sentiment using shared services."""
    
    # Retrieve customer memories
    memories = await shared_services.memory.retrieve_memories(
        query=f"customer {customer_id} feedback sentiment",
        limit=10
    )
    
    # Search knowledge base
    entities = await shared_services.knowledge.search_entities(
        query=f"customer {customer_id}",
        limit=5
    )
    
    # Perform analysis
    sentiment_score = analyze_sentiment_from_data(memories, entities)
    
    # Store result
    await shared_services.memory.store_memory(
        content=f"Customer {customer_id} sentiment: {sentiment_score}",
        metadata={"type": "sentiment_analysis", "customer_id": customer_id}
    )
    
    return f"Customer sentiment score: {sentiment_score}"
```

### Workflow Customization

Create complex conditional workflows:

```python
def create_adaptive_workflow(langchain_integration):
    """Create workflow that adapts based on input analysis."""
    
    # Define workflow logic
    def should_run_compliance(state):
        return "compliance" in state.message.lower()
    
    def should_run_analytics(state):
        return "analyze" in state.message.lower() or "trend" in state.message.lower()
    
    def should_run_quality_check(state):
        return "data" in state.message.lower() and "quality" in state.message.lower()
    
    # Create conditional workflow
    workflow_config = {
        "name": "adaptive_analysis",
        "type": "conditional",
        "conditions": [
            {
                "condition": should_run_quality_check,
                "agent_id": "data_quality_agent",
                "priority": 1
            },
            {
                "condition": should_run_compliance,
                "agent_id": "compliance_agent", 
                "priority": 2
            },
            {
                "condition": should_run_analytics,
                "agent_id": "analytics_agent",
                "priority": 3
            }
        ]
    }
    
    return langchain_integration.create_conditional_workflow(workflow_config)
```

### Error Handling and Retry Logic

The integration includes robust error handling:

```python
# Automatic retry for failed agents
workflow_config = {
    "retry_failed_agents": True,
    "max_retries": 3,
    "retry_delay": 2.0,  # seconds
    "failure_strategy": "continue"  # or "stop"
}

# Custom error handling in agents
class RobustAgent(LangChainAgent):
    async def _execute_capability(self, capability: str, state):
        try:
            return await super()._execute_capability(capability, state)
        except Exception as e:
            # Log error and provide fallback
            logger.error(f"Capability {capability} failed: {e}")
            return {"error": str(e), "fallback_result": "default_response"}
```

## Production Deployment

### Configuration

Use the production configuration example:

```python
from examples.production_config import get_production_config

config = get_production_config()
shared_services = await create_shared_services(config)
langchain_integration = await create_langchain_integration(
    shared_services, 
    config.langchain_config
)
```

### Environment Variables

Required environment variables:

```bash
# Core Configuration
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@localhost/edgp_ai
REDIS_URL=redis://localhost:6379

# LLM Configuration
OPENAI_API_KEY=your-api-key
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7

# Workflow Configuration
WORKFLOW_TIMEOUT=300
MAX_PARALLEL_AGENTS=5
ENABLE_MONITORING=true

# Security
JWT_SECRET=your-secret
API_KEY_REQUIRED=true
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
```

### Kubernetes Deployment

Use the Kubernetes configuration from `examples/production_config.py`:

```bash
kubectl apply -f kubernetes-deployment.yml
kubectl apply -f kubernetes-secrets.yml
```

## Best Practices

### Agent Development

1. **Capability-Based Design**: Design agents around specific capabilities
2. **Shared Services Integration**: Leverage shared services for memory, knowledge, etc.
3. **Error Handling**: Implement robust error handling and fallback mechanisms
4. **Performance Monitoring**: Use metrics to track agent performance
5. **Tool Integration**: Create custom tools that enhance agent capabilities

### Workflow Design

1. **Modular Workflows**: Design workflows as composable components
2. **State Management**: Use workflow state effectively for data passing
3. **Conditional Logic**: Implement smart routing based on input analysis
4. **Performance Optimization**: Monitor and optimize workflow execution times
5. **Error Recovery**: Design workflows to handle agent failures gracefully

### Production Considerations

1. **Resource Management**: Monitor memory and CPU usage
2. **Scalability**: Design for horizontal scaling
3. **Security**: Implement proper authentication and authorization
4. **Monitoring**: Use comprehensive logging and metrics
5. **Testing**: Implement thorough unit and integration tests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure LangChain dependencies are installed
2. **Configuration Issues**: Check environment variables and config files
3. **Workflow Failures**: Check agent logs and error messages
4. **Performance Issues**: Monitor metrics and optimize bottlenecks
5. **Integration Issues**: Verify shared services are properly initialized

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable LangChain debugging
from langchain.globals import set_debug
set_debug(True)
```

Monitor workflow execution:

```python
# Check workflow status
status = await langchain_integration.get_workflow_status("workflow_name")

# Get agent metrics
metrics = agent.get_metrics()

# Check shared services status
status = shared_services.get_status()
```

## Migration Guide

### From Basic Agents to LangChain Agents

1. **Extend LangChainAgent**: Replace base agent class
2. **Implement Required Methods**: Add capability analysis and execution
3. **Update Tool Usage**: Use LangChain tool decorators
4. **Migrate State Management**: Use enhanced state tracking
5. **Test Integration**: Verify shared services integration

Example migration:

```python
# Before: Basic agent
class OldAgent(BaseAgent):
    async def process(self, message):
        return "response"

# After: LangChain integrated agent
class NewAgent(LangChainAgent):
    def __init__(self, shared_services):
        super().__init__(
            agent_id="new_agent",
            agent_type="migrated",
            shared_services=shared_services,
            capabilities=["process"]
        )
    
    async def _execute_capability(self, capability, state):
        return "enhanced response"
```

## Examples

See the `examples/` directory for complete examples:

- `langchain_integration_demo.py` - Comprehensive integration demonstration
- `production_config.py` - Production configuration examples
- Agent implementations in `agents/` directory

## Contributing

When contributing to the LangChain/LangGraph integration:

1. Follow existing patterns and conventions
2. Include comprehensive tests
3. Update documentation for new features
4. Ensure backward compatibility
5. Add examples for new functionality

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review example implementations
3. Check logs for detailed error messages
4. Verify configuration and environment variables
5. Test with simplified examples first
