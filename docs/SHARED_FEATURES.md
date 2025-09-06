# Core Shared Features Documentation

## Overview

The `core/shared/` module provides enterprise-grade shared services that enable seamless collaboration between AI agents and comprehensive feature sharing across the platform. This includes standardized communication, prompt engineering, RAG management, memory systems, knowledge bases, tool management, and context management.

## Architecture

The shared features architecture follows enterprise patterns with:
- **Modular Design**: Each feature area has its own dedicated module
- **Type Safety**: Comprehensive type definitions with validation
- **Configurability**: Flexible configuration systems for different deployment scenarios
- **Extensibility**: Plugin architectures and factory patterns for customization
- **Error Handling**: Robust error handling and logging throughout
- **Performance**: Asynchronous operations with connection pooling and caching

## Module Structure

```
core/shared/
├── __init__.py              # Main exports and convenience functions
├── communication/           # Standardized agent communication
│   ├── types.py            # Communication type definitions
│   └── __init__.py         # Communication exports
├── prompt/                 # Prompt engineering system
│   ├── types.py            # Prompt type definitions
│   ├── manager.py          # Prompt management engine
│   ├── templates.py        # Template management
│   ├── optimization.py     # Prompt optimization
│   ├── validation.py       # Prompt validation
│   ├── registry.py         # Prompt registry
│   └── __init__.py         # Prompt system exports
├── rag/                    # RAG management system
│   ├── types.py            # RAG type definitions
│   ├── manager.py          # RAG management engine
│   ├── processors.py       # Document processors
│   ├── embeddings.py       # Embedding management
│   ├── stores.py           # Vector storage
│   ├── retrievers.py       # Retrieval engines
│   └── __init__.py         # RAG system exports
├── memory/                 # Memory management system
│   ├── types.py            # Memory type definitions
│   ├── manager.py          # Memory management engine
│   ├── stores.py           # Memory storage backends
│   ├── consolidation.py    # Memory consolidation
│   ├── indexing.py         # Memory indexing
│   └── __init__.py         # Memory system exports
├── knowledge/              # Knowledge base management
│   ├── types.py            # Knowledge type definitions
│   ├── manager.py          # Knowledge management engine
│   ├── stores.py           # Knowledge storage
│   ├── extraction.py       # Knowledge extraction
│   ├── inference.py        # Knowledge inference
│   └── __init__.py         # Knowledge system exports
├── tools/                  # Tool management system
│   ├── types.py            # Tool type definitions
│   ├── manager.py          # Tool management engine
│   └── __init__.py         # Tool system exports
└── context/                # Context management system
    ├── types.py            # Context type definitions
    ├── manager.py          # Context management engine
    ├── stores.py           # Context storage
    └── __init__.py         # Context system exports
```

## Quick Start

### Initialize Shared Services

```python
from core.shared import (
    create_communication_system,
    create_prompt_manager,
    create_rag_manager,
    create_memory_manager,
    create_knowledge_manager,
    create_tool_manager,
    create_context_manager
)

# Initialize core systems
communication = create_communication_system()
prompt_manager = await create_prompt_manager()
rag_manager = await create_rag_manager()
memory_manager = await create_memory_manager()
knowledge_manager = await create_knowledge_manager()
tool_manager = await create_tool_manager()
context_manager = await create_context_manager()
```

### Standardized Agent Communication

```python
from core.shared.communication import StandardAgentInput, StandardAgentOutput

# Standardized input/output
agent_input = StandardAgentInput[Dict[str, Any]](
    request_id="req_123",
    user_id="user_456",
    session_id="session_789",
    agent_id="analytics_agent",
    data={"query": "Analyze user engagement metrics"},
    metadata={"priority": "high"}
)

agent_output = StandardAgentOutput[Dict[str, str]](
    request_id="req_123",
    agent_id="analytics_agent",
    data={"result": "Analysis complete"},
    success=True
)
```

### Prompt Engineering

```python
# Create and manage prompts
prompt_id = await prompt_manager.create_prompt(
    name="user_analysis_prompt",
    content="Analyze user behavior data: {data}",
    category="analytics",
    agent_ids=["analytics_agent"]
)

# Use prompt with variables
rendered = await prompt_manager.render_prompt(
    prompt_id,
    variables={"data": "engagement metrics"}
)
```

### RAG Management

```python
# Add documents to RAG system
doc_id = await rag_manager.add_document(
    content="User engagement analysis guidelines...",
    metadata={"type": "guidelines", "category": "analytics"}
)

# Search and retrieve
results = await rag_manager.search(
    query="How to analyze user engagement?",
    collection_name="analytics_docs",
    top_k=5
)
```

### Memory Management

```python
# Store and retrieve memories
memory_id = await memory_manager.store_memory(
    content="User prefers detailed analytics reports",
    memory_type=MemoryType.SEMANTIC,
    metadata={"user_id": "user_456", "category": "preferences"}
)

memories = await memory_manager.retrieve_memories(
    query="user preferences for reports",
    memory_types=[MemoryType.SEMANTIC],
    limit=10
)
```

### Knowledge Management

```python
# Create knowledge entities
entity_id = await knowledge_manager.create_entity(
    name="User Engagement",
    entity_type="concept",
    properties={"definition": "Measure of user interaction"}
)

# Create relationships
await knowledge_manager.create_relationship(
    source_id=entity_id,
    target_id="metrics_entity_id",
    relationship_type="measured_by"
)

# Query knowledge
entities = await knowledge_manager.search_entities(
    query="user engagement metrics"
)
```

### Tool Management

```python
# Register tools
tool_id = await tool_manager.register_tool(
    name="engagement_analyzer",
    description="Analyze user engagement metrics",
    function=analyze_engagement_function,
    parameters={"metrics": "List of metrics to analyze"}
)

# Execute tools
result = await tool_manager.execute_tool(
    tool_id,
    arguments={"metrics": ["clicks", "time_spent", "conversion"]}
)
```

### Context Management

```python
# Manage sessions and conversations
session_id = await context_manager.create_session(
    user_id="user_456",
    session_type="analytics_session"
)

conversation_id = await context_manager.start_conversation(
    session_id=session_id,
    title="User Engagement Analysis"
)

# Store shared context
await context_manager.store_shared_context(
    name="current_analysis_context",
    data={"focus": "engagement", "timeframe": "last_30_days"},
    scope=ContextScope.SESSION
)
```

## Advanced Features

### Multi-Agent Collaboration

```python
# Shared prompt templates across agents
await prompt_manager.share_prompt_with_agents(
    prompt_id="analysis_template",
    agent_ids=["analytics_agent", "reporting_agent", "insights_agent"]
)

# Shared memory across sessions
await memory_manager.share_memory(
    memory_id="user_preferences",
    scope=MemoryScope.GLOBAL
)

# Cross-agent knowledge sharing
await knowledge_manager.share_knowledge_base(
    knowledge_base_id="company_metrics",
    agent_ids=["all_analytics_agents"]
)
```

### Advanced RAG Operations

```python
# Hybrid search combining vector and keyword search
results = await rag_manager.hybrid_search(
    query="user engagement optimization strategies",
    vector_weight=0.7,
    keyword_weight=0.3,
    collections=["best_practices", "case_studies"]
)

# Multi-modal document processing
await rag_manager.process_multimodal_document(
    file_path="engagement_report.pdf",
    extract_images=True,
    extract_tables=True
)
```

### Memory Consolidation

```python
# Automatic memory consolidation
await memory_manager.consolidate_memories(
    user_id="user_456",
    timeframe_days=7,
    consolidation_strategy="importance_based"
)

# Memory hierarchy management
await memory_manager.promote_memory(
    memory_id="important_insight",
    new_type=MemoryType.LONG_TERM
)
```

### Knowledge Inference

```python
# Infer new relationships
new_relationships = await knowledge_manager.infer_relationships(
    entity_id="user_engagement",
    inference_types=["correlation", "causation"]
)

# Knowledge consistency validation
inconsistencies = await knowledge_manager.validate_consistency()
```

### Workflow Automation

```python
# Create tool workflows
workflow_id = await tool_manager.create_workflow(
    name="complete_analysis_workflow",
    steps=[
        {"tool": "data_collector", "inputs": {"source": "analytics_db"}},
        {"tool": "engagement_analyzer", "inputs": {"data": "@step1.output"}},
        {"tool": "report_generator", "inputs": {"analysis": "@step2.output"}}
    ]
)

# Execute workflow
workflow_result = await tool_manager.execute_workflow(workflow_id)
```

## Configuration

### Environment-Based Configuration

```python
# Production configuration
config = {
    "prompt": {
        "cache_enabled": True,
        "optimization_enabled": True,
        "storage_path": "/data/prompts"
    },
    "rag": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store": "chroma",
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "memory": {
        "max_short_term_memories": 1000,
        "consolidation_interval_hours": 24,
        "storage_backend": "sqlite"
    },
    "knowledge": {
        "auto_extraction": True,
        "inference_enabled": True,
        "storage_backend": "sqlite"
    },
    "context": {
        "session_timeout_hours": 24,
        "message_retention_days": 30,
        "storage_backend": "sqlite"
    }
}
```

### Development Configuration

```python
# Development/testing configuration
dev_config = {
    "prompt": {"storage_path": ":memory:"},
    "rag": {"vector_store": "in_memory"},
    "memory": {"storage_backend": "in_memory"},
    "knowledge": {"storage_backend": "in_memory"},
    "context": {"storage_backend": "in_memory"}
}
```

## Integration Patterns

### Agent Integration

```python
from core.agents.base import BaseAgent
from core.shared import SharedServices

class AnalyticsAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.shared = SharedServices()
    
    async def process_request(self, input_data: StandardAgentInput):
        # Use shared prompt
        prompt = await self.shared.prompt.get_prompt("analytics_prompt")
        
        # Retrieve relevant documents
        docs = await self.shared.rag.search(input_data.data["query"])
        
        # Access relevant memories
        memories = await self.shared.memory.retrieve_memories(
            query=input_data.data["query"]
        )
        
        # Process with context
        result = await self.analyze_with_context(docs, memories)
        
        # Store new memories
        await self.shared.memory.store_memory(
            content=f"Analysis completed: {result}",
            metadata={"request_id": input_data.request_id}
        )
        
        return StandardAgentOutput(
            request_id=input_data.request_id,
            agent_id=self.agent_id,
            data=result,
            success=True
        )
```

### Multi-Agent Workflows

```python
async def multi_agent_analysis_workflow(query: str, user_id: str):
    """Example multi-agent workflow using shared services."""
    
    # Create shared session
    session_id = await context_manager.create_session(
        user_id=user_id,
        session_type="multi_agent_analysis"
    )
    
    # Store shared context
    await context_manager.store_shared_context(
        name="analysis_context",
        data={"query": query, "status": "starting"},
        scope=ContextScope.SESSION,
        session_id=session_id
    )
    
    # Step 1: Data Collection Agent
    data_agent = DataCollectionAgent()
    raw_data = await data_agent.collect_data(query, session_id)
    
    # Step 2: Analytics Agent
    analytics_agent = AnalyticsAgent()
    analysis = await analytics_agent.analyze_data(raw_data, session_id)
    
    # Step 3: Insights Agent
    insights_agent = InsightsAgent()
    insights = await insights_agent.generate_insights(analysis, session_id)
    
    # Step 4: Reporting Agent
    reporting_agent = ReportingAgent()
    report = await reporting_agent.create_report(insights, session_id)
    
    # Update shared context
    await context_manager.update_shared_context(
        context_id="analysis_context",
        data={"query": query, "status": "completed", "report_id": report.id}
    )
    
    return report
```

## Best Practices

### Performance Optimization

1. **Connection Pooling**: Use connection pools for database operations
2. **Caching**: Implement caching for frequently accessed data
3. **Batch Operations**: Use batch operations for bulk data processing
4. **Asynchronous Operations**: Use async/await for I/O operations
5. **Resource Management**: Proper cleanup and resource management

### Error Handling

1. **Graceful Degradation**: Handle service failures gracefully
2. **Retry Logic**: Implement retry logic for transient failures
3. **Circuit Breakers**: Use circuit breakers for external dependencies
4. **Comprehensive Logging**: Log errors with sufficient context
5. **Monitoring**: Monitor service health and performance

### Security Considerations

1. **Input Validation**: Validate all inputs thoroughly
2. **Access Control**: Implement proper access controls
3. **Data Encryption**: Encrypt sensitive data at rest and in transit
4. **Audit Logging**: Log security-relevant events
5. **Principle of Least Privilege**: Grant minimal necessary permissions

### Scalability

1. **Horizontal Scaling**: Design for horizontal scaling
2. **Load Balancing**: Implement load balancing strategies
3. **Data Partitioning**: Partition data appropriately
4. **Microservices**: Consider microservices architecture for large deployments
5. **Resource Monitoring**: Monitor resource usage and scale accordingly

## Testing

### Unit Testing

```python
import pytest
from core.shared import create_prompt_manager

@pytest.mark.asyncio
async def test_prompt_creation():
    manager = await create_prompt_manager(storage_path=":memory:")
    
    prompt_id = await manager.create_prompt(
        name="test_prompt",
        content="Test content: {variable}",
        category="test"
    )
    
    assert prompt_id is not None
    
    prompt = await manager.get_prompt(prompt_id)
    assert prompt.name == "test_prompt"
    assert prompt.content == "Test content: {variable}"
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_multi_service_integration():
    # Initialize services
    prompt_manager = await create_prompt_manager(storage_path=":memory:")
    rag_manager = await create_rag_manager(storage_path=":memory:")
    memory_manager = await create_memory_manager(storage_path=":memory:")
    
    # Test prompt and RAG integration
    prompt_id = await prompt_manager.create_prompt(
        name="search_prompt",
        content="Search for: {query}",
        category="search"
    )
    
    doc_id = await rag_manager.add_document(
        content="This is a test document about AI",
        metadata={"category": "ai"}
    )
    
    # Test search functionality
    results = await rag_manager.search("AI", top_k=1)
    assert len(results) == 1
    assert "AI" in results[0].content
```

## Monitoring and Analytics

The shared services provide comprehensive monitoring and analytics capabilities:

- **Usage Metrics**: Track usage patterns across all services
- **Performance Metrics**: Monitor latency, throughput, and resource usage
- **Error Tracking**: Track and analyze errors across services
- **Health Checks**: Automated health monitoring
- **Usage Analytics**: Detailed analytics on feature usage

## Support and Maintenance

- **Documentation**: Comprehensive API documentation
- **Examples**: Extensive examples and tutorials
- **Community**: Active community support
- **Updates**: Regular updates and feature additions
- **Migration**: Migration guides for updates

This comprehensive shared services architecture enables powerful, scalable, and maintainable AI agent systems with enterprise-grade features and collaboration capabilities.
