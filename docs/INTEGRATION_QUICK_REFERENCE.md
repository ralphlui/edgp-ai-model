# Quick Reference: External Microservice Integration

## 🚀 **Quick Start**

### 1. Register External Service
```python
POST /external/services/register
{
    "service_name": "my_service",
    "base_url": "http://my-service:8001",
    "pattern": "sync_api",  # or async_api, message_queue
    "config": {
        "headers": {"Authorization": "Bearer token"},
        "timeout": 30
    }
}
```

### 2. Use Enhanced Agent
```python
POST /enhanced-agents/data_quality_agent/process
{
    "data": {"dataset": "customer_data"},
    "operation_type": "data_quality_assessment",
    "use_external_services": true
}
```

### 3. Cross-Agent Workflow
```python
POST /cross-agent-workflows/assessment_pipeline
{
    "agents_sequence": ["data_quality_agent", "compliance_agent"],
    "initial_data": {"dataset": {...}}
}
```

## 📡 **Integration Patterns**

| Pattern | Use Case | Response Type |
|---------|----------|---------------|
| `sync_api` | Real-time validation, quick lookups | Immediate |
| `async_api` | Long-running analysis, complex processing | Callback |
| `message_queue` | Reliable delivery, batch processing | Queue response |
| `webhook_callback` | Event-driven updates | Push notification |

## 🤖 **Agent-LLM Gateway Integration**

### Direct LLM Access
```python
POST /llm/generate
{
    "agent_id": "compliance_agent",
    "prompt": "Assess GDPR compliance for: {...}",
    "template": "compliance",
    "response_format": "json"
}
```

### Data Analysis
```python
POST /llm/analyze  
{
    "agent_id": "analytics_agent",
    "data": {"metrics": [...]},
    "analysis_type": "trend_analysis"
}
```

### Agent Collaboration
```python
POST /llm/collaborate
{
    "initiating_agent": "data_quality_agent",
    "target_agent": "remediation_agent",
    "collaboration_prompt": "Create remediation plan for quality issues",
    "context_data": {"issues": [...]}
}
```

## 🔄 **Async Operation Tracking**

### Make Async Request
```python
POST /external/request
{
    "service_name": "regulatory_service",
    "pattern": "async_api",
    "endpoint": "/compliance/check",
    "payload": {...},
    "callback_url": "http://edgp:8000/webhooks/regulatory_service/callback/{correlation_id}"
}
# Returns: {"request_id": "uuid-123", "status": "pending"}
```

### Check Status
```python
GET /external/request/uuid-123/status
# Returns: {"status": "completed", "response_data": {...}}
```

### Handle Webhook
```python
# Automatic - webhook received at:
POST /webhooks/regulatory_service/callback/uuid-123
{
    "result": "compliant",
    "violations": [],
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## 📊 **Monitoring**

### System Health
```bash
GET /health                 # Overall system status
GET /external/metrics      # Integration metrics
GET /llm/metrics          # LLM bridge performance
```

### Request Tracking
```bash
GET /external/request/{id}/status        # Individual request
GET /aggregation/{task_id}/status       # Batch operation
GET /cross-agent-workflows/{id}/status  # Workflow progress
```

## ⚙️ **Configuration**

### Environment Variables
```bash
# External Services
DATA_VALIDATOR_URL=http://validator:8001
REGULATORY_SERVICE_URL=http://compliance:8002
AUTOMATION_ENGINE_URL=http://automation:8003

# API Keys
DATA_VALIDATOR_API_KEY=key123
REGULATORY_SERVICE_TOKEN=token456
```

### Dynamic Registration
Services can be registered at runtime via API, allowing for flexible deployment configurations.

## 🔧 **Development Examples**

### Custom External Service
```python
# 1. Register your service
await service_registry.register_service(
    "my_custom_service",
    "http://my-service:8080",
    IntegrationPattern.SYNC_API,
    {"headers": {"X-API-Key": "key"}}
)

# 2. Use in agent
result = await agent.call_external_microservice(
    "my_custom_service",
    "/api/process",
    {"data": "..."},
    IntegrationPattern.SYNC_API
)
```

### Custom Agent with External Integration
```python
from core.enhanced_agents import EnhancedAgentBase

class MyCustomAgent(EnhancedAgentBase):
    def _setup_default_templates(self):
        return {"analysis": PromptTemplate.ANALYTICS}
    
    async def my_custom_operation(self, data):
        # Use LLM
        llm_result = await self.process_with_external_llm(
            data, "analysis"
        )
        
        # Call external service
        external_result = await self.call_external_microservice(
            "my_service", "/analyze", data
        )
        
        # Combine results
        return {"llm": llm_result, "external": external_result}
```

## 📝 **Best Practices**

1. **Always use correlation IDs** for async operations
2. **Set appropriate timeouts** based on service characteristics  
3. **Implement proper error handling** for external service failures
4. **Use caching** for frequently requested LLM operations
5. **Monitor integration metrics** for performance optimization
6. **Configure fallback strategies** for critical operations

## 🔍 **Troubleshooting**

### Common Issues
- **Service not registered**: Use `GET /external/services` to verify
- **Correlation timeout**: Check service responsiveness and timeout settings
- **LLM errors**: Verify templates and check `GET /llm/metrics`
- **Agent not found**: Ensure agent is in enhanced agents registry

### Debug Endpoints
```bash
GET /monitoring/metrics     # System-wide metrics
GET /external/metrics      # Integration-specific metrics
GET /llm/metrics          # LLM bridge metrics
```

This implementation provides a complete foundation for external microservice interactions with comprehensive async support, LLM gateway integration, and shared utility functions.
