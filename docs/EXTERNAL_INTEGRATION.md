# External Microservice Integration Guide

## Overview

The EDGP AI Model now includes comprehensive external microservice integration capabilities with shared functions for async, message queue (MQ), and API approaches. This guide covers the implementation of missing components and how external modules can interact with this microservice.

## 🔧 **Implemented Integration Patterns**

### 1. **Synchronous API Integration**
- Direct HTTP calls with immediate responses
- Used for real-time validations and quick data retrievals
- Includes automatic retry and circuit breaker patterns

### 2. **Asynchronous API Integration** 
- HTTP calls with callback handling for long-running operations
- Correlation ID tracking for request-response matching
- Webhook endpoints for receiving async responses

### 3. **Message Queue Integration**
- AWS SQS for reliable message delivery
- Priority-based message routing
- Dead letter queue handling for failed messages

### 4. **Event Broadcasting**
- AWS SNS for publishing events to multiple subscribers
- Topic-based event routing
- Priority levels for event processing

### 5. **Webhook Callback Handling**
- Generic and service-specific webhook endpoints
- Automatic correlation with pending requests
- Background processing for webhook payloads

## 🛠 **New Shared Functions and Components**

### Core Integration Modules

#### 1. `core/integration_patterns.py`
- **RequestCorrelationManager**: Tracks async requests and responses
- **ExternalServiceAdapter**: Base adapter for different integration patterns
- **SyncAPIAdapter**: Handles synchronous API calls
- **AsyncAPIAdapter**: Manages async API calls with callbacks
- **MessageQueueAdapter**: Integrates with message queue systems
- **WebhookHandler**: Processes incoming webhooks
- **IntegrationOrchestrator**: Coordinates all integration patterns
- **SharedIntegrationFunctions**: Common utilities for all integrations

#### 2. `core/llm_gateway_bridge.py`
- **LLMGatewayBridge**: Standardized interface between agents and LLM gateway
- **PromptTemplateManager**: Manages agent-specific prompt templates
- **AgentLLMInterface**: Simplified LLM interface for agents
- **BatchLLMProcessor**: Processes multiple LLM requests efficiently
- **CrossAgentCommunicationBridge**: Facilitates agent-to-agent communication

#### 3. `core/shared_integration.py`
- **AgentIntegrationHelper**: Helper class for agent external integrations
- **AgentOperationPipeline**: Multi-step operation pipeline for complex workflows
- **SharedAgentFunctions**: Common functions for all agents

#### 4. `core/enhanced_agents.py`
- **EnhancedAgentBase**: Extended agent base with integration capabilities
- **Agent-specific enhanced classes**: DataQuality, Compliance, Remediation, Analytics, Policy
- **AgentFactory**: Creates enhanced agents with integration capabilities
- **CrossAgentOperationManager**: Manages multi-agent workflows

#### 5. `core/external_endpoints.py`
- **External Service Router**: `/external/*` endpoints for service integration
- **Webhook Router**: `/webhooks/*` endpoints for callback handling  
- **LLM Router**: `/llm/*` endpoints for LLM gateway interactions
- **Integration Router**: `/integration/*` endpoints for utility functions

#### 6. `core/integration_config.py`
- **IntegrationConfigManager**: Manages external service configurations
- **ExternalServiceConfig**: Configuration dataclass for services
- **IntegrationWorkflowTemplates**: Pre-defined integration workflows
- **SharedUtilityFunctions**: Common utility functions

## 📋 **API Endpoints for External Microservices**

### External Service Management
```bash
POST /external/services/register      # Register new external service
GET  /external/services              # List registered services
POST /external/request               # Make request to external service
GET  /external/request/{id}/status   # Check request status
POST /external/batch-request         # Make multiple requests
GET  /external/aggregation/{id}/status # Check aggregation status
GET  /external/metrics               # Get integration metrics
```

### Webhook Handling
```bash
POST /webhooks/{service}/callback/{correlation_id}  # Service-specific webhook
POST /webhooks/generic                              # Generic webhook handler
```

### LLM Gateway Integration
```bash
POST /llm/generate                   # Generate LLM response for agent
POST /llm/analyze                   # Analyze data via LLM
POST /llm/collaborate               # Facilitate agent collaboration
GET  /llm/metrics                   # Get LLM bridge metrics
POST /llm/cache/clear               # Clear LLM cache
```

### Enhanced Agent Operations  
```bash
POST /enhanced-agents/{agent_id}/process           # Process with enhanced agent
POST /enhanced-agents/collaborate                  # Agent collaboration
POST /cross-agent-workflows/{workflow}             # Execute cross-agent workflow
GET  /cross-agent-workflows/{id}/status           # Get workflow status
```

### Integration Utilities
```bash
POST /integration/correlation/create               # Create correlation ID
POST /integration/payload/standardize             # Standardize payloads
GET  /integration/patterns                        # List integration patterns
```

## 🔗 **How External Modules Interact**

### 1. **For External Data Validation Services**

```python
# Register the service
POST /external/services/register
{
    "service_name": "data_validator_service",
    "base_url": "http://your-validator-service:8001",
    "pattern": "sync_api",
    "config": {
        "headers": {"X-API-Key": "your-api-key"},
        "endpoints": {
            "validate_schema": "/api/v1/validate/schema",
            "validate_quality": "/api/v1/validate/quality"
        }
    }
}

# Use in data quality agent
POST /enhanced-agents/data_quality_agent/process
{
    "data": {"dataset": "customer_data", "schema": {...}},
    "operation_type": "data_quality_assessment",
    "use_external_services": true
}
```

### 2. **For Regulatory Compliance Services**

```python
# Register async compliance service
POST /external/services/register  
{
    "service_name": "regulatory_service",
    "base_url": "http://compliance-api:8002",
    "pattern": "async_api",
    "config": {
        "headers": {"Authorization": "Bearer token"},
        "webhook_config": {
            "callback_path": "/webhooks/regulatory_service/callback"
        }
    }
}

# Make async compliance check
POST /external/request
{
    "service_name": "regulatory_service",
    "endpoint": "/api/v1/compliance/check",
    "pattern": "async_api", 
    "payload": {"data": {...}, "regulations": ["GDPR", "CCPA"]},
    "callback_url": "http://edgp-ai-model:8000/webhooks/regulatory_service/callback/{correlation_id}"
}
```

### 3. **For Automation Engine Integration**

```python
# Message queue pattern for automation
POST /external/queue-message
{
    "queue_name": "automation_engine",
    "message": {
        "action": "execute_remediation",
        "payload": {"issues": [...], "remediation_plan": {...}},
        "correlation_id": "req-123"
    }
}

# Event broadcasting for automation triggers
POST /external/send-event
{
    "topic": "agent.remediation_agent.remediation_completed",
    "data": {"remediation_id": "rem-123", "status": "completed"},
    "priority": "high"
}
```

### 4. **For Business Intelligence Integration**

```python
# Sync API for BI analytics
POST /external/request
{
    "service_name": "bi_analytics_service",
    "endpoint": "/api/v1/reports/generate",
    "pattern": "sync_api",
    "payload": {
        "report_type": "data_quality_dashboard",
        "data_source": "quality_assessments",
        "time_range": "last_30_days"
    }
}
```

## 🤖 **LLM Gateway Integration with Agents**

### How Agents Interact with LLM Gateway

#### 1. **Direct LLM Generation**
```python
# Via enhanced agent
POST /llm/generate
{
    "agent_id": "data_quality_agent",
    "prompt": "Analyze this dataset for quality issues: {...}",
    "template": "data_quality",
    "response_format": "json",
    "temperature": 0.3
}
```

#### 2. **Data Analysis via LLM**
```python
POST /llm/analyze
{
    "agent_id": "analytics_agent",
    "data": {"metrics": [...], "trends": [...]},
    "analysis_type": "trend_analysis",
    "context": {"time_period": "Q1_2024"}
}
```

#### 3. **Agent Collaboration**
```python
POST /llm/collaborate
{
    "initiating_agent": "compliance_agent",
    "target_agent": "remediation_agent", 
    "collaboration_prompt": "Develop remediation plan for compliance violations",
    "context_data": {"violations": [...], "priority": "high"}
}
```

### Enhanced Agent Features

#### 1. **Operation Pipelines**
Agents can create multi-step pipelines combining LLM processing and external service calls:

```python
# Example: Data Quality Assessment Pipeline
1. LLM Initial Assessment → 
2. External Schema Validation →
3. External Quality Validation →
4. LLM Final Synthesis
```

#### 2. **Cross-Agent Workflows**
```python
POST /cross-agent-workflows/comprehensive_assessment
{
    "agents_sequence": ["data_quality_agent", "compliance_agent", "remediation_agent"],
    "initial_data": {"dataset": {...}},
    "operation_config": {
        "data_quality_agent": {"operation": "quality_assessment"},
        "compliance_agent": {"operation": "compliance_check"},
        "remediation_agent": {"operation": "remediation_planning"}
    }
}
```

## 🔄 **Message Flow Patterns**

### 1. **Synchronous Request-Response**
```
External Service → POST /external/request → Agent Processing → LLM Gateway → Immediate Response
```

### 2. **Asynchronous with Callbacks**
```
External Service → POST /external/request (async) → Correlation ID Returned
                ↓
External Service Processing → Webhook Callback → Response Correlation → Agent Notification
```

### 3. **Message Queue Pattern**
```
External Service → SQS Message → Agent Processing → LLM Gateway → Response Queue → External Service
```

### 4. **Event Broadcasting**
```
Agent Event → SNS Topic → Multiple External Services → Independent Processing
```

## 🛡️ **Error Handling and Resilience**

### 1. **Retry Mechanisms**
- Automatic retry with exponential backoff
- Configurable retry counts per service
- Circuit breaker pattern for failing services

### 2. **Timeout Management**
- Request-level timeouts
- Correlation tracking with timeout cleanup
- Graceful degradation when services are unavailable

### 3. **Fallback Strategies**
- LLM-only processing when external services fail
- Default responses for critical operations
- Error aggregation and reporting

## 📊 **Monitoring and Observability**

### Metrics Available
- Request/response correlation tracking
- External service response times
- LLM gateway performance metrics
- Agent operation success rates
- Integration pattern usage statistics

### Health Checks
```bash
GET /health                    # Overall system health
GET /external/metrics         # External integration metrics  
GET /llm/metrics             # LLM gateway metrics
GET /monitoring/metrics      # Comprehensive metrics
```

## 🚀 **Usage Examples**

### Complete Integration Flow

1. **Register External Service**
```bash
curl -X POST http://localhost:8000/external/services/register \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "my_validation_service",
    "base_url": "http://my-service:8001", 
    "pattern": "sync_api",
    "config": {"headers": {"X-API-Key": "key123"}}
  }'
```

2. **Process Data with Enhanced Agent**
```bash
curl -X POST http://localhost:8000/enhanced-agents/data_quality_agent/process \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"dataset": "customer_data", "schema": {...}},
    "operation_type": "data_quality_assessment",
    "use_external_services": true
  }'
```

3. **Cross-Agent Workflow**
```bash
curl -X POST http://localhost:8000/cross-agent-workflows/comprehensive_assessment \
  -H "Content-Type: application/json" \
  -d '{
    "agents_sequence": ["data_quality_agent", "compliance_agent"],
    "initial_data": {"dataset": {...}},
    "operation_config": {}
  }'
```

## 🔧 **Configuration**

### Environment Variables
See updated `.env.example` for all external service configuration options.

### Service Configuration
External services are automatically configured based on environment variables, but can be dynamically registered via the API.

### Agent Configuration
Each agent type has default external service mappings that can be customized through the configuration system.

## 🎯 **Key Benefits**

1. **Standardized Integration**: Consistent patterns for all external service interactions
2. **Async Support**: Full support for long-running external operations
3. **Correlation Tracking**: Automatic tracking of request-response cycles
4. **LLM Integration**: Seamless integration between agents and LLM gateway
5. **Cross-Agent Workflows**: Orchestrated operations across multiple agents
6. **Resilience**: Built-in error handling, retries, and fallback mechanisms
7. **Monitoring**: Comprehensive metrics and observability
8. **Flexibility**: Support for multiple integration patterns in one system

This implementation provides a complete foundation for external microservice integration while maintaining the existing MCP internal communication and AWS SQS/SNS external messaging capabilities.
