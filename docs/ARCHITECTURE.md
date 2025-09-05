# EDGP AI Model - Architecture Documentation

## System Overview

The EDGP AI Model is an agentic AI microservice designed to solve master data management problems through specialized AI agents. The system leverages LangChain, LangGraph, MCP (Model Context Protocol), and RAG (Retrieval-Augmented Generation) to provide intelligent data governance solutions.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway                         │
│                     (main.py + core/)                          │
├─────────────────────────────────────────────────────────────────┤
│                      Core Components                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   LLM Gateway   │  │ Orchestration   │  │   RAG System    │ │
│  │                 │  │     Layer       │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Config Mgmt   │  │   Agent Base    │  │  Type System    │ │
│  │                 │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     AI Agent Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Policy Suggest  │  │ Data Privacy &  │  │  Data Quality   │ │
│  │     Agent       │  │ Compliance Agt  │  │     Agent       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │ Data Remediation│  │   Analytics     │                      │
│  │     Agent       │  │     Agent       │                      │
│  └─────────────────┘  └─────────────────┘                      │
├─────────────────────────────────────────────────────────────────┤
│                     External Services                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   AWS Bedrock   │  │   Vector Store  │  │   PostgreSQL    │ │
│  │ (Claude, Titan) │  │ (ChromaDB/FAISS)│  │   Database      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐                                           │
│  │   Redis Cache   │                                           │
│  │                 │                                           │
│  └─────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. LLM Gateway (`core/llm_gateway.py`)

**Purpose**: Centralized interface for multiple LLM providers with failover capabilities.

**Key Features**:
- Multi-provider support (AWS Bedrock, OpenAI, Anthropic)
- Automatic failover and load balancing
- Response caching and rate limiting
- Model-specific optimization

**Supported Models**:
- **AWS Bedrock**: Claude 3 (Sonnet/Haiku), Amazon Titan, Meta Llama 2
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude 3 (Direct API)

### 2. Orchestration Layer (`core/orchestration.py`)

**Purpose**: Manages multi-agent workflows and inter-agent communication.

**Capabilities**:
- Workflow execution and monitoring
- Agent dependency resolution
- Task routing and load balancing
- Error handling and recovery

### 3. RAG System (`core/rag_system.py`)

**Purpose**: Provides context-aware responses using retrieval-augmented generation.

**Components**:
- Document ingestion and chunking
- Vector embedding and storage
- Similarity search and retrieval
- Context injection for LLM queries

### 4. Type System (`core/types/`)

**Purpose**: Ensures type safety and standardized communication across agents.

**Features**:
- Pydantic-based validation
- Standardized request/response formats
- Agent-specific type definitions
- Validation utilities and error handling

## Agent Architecture

### Agent Structure

Each agent follows a consistent structure:

```python
class BaseAgent:
    """Base class for all AI agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_gateway = LLMGateway()
        self.rag_system = RAGSystem()
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Main processing method - must be implemented by subclasses"""
        raise NotImplementedError
    
    async def validate_input(self, request: AgentRequest) -> bool:
        """Validate input data"""
        pass
    
    async def enhance_with_context(self, request: AgentRequest) -> str:
        """Add RAG context to request"""
        pass
```

### Specialized Agents

#### 1. Policy Suggestion Agent
- **Purpose**: Generate data governance policies and validation rules
- **Inputs**: Business context, compliance requirements, data schemas
- **Outputs**: Policy recommendations, validation rules, implementation guides

#### 2. Data Privacy & Compliance Agent  
- **Purpose**: Ensure regulatory compliance and identify privacy risks
- **Inputs**: Data schemas, regulations, existing policies
- **Outputs**: Compliance assessments, privacy risk reports, violation alerts

#### 3. Data Quality Agent
- **Purpose**: Assess and monitor data quality across datasets
- **Inputs**: Datasets, quality rules, historical metrics
- **Outputs**: Quality scores, anomaly detection, quality reports

#### 4. Data Remediation Agent
- **Purpose**: Provide recommendations for fixing data quality issues
- **Inputs**: Quality issues, business rules, data constraints
- **Outputs**: Remediation strategies, automated fixes, manual procedures

#### 5. Analytics Agent
- **Purpose**: Generate insights and reports from data governance activities
- **Inputs**: Historical data, metrics, business questions
- **Outputs**: Analytics reports, trend analysis, recommendations

## Communication Patterns

### 1. Direct Agent Invocation

```python
# Single agent request
request = PolicySuggestionRequest(
    business_context="Financial services",
    compliance_requirements=[ComplianceRegulation.GDPR]
)
response = await policy_agent.process(request)
```

### 2. Multi-Agent Workflows

```python
# Orchestrated workflow
workflow_request = WorkflowRequest(
    workflow_id="data-governance-assessment",
    agents=[AgentType.DATA_QUALITY, AgentType.COMPLIANCE],
    data_source=data_source,
    parallel_execution=False
)
workflow_response = await orchestrator.execute_workflow(workflow_request)
```

### 3. Inter-Agent Communication

```python
# Agent-to-agent messaging
message = AgentMessage(
    from_agent=AgentType.DATA_QUALITY,
    to_agent=AgentType.DATA_REMEDIATION,
    message_type="quality_issues_detected",
    payload={"issues": quality_issues}
)
await orchestrator.send_message(message)
```

## Data Flow

### 1. Request Processing Flow

```
Client Request
    ↓
FastAPI Endpoint
    ↓
Input Validation (Pydantic)
    ↓
Agent Selection/Routing
    ↓
Context Enhancement (RAG)
    ↓
LLM Processing (Bedrock/OpenAI)
    ↓
Response Validation
    ↓
Client Response
```

### 2. Multi-Agent Workflow

```
Workflow Request
    ↓
Orchestration Layer
    ↓
┌─────────────────┐    ┌─────────────────┐
│   Agent A       │    │   Agent B       │
│   Processing    │ ←→ │   Processing    │
└─────────────────┘    └─────────────────┘
    ↓                      ↓
Result Aggregation
    ↓
Workflow Response
```

## Configuration Management

### Environment Variables

```bash
# Core Configuration
ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://localhost:5432/edgp_ai
REDIS_URL=redis://localhost:6379

# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# OpenAI Configuration (Fallback)
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4

# Vector Store Configuration
VECTOR_STORE_TYPE=chromadb
CHROMA_PERSIST_DIR=./data/chroma

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000"]
```

### Agent Configuration

```python
# Agent-specific settings
AGENT_CONFIGS = {
    "policy_suggestion": {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "temperature": 0.1,
        "max_tokens": 4096
    },
    "data_quality": {
        "model": "anthropic.claude-3-haiku-20240307-v1:0", 
        "temperature": 0.0,
        "max_tokens": 2048
    }
}
```

## Security Considerations

### 1. Data Protection
- PII data encryption in transit and at rest
- Role-based access control (RBAC)
- API authentication and authorization
- Audit logging for all operations

### 2. Model Security
- Input sanitization and validation
- Output filtering for sensitive content
- Rate limiting and abuse prevention
- Model response monitoring

### 3. Infrastructure Security
- TLS/SSL encryption for all communications
- VPC isolation for cloud deployments
- Secrets management for API keys
- Container security scanning

## Scalability Design

### 1. Horizontal Scaling
- Stateless agent design for easy replication
- Load balancing across agent instances
- Database connection pooling
- Caching layers (Redis) for performance

### 2. Performance Optimization
- Async/await patterns throughout
- Lazy loading of models and data
- Response caching for repeated queries
- Efficient vector similarity search

### 3. Resource Management
- Memory-efficient model loading
- CPU/GPU resource allocation
- Queue management for high-throughput scenarios
- Auto-scaling based on demand

## Development Workflow

### 1. Adding New Agents

1. Create agent folder: `agents/new_agent/`
2. Define agent-specific types: `core/types/agents/new_agent.py`
3. Implement agent class: Inherit from `BaseAgent`
4. Add endpoint: Update `main.py` with new routes
5. Update configuration: Add agent config settings
6. Write tests: Unit and integration tests

### 2. Extending Types

1. Add new enums to `core/types/base.py`
2. Create new data models in `core/types/data.py`
3. Update agent types in `core/types/agent_types.py`
4. Update validation logic in `core/types/validation.py`
5. Update documentation

### 3. Testing Strategy

```python
# Unit tests for individual agents
async def test_policy_suggestion_agent():
    agent = PolicySuggestionAgent()
    request = PolicySuggestionRequest(...)
    response = await agent.process(request)
    assert response.success
    assert isinstance(response, PolicySuggestionResponse)

# Integration tests for workflows
async def test_data_governance_workflow():
    workflow = DataGovernanceWorkflow()
    result = await workflow.execute(workflow_request)
    assert result.success
```

## Monitoring and Observability

### 1. Metrics Collection
- Request/response latency
- Agent success/failure rates
- LLM token usage and costs
- Cache hit/miss ratios

### 2. Logging Strategy
- Structured logging with JSON format
- Request tracing with correlation IDs
- Agent decision logging
- Error tracking and alerting

### 3. Health Checks
- Agent availability monitoring
- Database connectivity checks
- External service health verification
- Resource utilization tracking

## Deployment

### Development Environment
```bash
# Local development
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Production Considerations
- Container orchestration (Kubernetes)
- Load balancing and auto-scaling
- Database migrations and backups
- Secrets management
- Monitoring and alerting setup

## Next Steps

1. **Phase 1**: Complete agent implementations with LLM integration
2. **Phase 2**: Implement comprehensive RAG system
3. **Phase 3**: Add MCP protocol support
4. **Phase 4**: Implement inter-agent workflows
5. **Phase 5**: Add monitoring and observability
6. **Phase 6**: Production deployment and scaling

This architecture provides a solid foundation for building a comprehensive, scalable, and maintainable agentic AI system for master data management.
