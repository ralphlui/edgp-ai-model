# EDGP AI Model - Agentic Master Data Management

An intelligent, agentic AI microservice architecture for solving master data management problems using **LangChain, LangGraph**, Model Context Protocol (MCP), and RAG technologies with comprehensive shared services and enterprise-grade infrastructure.

## 🚀 Latest Updates

### 🔗 **LangChain/LangGraph Integration** 
Complete integration with LangChain and LangGraph frameworks providing:
- **Sophisticated Workflow Orchestration**: Sequential, parallel, and conditional agent workflows
- **Enhanced Agent Architecture**: LangChain-integrated base classes with advanced capabilities
- **Tool Integration**: Native LangChain tools for all shared services
- **State Management**: Advanced state tracking across workflow execution
- **Performance Monitoring**: Comprehensive metrics and callback tracking

### 📨 **Message Queue Service** (NEW)
Enterprise-grade message queue integration with AWS SQS:
- **Multiple Queue Support**: Handle both request and response queues
- **Runtime AWS Credentials**: Support for environment variable-based AWS authentication
- **Message Processing**: Automated message consumption and processing
- **Workflow Triggers**: API endpoints for triggering agent workflows via messages
- **Queue Monitoring**: Real-time queue statistics and health monitoring

### 🌟 **Enterprise Shared Services**
- **Standardized Communication**: Unified agent input/output types
- **Prompt Engineering**: Template management, A/B testing, and optimization
- **Advanced RAG**: Multi-modal processing with intelligent chunking
- **Memory Management**: Conversation, working, and long-term memory with consolidation
- **Knowledge Base**: Entity management with graph relationships and inference
- **Tool Management**: Dynamic registration and workflow integration
- **Context Management**: Session management with state persistence

## 🏗️ Architecture Overview

This system implements an agentic AI layer with 5 specialized agents that work together to provide comprehensive master data management:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Agentic AI Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────────────┐ │
│  │ Policy          │    │ Data Privacy &                          │ │
│  │ Suggestion      │───▶│ Compliance Agent                        │ │
│  │ Agent           │    └─────────────────────────────────────────┘ │
│  └─────────────────┘                    │                           │
│           │                             ▼                           │
│           ▼                    ┌─────────────────┐                  │
│  ┌─────────────────┐           │ Data            │                  │
│  │ Data Quality    │──────────▶│ Remediation     │                  │
│  │ Agent           │           │ Agent           │                  │
│  └─────────────────┘           └─────────────────┘                  │
│           │                             │                           │
│           └─────────────┐               │                           │
│                        ▼               ▼                           │
│                    ┌─────────────────────────────────────────────┐  │
│                    │ Analytics Agent                             │  │
│                    │ (Tabular/Chart Generation)                  │  │
│                    └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🤖 Agents

### 1. Policy Suggestion Agent
- **Purpose**: Suggests rules and policies for data governance
- **Capabilities**:
  - Policy creation assistance
  - Data validation rule suggestions
  - Governance framework recommendations
  - Compliance rule generation

### 2. Data Privacy & Compliance Agent
- **Purpose**: Monitors data privacy and compliance risks
- **Capabilities**:
  - Privacy risk detection
  - Compliance violation identification
  - Remediation task generation
  - Regulatory compliance monitoring

### 3. Data Quality Agent
- **Purpose**: Detects and reports data quality issues
- **Capabilities**:
  - Anomaly detection
  - Duplicate record identification
  - Data quality metrics calculation
  - Quality trend analysis

### 4. Data Remediation Agent
- **Purpose**: Handles data remediation and provides guidance
- **Capabilities**:
  - Automated data correction
  - Manual remediation guidance
  - Remediation outcome tracking
  - Quality improvement measurement

### 5. Analytics Agent
- **Purpose**: Provides analytics, reporting, and visualization
- **Capabilities**:
  - Dashboard generation
  - Chart and table creation
  - Metrics compilation
  - Report generation

## 🔗 LangChain/LangGraph Integration

The platform features comprehensive integration with LangChain and LangGraph frameworks, providing sophisticated workflow orchestration and enhanced agent capabilities.

### 🌟 Key Integration Features

#### **LangGraph Workflows**
- **Sequential Workflows**: Execute agents one after another with result passing
- **Parallel Workflows**: Run multiple agents simultaneously for faster processing
- **Conditional Workflows**: Smart routing based on input analysis and conditions
- **State Management**: Advanced state tracking across entire workflow execution

#### **Enhanced Agent Architecture**
- **LangChainAgent Base Class**: All agents inherit from enhanced base with LangChain integration
- **Capability-Based Processing**: Agents analyze input and route to appropriate capabilities
- **Tool Integration**: Native LangChain tools for all shared services
- **Workflow Nodes**: Custom nodes for capability routing, execution, and response compilation

#### **Shared Services Toolkit**
LangChain tools are automatically created for all shared services:
- **Memory Tools**: Store, retrieve, and search memories across conversations
- **RAG Tools**: Add documents and perform semantic searches
- **Knowledge Tools**: Manage entities, relationships, and graph queries
- **Prompt Tools**: Create, version, and optimize prompt templates
- **Context Tools**: Manage session state and conversation context
- **Tool Tools**: Register and execute custom tools within workflows

#### **Performance Monitoring**
- **Callback Integration**: SharedServicesCallback tracks all LangChain operations
- **Workflow Metrics**: Execution time, step count, and agent performance tracking
- **Tool Usage Analytics**: Monitor which tools are used and their effectiveness
- **Error Tracking**: Comprehensive error logging and retry mechanisms

### 🔄 Workflow Examples

#### Sequential Analysis Workflow
```python
# Execute agents in sequence: Data Quality → Compliance → Analytics
workflow = [
    {"agent_id": "data_quality_agent", "type": "data_quality"},
    {"agent_id": "compliance_agent", "type": "compliance"},
    {"agent_id": "analytics_agent", "type": "analytics"}
]

result = await langchain_integration.execute_workflow(
    "comprehensive_analysis", 
    workflow, 
    "sequential"
)
```

#### Parallel Assessment Workflow
```python
# Run compliance and quality checks simultaneously
workflow = [
    {"agent_id": "data_quality_agent", "type": "data_quality"},
    {"agent_id": "compliance_agent", "type": "compliance"}
]

result = await langchain_integration.execute_workflow(
    "parallel_assessment",
    workflow,
    "parallel"
)
```

#### Conditional Processing Workflow
```python
# Route to different agents based on input analysis
workflow_config = {
    "name": "smart_routing",
    "type": "conditional", 
    "conditions": [
        {"condition": "compliance_required", "agent": "compliance_agent"},
        {"condition": "quality_issues", "agent": "data_quality_agent"},
        {"condition": "analytics_needed", "agent": "analytics_agent"}
    ]
}
```

### 🛠️ Integration Components

#### **LangGraphState** 
Enhanced state management tracking:
- Message and user context
- Workflow execution progress  
- Agent outputs and tool results
- Performance metrics and timing
- Error handling and retry logic

#### **SharedServicesToolkit**
Provides LangChain-compatible tools for:
- Memory management and retrieval
- RAG document processing and search
- Knowledge base entity management
- Prompt template operations
- Context and session management

#### **LangGraphWorkflowBuilder**
Creates and manages complex workflows:
- Workflow type detection and routing
- Agent dependency management
- State transitions and data flow
- Error handling and recovery
- Performance optimization

For detailed integration documentation, see [`docs/LANGCHAIN_INTEGRATION.md`](docs/LANGCHAIN_INTEGRATION.md).

## 🏛️ Core Architecture Components

### LLM Gateway
Centralized gateway supporting multiple LLM providers:
- **AWS Bedrock** (Primary) - Claude 3, Titan, Llama2
- **OpenAI** - GPT-4, GPT-3.5
- **Anthropic** - Claude models
- Load balancing and failover
- Unified interface for all agents

### Orchestration Layer (LangGraph)
- Agent workflow management
- Inter-agent communication
- State management
- Parallel processing capabilities

### RAG System
- Vector database integration (ChromaDB, FAISS)
- Semantic search capabilities
- Context-aware responses
- Knowledge base management

### Configuration Management
- Environment-based configuration
- Multi-provider LLM settings
- Agent-specific parameters
- Security and compliance settings

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- AWS Account (for Bedrock)
- Optional: OpenAI/Anthropic API keys

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd edgp-ai-model
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Run the application**:

**Option 1: Standard Python execution**
```bash
python main.py
```

**Option 2: Using the startup script**
```bash
python start.py
```

**Option 3: Runtime environment variables (similar to EDGP rules engine)**
```bash
APP_ENV="DEV" AWS_REGION="ap-southeast-1" AWS_ACCESS_KEY_ID="AKIAXXXXXXXX" AWS_SECRET_ACCESS_KEY="your_secret_key" WORKFLOW_RULES_REQUEST_SQS_URL="https://sqs.ap-southeast-1.amazonaws.com/891377185630/test_sqs_queue_input" WORKFLOW_RULES_RESPONSE_SQS_URL="https://sqs.ap-southeast-1.amazonaws.com/891377185630/test_sqs_queue_output" python -m app.main
```

**Option 4: Using command line arguments**
```bash
python start.py --env DEV --aws-region ap-southeast-1 --aws-access-key-id "AKIAXXXXXXXX" --aws-secret-access-key "your_secret_key" --workflow-request-sqs-url "https://sqs.ap-southeast-1.amazonaws.com/891377185630/test_sqs_queue_input"
```

5. **Access the application**:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Message Queue Status**: http://localhost:8000/message-queue/status

### Message Queue Service Usage

**Send a workflow request message**:
```bash
curl -X POST "http://localhost:8000/message-queue/trigger-data-quality-assessment" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "dataset_123",
    "assessment_type": "full"
  }'
```

**Check queue status**:
```bash
curl "http://localhost:8000/message-queue/status"
```

**Send custom message**:
```bash
curl -X POST "http://localhost:8000/message-queue/send-message" \
  -H "Content-Type: application/json" \
  -d '{
    "queue_name": "workflow_request",
    "message_type": "workflow_request", 
    "source": "api_client",
    "payload": {
      "workflow_type": "compliance_check",
      "data": {"dataset_id": "dataset_456"}
    }
  }'
```

6. **Run LangChain Integration Demo**:
```bash
# Run the comprehensive LangChain/LangGraph integration demonstration
python examples/langchain_integration_demo.py
```

This will demonstrate:
- Initialization of all shared services
- Creation of LangChain-integrated agents
- Execution of various workflow types (sequential, parallel, conditional)
- Shared services integration (memory, RAG, knowledge, etc.)
- Performance monitoring and metrics collection

## 📁 Project Structure

```
edgp-ai-model/
├── agents/                          # All agent implementations
│   ├── __init__.py
│   ├── policy_suggestion/           # Policy Suggestion Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── data_privacy_compliance/     # Privacy & Compliance Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── data_quality/               # Data Quality Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── data_remediation/           # Data Remediation Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   └── analytics/                  # Analytics Agent
│       ├── __init__.py
│       └── agent.py
├── core/                           # Core infrastructure & shared services
│   ├── __init__.py
│   ├── config.py                   # Configuration management
│   ├── shared.py                   # Shared services orchestration
│   ├── agents/                     # Enhanced agent architecture
│   │   ├── base.py                 # Traditional base agent
│   │   ├── enhanced.py             # Enhanced agent features
│   │   ├── mcp_enabled.py          # MCP integration
│   │   └── enhanced_base.py        # LangChain-integrated base class
│   ├── communication/              # Agent communication system
│   │   ├── external.py             # External system communication
│   │   └── mcp.py                  # Model Context Protocol
│   ├── infrastructure/             # Enterprise infrastructure
│   │   ├── auth.py                 # Authentication & authorization
│   │   ├── monitoring.py           # System monitoring
│   │   ├── error_handling.py       # Error handling & recovery
│   │   └── config.py               # Infrastructure configuration
│   ├── integrations/               # LangChain & external integrations
│   │   ├── config.py               # Integration configuration
│   │   ├── endpoints.py            # API endpoints
│   │   ├── patterns.py             # Integration patterns
│   │   ├── shared.py               # Shared integration utilities
│   │   └── langchain_integration.py # Complete LangChain/LangGraph integration
│   ├── services/                   # Shared services
│   │   ├── llm_bridge.py           # LLM abstraction layer
│   │   ├── llm_gateway.py          # Multi-provider LLM gateway
│   │   ├── rag_system.py           # Advanced RAG system
│   │   └── message_queue.py        # AWS SQS message queue service
│   ├── api/                        # API endpoints
│   │   └── message_queue_endpoints.py # Message queue REST API
│   └── types/                      # Type definitions & validation
│       ├── base.py                 # Base types
│       ├── data.py                 # Data types
│       ├── responses.py            # Response types
│       ├── validation.py           # Validation utilities
│       ├── agent_types.py          # Agent-specific types
│       └── agents/                 # Agent type definitions
├── docs/                           # Comprehensive documentation
│   ├── API_DOCUMENTATION.md        # API reference
│   ├── ARCHITECTURE.md             # System architecture
│   ├── DEPLOYMENT.md               # Deployment guide
│   ├── EXTERNAL_INTEGRATION.md     # External integration guide
│   ├── INTEGRATION_QUICK_REFERENCE.md # Quick integration reference
│   ├── REPOSITORY_STRUCTURE.md     # Repository structure guide
│   ├── TYPE_SYSTEM.md              # Type system documentation
│   └── LANGCHAIN_INTEGRATION.md    # LangChain/LangGraph integration guide
├── examples/                       # Example implementations & demos
│   ├── langchain_integration_demo.py # Comprehensive LangChain demo
│   └── production_config.py        # Production configuration examples
├── graphs/                         # LangGraph workflow definitions
│   └── remediation/                # Remediation workflow graphs
│       ├── __init__.py
│       ├── base_workflow.py        # Base remediation workflow
│       ├── automated_workflow.py   # Automated remediation workflows
│       └── orchestrator.py         # Workflow orchestrator
├── app/                            # Module entry point
│   ├── __init__.py
│   └── main.py                     # Module main entry point
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Test configuration
│   ├── unit/                       # Unit tests
│   │   └── test_agent_types.py
│   └── integration/                # Integration tests
│       └── test_endpoints.py
├── requirements.txt                # Python dependencies (includes LangChain/LangGraph)
├── docker-compose.yml              # Docker composition
├── Dockerfile                      # Container definition
├── main.py                         # Application entry point
├── start.py                        # Enhanced startup script with runtime env support
├── .env.example                    # Environment template
└── README.md                       # This file
```

## 🔧 Development Status

### ✅ **Completed Features**
- **LangChain/LangGraph Integration**: Complete workflow orchestration and agent architecture
- **Message Queue Service**: AWS SQS integration with multiple queue support
- **Shared Services**: All 7 core services implemented and integrated
- **Agent Architecture**: 5 specialized agents with MCP communication
- **Runtime Configuration**: Support for environment variable-based deployment
- **API Endpoints**: RESTful API with comprehensive message queue management
- **Documentation**: Extensive documentation and examples

### 🔄 **Message Queue Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Message Queue Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   AWS SQS       │    │  Message Queue  │    │   API Gateway   │ │
│  │   Queues        │───▶│    Service      │◀───│   Endpoints     │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent Layer                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ Data Quality    │    │   Compliance    │    │   Remediation   │ │
│  │    Agent        │    │     Agent       │    │     Agent       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Message Queue Features:**
- **Multiple Queue Support**: Handle request, response, and custom queues
- **Message Processing**: Automated consumption and processing of SQS messages
- **Runtime AWS Authentication**: Support for environment-based AWS credentials
- **Workflow Triggers**: API endpoints for triggering workflows via messages
- **Queue Monitoring**: Real-time queue statistics and health monitoring
- **Message Types**: Support for 7 different message types (workflow, compliance, etc.)
- **Error Handling**: Dead letter queues and retry mechanisms

### 🚨 **Troubleshooting**

**Common Issues and Solutions:**

1. **AWS Credentials Error**
   ```bash
   # Error: NoCredentialsError: Unable to locate credentials
   # Solution: Set AWS credentials
   export AWS_ACCESS_KEY_ID="your_key"
   export AWS_SECRET_ACCESS_KEY="your_secret"
   export AWS_REGION="your_region"
   ```

2. **SQS Queue Access Denied**
   ```bash
   # Error: AccessDenied when accessing SQS queue
   # Solution: Verify queue URL and IAM permissions
   # Ensure the AWS credentials have sqs:SendMessage, sqs:ReceiveMessage permissions
   ```

3. **Module Import Error**
   ```bash
   # Error: ModuleNotFoundError when running python -m app.main
   # Solution: Ensure you're in the project root directory
   cd /path/to/edgp-ai-model
   python -m app.main
   ```

4. **Port Already in Use**
   ```bash
   # Error: Address already in use
   # Solution: Use a different port
   python start.py --port 8001
   ```

5. **Message Queue Not Processing**
   ```bash
   # Check queue status
   curl "http://localhost:8000/message-queue/status"
   # Verify AWS credentials and queue URLs in logs
   ```

**Environment Variable Examples:**
```bash
# Minimal setup for testing
APP_ENV="DEV" AWS_REGION="us-east-1" python -m app.main

# Full production setup
APP_ENV="PROD" \
AWS_REGION="ap-southeast-1" \
AWS_ACCESS_KEY_ID="AKIAXXXXXXXX" \
AWS_SECRET_ACCESS_KEY="your_secret_key" \
WORKFLOW_RULES_REQUEST_SQS_URL="https://sqs.ap-southeast-1.amazonaws.com/123456789/requests" \
WORKFLOW_RULES_RESPONSE_SQS_URL="https://sqs.ap-southeast-1.amazonaws.com/123456789/responses" \
python -m app.main
```

This is currently a **repository skeleton** with placeholder implementations. Each agent contains:
- ✅ Agent class structure
- ✅ Method signatures and documentation
- ✅ Placeholder implementations
- ✅ AWS Bedrock integration setup
- 🚧 Full LLM-powered implementations (coming next)

### Next Steps for Implementation:

1. **Core Infrastructure**:
   - Complete LLM Gateway implementation
   - Implement orchestration workflows
   - Set up RAG system with vector databases

2. **Agent Development**:
   - Implement actual LLM-powered agent logic
   - Add inter-agent communication
   - Implement MCP integration

3. **API Development**:
   - Create FastAPI endpoints
   - Add authentication and authorization
   - Implement monitoring and logging

4. **Testing & Deployment**:
   - Unit and integration tests
   - Docker containerization
   - CI/CD pipelines

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support, please open an issue in the repository.

---

**Note**: This is a development version. The agents currently contain placeholder implementations and will be enhanced with full LLM-powered capabilities in subsequent releases.
