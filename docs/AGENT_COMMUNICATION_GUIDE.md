# Agent Communication Patterns Guide

## 🔄 Overview

This guide provides comprehensive documentation on how agents communicate with each other in the agentic AI framework using **Model Context Protocol (MCP)**, **LangGraph workflows**, and **message-driven architectures**.

**NEW**: The framework now supports **standardized communication interface (v2)** following agentic AI best practices with comprehensive type safety, traceability, and performance monitoring.

## 🚀 Communication Interfaces

### Legacy Interface (v1)
- Basic MCP message passing
- Simple parameter-based communication
- Limited type safety

### Standardized Interface (v2) - **Recommended**
- Agentic AI best practices implementation
- Comprehensive Pydantic type system
- Full request traceability with correlation IDs
- Built-in performance metrics
- Structured error handling
- Generic type support for extensibility

## 🏗️ Communication Architecture

### Core Communication Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent Communication Layer                         │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │      MCP        │  │   LangGraph     │  │   Message Queue     │ │
│  │   Message Bus   │  │   Workflows     │  │                     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│           │                     │                       │           │
│           ▼                     ▼                       ▼           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │   Direct P2P    │  │  Workflow State │  │   Pub/Sub Events    │ │
│  │  Communication  │  │   Management    │  │                     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 Standardized Communication Interface (v2)

### Overview

The standardized interface implements agentic AI best practices with a comprehensive type system that ensures type safety, traceability, and performance monitoring across all agent communications.

### Core Components

#### 1. Generic Type System

```python
from typing import TypeVar, Generic
from core.types.communication import StandardAgentInput, StandardAgentOutput

T = TypeVar('T')  # Domain-specific input type
U = TypeVar('U')  # Domain-specific output type

class StandardAgentInput(Generic[T]):
    message_id: str
    source_agent_id: str
    target_agent_id: str
    capability_name: str
    data: T  # Domain-specific input (e.g., DataQualityInput)
    context: AgentContext
    execution_mode: ExecutionMode
    priority: MessagePriority

class StandardAgentOutput(Generic[U]):
    message_id: str
    source_agent_id: str
    target_agent_id: str
    capability_name: str
    status: TaskStatus
    success: bool
    data: Optional[U]  # Domain-specific output (e.g., DataQualityOutput)
    context: AgentContext
    timestamp: datetime
    performance_metrics: Dict[str, Any]
```

#### 2. Domain-Specific Types

Each agent domain has specialized input/output types:

```python
# Data Quality Agent Types
class DataQualityInput(BaseModel):
    data_payload: DataPayload
    operation_parameters: OperationParameters
    quality_dimensions: List[QualityDimension]
    include_anomaly_detection: bool = False
    include_profiling: bool = False

class DataQualityOutput(BaseModel):
    processing_result: ProcessingResult
    overall_quality_score: float
    dimension_scores: Dict[QualityDimension, float]
    quality_issues: List[Dict[str, Any]]
    anomalies_detected: List[Dict[str, Any]]
    data_profile: Dict[str, Any]
    improvement_recommendations: List[str]
    priority_actions: List[Dict[str, str]]
```

#### 3. Agent Context and Traceability

```python
class AgentContext(BaseModel):
    session_id: str
    correlation_id: str
    trace_id: str
    user_id: Optional[str]
    request_metadata: Dict[str, Any]
    
    # Distributed tracing support
    parent_span_id: Optional[str] = None
    trace_state: Dict[str, str] = Field(default_factory=dict)
```

#### 4. Processing Results and Metrics

```python
class ProcessingResult(BaseModel):
    operation_type: OperationType
    processing_stage: ProcessingStage
    primary_output: Dict[str, Any]
    secondary_outputs: Dict[str, Any] = Field(default_factory=dict)
    key_findings: List[str] = Field(default_factory=list)
    issues_found: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
```

### Implementation Pattern

#### 1. BaseAgent Implementation

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from core.types.communication import StandardAgentInput, StandardAgentOutput

T = TypeVar('T')
U = TypeVar('U')

class BaseAgent(Generic[T], ABC):
    """Base agent class with standardized communication interface."""
    
    @abstractmethod
    async def process_standardized_input(
        self, 
        agent_input: StandardAgentInput[T]
    ) -> StandardAgentOutput[U]:
        """Process standardized input and return typed output."""
        pass
    
    @abstractmethod
    async def handle_standardized_message(
        self, 
        message: StandardizedAgentMessage
    ) -> Optional[StandardizedAgentMessage]:
        """Handle incoming standardized messages."""
        pass
```

#### 2. Agent Implementation Example

```python
class DataQualityAgent(BaseAgent[DataQualityInput]):
    """Data quality agent with standardized interface."""
    
    async def process_standardized_input(
        self, 
        agent_input: StandardAgentInput[DataQualityInput]
    ) -> StandardAgentOutput[DataQualityOutput]:
        """Process data quality assessment request."""
        
        # Extract domain-specific input
        quality_input = agent_input.data
        
        # Perform processing based on capability
        if agent_input.capability_name == "data_quality_assessment":
            result = await self._perform_quality_assessment(quality_input)
        elif agent_input.capability_name == "anomaly_detection":
            result = await self._perform_anomaly_detection(quality_input)
        else:
            raise ValueError(f"Unsupported capability: {agent_input.capability_name}")
        
        # Create standardized output
        return create_standard_output(
            request_id=agent_input.message_id,
            source_agent_id=self.agent_id,
            capability_used=agent_input.capability_name,
            status=TaskStatus.COMPLETED,
            success=True,
            result=result,
            metadata=AgentMetadata(
                agent_id=self.agent_id,
                capability_used=agent_input.capability_name,
                operation_type=OperationType.ASSESSMENT
            )
        )
```

### Communication Flow

```
1. Request Creation
   ├── Create domain-specific input (e.g., DataQualityInput)
   ├── Wrap in StandardAgentInput[T]
   └── Add context and tracing information

2. Agent Processing
   ├── Validate input types
   ├── Extract domain-specific data
   ├── Perform capability-specific processing
   ├── Track performance metrics
   └── Handle errors gracefully

3. Response Generation
   ├── Create domain-specific output (e.g., DataQualityOutput)
   ├── Wrap in StandardAgentOutput[U]
   ├── Add performance metrics
   └── Include tracing information

4. Result Delivery
   ├── Type-safe response handling
   ├── Distributed tracing propagation
   └── Performance monitoring
```

### Usage Examples

#### Creating a Standardized Request

```python
from core.types.communication import (
    create_standard_input, DataQualityInput, 
    DataPayload, OperationParameters, QualityDimension
)

# Create data payload
data_payload = DataPayload(
    data_type="tabular",
    data_format="json",
    content={"customers": [...]},
    source_system="crm_database"
)

# Create operation parameters
operation_params = OperationParameters(
    quality_threshold=0.8,
    include_recommendations=True
)

# Create domain-specific input
quality_input = DataQualityInput(
    data_payload=data_payload,
    operation_parameters=operation_params,
    quality_dimensions=[
        QualityDimension.COMPLETENESS,
        QualityDimension.ACCURACY,
        QualityDimension.CONSISTENCY
    ],
    include_anomaly_detection=True,
    include_profiling=True
)

# Create standardized input
agent_input = create_standard_input(
    source_agent_id="client_app",
    target_agent_id="data_quality_agent",
    capability_name="data_quality_assessment",
    data=quality_input
)

# Process request
result = await agent.process_standardized_input(agent_input)
```

#### Processing the Response

```python
# Response is fully typed
if result.success:
    quality_output = result.data  # Type: DataQualityOutput
    
    print(f"Overall Quality Score: {quality_output.overall_quality_score}")
    print(f"Issues Found: {len(quality_output.quality_issues)}")
    
    # Access performance metrics
    metrics = result.performance_metrics
    print(f"Processing Time: {metrics.get('processing_time_ms')}ms")
    
    # Access tracing information
    trace_id = result.context.trace_id
    print(f"Trace ID: {trace_id}")
else:
    print(f"Processing failed: {result.error_message}")
```

### Benefits of Standardized Interface

1. **Type Safety**: Full Pydantic validation and type hints
2. **Traceability**: Built-in correlation and trace IDs
3. **Performance Monitoring**: Automatic metrics collection
4. **Error Handling**: Structured error responses with context
5. **Extensibility**: Generic type system for new domains
6. **Testing**: Easier unit and integration testing
7. **Documentation**: Self-documenting through type annotations

## 📨 MCP Message Protocol (Legacy)

### Message Structure

Every communication uses the standardized MCP message format:

```python
@dataclass
class MCPMessage:
    """Model Context Protocol message for agent communication."""
    
    # Message identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None  # For request-response pairing
    
    # Routing information
    sender_id: str
    recipient_id: str  # "*" for broadcast
    route_path: List[str] = field(default_factory=list)
    
    # Message classification
    message_type: str  # "request", "response", "broadcast", "notification"
    capability: AgentCapability
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Content
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Response handling
    expects_response: bool = False
    response_timeout: Optional[int] = 30  # seconds
```

### Message Types

| Type | Purpose | Example Use Case |
|------|---------|------------------|
| **request** | Request specific action from another agent | "Please assess data quality for dataset X" |
| **response** | Reply to a request message | "Quality assessment complete: 85% score" |
| **broadcast** | Announce information to multiple agents | "New compliance rule updated" |
| **notification** | Inform about state changes | "Task completed successfully" |
| **event** | Trigger reactive behaviors | "Data anomaly detected" |

## 🔄 Communication Patterns

### 1. Direct Request-Response Pattern

```python
class DirectCommunicationExample:
    """Direct agent-to-agent communication."""
    
    async def request_data_quality_assessment(self, dataset_id: str) -> Dict:
        """Request quality assessment from data quality agent."""
        
        # Create request message
        request = MCPMessage(
            sender_id=self.agent_id,
            recipient_id="data_quality_agent",
            message_type="request",
            capability=AgentCapability.DATA_QUALITY_ASSESSMENT,
            content={
                "dataset_id": dataset_id,
                "assessment_level": "comprehensive",
                "include_recommendations": True
            },
            expects_response=True,
            response_timeout=60
        )
        
        # Send request and wait for response
        response = await self.message_bus.send_message_with_response(request)
        
        if response and response.message_type == "response":
            return response.content
        else:
            raise TimeoutError("No response received from data quality agent")
    
    async def handle_quality_assessment_request(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming quality assessment request."""
        
        try:
            # Extract parameters
            dataset_id = message.content.get("dataset_id")
            assessment_level = message.content.get("assessment_level", "basic")
            
            # Perform assessment
            assessment_result = await self.assess_data_quality(
                dataset_id=dataset_id,
                level=assessment_level
            )
            
            # Create response
            response = MCPMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                message_type="response",
                capability=message.capability,
                content={
                    "status": "success",
                    "assessment": assessment_result,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return response
            
        except Exception as e:
            # Return error response
            return MCPMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                message_type="response",
                capability=message.capability,
                content={
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
```

### 2. Broadcast Communication Pattern

```python
class BroadcastCommunicationExample:
    """Broadcast communication to multiple agents."""
    
    async def announce_policy_update(self, policy_data: Dict):
        """Broadcast policy update to all interested agents."""
        
        broadcast_message = MCPMessage(
            sender_id=self.agent_id,
            recipient_id="*",  # Broadcast to all
            message_type="broadcast",
            capability=AgentCapability.POLICY_UPDATE,
            content={
                "policy_id": policy_data["id"],
                "policy_type": policy_data["type"],
                "effective_date": policy_data["effective_date"],
                "changes": policy_data["changes"],
                "impact_areas": policy_data["impact_areas"]
            },
            metadata={
                "broadcast_scope": ["compliance", "quality", "remediation"],
                "priority": "high"
            }
        )
        
        # Send broadcast
        responses = await self.message_bus.broadcast_message(broadcast_message)
        
        # Process acknowledgments
        acknowledged_agents = []
        for response in responses:
            if response.content.get("acknowledged"):
                acknowledged_agents.append(response.sender_id)
        
        return {
            "broadcast_sent": True,
            "total_recipients": len(responses),
            "acknowledged_by": acknowledged_agents
        }
    
    async def handle_policy_update_broadcast(self, message: MCPMessage) -> MCPMessage:
        """Handle policy update broadcast."""
        
        # Check if this agent cares about this policy type
        policy_type = message.content.get("policy_type")
        impact_areas = message.content.get("impact_areas", [])
        
        if self._is_policy_relevant(policy_type, impact_areas):
            # Update internal policy cache
            await self._update_policy_cache(message.content)
            
            # Send acknowledgment
            return MCPMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                message_type="notification",
                capability=message.capability,
                content={
                    "acknowledged": True,
                    "policy_applied": True,
                    "agent_type": self.agent_type.value
                }
            )
        
        return None  # No response needed
```

### 3. Event-Driven Communication Pattern

```python
class EventDrivenCommunicationExample:
    """Event-driven reactive communication."""
    
    async def emit_data_anomaly_event(self, anomaly_data: Dict):
        """Emit data anomaly event for reactive processing."""
        
        event_message = MCPMessage(
            sender_id=self.agent_id,
            recipient_id="*",
            message_type="event",
            capability=AgentCapability.ANOMALY_DETECTED,
            content={
                "event_type": "data_anomaly",
                "severity": anomaly_data["severity"],
                "dataset_id": anomaly_data["dataset_id"],
                "anomaly_type": anomaly_data["type"],
                "confidence_score": anomaly_data["confidence"],
                "detected_at": datetime.now().isoformat(),
                "requires_immediate_action": anomaly_data["severity"] == "critical"
            },
            metadata={
                "event_source": "data_quality_agent",
                "trigger_workflow": anomaly_data["severity"] in ["high", "critical"]
            }
        )
        
        # Emit event
        await self.message_bus.emit_event(event_message)
    
    async def handle_data_anomaly_event(self, message: MCPMessage):
        """React to data anomaly events."""
        
        severity = message.content.get("severity")
        dataset_id = message.content.get("dataset_id")
        
        # React based on severity
        if severity == "critical":
            # Immediate action required
            await self._initiate_emergency_remediation(dataset_id)
            
            # Notify compliance team
            await self._notify_compliance_team(message.content)
            
        elif severity == "high":
            # Schedule remediation
            await self._schedule_remediation_task(dataset_id, priority="high")
            
        else:
            # Log for monitoring
            await self._log_anomaly_for_tracking(message.content)
```

## 🔗 LangGraph Workflow Communication

### Workflow State Management

Agents communicate through shared workflow state in LangGraph:

```python
@dataclass
class AgentWorkflowState:
    """Shared state across agents in a workflow."""
    
    # Core identifiers
    workflow_id: str
    execution_id: str
    current_step: str
    
    # Data pipeline
    input_data: Dict[str, Any]
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_output: Optional[Dict[str, Any]] = None
    
    # Agent execution tracking
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Communication context
    messages: List[MCPMessage] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    started_at: datetime = field(default_factory=datetime.now)
    current_agent: Optional[str] = None
    next_agent: Optional[str] = None
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    def add_agent_result(self, agent_id: str, result: Dict[str, Any]):
        """Add agent execution result to state."""
        self.agent_outputs[agent_id] = result
        self.completed_steps.append(agent_id)
        
    def add_error(self, agent_id: str, error: str):
        """Add error to state."""
        self.errors.append(f"{agent_id}: {error}")
        self.failed_steps.append(agent_id)
        
    def get_previous_results(self, agent_type: str) -> Optional[Dict]:
        """Get results from previous agent of specific type."""
        return self.agent_outputs.get(agent_type)
```

### Multi-Agent Workflow Example

```python
def create_comprehensive_data_assessment_workflow() -> StateGraph:
    """Create multi-agent workflow for comprehensive data assessment."""
    
    def quality_assessment_node(state: AgentWorkflowState) -> AgentWorkflowState:
        """Data quality assessment node."""
        
        try:
            # Get quality agent
            quality_agent = get_agent("data_quality_agent")
            
            # Execute quality assessment
            quality_result = await quality_agent.execute_capability(
                AgentCapability.DATA_QUALITY_ASSESSMENT,
                {
                    "dataset": state.input_data,
                    "context": state.shared_context
                }
            )
            
            # Update state
            state.add_agent_result("quality_assessment", quality_result)
            state.shared_context["quality_score"] = quality_result.get("overall_score")
            state.shared_context["quality_issues"] = quality_result.get("issues", [])
            
            # Add inter-agent message
            quality_message = MCPMessage(
                sender_id="data_quality_agent",
                recipient_id="compliance_agent",
                message_type="notification",
                capability=AgentCapability.QUALITY_RESULTS_AVAILABLE,
                content=quality_result
            )
            state.messages.append(quality_message)
            
            return state
            
        except Exception as e:
            state.add_error("quality_assessment", str(e))
            return state
    
    def compliance_check_node(state: AgentWorkflowState) -> AgentWorkflowState:
        """Compliance check node."""
        
        try:
            # Get compliance agent
            compliance_agent = get_agent("compliance_agent")
            
            # Get quality results from previous step
            quality_results = state.get_previous_results("quality_assessment")
            
            # Execute compliance check with quality context
            compliance_result = await compliance_agent.execute_capability(
                AgentCapability.COMPLIANCE_CHECK,
                {
                    "dataset": state.input_data,
                    "quality_context": quality_results,
                    "context": state.shared_context
                }
            )
            
            # Update state
            state.add_agent_result("compliance_check", compliance_result)
            state.shared_context["compliance_violations"] = compliance_result.get("violations", [])
            state.shared_context["risk_level"] = compliance_result.get("risk_level")
            
            return state
            
        except Exception as e:
            state.add_error("compliance_check", str(e))
            return state
    
    def remediation_planning_node(state: AgentWorkflowState) -> AgentWorkflowState:
        """Remediation planning node."""
        
        try:
            # Get remediation agent
            remediation_agent = get_agent("remediation_agent")
            
            # Combine context from previous agents
            combined_context = {
                "quality_issues": state.shared_context.get("quality_issues", []),
                "compliance_violations": state.shared_context.get("compliance_violations", []),
                "risk_level": state.shared_context.get("risk_level", "low")
            }
            
            # Create remediation plan
            remediation_result = await remediation_agent.execute_capability(
                AgentCapability.REMEDIATION_PLANNING,
                {
                    "dataset": state.input_data,
                    "issues_context": combined_context,
                    "context": state.shared_context
                }
            )
            
            # Update state
            state.add_agent_result("remediation_planning", remediation_result)
            state.final_output = {
                "quality_assessment": state.get_previous_results("quality_assessment"),
                "compliance_check": state.get_previous_results("compliance_check"),
                "remediation_plan": remediation_result
            }
            
            return state
            
        except Exception as e:
            state.add_error("remediation_planning", str(e))
            return state
    
    def route_next_step(state: AgentWorkflowState) -> str:
        """Route to next step based on results."""
        
        if "quality_assessment" not in state.completed_steps:
            return "quality_assessment"
        elif "compliance_check" not in state.completed_steps:
            return "compliance_check"
        elif "remediation_planning" not in state.completed_steps:
            # Only proceed to remediation if issues found
            quality_issues = state.shared_context.get("quality_issues", [])
            compliance_violations = state.shared_context.get("compliance_violations", [])
            
            if quality_issues or compliance_violations:
                return "remediation_planning"
            else:
                return "complete"
        else:
            return "complete"
    
    # Build workflow graph
    workflow = StateGraph(AgentWorkflowState)
    
    # Add nodes
    workflow.add_node("quality_assessment", quality_assessment_node)
    workflow.add_node("compliance_check", compliance_check_node)
    workflow.add_node("remediation_planning", remediation_planning_node)
    
    # Define routing
    workflow.add_edge(START, "quality_assessment")
    workflow.add_edge("quality_assessment", "compliance_check")
    workflow.add_conditional_edges(
        "compliance_check",
        route_next_step,
        {
            "remediation_planning": "remediation_planning",
            "complete": END
        }
    )
    workflow.add_edge("remediation_planning", END)
    
    return workflow.compile()
```

## 🔀 Advanced Communication Patterns

### 1. Conversation Threading

Maintain conversation context across multiple exchanges:

```python
class ConversationManager:
    """Manage threaded conversations between agents."""
    
    def __init__(self):
        self.conversations: Dict[str, List[MCPMessage]] = {}
    
    async def start_conversation(self, participants: List[str], topic: str) -> str:
        """Start a new conversation thread."""
        conversation_id = str(uuid.uuid4())
        
        # Initialize conversation
        self.conversations[conversation_id] = []
        
        # Send conversation start message
        start_message = MCPMessage(
            sender_id="conversation_manager",
            recipient_id="*",
            message_type="notification",
            capability=AgentCapability.CONVERSATION_START,
            content={
                "conversation_id": conversation_id,
                "participants": participants,
                "topic": topic,
                "started_at": datetime.now().isoformat()
            },
            metadata={"conversation_id": conversation_id}
        )
        
        await self.message_bus.broadcast_message(start_message)
        return conversation_id
    
    async def add_to_conversation(self, conversation_id: str, message: MCPMessage):
        """Add message to conversation thread."""
        if conversation_id in self.conversations:
            message.metadata["conversation_id"] = conversation_id
            self.conversations[conversation_id].append(message)
    
    def get_conversation_history(self, conversation_id: str) -> List[MCPMessage]:
        """Get conversation history."""
        return self.conversations.get(conversation_id, [])
```

### 2. Agent Coordination Patterns

Coordinate multiple agents for complex tasks:

```python
class AgentCoordinator:
    """Coordinate multiple agents for complex tasks."""
    
    async def coordinate_parallel_assessment(self, dataset_info: Dict) -> Dict:
        """Coordinate parallel assessment by multiple agents."""
        
        # Create coordination session
        session_id = str(uuid.uuid4())
        
        # Define parallel tasks
        parallel_tasks = [
            ("data_quality_agent", AgentCapability.DATA_QUALITY_ASSESSMENT),
            ("compliance_agent", AgentCapability.COMPLIANCE_CHECK),
            ("analytics_agent", AgentCapability.STATISTICAL_ANALYSIS)
        ]
        
        # Send parallel requests
        futures = []
        for agent_id, capability in parallel_tasks:
            task_message = MCPMessage(
                sender_id=self.agent_id,
                recipient_id=agent_id,
                message_type="request",
                capability=capability,
                content={
                    "dataset": dataset_info,
                    "session_id": session_id,
                    "coordination_mode": "parallel"
                },
                expects_response=True,
                response_timeout=120
            )
            
            future = self.message_bus.send_message_with_response(task_message)
            futures.append((agent_id, future))
        
        # Collect results
        results = {}
        for agent_id, future in futures:
            try:
                response = await future
                results[agent_id] = response.content
            except Exception as e:
                results[agent_id] = {"error": str(e), "status": "failed"}
        
        # Synthesize results
        synthesis_result = await self._synthesize_parallel_results(results)
        
        return {
            "session_id": session_id,
            "individual_results": results,
            "synthesized_assessment": synthesis_result
        }
    
    async def coordinate_sequential_pipeline(self, initial_data: Dict) -> Dict:
        """Coordinate sequential data processing pipeline."""
        
        pipeline_steps = [
            ("data_quality_agent", AgentCapability.DATA_PROFILING),
            ("compliance_agent", AgentCapability.PRIVACY_SCAN),
            ("remediation_agent", AgentCapability.DATA_CLEANING),
            ("analytics_agent", AgentCapability.REPORT_GENERATION)
        ]
        
        current_data = initial_data
        pipeline_results = {}
        
        for step_index, (agent_id, capability) in enumerate(pipeline_steps):
            # Send request to current agent
            request = MCPMessage(
                sender_id=self.agent_id,
                recipient_id=agent_id,
                message_type="request",
                capability=capability,
                content={
                    "input_data": current_data,
                    "pipeline_step": step_index + 1,
                    "previous_results": pipeline_results
                },
                expects_response=True
            )
            
            # Get response
            response = await self.message_bus.send_message_with_response(request)
            
            if response and response.content.get("status") == "success":
                step_result = response.content
                pipeline_results[agent_id] = step_result
                
                # Use output as input for next step
                current_data = step_result.get("output_data", current_data)
            else:
                # Handle pipeline failure
                pipeline_results[agent_id] = {"status": "failed", "error": "Agent did not respond"}
                break
        
        return {
            "pipeline_complete": len(pipeline_results) == len(pipeline_steps),
            "step_results": pipeline_results,
            "final_data": current_data
        }
```

### 3. Fault-Tolerant Communication

Handle communication failures gracefully:

```python
class FaultTolerantCommunication:
    """Fault-tolerant communication patterns."""
    
    async def send_with_circuit_breaker(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Send message with circuit breaker pattern."""
        
        agent_id = message.recipient_id
        circuit_state = await self._get_circuit_state(agent_id)
        
        if circuit_state == "open":
            # Circuit is open, don't attempt
            return await self._handle_circuit_open(message)
        
        try:
            response = await self.message_bus.send_message_with_response(message)
            await self._record_success(agent_id)
            return response
            
        except Exception as e:
            await self._record_failure(agent_id)
            
            # Check if circuit should open
            if await self._should_open_circuit(agent_id):
                await self._open_circuit(agent_id)
            
            raise e
    
    async def send_with_fallback(self, primary_message: MCPMessage, 
                                fallback_agents: List[str]) -> MCPMessage:
        """Send message with fallback agents."""
        
        # Try primary agent first
        try:
            return await self.message_bus.send_message_with_response(primary_message)
        except Exception as primary_error:
            logging.warning(f"Primary agent {primary_message.recipient_id} failed: {primary_error}")
            
            # Try fallback agents
            for fallback_agent_id in fallback_agents:
                try:
                    fallback_message = MCPMessage(
                        sender_id=primary_message.sender_id,
                        recipient_id=fallback_agent_id,
                        message_type=primary_message.message_type,
                        capability=primary_message.capability,
                        content=primary_message.content,
                        metadata={**primary_message.metadata, "fallback_for": primary_message.recipient_id}
                    )
                    
                    return await self.message_bus.send_message_with_response(fallback_message)
                    
                except Exception as fallback_error:
                    logging.warning(f"Fallback agent {fallback_agent_id} failed: {fallback_error}")
                    continue
            
            # All agents failed
            raise Exception(f"All agents failed. Primary: {primary_error}")
```

## 📊 Communication Monitoring

Monitor and analyze agent communication patterns:

```python
class CommunicationMonitor:
    """Monitor agent communication patterns."""
    
    def __init__(self):
        self.message_metrics = {}
        self.communication_graph = {}
    
    async def record_message(self, message: MCPMessage):
        """Record message for monitoring."""
        
        # Update metrics
        metric_key = f"{message.sender_id}->{message.recipient_id}"
        if metric_key not in self.message_metrics:
            self.message_metrics[metric_key] = {
                "count": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_response_time": 0
            }
        
        self.message_metrics[metric_key]["count"] += 1
        
        # Update communication graph
        if message.sender_id not in self.communication_graph:
            self.communication_graph[message.sender_id] = set()
        self.communication_graph[message.sender_id].add(message.recipient_id)
    
    def get_communication_stats(self) -> Dict:
        """Get communication statistics."""
        return {
            "total_messages": sum(m["count"] for m in self.message_metrics.values()),
            "agent_pairs": len(self.message_metrics),
            "communication_graph": {
                agent: list(targets) 
                for agent, targets in self.communication_graph.items()
            },
            "top_communicators": self._get_top_communicators(),
            "communication_patterns": self._analyze_patterns()
        }
    
    def _get_top_communicators(self) -> List[Dict]:
        """Get agents with highest communication volume."""
        agent_counts = {}
        
        for metric_key, stats in self.message_metrics.items():
            sender, recipient = metric_key.split("->")
            agent_counts[sender] = agent_counts.get(sender, 0) + stats["count"]
            if recipient != "*":  # Don't count broadcasts as received
                agent_counts[recipient] = agent_counts.get(recipient, 0) + stats["count"]
        
        return sorted(
            [{"agent": agent, "message_count": count} for agent, count in agent_counts.items()],
            key=lambda x: x["message_count"],
            reverse=True
        )[:10]
```

This comprehensive guide provides all the patterns and tools needed for effective agent communication in the framework. The MCP protocol ensures standardized messaging, LangGraph workflows enable complex coordination, and the monitoring tools help maintain system health and performance.
