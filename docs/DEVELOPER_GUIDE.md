# Developer Guide: Leveraging Common Framework Features

## 🎯 Quick Start for Agent Development

This guide shows you how to leverage the common features of the agentic AI framework to build powerful, collaborative agents.

## 🏗️ Framework Components Overview

### Core Services Available to All Agents

| Service | Purpose | Usage |
|---------|---------|-------|
| **RAG System** | Context-aware knowledge retrieval | `self.rag_system.retrieve_context(query)` |
| **LLM Gateway** | Multi-provider AI model access | `self.llm_chain.ainvoke(prompt)` |
| **MCP Message Bus** | Agent communication | `await self.message_bus.send_message(msg)` |
| **Vector Store** | Semantic search and storage | `await self.vector_store.add_documents(docs)` |
| **Workflow Orchestrator** | LangGraph workflow execution | `await orchestrator.execute_workflow()` |
| **Configuration Manager** | Environment settings | `config.get_llm_config()` |
| **Monitoring System** | Metrics and observability | `metrics.increment_counter()` |

## 🤖 Creating Your First Agent

### Step 1: Agent Class Structure

```python
from core.agents.base import BaseAgent
from core.types.agent_types import AgentType, AgentCapability
from core.types.base import AgentMessage, AgentTask
from typing import Dict, List, Optional

class MyCustomAgent(BaseAgent):
    """Custom agent for specific business logic."""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.CUSTOM,  # Add to enum
            capabilities=[
                AgentCapability.CUSTOM_PROCESSING,  # Define your capabilities
                AgentCapability.CUSTOM_ANALYSIS
            ]
        )
        # Initialize agent-specific resources
        self._initialize_domain_knowledge()
        self._setup_custom_tools()
    
    async def execute_capability(self, capability: AgentCapability, parameters: Dict) -> Dict:
        """Execute a specific capability with parameters."""
        if capability == AgentCapability.CUSTOM_PROCESSING:
            return await self._process_custom_logic(parameters)
        elif capability == AgentCapability.CUSTOM_ANALYSIS:
            return await self._analyze_data(parameters)
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def process_message(self, message: str, context: Dict) -> Dict:
        """Process incoming text message with context."""
        # Use RAG for context-aware processing
        relevant_context = await self.rag_system.retrieve_context(message)
        
        # Prepare LLM prompt with context
        prompt = self._build_prompt(message, context, relevant_context)
        
        # Get LLM response
        response = await self.llm_chain.ainvoke({"input": prompt})
        
        return {"response": response.content, "context": context}
    
    async def process_task(self, task: AgentTask) -> Dict:
        """Process a structured task."""
        # Implementation for task processing
        return await self.execute_capability(task.capability, task.parameters)
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle MCP messages from other agents."""
        # Process the message and optionally return a response
        result = await self.process_message(message.content, message.context)
        
        # Return response message if needed
        if result.get("requires_response"):
            return AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content=result["response"],
                message_type="response"
            )
        return None
```

### Step 2: Leveraging RAG System

The RAG system provides context-aware knowledge retrieval:

```python
class RAGIntegrationExample:
    """Examples of using the RAG system effectively."""
    
    async def add_domain_knowledge(self):
        """Add domain-specific knowledge to your agent."""
        # Add documents to the knowledge base
        documents = [
            Document(
                content="Best practices for data validation include...",
                metadata={
                    "type": "best_practice",
                    "domain": "data_validation",
                    "source": "internal_guidelines"
                }
            ),
            Document(
                content="Common data quality issues include missing values...",
                metadata={
                    "type": "issue_catalog",
                    "domain": "data_quality",
                    "severity": "high"
                }
            )
        ]
        
        await self.rag_system.add_documents(documents)
    
    async def retrieve_contextual_information(self, query: str) -> List[Document]:
        """Retrieve relevant context for a query."""
        # Basic similarity search
        results = await self.rag_system.similarity_search(query, k=5)
        
        # Filtered search with metadata
        filtered_results = await self.rag_system.similarity_search_with_score(
            query=query,
            k=10,
            filter={"type": "best_practice"}
        )
        
        # Get documents above threshold
        relevant_docs = [doc for doc, score in filtered_results if score > 0.7]
        
        return relevant_docs
    
    async def enhance_response_with_context(self, user_query: str) -> str:
        """Generate context-enhanced response."""
        # 1. Retrieve relevant context
        context_docs = await self.retrieve_contextual_information(user_query)
        
        # 2. Build context-aware prompt
        context_text = "\n".join([doc.content for doc in context_docs])
        enhanced_prompt = f"""
        Context Information:
        {context_text}
        
        User Query: {user_query}
        
        Please provide a comprehensive response using the context above.
        """
        
        # 3. Generate response
        response = await self.llm_chain.ainvoke({"input": enhanced_prompt})
        return response.content
```

### Step 3: Agent Communication via MCP

Use the MCP message bus for agent-to-agent communication:

```python
class AgentCommunicationExample:
    """Examples of agent communication patterns."""
    
    async def send_direct_message(self, target_agent_id: str, capability: AgentCapability, data: Dict):
        """Send a direct message to another agent."""
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=target_agent_id,
            capability=capability,
            content=data,
            message_type="request",
            priority=MessagePriority.NORMAL
        )
        
        response = await self.message_bus.send_message(message)
        return response
    
    async def broadcast_to_interested_agents(self, capability: AgentCapability, data: Dict):
        """Broadcast message to all agents with a specific capability."""
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id="*",  # Broadcast
            capability=capability,
            content=data,
            message_type="broadcast"
        )
        
        responses = await self.message_bus.broadcast_message(message)
        return responses
    
    async def request_collaboration(self, task_data: Dict) -> Dict:
        """Request collaboration from multiple agents."""
        # Request quality check from data quality agent
        quality_response = await self.send_direct_message(
            target_agent_id="data_quality_agent",
            capability=AgentCapability.DATA_QUALITY_ASSESSMENT,
            data=task_data
        )
        
        # Request compliance check from compliance agent
        compliance_response = await self.send_direct_message(
            target_agent_id="compliance_agent", 
            capability=AgentCapability.COMPLIANCE_CHECK,
            data=task_data
        )
        
        # Combine results
        return {
            "quality_results": quality_response,
            "compliance_results": compliance_response,
            "combined_assessment": self._combine_assessments(
                quality_response, compliance_response
            )
        }
```

### Step 4: LangGraph Workflow Integration

Integrate your agent into LangGraph workflows:

```python
from langgraph import StateGraph, START, END

class WorkflowIntegrationExample:
    """Examples of LangGraph workflow integration."""
    
    def create_custom_workflow(self) -> StateGraph:
        """Create a workflow that includes your agent."""
        workflow = StateGraph()
        
        # Add nodes for different agents
        workflow.add_node("my_agent", self.my_agent_node)
        workflow.add_node("quality_agent", self.quality_agent_node)
        workflow.add_node("compliance_agent", self.compliance_agent_node)
        
        # Define workflow flow
        workflow.add_edge(START, "my_agent")
        workflow.add_conditional_edges(
            "my_agent",
            self.route_next_step,
            {
                "needs_quality_check": "quality_agent",
                "needs_compliance_check": "compliance_agent",
                "complete": END
            }
        )
        workflow.add_edge("quality_agent", "compliance_agent")
        workflow.add_edge("compliance_agent", END)
        
        return workflow.compile()
    
    async def my_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Your agent's processing node in the workflow."""
        # Execute your agent's logic
        result = await self.execute_capability(
            AgentCapability.CUSTOM_PROCESSING,
            {"data": state.input_data}
        )
        
        # Update workflow state
        state.custom_results = result
        state.completed_steps.append("custom_processing")
        
        # Determine next step
        if result.get("quality_issues"):
            state.next_step = "quality_agent"
        elif result.get("compliance_concerns"):
            state.next_step = "compliance_agent"
        else:
            state.next_step = "complete"
            
        return state
    
    def route_next_step(self, state: WorkflowState) -> str:
        """Route to the next step based on state."""
        return state.next_step
```

## 🔧 Advanced Features

### 1. Custom LLM Tools

Create domain-specific tools for your agent:

```python
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class CustomAnalysisTool(BaseTool):
    """Custom tool for domain-specific analysis."""
    name = "custom_analysis"
    description = "Performs custom analysis on data"
    
    class CustomAnalysisInput(BaseModel):
        data: dict = Field(description="Data to analyze")
        analysis_type: str = Field(description="Type of analysis to perform")
    
    args_schema: Type[BaseModel] = CustomAnalysisInput
    
    def _run(self, data: dict, analysis_type: str) -> str:
        """Execute the analysis."""
        # Your custom analysis logic here
        return f"Analysis of type {analysis_type} completed"
    
    async def _arun(self, data: dict, analysis_type: str) -> str:
        """Async version of the analysis."""
        # Your async analysis logic here
        return f"Async analysis of type {analysis_type} completed"

# Add tool to your agent
class MyAgent(BaseAgent):
    def _setup_custom_tools(self):
        """Setup agent-specific tools."""
        self.tools.append(CustomAnalysisTool())
        
        # Update LLM chain to use tools
        self.llm_chain = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt_template
        )
```

### 2. Custom Memory Integration

Implement persistent memory for your agent:

```python
from langchain.memory import ConversationSummaryBufferMemory

class AgentMemoryExample:
    """Examples of memory integration."""
    
    def __init__(self):
        # Conversation memory
        self.conversation_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        # Long-term memory via RAG
        self.long_term_memory = self.rag_system
    
    async def process_with_memory(self, message: str, context: Dict) -> Dict:
        """Process message with memory context."""
        # Get conversation history
        conversation_context = self.conversation_memory.chat_memory.messages
        
        # Get relevant long-term context
        long_term_context = await self.long_term_memory.retrieve_context(message)
        
        # Combine contexts
        full_context = {
            "conversation_history": conversation_context,
            "domain_knowledge": long_term_context,
            "current_context": context
        }
        
        # Process with full context
        response = await self.llm_chain.ainvoke({
            "input": message,
            "context": full_context
        })
        
        # Save to memory
        self.conversation_memory.chat_memory.add_user_message(message)
        self.conversation_memory.chat_memory.add_ai_message(response.content)
        
        return {"response": response.content, "context": full_context}
```

### 3. Error Handling and Resilience

Implement robust error handling:

```python
import asyncio
from typing import Optional
import logging

class ResilientAgentExample:
    """Examples of resilient agent patterns."""
    
    async def execute_with_retry(self, capability: AgentCapability, 
                                parameters: Dict, max_retries: int = 3) -> Dict:
        """Execute capability with retry logic."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await self.execute_capability(capability, parameters)
                return result
                
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        # All retries failed
        raise Exception(f"All {max_retries} attempts failed. Last error: {last_error}")
    
    async def safe_llm_call(self, prompt: str) -> Optional[str]:
        """Make a safe LLM call with error handling."""
        try:
            response = await self.llm_chain.ainvoke({"input": prompt})
            return response.content
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            # Fallback to RAG-only response
            context_docs = await self.rag_system.retrieve_context(prompt)
            if context_docs:
                return f"Based on available knowledge: {context_docs[0].content[:500]}..."
            return "I apologize, but I'm unable to process your request at this time."
    
    async def graceful_capability_execution(self, capability: AgentCapability, 
                                          parameters: Dict) -> Dict:
        """Execute capability with graceful degradation."""
        try:
            # Try full capability execution
            return await self.execute_capability(capability, parameters)
            
        except Exception as e:
            logging.error(f"Full capability execution failed: {e}")
            
            # Try degraded mode (without LLM)
            try:
                return await self._execute_degraded_mode(capability, parameters)
            except Exception as e2:
                logging.error(f"Degraded mode failed: {e2}")
                
                # Return error response
                return {
                    "status": "error",
                    "message": "Capability temporarily unavailable",
                    "capability": capability.value,
                    "error_id": str(uuid.uuid4())
                }
```

## 📊 Monitoring and Metrics

Integrate monitoring into your agent:

```python
from core.infrastructure.monitoring import SystemMonitor
import time

class MonitoredAgentExample:
    """Examples of agent monitoring integration."""
    
    def __init__(self):
        super().__init__()
        self.monitor = SystemMonitor()
    
    async def execute_capability_with_metrics(self, capability: AgentCapability, 
                                            parameters: Dict) -> Dict:
        """Execute capability with metrics collection."""
        start_time = time.time()
        
        # Increment execution counter
        self.monitor.increment_counter(
            f"agent.{self.agent_id}.capability.{capability.value}.executions"
        )
        
        try:
            result = await self.execute_capability(capability, parameters)
            
            # Record success
            self.monitor.increment_counter(
                f"agent.{self.agent_id}.capability.{capability.value}.success"
            )
            
            return result
            
        except Exception as e:
            # Record error
            self.monitor.increment_counter(
                f"agent.{self.agent_id}.capability.{capability.value}.errors"
            )
            raise
            
        finally:
            # Record execution time
            execution_time = time.time() - start_time
            self.monitor.record_histogram(
                f"agent.{self.agent_id}.capability.{capability.value}.duration",
                execution_time
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health_status = await super().health_check()
        
        # Add custom health metrics
        health_status.update({
            "custom_metrics": {
                "total_executions": self.monitor.get_counter_value(
                    f"agent.{self.agent_id}.total_executions"
                ),
                "success_rate": self._calculate_success_rate(),
                "avg_response_time": self._calculate_avg_response_time()
            }
        })
        
        return health_status
```

## 🚀 Best Practices

### 1. Agent Design Principles

- **Single Responsibility**: Each agent should have a focused domain
- **Loose Coupling**: Communicate via MCP, avoid direct dependencies
- **Stateless Design**: Use workflow state, not internal state
- **Error Resilience**: Handle failures gracefully
- **Observable**: Include comprehensive logging and metrics

### 2. Performance Optimization

```python
class PerformanceOptimizedAgent:
    """Performance optimization examples."""
    
    async def batch_process_documents(self, documents: List[Document]) -> List[Dict]:
        """Process multiple documents efficiently."""
        # Batch RAG operations
        contexts = await self.rag_system.batch_similarity_search(
            queries=[doc.content[:100] for doc in documents],
            k=3
        )
        
        # Batch LLM operations
        prompts = [
            self._build_prompt(doc.content, context) 
            for doc, context in zip(documents, contexts)
        ]
        
        responses = await self.llm_chain.abatch([
            {"input": prompt} for prompt in prompts
        ])
        
        return [{"document": doc, "response": resp} 
                for doc, resp in zip(documents, responses)]
    
    async def cache_expensive_operations(self, query: str) -> str:
        """Cache expensive operations."""
        cache_key = f"agent:{self.agent_id}:query:{hash(query)}"
        
        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Perform expensive operation
        result = await self._expensive_operation(query)
        
        # Cache result
        await self.cache.set(cache_key, result, expire=3600)
        
        return result
```

### 3. Testing Your Agent

```python
import pytest
from unittest.mock import AsyncMock, Mock

class TestMyCustomAgent:
    """Test suite for custom agent."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent for testing."""
        agent = MyCustomAgent()
        
        # Mock external dependencies
        agent.rag_system = AsyncMock()
        agent.llm_chain = AsyncMock()
        agent.message_bus = AsyncMock()
        
        return agent
    
    async def test_capability_execution(self, agent):
        """Test capability execution."""
        # Setup mocks
        agent.rag_system.retrieve_context.return_value = [
            Mock(content="test context")
        ]
        agent.llm_chain.ainvoke.return_value = Mock(content="test response")
        
        # Execute capability
        result = await agent.execute_capability(
            AgentCapability.CUSTOM_PROCESSING,
            {"test": "data"}
        )
        
        # Verify results
        assert result is not None
        assert "response" in result
    
    async def test_message_handling(self, agent):
        """Test message handling."""
        message = AgentMessage(
            sender_id="test_sender",
            recipient_id=agent.agent_id,
            content="test message",
            message_type="request"
        )
        
        response = await agent.handle_message(message)
        
        assert response is not None or response is None  # Depending on logic
```

This guide provides you with all the tools and patterns needed to build powerful, collaborative agents that leverage the full capabilities of the framework. Remember to follow the established patterns for consistency and maintainability.
