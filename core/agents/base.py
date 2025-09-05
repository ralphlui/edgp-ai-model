"""
Base agent class and common agent utilities.
Provides standardized interface and shared functionality for all agents.
Enhanced with LangChain, LangGraph, and RAG integration.
Updated with standardized communication types following agentic AI best practices.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from enum import Enum
import logging

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Local imports - avoid circular dependencies
from ..infrastructure.config import settings
from ..types.agent_types import AgentCapability, AgentStatus
from ..types.base import AgentType
from ..types.communication import (
    StandardAgentInput, StandardAgentOutput, AgentContext, AgentMetadata,
    ProcessingResult, AgentError, MessageType, OperationType, ProcessingStage,
    TaskStatus, Priority, ConfidenceLevel, create_standard_output, create_agent_error
)

logger = logging.getLogger(__name__)


T = TypeVar('T')  # Generic type for typed inputs/outputs

class AgentState(BaseModel):
    """LangGraph state for agent workflows."""
    messages: List[BaseMessage] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    current_task: Optional[str] = None
    agent_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    class Config:
        arbitrary_types_allowed = True


class StandardizedAgentMessage(BaseModel):
    """Standardized message format using new communication types."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    sender_id: str
    recipient_id: str
    capability_name: str
    content: Dict[str, Any]
    context: AgentContext
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: Priority = Priority.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class StandardizedAgentTask(BaseModel):
    """Standardized task representation using new communication types."""
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    capability: AgentCapability
    input_data: StandardAgentInput
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    
    # Execution tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    output_data: Optional[StandardAgentOutput] = None
    error: Optional[AgentError] = None
    
    def mark_started(self):
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def mark_completed(self, output: StandardAgentOutput):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.output_data = output
    
    def mark_failed(self, error: AgentError):
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error


class BaseAgent(ABC, Generic[T]):
    """
    Base class for all agents in the system with LangChain/LangGraph integration.
    Provides common functionality and standardized interface.
    Enhanced with standardized communication types following agentic AI best practices.
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        description: str,
        system_prompt: Optional[str] = None,
        capabilities: List[AgentCapability] = None,
        enable_rag: bool = True
    ):
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        
        # Set agent_id if not defined as a property in subclass
        if not hasattr(type(self), 'agent_id') or not isinstance(getattr(type(self), 'agent_id'), property):
            self.agent_id = str(uuid.uuid4())
        
        self.capabilities = capabilities or []
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        
        # Initialize core components (lazy imports to avoid circular dependencies)
        self._llm_gateway = None
        self._rag_system = None
        self.enable_rag = enable_rag
        
        # Queues for standardized messages and tasks
        self.message_queue: List[StandardizedAgentMessage] = []
        self.task_queue: List[StandardizedAgentTask] = []
        
        # Performance tracking
        self.execution_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0
        }
        
        # Initialize workflow
        self._initialize_workflow()
        
        # Set status to idle after initialization
        self.status = AgentStatus.IDLE
        
        logger.info(f"Initialized standardized agent: {self.name} (ID: {self.agent_id})")
    
    @property
    def llm_gateway(self):
        """Lazy load LLM gateway to avoid circular imports."""
        if self._llm_gateway is None:
            from ..services.llm_gateway import llm_gateway
            self._llm_gateway = llm_gateway
        return self._llm_gateway
    
    @property
    def rag_system(self):
        """Lazy load RAG system to avoid circular imports."""
        if self._rag_system is None:
            from ..services.rag_system import rag_system
            self._rag_system = rag_system
        return self._rag_system
    
    def _initialize_workflow(self):
        """Initialize the LangGraph workflow for this agent."""
        try:
            self._workflow = StateGraph(AgentState)
            
            # Add default nodes
            self._workflow.add_node("process_input", self._process_input_node)
            self._workflow.add_node("generate_response", self._generate_response_node)
            self._workflow.add_node("execute_capability", self._execute_capability_node)
            
            # Add edges
            self._workflow.add_edge(START, "process_input")
            self._workflow.add_conditional_edges(
                "process_input",
                self._route_decision,
                {
                    "capability": "execute_capability",
                    "generate": "generate_response",
                    "end": END
                }
            )
            self._workflow.add_edge("execute_capability", "generate_response")
            self._workflow.add_edge("generate_response", END)
            
            # Compile the workflow
            self._compiled_workflow = self._workflow.compile()
            
        except Exception as e:
            logger.error(f"Error initializing workflow for agent {self.name}: {e}")
            # Create a minimal workflow if full initialization fails
            self._compiled_workflow = None
    
    async def _process_input_node(self, state: AgentState) -> AgentState:
        """Process input and prepare for next step."""
        if state.messages:
            last_message = state.messages[-1]
            if isinstance(last_message, HumanMessage):
                state.context["user_input"] = last_message.content
                
                # Add RAG context if enabled
                if self.enable_rag and hasattr(self, '_should_use_rag'):
                    if await self._should_use_rag(last_message.content):
                        try:
                            rag_results = await self.rag_system.query(
                                last_message.content,
                                top_k=settings.rag_top_k
                            )
                            state.context["rag_context"] = rag_results.get("contexts", [])
                        except Exception as e:
                            logger.warning(f"RAG query failed: {e}")
        
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Decide next step in workflow."""
        user_input = state.context.get("user_input", "")
        
        # Check if this requires a specific capability
        for capability in self.capabilities:
            if capability.value.lower() in user_input.lower():
                state.context["target_capability"] = capability
                return "capability"
        
        # Default to generate response
        return "generate"
    
    async def _execute_capability_node(self, state: AgentState) -> AgentState:
        """Execute a specific agent capability."""
        capability = state.context.get("target_capability")
        if capability:
            try:
                result = await self.execute_capability(
                    capability, 
                    state.context.get("parameters", {})
                )
                state.context["capability_result"] = result
            except Exception as e:
                state.context["capability_error"] = str(e)
                logger.error(f"Error executing capability {capability}: {e}")
        
        return state
    
    async def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response using LLM."""
        try:
            # Prepare context
            context_parts = []
            
            if "rag_context" in state.context:
                context_parts.append(f"Relevant context: {state.context['rag_context']}")
            
            if "capability_result" in state.context:
                context_parts.append(f"Capability result: {state.context['capability_result']}")
            
            if "capability_error" in state.context:
                context_parts.append(f"Error occurred: {state.context['capability_error']}")
            
            context_str = "\n".join(context_parts) if context_parts else ""
            
            # Generate response
            user_input = state.context.get("user_input", "")
            system_prompt = self._get_system_prompt()
            
            full_prompt = f"{user_input}\n\n{context_str}" if context_str else user_input
            
            response = await self.generate_llm_response(
                full_prompt,
                system_message=system_prompt
            )
            
            # Add AI response to messages
            ai_message = AIMessage(content=response.get("content", ""))
            state.messages.append(ai_message)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_message = AIMessage(content=f"I encountered an error: {str(e)}")
            state.messages.append(error_message)
        
        return state
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for this agent."""
        capabilities_str = ", ".join([cap.value for cap in self.capabilities])
        return f"""You are {self.name}, a specialized AI agent.
        
        Description: {self.description}
        Capabilities: {capabilities_str}

        You should respond helpfully and accurately based on your capabilities and the provided context."""
    
    async def _should_use_rag(self, query: str) -> bool:
        """Determine if RAG should be used for this query."""
        # Simple heuristic - can be overridden by subclasses
        return len(query.split()) > 3
    
    async def invoke_workflow(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke the agent's LangGraph workflow."""
        if not self._compiled_workflow:
            # Fallback to direct message processing
            response = await self.process_message(message, context or {})
            return {
                "response": response,
                "agent_id": self.id,
                "status": "completed"
            }
        
        try:
            # Create initial state
            initial_state = AgentState(
                agent_id=self.id,
                messages=[HumanMessage(content=message)],
                context=context or {}
            )
            
            # Run workflow
            result = await self._compiled_workflow.ainvoke(initial_state)
            
            # Extract response
            response_content = ""
            if result.messages:
                last_message = result.messages[-1]
                if isinstance(last_message, AIMessage):
                    response_content = last_message.content
            
            return {
                "response": response_content,
                "agent_id": self.id,
                "session_id": result.session_id,
                "context": result.context,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            return {
                "response": f"Error processing request: {str(e)}",
                "agent_id": self.id,
                "status": "error",
                "error": str(e)
            }
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to this agent."""
        self._tools.append(tool)
        logger.info(f"Added tool {tool.name} to agent {self.name}")
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add knowledge to the RAG system."""
        if not self.enable_rag:
            logger.warning(f"RAG not enabled for agent {self.name}")
            return False
        
        try:
            await self.rag_system.add_document(content, metadata)
            logger.info(f"Added knowledge to agent {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to this agent."""
        if capability not in self._capabilities:
            self._capabilities.append(capability)
            logger.info(f"Added capability {capability} to agent {self.agent_id}")
            
    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove a capability from this agent."""
        if capability in self._capabilities:
            self._capabilities.remove(capability)
            logger.info(f"Removed capability {capability} from agent {self.agent_id}")
            
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if this agent has a specific capability."""
        return capability in self._capabilities
    
    # ==================== STANDARDIZED ABSTRACT METHODS ====================
    
    @abstractmethod
    async def execute_capability(
        self, 
        capability: AgentCapability, 
        parameters: Dict[str, Any]
    ) -> StandardAgentOutput:
        """Execute a specific capability with standardized input/output."""
        pass
    
    @abstractmethod
    async def process_message(
        self, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process a message and return a response. Legacy method for backward compatibility."""
        pass
    
    @abstractmethod
    async def process_standardized_input(
        self, 
        agent_input: StandardAgentInput[T]
    ) -> StandardAgentOutput[T]:
        """Process standardized input and return standardized output."""
        pass
    
    @abstractmethod
    async def handle_standardized_message(
        self, 
        message: StandardizedAgentMessage
    ) -> Optional[StandardizedAgentMessage]:
        """Handle incoming standardized messages."""
        pass
    
    # ==================== TASK EXECUTION METHODS ====================
    
    async def execute_standardized_task(
        self, 
        task: StandardizedAgentTask
    ) -> StandardizedAgentTask:
        """
        Execute a standardized task with proper error handling and status updates.
        """
        task.mark_started()
        self.status = AgentStatus.RUNNING
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Agent {self.name} executing standardized task {task.task_id}")
            
            # Process the standardized input
            result = await self.process_standardized_input(task.input_data)
            
            # Update execution metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.metadata.execution_time_ms = execution_time
            
            task.mark_completed(result)
            
            # Update performance metrics
            self.execution_metrics["successful_requests"] += 1
            self._update_average_response_time(execution_time)
            
            logger.info(f"Agent {self.name} completed standardized task {task.task_id}")
            
        except Exception as e:
            # Create standardized error
            error = create_agent_error(
                error_code="TASK_EXECUTION_FAILED",
                error_message=str(e),
                agent_id=self.agent_id,
                capability_name=task.capability.value,
                processing_stage=ProcessingStage.PROCESSING
            )
            
            task.mark_failed(error)
            self.execution_metrics["failed_requests"] += 1
            
            logger.error(f"Agent {self.name} standardized task {task.task_id} failed: {e}")
            
        finally:
            self.status = AgentStatus.IDLE
            self.execution_metrics["total_requests"] += 1
        
        return task
    
    # Legacy method for backward compatibility
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy task execution method for backward compatibility.
        Converts to standardized format internally.
        """
        # Convert to standardized format
        agent_input = StandardAgentInput(
            source_agent_id="legacy",
            target_agent_id=self.agent_id,
            capability_name="legacy_processing",
            data=task_data,
            context=AgentContext()
        )
        
        standardized_task = StandardizedAgentTask(
            capability=AgentCapability.DATA_QUALITY_ASSESSMENT,  # Default capability
            input_data=agent_input
        )
        
        # Execute standardized task
        completed_task = await self.execute_standardized_task(standardized_task)
        
        # Return legacy format
        if completed_task.output_data:
            return completed_task.output_data.dict()
        elif completed_task.error:
            return {"error": completed_task.error.error_message}
        else:
            return {"error": "Unknown execution error"}
    
    # ==================== MESSAGE HANDLING METHODS ====================
    
    async def send_standardized_message(
        self,
        recipient_id: str,
        capability_name: str,
        content: Dict[str, Any],
        context: Optional[AgentContext] = None,
        priority: Priority = Priority.MEDIUM
    ) -> StandardizedAgentMessage:
        """Send a standardized message to another agent."""
        message = StandardizedAgentMessage(
            message_type=MessageType.REQUEST,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            capability_name=capability_name,
            content=content,
            context=context or AgentContext(),
            priority=priority
        )
        
        # Add to message queue (in real implementation, would go through message bus)
        self.message_queue.append(message)
        
        logger.info(f"Agent {self.name} sending standardized message to {recipient_id}: {capability_name}")
        
        return message
    
    def _update_average_response_time(self, new_time_ms: float):
        """Update the rolling average response time."""
        total_requests = self.execution_metrics["total_requests"]
        if total_requests == 0:
            self.execution_metrics["average_response_time"] = new_time_ms
        else:
            current_avg = self.execution_metrics["average_response_time"]
            self.execution_metrics["average_response_time"] = (
                (current_avg * total_requests + new_time_ms) / (total_requests + 1)
            )
    
    async def generate_llm_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate LLM response using the gateway."""
        try:
            response = await self.llm_gateway.generate(
                prompt=prompt,
                system_prompt=system_message,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {"content": f"Error: {str(e)}", "error": True}
    
    async def analyze_data(
        self,
        data: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Common data analysis functionality.
        Can be overridden by subclasses for specific analysis types.
        """
        prompt = f"""
        Analyze the following data for {analysis_type}:
        
        Data: {data}
        
        Provide a structured analysis with insights and recommendations.
        """
        
        response = await self.generate_llm_response(prompt)
        
        return {
            "analysis_type": analysis_type,
            "data_analyzed": data,
            "insights": response.get("content", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.agent_type.value,
            "status": self.status.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "message_queue_size": len(self.message_queue),
            "task_queue_size": len(self.task_queue),
            "execution_metrics": self.execution_metrics
        }
    
    def add_standardized_task(self, task: StandardizedAgentTask):
        """Add a standardized task to the agent's queue."""
        self.task_queue.append(task)
        # Sort by priority (higher enum value = higher priority)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
    
    def add_standardized_message(self, message: StandardizedAgentMessage):
        """Add a standardized message to the agent's queue."""
        self.message_queue.append(message)
    
    # Legacy methods for backward compatibility
    def add_task(self, task_data: Dict[str, Any]):
        """Add a legacy task (converted to standardized format)."""
        # Convert to standardized format
        agent_input = StandardAgentInput(
            source_agent_id="legacy",
            target_agent_id=self.agent_id,
            capability_name="legacy_processing",
            data=task_data,
            context=AgentContext()
        )
        
        standardized_task = StandardizedAgentTask(
            capability=AgentCapability.DATA_QUALITY_ASSESSMENT,  # Default capability
            input_data=agent_input,
            priority=Priority.MEDIUM
        )
        
        self.add_standardized_task(standardized_task)
    
    def add_message(self, message_data: Dict[str, Any]):
        """Add a legacy message (converted to standardized format)."""
        standardized_message = StandardizedAgentMessage(
            message_type=MessageType.REQUEST,
            sender_id=message_data.get("sender", "unknown"),
            recipient_id=self.agent_id,
            capability_name=message_data.get("capability", "legacy_processing"),
            content=message_data.get("content", {}),
            context=AgentContext()
        )
        
        self.add_standardized_message(standardized_message)
    
    async def process_queues(self):
        """Process pending standardized tasks and messages."""
        # Process messages first
        while self.message_queue:
            message = self.message_queue.pop(0)
            try:
                response = await self.handle_standardized_message(message)
                if response:
                    # In a real implementation, send response through message bus
                    logger.info(f"Agent {self.name} generated response to message {message.message_id}")
            except Exception as e:
                logger.error(f"Error handling message {message.message_id}: {e}")
        
        # Process tasks
        while self.task_queue and self.status == AgentStatus.IDLE:
            task = self.task_queue.pop(0)
            await self.execute_standardized_task(task)


class AgentRegistry:
    """Registry for managing agent instances."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent instance."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        for agent in self.agents.values():
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [agent.get_status() for agent in self.agents.values()]
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        return [
            agent for agent in self.agents.values()
            if agent.agent_type == agent_type
        ]


# Global agent registry
agent_registry = AgentRegistry()
