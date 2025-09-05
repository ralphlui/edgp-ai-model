"""
Base agent class and common agent utilities.
Provides standardized interface and shared functionality for all agents.
Enhanced with LangChain, LangGraph, and RAG integration.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
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

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """LangGraph state for agent workflows."""
    messages: List[BaseMessage] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    current_task: Optional[str] = None
    agent_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    class Config:
        arbitrary_types_allowed = True


class AgentMessage:
    """Standardized message format for inter-agent communication."""
    
    def __init__(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        content: Dict[str, Any],
        correlation_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type
        self.content = content
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "content": self.content,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat()
        }


class AgentTask:
    """Represents a task to be executed by an agent."""
    
    def __init__(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: int = 1,
        timeout: Optional[int] = None
    ):
        self.id = str(uuid.uuid4())
        self.task_type = task_type
        self.data = data
        self.priority = priority
        self.timeout = timeout or settings.agent_timeout
        self.created_at = datetime.utcnow()
        self.status = AgentStatus.IDLE
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class for all agents in the system with LangChain/LangGraph integration.
    Provides common functionality and standardized interface.
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
        self.id = str(uuid.uuid4())
        self.capabilities = capabilities or []
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        
        # Initialize core components (lazy imports to avoid circular dependencies)
        self._llm_gateway = None
        self._rag_system = None
        self.enable_rag = enable_rag
        
        # Queues for messages and tasks
        self.message_queue: List[AgentMessage] = []
        self.task_queue: List[AgentTask] = []
        
        # Initialize workflow
        self._initialize_workflow()
        
        # Set status to idle after initialization
        self.status = AgentStatus.IDLE
        
        logger.info(f"Initialized agent: {self.name} (ID: {self.id})")
    
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
    
    @abstractmethod
    async def execute_capability(
        self, 
        capability: AgentCapability, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific capability. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a message and return a response. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a specific task. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages. Must be implemented by subclasses."""
        pass
    
    async def execute_task(self, task: AgentTask) -> AgentTask:
        """
        Execute a task with proper error handling and status updates.
        """
        task.status = AgentStatus.RUNNING
        self.status = AgentStatus.RUNNING
        
        try:
            logger.info(f"Agent {self.name} executing task {task.id}")
            
            # Set timeout
            result = await asyncio.wait_for(
                self.process_task(task),
                timeout=task.timeout
            )
            
            task.result = result
            task.status = AgentStatus.COMPLETED
            
            logger.info(f"Agent {self.name} completed task {task.id}")
            
        except asyncio.TimeoutError:
            task.status = AgentStatus.TIMEOUT
            task.error = f"Task timed out after {task.timeout} seconds"
            logger.error(f"Agent {self.name} task {task.id} timed out")
            
        except Exception as e:
            task.status = AgentStatus.FAILED
            task.error = str(e)
            logger.error(f"Agent {self.name} task {task.id} failed: {e}")
            
        finally:
            self.status = AgentStatus.IDLE
        
        return task
    
    async def send_message(
        self,
        recipient: str,
        message_type: str,
        content: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id
        )
        
        # In a real implementation, this would go through the orchestrator
        logger.info(f"Agent {self.name} sending message to {recipient}: {message_type}")
        
        return message
    
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
            "task_queue_size": len(self.task_queue)
        }
    
    def add_task(self, task: AgentTask):
        """Add a task to the agent's queue."""
        self.task_queue.append(task)
        # Sort by priority (higher number = higher priority)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
    
    def add_message(self, message: AgentMessage):
        """Add a message to the agent's queue."""
        self.message_queue.append(message)
    
    async def process_queues(self):
        """Process pending tasks and messages."""
        # Process messages first
        while self.message_queue:
            message = self.message_queue.pop(0)
            try:
                response = await self.handle_message(message)
                if response:
                    # In a real implementation, send response through orchestrator
                    logger.info(f"Agent {self.name} generated response to message {message.id}")
            except Exception as e:
                logger.error(f"Error handling message {message.id}: {e}")
        
        # Process tasks
        while self.task_queue and self.status == AgentStatus.IDLE:
            task = self.task_queue.pop(0)
            await self.execute_task(task)


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
