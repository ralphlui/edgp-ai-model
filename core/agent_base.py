"""
Base agent class and common agent utilities.
Provides standardized interface and shared functionality for all agents.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging

from .llm_gateway import llm_gateway, LLMResponse
from .config import settings

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the system."""
    POLICY_SUGGESTION = "policy_suggestion"
    DATA_PRIVACY_COMPLIANCE = "data_privacy_compliance"
    DATA_QUALITY = "data_quality"
    DATA_REMEDIATION = "data_remediation"
    ANALYTICS = "analytics"


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


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
    Base class for all agents in the system.
    Provides common functionality and standardized interface.
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        description: str,
        system_prompt: Optional[str] = None
    ):
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.id = str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self.created_at = datetime.utcnow()
        
        # Message handling
        self.message_queue: List[AgentMessage] = []
        self.task_queue: List[AgentTask] = []
        
        # Configuration
        self.llm_provider = None  # Can be overridden by subclasses
        
        logger.info(f"Initialized {self.name} agent (ID: {self.id})")
    
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
    ) -> LLMResponse:
        """Generate LLM response using the gateway."""
        system_msg = system_message or self.system_prompt
        
        return await llm_gateway.generate(
            prompt=prompt,
            provider=self.llm_provider,
            system_message=system_msg,
            **kwargs
        )
    
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
            "insights": response.content,
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
