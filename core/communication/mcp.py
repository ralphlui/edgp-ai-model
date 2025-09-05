"""
Model Context Protocol (MCP) Communication System
Handles internal agent communication using MCP and LangGraph.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel

# Handle MCP and LangGraph imports gracefully
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain.schema import BaseRetriever
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Mock implementations for development
    LANGGRAPH_AVAILABLE = False
    BaseMessage = object
    HumanMessage = object  
    AIMessage = object
    BaseRetriever = object
    
    class MockStateGraph:
        def __init__(self):
            pass
        def add_node(self, *args, **kwargs): pass
        def add_edge(self, *args, **kwargs): pass
        def set_entry_point(self, *args): pass
        def compile(self): return self
        async def invoke(self, *args, **kwargs): return {}
    
    StateGraph = MockStateGraph
    START = "START"
    END = "END"

from ..types.agent_types import AgentCapability

import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from ..types.agent_types import AgentCapability
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
# LangGraph imports are handled in the try/except block above
# LangGraph imports handled in try/except block above

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """MCP message types for internal communication."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(str, Enum):
    """Message priority levels for MCP communication."""
    LOW = "low"
    NORMAL = "normal" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MCPMessage:
    """MCP message structure for internal communication."""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: Optional[str]
    capability: Optional[AgentCapability]
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        return cls(**data)


class MCPResourceProvider(ABC):
    """Abstract base class for MCP resource providers."""
    
    @abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get the capabilities this provider offers."""
        pass
    
    @abstractmethod
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle an incoming MCP request."""
        pass
    
    @abstractmethod
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get available resources from this provider."""
        pass


class MCPClient:
    """MCP client for sending requests and handling responses."""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._message_handlers: Dict[MessageType, Callable] = {}
        self.is_connected = False
    
    async def send_request(
        self, 
        receiver_id: str, 
        capability: AgentCapability, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> MCPMessage:
        """Send a request and wait for response."""
        message_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        
        message = MCPMessage(
            message_id=message_id,
            message_type=MessageType.REQUEST,
            sender_id=self.client_id,
            receiver_id=receiver_id,
            capability=capability,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id
        )
        
        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[correlation_id] = future
        
        try:
            # Send message through MCP bus
            await self._send_message(message)
            
            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for message {message_id}")
            raise
        finally:
            # Clean up pending request
            self._pending_requests.pop(correlation_id, None)
    
    async def send_notification(
        self, 
        receiver_id: Optional[str], 
        capability: AgentCapability, 
        payload: Dict[str, Any]
    ):
        """Send a notification (no response expected)."""
        message = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.NOTIFICATION,
            sender_id=self.client_id,
            receiver_id=receiver_id,
            capability=capability,
            payload=payload,
            timestamp=datetime.utcnow().isoformat()
        )
        
        await self._send_message(message)
    
    async def _send_message(self, message: MCPMessage):
        """Send message through the MCP message bus."""
        # This would integrate with the actual MCP transport layer
        await MCPMessageBus.instance.route_message(message)
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for specific message types."""
        self._message_handlers[message_type] = handler
    
    async def handle_incoming_message(self, message: MCPMessage):
        """Handle incoming message from MCP bus."""
        if message.message_type == MessageType.RESPONSE:
            # Handle response to pending request
            if message.correlation_id in self._pending_requests:
                future = self._pending_requests[message.correlation_id]
                if not future.done():
                    future.set_result(message)
        
        # Call registered handler
        handler = self._message_handlers.get(message.message_type)
        if handler:
            await handler(message)


class MCPMessageBus:
    """Central message bus for MCP communication."""
    
    instance: 'MCPMessageBus' = None
    
    def __init__(self):
        self._providers: Dict[str, MCPResourceProvider] = {}
        self._clients: Dict[str, MCPClient] = {}
        self._capabilities_registry: Dict[AgentCapability, List[str]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        # Set singleton instance
        MCPMessageBus.instance = self
    
    async def register_provider(self, provider_id: str, provider: MCPResourceProvider):
        """Register an MCP resource provider."""
        self._providers[provider_id] = provider
        
        # Register capabilities
        capabilities = await provider.get_capabilities()
        for capability in capabilities:
            if capability not in self._capabilities_registry:
                self._capabilities_registry[capability] = []
            self._capabilities_registry[capability].append(provider_id)
        
        logger.info(f"Registered MCP provider {provider_id} with capabilities: {capabilities}")
    
    def register_client(self, client: MCPClient):
        """Register an MCP client."""
        self._clients[client.client_id] = client
        client.is_connected = True
        logger.info(f"Registered MCP client {client.client_id}")
    
    async def route_message(self, message: MCPMessage):
        """Route message to appropriate destination."""
        await self._message_queue.put(message)
    
    async def start_message_processing(self):
        """Start processing messages from the queue."""
        self._running = True
        logger.info("Starting MCP message bus processing")
        
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def stop_message_processing(self):
        """Stop message processing."""
        self._running = False
        logger.info("Stopping MCP message bus processing")
    
    async def _process_message(self, message: MCPMessage):
        """Process a single message."""
        try:
            if message.receiver_id:
                # Direct message to specific receiver
                await self._route_to_receiver(message)
            elif message.capability:
                # Route based on capability
                await self._route_by_capability(message)
            else:
                # Broadcast message
                await self._broadcast_message(message)
                
        except Exception as e:
            logger.error(f"Failed to process message {message.message_id}: {e}")
            await self._send_error_response(message, str(e))
    
    async def _route_to_receiver(self, message: MCPMessage):
        """Route message to specific receiver."""
        receiver_id = message.receiver_id
        
        # Check if receiver is a provider
        if receiver_id in self._providers:
            provider = self._providers[receiver_id]
            try:
                response = await provider.handle_request(message)
                if message.message_type == MessageType.REQUEST:
                    await self._send_response(message, response)
            except Exception as e:
                await self._send_error_response(message, str(e))
        
        # Check if receiver is a client
        elif receiver_id in self._clients:
            client = self._clients[receiver_id]
            await client.handle_incoming_message(message)
        
        else:
            logger.warning(f"Unknown receiver: {receiver_id}")
            await self._send_error_response(message, f"Unknown receiver: {receiver_id}")
    
    async def _route_by_capability(self, message: MCPMessage):
        """Route message based on capability."""
        capability = message.capability
        
        if capability not in self._capabilities_registry:
            await self._send_error_response(message, f"No providers for capability: {capability}")
            return
        
        providers = self._capabilities_registry[capability]
        
        if message.message_type == MessageType.REQUEST:
            # Route to first available provider (could implement load balancing here)
            provider_id = providers[0]
            provider = self._providers[provider_id]
            
            try:
                response = await provider.handle_request(message)
                await self._send_response(message, response)
            except Exception as e:
                await self._send_error_response(message, str(e))
        
        else:
            # Broadcast to all providers with this capability
            for provider_id in providers:
                provider = self._providers[provider_id]
                try:
                    await provider.handle_request(message)
                except Exception as e:
                    logger.error(f"Error sending to provider {provider_id}: {e}")
    
    async def _broadcast_message(self, message: MCPMessage):
        """Broadcast message to all connected clients and providers."""
        # Send to all clients
        for client in self._clients.values():
            try:
                await client.handle_incoming_message(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client.client_id}: {e}")
        
        # Send to all providers
        for provider_id, provider in self._providers.items():
            try:
                await provider.handle_request(message)
            except Exception as e:
                logger.error(f"Error broadcasting to provider {provider_id}: {e}")
    
    async def _send_response(self, original_message: MCPMessage, response: MCPMessage):
        """Send response back to original sender."""
        response.message_type = MessageType.RESPONSE
        response.receiver_id = original_message.sender_id
        response.correlation_id = original_message.correlation_id
        response.reply_to = original_message.message_id
        
        await self.route_message(response)
    
    async def _send_error_response(self, original_message: MCPMessage, error_msg: str):
        """Send error response back to original sender."""
        error_response = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            sender_id="mcp_bus",
            receiver_id=original_message.sender_id,
            capability=None,
            payload={"error": error_msg},
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=original_message.correlation_id,
            reply_to=original_message.message_id
        )
        
        await self.route_message(error_response)
    
    async def discover_capabilities(self) -> Dict[AgentCapability, List[str]]:
        """Discover all available capabilities and their providers."""
        return self._capabilities_registry.copy()
    
    async def get_provider_resources(self, provider_id: str) -> List[Dict[str, Any]]:
        """Get resources from a specific provider."""
        if provider_id not in self._providers:
            raise ValueError(f"Provider {provider_id} not found")
        
        provider = self._providers[provider_id]
        return await provider.get_resources()


class LangGraphAgentOrchestrator:
    """LangGraph-based agent orchestration with MCP integration."""
    
    def __init__(self, mcp_bus: 'MCPMessageBus' = None):
        self.mcp_bus = mcp_bus
        self.mcp_client = MCPClient("orchestrator")
        self.workflows: Dict[str, Any] = {}
        self.tool_executors: Dict[str, Any] = {}  # Use Any since ToolExecutor might not be available
    
    async def initialize(self):
        """Initialize the orchestrator with MCP bus."""
        if MCPMessageBus.instance:
            MCPMessageBus.instance.register_client(self.mcp_client)
        
        # Register message handlers
        self.mcp_client.register_handler(MessageType.RESPONSE, self._handle_agent_response)
        self.mcp_client.register_handler(MessageType.NOTIFICATION, self._handle_agent_notification)
    
    def create_workflow(self, workflow_name: str, workflow_definition: Dict[str, Any]):
        """Create a LangGraph workflow for agent orchestration."""
        # Create state graph
        workflow = StateGraph()
        
        # Add nodes based on workflow definition
        for node_name, node_config in workflow_definition.get("nodes", {}).items():
            workflow.add_node(node_name, self._create_node_function(node_config))
        
        # Add edges
        for edge in workflow_definition.get("edges", []):
            workflow.add_edge(edge["from"], edge["to"])
        
        # Set entry point
        entry_point = workflow_definition.get("entry_point", "start")
        workflow.set_entry_point(entry_point)
        
        # Compile workflow
        compiled_workflow = workflow.compile()
        self.workflows[workflow_name] = compiled_workflow
        
        logger.info(f"Created workflow: {workflow_name}")
    
    def _create_node_function(self, node_config: Dict[str, Any]):
        """Create a node function for LangGraph based on configuration."""
        async def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
            agent_capability = AgentCapability(node_config["capability"])
            agent_id = node_config.get("agent_id")
            
            # Prepare payload for MCP request
            payload = {
                "input": state.get("input"),
                "context": state.get("context", {}),
                "parameters": node_config.get("parameters", {})
            }
            
            try:
                # Send request via MCP
                response = await self.mcp_client.send_request(
                    receiver_id=agent_id,
                    capability=agent_capability,
                    payload=payload
                )
                
                # Update state with response
                state["output"] = response.payload.get("result")
                state["metadata"] = response.payload.get("metadata", {})
                state["last_agent"] = response.sender_id
                
                return state
                
            except Exception as e:
                logger.error(f"Node {node_config} execution failed: {e}")
                state["error"] = str(e)
                return state
        
        return node_function
    
    async def execute_workflow(self, workflow_name: str, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a LangGraph workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        
        try:
            # Execute workflow
            result = await workflow.ainvoke(initial_state)
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_name} execution failed: {e}")
            raise
    
    async def _handle_agent_response(self, message: MCPMessage):
        """Handle responses from agents."""
        logger.info(f"Received response from {message.sender_id}: {message.payload}")
    
    async def _handle_agent_notification(self, message: MCPMessage):
        """Handle notifications from agents."""
        logger.info(f"Received notification from {message.sender_id}: {message.payload}")


# Global MCP message bus instance
mcp_bus = MCPMessageBus()
