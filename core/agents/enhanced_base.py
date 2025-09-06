"""
Enhanced Agent Base Class with LangChain/LangGraph Integration

Enhanced base agent class that fully integrates LangChain capabilities
with the shared services platform for collaborative AI.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Union, Callable
import uuid

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models import BaseLanguageModel

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Our shared services and types
from core.shared import (
    StandardAgentInput, StandardAgentOutput, Priority,
    create_standard_input, create_standard_output, create_error_output,
    SharedServices
)
from core.integrations.langchain_integration import (
    LangGraphState, SharedServicesToolkit, SharedServicesCallback
)

logger = logging.getLogger(__name__)


class EnhancedAgentState(LangGraphState):
    """Enhanced agent state with additional agent-specific fields."""
    
    # Agent-specific state (stored in shared_context to avoid field conflicts)
    # agent_type: accessed via shared_context["agent_type"]
    # capabilities: accessed via shared_context["capabilities"] 
    # current_capability: accessed via shared_context["current_capability"]
    
    # Task management (stored in shared_context)
    # current_task: accessed via shared_context["current_task"]
    
    @property
    def current_capability(self) -> Optional[str]:
        """Get current capability from shared context."""
        return self.shared_context.get("current_capability")
    
    @current_capability.setter
    def current_capability(self, value: Optional[str]):
        """Set current capability in shared context."""
        self.shared_context["current_capability"] = value
    
    @property
    def capabilities(self) -> List[str]:
        """Get capabilities from shared context."""
        return self.shared_context.get("capabilities", [])
    
    @property
    def agent_type(self) -> str:
        """Get agent type from shared context."""
        return self.shared_context.get("agent_type", "generic")
    task_queue: List[Dict[str, Any]] = None
    completed_tasks: List[str] = None
    
    # Agent memory and context
    working_memory: Dict[str, Any] = None
    long_term_memory_refs: List[str] = None
    
    # Performance tracking
    processing_times: List[float] = None
    tool_usage_count: Dict[str, int] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.task_queue is None:
            self.task_queue = []
        if self.completed_tasks is None:
            self.completed_tasks = []
        if self.working_memory is None:
            self.working_memory = {}
        if self.long_term_memory_refs is None:
            self.long_term_memory_refs = []
        if self.processing_times is None:
            self.processing_times = []
        if self.tool_usage_count is None:
            self.tool_usage_count = {}


class LangChainAgent(ABC):
    """
    Enhanced base class for agents with full LangChain/LangGraph integration.
    
    Provides comprehensive integration with shared services, standardized
    communication, and enterprise-grade agent capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        name: str,
        description: str,
        shared_services: SharedServices,
        capabilities: List[str] = None,
        llm: Optional[BaseLanguageModel] = None,
        system_prompt: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.shared_services = shared_services
        self.capabilities = capabilities or []
        self.llm = llm
        self.system_prompt = system_prompt
        
        # LangChain integration components
        self.toolkit: Optional[SharedServicesToolkit] = None
        self.callback_handler: Optional[SharedServicesCallback] = None
        self.workflow: Optional[StateGraph] = None
        self.compiled_workflow: Optional[Runnable] = None
        
        # Agent state and performance tracking
        self.current_session: Optional[str] = None
        self.active_tasks: Dict[str, Any] = {}
        self.metrics = {
            "tasks_completed": 0,
            "total_processing_time": 0.0,
            "successful_executions": 0,
            "failed_executions": 0,
            "tool_calls": 0
        }
        
        # Initialize components
        self._initialized = False
    
    async def initialize(self):
        """Initialize the agent with LangChain integration."""
        if self._initialized:
            return
        
        # Initialize shared services toolkit
        self.toolkit = SharedServicesToolkit(self.shared_services)
        
        # Initialize callback handler
        self.callback_handler = SharedServicesCallback(self.shared_services)
        
        # Create agent-specific tools
        agent_tools = await self._create_agent_tools()
        if agent_tools:
            self.toolkit.tools.extend(agent_tools)
        
        # Initialize LangGraph workflow
        await self._initialize_workflow()
        
        # Register agent with shared services if context is available
        if self.shared_services.is_feature_available("context"):
            await self._register_agent_context()
        
        self._initialized = True
        logger.info(f"Agent {self.agent_id} ({self.agent_type}) initialized")
    
    async def _initialize_workflow(self):
        """Initialize the LangGraph workflow for this agent."""
        try:
            # Create workflow graph
            self.workflow = StateGraph(EnhancedAgentState)
            
            # Add core nodes
            self.workflow.add_node("initialize", self._initialize_node)
            self.workflow.add_node("process_input", self._process_input_node)
            self.workflow.add_node("select_capability", self._select_capability_node)
            self.workflow.add_node("execute_capability", self._execute_capability_node)
            self.workflow.add_node("use_tools", self._use_tools_node)
            self.workflow.add_node("generate_response", self._generate_response_node)
            self.workflow.add_node("finalize", self._finalize_node)
            
            # Add workflow edges
            self._build_workflow_edges()
            
            # Add custom nodes if implemented by subclass
            await self._add_custom_nodes()
            
            # Compile workflow
            self.compiled_workflow = self.workflow.compile(
                checkpointer=MemorySaver()  # For state persistence
            )
            
            logger.info(f"Workflow initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow for {self.agent_id}: {str(e)}")
            raise
    
    def _build_workflow_edges(self):
        """Build the workflow edges for the agent."""
        # Basic workflow structure
        self.workflow.add_edge(START, "initialize")
        self.workflow.add_edge("initialize", "process_input")
        
        # Conditional routing based on input
        self.workflow.add_conditional_edges(
            "process_input",
            self._route_processing,
            {
                "capability": "select_capability",
                "tools": "use_tools",
                "direct": "generate_response",
                "error": "finalize"
            }
        )
        
        self.workflow.add_edge("select_capability", "execute_capability")
        self.workflow.add_edge("execute_capability", "generate_response")
        self.workflow.add_edge("use_tools", "generate_response")
        self.workflow.add_edge("generate_response", "finalize")
        self.workflow.add_edge("finalize", END)
    
    async def _add_custom_nodes(self):
        """Add custom nodes - implemented by subclasses."""
        pass
    
    def _route_processing(self, state: EnhancedAgentState) -> str:
        """Route processing based on the current state."""
        try:
            # Check if we need to use specific capabilities
            if state.current_capability:
                return "capability"
            
            # Check if tools are explicitly requested
            if any("tool" in msg.content.lower() for msg in state.messages[-2:]):
                return "tools"
            
            # Check for errors
            if state.errors:
                return "error"
            
            # Default to direct response generation
            return "direct"
            
        except Exception as e:
            logger.error(f"Routing error: {str(e)}")
            return "error"
    
    # Core workflow nodes
    async def _initialize_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Initialize the agent state for processing."""
        # Store agent info in shared_context to avoid field conflicts
        state.shared_context["agent_type"] = self.agent_type
        state.shared_context["capabilities"] = self.capabilities.copy()
        state.shared_context["current_capability"] = None
        state.current_agent = self.agent_id
        
        # Load agent context if available
        if self.shared_services.is_feature_available("context") and state.session_id:
            try:
                agent_context = await self.shared_services.context.get_agent_context(
                    self.agent_id, state.session_id
                )
                if agent_context:
                    state.working_memory.update(agent_context.working_memory)
                    state.long_term_memory_refs.extend(agent_context.facts)
            except Exception as e:
                state.add_warning(f"Failed to load agent context: {str(e)}", self.agent_id)
        
        return state
    
    async def _process_input_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Process the input and prepare for execution."""
        try:
            # Get the latest message
            if state.messages:
                latest_message = state.messages[-1]
                
                # Store in working memory
                state.working_memory["last_user_input"] = latest_message.content
                
                # Analyze input to determine processing needs
                input_analysis = await self._analyze_input(latest_message.content)
                state.working_memory["input_analysis"] = input_analysis
                
                # Determine required capability
                required_capability = self._determine_required_capability(input_analysis)
                if required_capability in self.capabilities:
                    state.current_capability = required_capability
            
            return state
            
        except Exception as e:
            state.add_error(f"Input processing failed: {str(e)}", self.agent_id)
            return state
    
    async def _select_capability_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Select and prepare the appropriate capability."""
        try:
            if state.current_capability:
                # Prepare capability execution
                capability_context = await self._prepare_capability_context(
                    state.current_capability, state
                )
                state.working_memory[f"{state.current_capability}_context"] = capability_context
                
                # Get relevant memories for this capability
                if self.shared_services.is_feature_available("memory"):
                    memories = await self.shared_services.memory.retrieve_memories(
                        query=f"{state.current_capability} {state.working_memory.get('last_user_input', '')}",
                        limit=5
                    )
                    state.working_memory["relevant_memories"] = [
                        {"content": mem.content, "metadata": mem.metadata} for mem in memories
                    ]
            
            return state
            
        except Exception as e:
            state.add_error(f"Capability selection failed: {str(e)}", self.agent_id)
            return state
    
    async def _execute_capability_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Execute the selected capability."""
        start_time = datetime.now()
        
        try:
            if state.current_capability:
                # Execute the capability
                result = await self._execute_capability(state.current_capability, state)
                
                # Store result
                state.agent_outputs[state.current_capability] = result
                state.working_memory[f"{state.current_capability}_result"] = result
                
                # Track completion
                if state.current_capability not in state.completed_tasks:
                    state.completed_tasks.append(state.current_capability)
                
                # Update metrics
                processing_time = (datetime.now() - start_time).total_seconds()
                state.processing_times.append(processing_time)
                self.metrics["tasks_completed"] += 1
                self.metrics["total_processing_time"] += processing_time
                self.metrics["successful_executions"] += 1
            
            return state
            
        except Exception as e:
            self.metrics["failed_executions"] += 1
            state.add_error(f"Capability execution failed: {str(e)}", self.agent_id)
            return state
    
    async def _use_tools_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Use tools to assist with processing."""
        try:
            # Get available tools
            tools = self.toolkit.get_tools()
            
            if tools:
                # Simple tool selection logic - can be enhanced
                tool_to_use = self._select_appropriate_tool(state, tools)
                
                if tool_to_use:
                    # Execute tool
                    tool_input = self._prepare_tool_input(state)
                    result = await tool_to_use.ainvoke(tool_input)
                    
                    # Store tool result
                    state.tool_results[tool_to_use.name] = result
                    state.working_memory[f"tool_{tool_to_use.name}_result"] = result
                    
                    # Update tool usage tracking
                    tool_name = tool_to_use.name
                    state.tool_usage_count[tool_name] = state.tool_usage_count.get(tool_name, 0) + 1
                    self.metrics["tool_calls"] += 1
            
            return state
            
        except Exception as e:
            state.add_error(f"Tool usage failed: {str(e)}", self.agent_id)
            return state
    
    async def _generate_response_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Generate the final response."""
        try:
            # Compile response based on execution results
            response_content = await self._compile_response(state)
            
            # Add response as AI message
            state.add_message(AIMessage(content=response_content, name=self.agent_id))
            
            # Store response in working memory
            state.working_memory["generated_response"] = response_content
            
            return state
            
        except Exception as e:
            state.add_error(f"Response generation failed: {str(e)}", self.agent_id)
            # Add error response
            state.add_message(AIMessage(
                content=f"I apologize, but I encountered an error: {str(e)}",
                name=self.agent_id
            ))
            return state
    
    async def _finalize_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Finalize processing and store results."""
        try:
            # Store important results in long-term memory
            if self.shared_services.is_feature_available("memory") and not state.errors:
                await self._store_execution_memory(state)
            
            # Update agent context
            if self.shared_services.is_feature_available("context"):
                await self._update_agent_context(state)
            
            # Store any new knowledge
            if self.shared_services.is_feature_available("knowledge"):
                await self._extract_and_store_knowledge(state)
            
            return state
            
        except Exception as e:
            state.add_warning(f"Finalization partially failed: {str(e)}", self.agent_id)
            return state
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        """Analyze input to understand requirements."""
        pass
    
    @abstractmethod
    def _determine_required_capability(self, input_analysis: Dict[str, Any]) -> Optional[str]:
        """Determine which capability is needed based on input analysis."""
        pass
    
    @abstractmethod
    async def _prepare_capability_context(self, capability: str, state: EnhancedAgentState) -> Dict[str, Any]:
        """Prepare context for capability execution."""
        pass
    
    @abstractmethod
    async def _execute_capability(self, capability: str, state: EnhancedAgentState) -> Any:
        """Execute a specific capability."""
        pass
    
    @abstractmethod
    async def _compile_response(self, state: EnhancedAgentState) -> str:
        """Compile the final response based on execution results."""
        pass
    
    # Optional methods that can be overridden
    async def _create_agent_tools(self) -> List[BaseTool]:
        """Create agent-specific tools."""
        return []
    
    def _select_appropriate_tool(self, state: EnhancedAgentState, tools: List[BaseTool]) -> Optional[BaseTool]:
        """Select the most appropriate tool for the current state."""
        # Default implementation - can be enhanced by subclasses
        if tools:
            return tools[0]
        return None
    
    def _prepare_tool_input(self, state: EnhancedAgentState) -> str:
        """Prepare input for tool execution."""
        return state.working_memory.get("last_user_input", "")
    
    # Helper methods
    async def _store_execution_memory(self, state: EnhancedAgentState):
        """Store execution results in long-term memory."""
        try:
            if state.agent_outputs:
                memory_content = f"Executed {len(state.agent_outputs)} capabilities: {list(state.agent_outputs.keys())}"
                await self.shared_services.memory.store_memory(
                    content=memory_content,
                    memory_type="episodic",
                    metadata={
                        "agent_id": self.agent_id,
                        "session_id": state.session_id,
                        "capabilities": list(state.agent_outputs.keys()),
                        "success": len(state.errors) == 0
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to store execution memory: {str(e)}")
    
    async def _update_agent_context(self, state: EnhancedAgentState):
        """Update agent context in shared services."""
        try:
            from core.shared.context.types import AgentContext
            
            agent_context = AgentContext(
                agent_id=self.agent_id,
                session_id=state.session_id,
                current_task=state.current_task,
                task_history=state.completed_tasks,
                goals=self.capabilities,
                working_memory=state.working_memory,
                facts=[msg.content for msg in state.messages[-5:]], # Last 5 messages as facts
                assumptions=[],
                available_tools=[tool.name for tool in self.toolkit.get_tools()],
                tool_usage_history=list(state.tool_usage_count.items()),
                performance_metrics=self.metrics,
                error_history=state.errors
            )
            
            await self.shared_services.context.store_agent_context(agent_context)
            
        except Exception as e:
            logger.warning(f"Failed to update agent context: {str(e)}")
    
    async def _extract_and_store_knowledge(self, state: EnhancedAgentState):
        """Extract and store knowledge from execution."""
        try:
            # This is a simplified implementation - can be enhanced
            if state.agent_outputs and not state.errors:
                for capability, result in state.agent_outputs.items():
                    if isinstance(result, dict) and result.get("insights"):
                        # Store insights as facts
                        fact_content = f"Agent {self.agent_id} capability {capability}: {result['insights']}"
                        await self.shared_services.knowledge.create_fact(
                            content=fact_content,
                            fact_type="agent_insight",
                            confidence=0.8
                        )
        except Exception as e:
            logger.warning(f"Failed to extract knowledge: {str(e)}")
    
    async def _register_agent_context(self):
        """Register agent with context management."""
        try:
            # This would register the agent for context tracking
            logger.info(f"Agent {self.agent_id} registered with context management")
        except Exception as e:
            logger.warning(f"Failed to register agent context: {str(e)}")
    
    # Public interface methods
    async def process_message(
        self,
        message: str,
        user_id: str = None,
        session_id: str = None,
        context: Dict[str, Any] = None
    ) -> StandardAgentOutput:
        """Process a message using the LangGraph workflow."""
        try:
            # Create initial state
            initial_state = EnhancedAgentState(
                messages=[HumanMessage(content=message)],
                user_id=user_id,
                session_id=session_id or str(uuid.uuid4()),
                shared_context=context or {}
            )
            
            # Execute workflow with proper configuration for checkpointer
            config = {
                "configurable": {
                    "thread_id": session_id or str(uuid.uuid4())
                }
            }
            result = await self.compiled_workflow.ainvoke(initial_state, config=config)
            
            # Handle case where result is None
            if result is None:
                logger.error(f"Workflow execution returned None for agent {self.agent_id}")
                return create_error_output(
                    request_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    error_code="WORKFLOW_ERROR",
                    error_message="Workflow execution returned no result"
                )
            
            # Extract response
            response_content = ""
            if result.messages:
                last_message = result.messages[-1]
                if isinstance(last_message, AIMessage):
                    response_content = last_message.content
            
            # Create standardized output
            return create_standard_output(
                request_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                data={
                    "response": response_content,
                    "agent_outputs": result.agent_outputs,
                    "capabilities_used": result.completed_tasks,
                    "tools_used": list(result.tool_results.keys()),
                    "processing_steps": result.step_count
                },
                success=len(result.errors) == 0,
                message=f"Processed successfully in {result.step_count} steps" if len(result.errors) == 0 else "Processing completed with errors"
            )
            
        except Exception as e:
            logger.error(f"Message processing failed for {self.agent_id}: {str(e)}")
            return create_error_output(
                request_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                error_code="PROCESSING_ERROR",
                error_message=str(e)
            )
    
    async def process_standardized_input(self, agent_input: StandardAgentInput) -> StandardAgentOutput:
        """Process standardized agent input."""
        try:
            # Extract message from standardized input
            message = agent_input.data.get("message", str(agent_input.data))
            
            # Use existing processing pipeline
            result = await self.process_message(
                message=message,
                user_id=agent_input.user_id,
                session_id=agent_input.session_id,
                context=agent_input.metadata
            )
            
            # Update result with original request ID
            result.request_id = agent_input.request_id
            return result
            
        except Exception as e:
            return create_error_output(
                request_id=agent_input.request_id,
                agent_id=self.agent_id,
                error_code="STANDARDIZED_INPUT_ERROR",
                error_message=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return self.capabilities.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return self.metrics.copy()
    
    async def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        logger.info(f"Shutting down agent {self.agent_id}")
        self._initialized = False


# Example implementation of a concrete agent
class ExampleAnalyticsAgent(LangChainAgent):
    """Example analytics agent implementation."""
    
    def __init__(self, shared_services: SharedServices):
        super().__init__(
            agent_id="analytics_agent",
            agent_type="analytics",
            name="Analytics Agent",
            description="Performs data analytics and generates insights",
            shared_services=shared_services,
            capabilities=["data_analysis", "trend_analysis", "report_generation"],
            system_prompt="You are an analytics agent that helps analyze data and generate insights."
        )
    
    async def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        """Analyze input for analytics requirements."""
        input_lower = input_text.lower()
        
        analysis = {
            "requires_data_analysis": any(word in input_lower for word in ["analyze", "data", "metrics"]),
            "requires_trends": any(word in input_lower for word in ["trend", "pattern", "over time"]),
            "requires_report": any(word in input_lower for word in ["report", "summary", "document"]),
            "complexity": "high" if len(input_text) > 200 else "medium" if len(input_text) > 50 else "low"
        }
        
        return analysis
    
    def _determine_required_capability(self, input_analysis: Dict[str, Any]) -> Optional[str]:
        """Determine required capability for analytics."""
        if input_analysis.get("requires_report"):
            return "report_generation"
        elif input_analysis.get("requires_trends"):
            return "trend_analysis"
        elif input_analysis.get("requires_data_analysis"):
            return "data_analysis"
        else:
            return "data_analysis"  # Default capability
    
    async def _prepare_capability_context(self, capability: str, state: EnhancedAgentState) -> Dict[str, Any]:
        """Prepare context for analytics capability."""
        context = {
            "capability": capability,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get relevant documents from RAG if available
        if self.shared_services.is_feature_available("rag"):
            try:
                docs = await self.shared_services.rag.search(
                    query=f"{capability} analytics best practices",
                    top_k=3
                )
                context["relevant_docs"] = [{"content": doc.content[:200]} for doc in docs]
            except Exception as e:
                context["rag_error"] = str(e)
        
        return context
    
    async def _execute_capability(self, capability: str, state: EnhancedAgentState) -> Any:
        """Execute analytics capability."""
        user_input = state.working_memory.get("last_user_input", "")
        
        if capability == "data_analysis":
            return await self._perform_data_analysis(user_input, state)
        elif capability == "trend_analysis":
            return await self._perform_trend_analysis(user_input, state)
        elif capability == "report_generation":
            return await self._generate_analytics_report(user_input, state)
        else:
            return {"error": f"Unknown capability: {capability}"}
    
    async def _perform_data_analysis(self, input_text: str, state: EnhancedAgentState) -> Dict[str, Any]:
        """Perform data analysis."""
        return {
            "analysis_type": "data_analysis",
            "insights": [
                "Data shows consistent patterns",
                "Key metrics are within expected ranges",
                "Recommendations: Continue monitoring trends"
            ],
            "confidence": 0.85,
            "data_points_analyzed": 150
        }
    
    async def _perform_trend_analysis(self, input_text: str, state: EnhancedAgentState) -> Dict[str, Any]:
        """Perform trend analysis."""
        return {
            "analysis_type": "trend_analysis",
            "trends": [
                "Upward trend in key metrics",
                "Seasonal variations detected",
                "Positive outlook for next quarter"
            ],
            "confidence": 0.92,
            "time_period": "last_6_months"
        }
    
    async def _generate_analytics_report(self, input_text: str, state: EnhancedAgentState) -> Dict[str, Any]:
        """Generate analytics report."""
        return {
            "analysis_type": "report_generation",
            "report_sections": [
                "Executive Summary",
                "Key Findings", 
                "Detailed Analysis",
                "Recommendations"
            ],
            "report_ready": True,
            "pages": 12
        }
    
    async def _compile_response(self, state: EnhancedAgentState) -> str:
        """Compile final response for analytics."""
        if state.errors:
            return f"I encountered some issues during analysis: {'; '.join(state.errors)}"
        
        response_parts = []
        
        # Add capability results
        for capability, result in state.agent_outputs.items():
            if isinstance(result, dict):
                if result.get("insights"):
                    response_parts.append(f"**{capability.replace('_', ' ').title()}:**")
                    for insight in result["insights"]:
                        response_parts.append(f"- {insight}")
                elif result.get("trends"):
                    response_parts.append(f"**{capability.replace('_', ' ').title()}:**")
                    for trend in result["trends"]:
                        response_parts.append(f"- {trend}")
                elif result.get("report_sections"):
                    response_parts.append(f"**Analytics Report Generated:**")
                    response_parts.append(f"Report includes {len(result['report_sections'])} sections covering comprehensive analysis.")
        
        # Add tool results if any
        if state.tool_results:
            response_parts.append("\n**Additional Information:**")
            for tool_name, result in state.tool_results.items():
                response_parts.append(f"- Used {tool_name}: {str(result)[:100]}...")
        
        return "\n".join(response_parts) if response_parts else "Analysis completed successfully."
