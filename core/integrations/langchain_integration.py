"""
LangChain and LangGraph Integration

Complete integration layer connecting LangChain/LangGraph frameworks
with the collaborative AI platform's shared services.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
import uuid

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

# LangGraph imports  
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
# Removed SqliteSaver import - using memory-based checkpointing for now
# from langgraph.checkpoint.sqlite import SqliteSaver

# Our shared services
from core.shared import (
    StandardAgentInput, StandardAgentOutput, Priority, 
    create_standard_input, create_standard_output, create_error_output,
    SharedServices, get_feature_status
)

logger = logging.getLogger(__name__)


@dataclass
class LangGraphState:
    """Enhanced state for LangGraph workflows with shared services integration."""
    
    # Core workflow state
    messages: List[BaseMessage] = field(default_factory=list)
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    
    # Agent and execution context
    current_agent: Optional[str] = None
    executed_agents: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Shared services integration
    shared_context: Dict[str, Any] = field(default_factory=dict)
    memory_keys: List[str] = field(default_factory=list)
    knowledge_refs: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow metadata
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Agent working memory and task tracking
    working_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_memory_refs: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    task_queue: List[str] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    
    # Workflow timing
    workflow_type: str = "sequential"
    
    def add_message(self, message: BaseMessage):
        """Add a message to the workflow state."""
        self.messages.append(message)
        self.last_activity = datetime.now()
        self.step_count += 1
    
    def set_agent_output(self, agent_id: str, output: Any):
        """Store output from an agent."""
        self.agent_outputs[agent_id] = output
        if agent_id not in self.executed_agents:
            self.executed_agents.append(agent_id)
        self.current_agent = agent_id
        self.last_activity = datetime.now()
    
    def add_error(self, error: str, agent_id: str = None):
        """Add an error to the workflow state."""
        error_msg = f"[{agent_id or 'WORKFLOW'}] {error}"
        self.errors.append(error_msg)
        logger.error(error_msg)
    
    def add_warning(self, warning: str, agent_id: str = None):
        """Add a warning to the workflow state."""
        warning_msg = f"[{agent_id or 'WORKFLOW'}] {warning}"
        self.warnings.append(warning_msg)
        logger.warning(warning_msg)


class SharedServicesCallback(BaseCallbackHandler):
    """LangChain callback handler for shared services integration."""
    
    def __init__(self, shared_services: SharedServices):
        super().__init__()
        self.shared_services = shared_services
        self.session_metrics = {}
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Handle chain start events."""
        run_id = kwargs.get('run_id')
        if run_id and self.shared_services.is_feature_available("memory"):
            # Store chain execution in memory
            await self.shared_services.memory.store_memory(
                content=f"Started chain: {serialized.get('name', 'unknown')}",
                memory_type="episodic",
                metadata={
                    "run_id": str(run_id),
                    "chain_type": "langchain",
                    "inputs": inputs
                }
            )
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Handle chain end events."""
        run_id = kwargs.get('run_id')
        if run_id and self.shared_services.is_feature_available("memory"):
            # Store chain results in memory
            await self.shared_services.memory.store_memory(
                content=f"Completed chain with outputs: {str(outputs)[:200]}...",
                memory_type="episodic",
                metadata={
                    "run_id": str(run_id),
                    "chain_type": "langchain",
                    "outputs": outputs
                }
            )
    
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Handle tool execution start."""
        if self.shared_services.is_feature_available("tools"):
            tool_name = serialized.get('name', 'unknown_tool')
            logger.info(f"LangChain tool started: {tool_name}")
    
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Handle tool execution end."""
        if self.shared_services.is_feature_available("tools"):
            logger.info(f"LangChain tool completed: {output[:100]}...")


class SharedServicesToolkit:
    """Toolkit for creating LangChain tools from shared services."""
    
    def __init__(self, shared_services: SharedServices):
        self.shared_services = shared_services
        self.tools: List[BaseTool] = []
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools from shared services."""
        self.tools = []
        
        # Memory tools
        if self.shared_services.is_feature_available("memory"):
            self.tools.extend(self._create_memory_tools())
        
        # RAG tools
        if self.shared_services.is_feature_available("rag"):
            self.tools.extend(self._create_rag_tools())
        
        # Knowledge tools
        if self.shared_services.is_feature_available("knowledge"):
            self.tools.extend(self._create_knowledge_tools())
        
        # Context tools
        if self.shared_services.is_feature_available("context"):
            self.tools.extend(self._create_context_tools())
        
        # Prompt tools
        if self.shared_services.is_feature_available("prompt"):
            self.tools.extend(self._create_prompt_tools())
    
    def _create_memory_tools(self) -> List[BaseTool]:
        """Create memory-related tools."""
        tools = []
        
        async def store_memory_func(content: str, memory_type: str = "semantic", metadata: Dict = None):
            """Store a memory in the shared memory system."""
            try:
                memory_id = await self.shared_services.memory.store_memory(
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata or {}
                )
                return f"Memory stored with ID: {memory_id}"
            except Exception as e:
                return f"Error storing memory: {str(e)}"
        
        async def retrieve_memories_func(query: str, limit: int = 5):
            """Retrieve relevant memories based on a query."""
            try:
                memories = await self.shared_services.memory.retrieve_memories(
                    query=query,
                    limit=limit
                )
                return [{"id": mem.memory_id, "content": mem.content, "metadata": mem.metadata} for mem in memories]
            except Exception as e:
                return f"Error retrieving memories: {str(e)}"
        
        tools.extend([
            StructuredTool.from_function(
                func=store_memory_func,
                name="store_memory",
                description="Store information in the shared memory system for future reference"
            ),
            StructuredTool.from_function(
                func=retrieve_memories_func,
                name="retrieve_memories", 
                description="Retrieve relevant memories based on a query"
            )
        ])
        
        return tools
    
    def _create_rag_tools(self) -> List[BaseTool]:
        """Create RAG-related tools."""
        tools = []
        
        async def search_documents_func(query: str, top_k: int = 5, collection: str = None):
            """Search documents in the RAG system."""
            try:
                results = await self.shared_services.rag.search(
                    query=query,
                    top_k=top_k,
                    collection_name=collection
                )
                return [{"content": doc.content, "metadata": doc.metadata, "score": doc.score} for doc in results]
            except Exception as e:
                return f"Error searching documents: {str(e)}"
        
        async def add_document_func(content: str, metadata: Dict = None):
            """Add a document to the RAG system."""
            try:
                doc_id = await self.shared_services.rag.add_document(
                    content=content,
                    metadata=metadata or {}
                )
                return f"Document added with ID: {doc_id}"
            except Exception as e:
                return f"Error adding document: {str(e)}"
        
        tools.extend([
            StructuredTool.from_function(
                func=search_documents_func,
                name="search_documents",
                description="Search for relevant documents in the knowledge base"
            ),
            StructuredTool.from_function(
                func=add_document_func,
                name="add_document",
                description="Add a new document to the knowledge base"
            )
        ])
        
        return tools
    
    def _create_knowledge_tools(self) -> List[BaseTool]:
        """Create knowledge management tools."""
        tools = []
        
        async def search_entities_func(query: str, limit: int = 10):
            """Search for entities in the knowledge base."""
            try:
                entities = await self.shared_services.knowledge.search_entities(
                    query=query,
                    limit=limit
                )
                return [{"id": ent.entity_id, "name": ent.name, "type": ent.entity_type, "properties": ent.properties} for ent in entities]
            except Exception as e:
                return f"Error searching entities: {str(e)}"
        
        async def create_entity_func(name: str, entity_type: str, properties: Dict = None):
            """Create a new entity in the knowledge base."""
            try:
                entity_id = await self.shared_services.knowledge.create_entity(
                    name=name,
                    entity_type=entity_type,
                    properties=properties or {}
                )
                return f"Entity created with ID: {entity_id}"
            except Exception as e:
                return f"Error creating entity: {str(e)}"
        
        async def get_relationships_func(entity_id: str):
            """Get relationships for an entity."""
            try:
                relationships = await self.shared_services.knowledge.get_entity_relationships(entity_id)
                return [{"id": rel.relationship_id, "type": rel.relationship_type, "target": rel.target_id} for rel in relationships]
            except Exception as e:
                return f"Error getting relationships: {str(e)}"
        
        tools.extend([
            StructuredTool.from_function(
                func=search_entities_func,
                name="search_entities",
                description="Search for entities in the knowledge graph"
            ),
            StructuredTool.from_function(
                func=create_entity_func,
                name="create_entity",
                description="Create a new entity in the knowledge graph"
            ),
            StructuredTool.from_function(
                func=get_relationships_func,
                name="get_entity_relationships",
                description="Get relationships for a specific entity"
            )
        ])
        
        return tools
    
    def _create_context_tools(self) -> List[BaseTool]:
        """Create context management tools."""
        tools = []
        
        async def get_session_context_func(session_id: str):
            """Get context for a session."""
            try:
                session = await self.shared_services.context.get_session(session_id)
                if session:
                    return {"session_id": session.session_id, "user_id": session.user_id, "state": session.state_data}
                return "Session not found"
            except Exception as e:
                return f"Error getting session context: {str(e)}"
        
        async def store_shared_context_func(name: str, data: Dict, scope: str = "session"):
            """Store shared context data."""
            try:
                context_id = await self.shared_services.context.store_shared_context(
                    name=name,
                    data=data,
                    scope=scope
                )
                return f"Shared context stored with ID: {context_id}"
            except Exception as e:
                return f"Error storing shared context: {str(e)}"
        
        tools.extend([
            StructuredTool.from_function(
                func=get_session_context_func,
                name="get_session_context",
                description="Get context information for a session"
            ),
            StructuredTool.from_function(
                func=store_shared_context_func,
                name="store_shared_context",
                description="Store data in shared context for cross-agent access"
            )
        ])
        
        return tools
    
    def _create_prompt_tools(self) -> List[BaseTool]:
        """Create prompt management tools."""
        tools = []
        
        async def get_prompt_func(name: str):
            """Get a prompt template by name."""
            try:
                prompt = await self.shared_services.prompt.get_prompt_by_name(name)
                if prompt:
                    return {"id": prompt.prompt_id, "name": prompt.name, "content": prompt.content}
                return "Prompt not found"
            except Exception as e:
                return f"Error getting prompt: {str(e)}"
        
        async def render_prompt_func(prompt_id: str, variables: Dict):
            """Render a prompt template with variables."""
            try:
                rendered = await self.shared_services.prompt.render_prompt(prompt_id, variables)
                return rendered
            except Exception as e:
                return f"Error rendering prompt: {str(e)}"
        
        tools.extend([
            StructuredTool.from_function(
                func=get_prompt_func,
                name="get_prompt_template",
                description="Get a prompt template by name"
            ),
            StructuredTool.from_function(
                func=render_prompt_func,
                name="render_prompt",
                description="Render a prompt template with variables"
            )
        ])
        
        return tools
    
    def get_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


class LangGraphWorkflowBuilder:
    """Builder for creating LangGraph workflows with shared services integration."""
    
    def __init__(self, shared_services: SharedServices):
        self.shared_services = shared_services
        self.toolkit = SharedServicesToolkit(shared_services)
        self.workflows: Dict[str, Any] = {}
    
    def create_collaborative_workflow(
        self,
        workflow_name: str,
        agents: List[Dict[str, Any]],
        workflow_type: str = "sequential"
    ):
        """Create a collaborative workflow with multiple agents."""
        
        # Create the state graph
        workflow = StateGraph(LangGraphState)
        
        # Add agent nodes
        for agent_config in agents:
            agent_id = agent_config["agent_id"]
            node_func = self._create_agent_node(agent_config)
            workflow.add_node(agent_id, node_func)
        
        # Add shared services nodes
        workflow.add_node("initialize_context", self._initialize_context_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        # Define workflow structure based on type
        if workflow_type == "sequential":
            self._build_sequential_workflow(workflow, agents)
        elif workflow_type == "parallel":
            self._build_parallel_workflow(workflow, agents)
        elif workflow_type == "conditional":
            self._build_conditional_workflow(workflow, agents)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        self.workflows[workflow_name] = compiled_workflow
        
        logger.info(f"Created {workflow_type} workflow: {workflow_name}")
        return compiled_workflow
    
    def _create_agent_node(self, agent_config: Dict[str, Any]):
        """Create a node function for an agent."""
        agent_id = agent_config["agent_id"]
        agent_type = agent_config.get("type", "generic")
        
        async def agent_node(state: LangGraphState) -> LangGraphState:
            try:
                # Prepare input for the agent using standardized communication
                agent_input = create_standard_input(
                    data={
                        "messages": [msg.content for msg in state.messages[-3:]], # Last 3 messages for context
                        "shared_context": state.shared_context,
                        "agent_outputs": state.agent_outputs
                    },
                    user_id=state.user_id,
                    session_id=state.session_id,
                    agent_id=agent_id,
                    priority=state.priority
                )
                
                # Execute agent logic (this would integrate with actual agents)
                result = await self._execute_agent_logic(agent_id, agent_input, agent_config)
                
                # Update state with agent output
                state.set_agent_output(agent_id, result.data)
                
                # Add agent response as a message
                if result.success and result.data:
                    response_content = str(result.data)
                    state.add_message(AIMessage(content=response_content, name=agent_id))
                
                # Store important results in shared services
                await self._store_agent_results(agent_id, result, state)
                
                return state
                
            except Exception as e:
                state.add_error(f"Agent execution failed: {str(e)}", agent_id)
                return state
        
        return agent_node
    
    async def _execute_agent_logic(
        self,
        agent_id: str,
        agent_input: StandardAgentInput,
        agent_config: Dict[str, Any]
    ) -> StandardAgentOutput:
        """Execute agent logic with shared services integration."""
        
        try:
            # This is where you would integrate with your actual agent implementations
            # For now, we'll simulate agent processing
            
            # Use shared services based on agent type
            agent_type = agent_config.get("type", "generic")
            
            if agent_type == "analytics":
                result = await self._execute_analytics_agent(agent_input)
            elif agent_type == "compliance":
                result = await self._execute_compliance_agent(agent_input)
            elif agent_type == "data_quality":
                result = await self._execute_data_quality_agent(agent_input)
            else:
                result = await self._execute_generic_agent(agent_input)
            
            return create_standard_output(
                request_id=agent_input.request_id,
                agent_id=agent_id,
                data=result,
                success=True
            )
            
        except Exception as e:
            return create_error_output(
                request_id=agent_input.request_id,
                agent_id=agent_id,
                error_code="AGENT_EXECUTION_ERROR",
                error_message=str(e)
            )
    
    async def _execute_analytics_agent(self, agent_input: StandardAgentInput) -> Dict[str, Any]:
        """Execute analytics agent logic."""
        results = {"agent_type": "analytics", "analysis": "performed"}
        
        # Use RAG to get relevant documents
        if self.shared_services.is_feature_available("rag"):
            docs = await self.shared_services.rag.search(
                query="analytics best practices",
                top_k=3
            )
            results["relevant_docs"] = len(docs)
        
        # Store analysis in memory
        if self.shared_services.is_feature_available("memory"):
            await self.shared_services.memory.store_memory(
                content="Analytics agent executed successfully",
                memory_type="episodic",
                metadata={"agent_type": "analytics"}
            )
        
        return results
    
    async def _execute_compliance_agent(self, agent_input: StandardAgentInput) -> Dict[str, Any]:
        """Execute compliance agent logic."""
        results = {"agent_type": "compliance", "compliance_check": "passed"}
        
        # Check knowledge base for compliance rules
        if self.shared_services.is_feature_available("knowledge"):
            entities = await self.shared_services.knowledge.search_entities(
                query="compliance rules",
                limit=5
            )
            results["compliance_rules_found"] = len(entities)
        
        return results
    
    async def _execute_data_quality_agent(self, agent_input: StandardAgentInput) -> Dict[str, Any]:
        """Execute data quality agent logic."""
        results = {"agent_type": "data_quality", "quality_score": 85}
        
        # Use tools for data quality assessment
        if self.shared_services.is_feature_available("tools"):
            # This would use registered data quality tools
            results["tools_available"] = True
        
        return results
    
    async def _execute_generic_agent(self, agent_input: StandardAgentInput) -> Dict[str, Any]:
        """Execute generic agent logic."""
        return {
            "agent_type": "generic",
            "processed": True,
            "message_count": len(agent_input.data.get("messages", []))
        }
    
    async def _store_agent_results(
        self,
        agent_id: str,
        result: StandardAgentOutput,
        state: LangGraphState
    ):
        """Store agent results in shared services."""
        
        # Store in memory if available
        if self.shared_services.is_feature_available("memory") and result.success:
            await self.shared_services.memory.store_memory(
                content=f"Agent {agent_id} completed: {str(result.data)[:200]}",
                memory_type="episodic",
                metadata={
                    "agent_id": agent_id,
                    "workflow_id": state.workflow_id,
                    "session_id": state.session_id
                }
            )
        
        # Update shared context
        if self.shared_services.is_feature_available("context"):
            state.shared_context[f"{agent_id}_result"] = result.data
    
    def _initialize_context_node(self, state: LangGraphState) -> LangGraphState:
        """Initialize workflow context with shared services."""
        state.add_message(SystemMessage(content="Initializing collaborative workflow"))
        
        # Load relevant context from shared services
        if state.session_id and self.shared_services.is_feature_available("context"):
            # This would load session context
            state.shared_context["workflow_initialized"] = True
        
        return state
    
    def _finalize_results_node(self, state: LangGraphState) -> LangGraphState:
        """Finalize workflow results and store in shared services."""
        
        # Compile final results
        final_results = {
            "workflow_id": state.workflow_id,
            "session_id": state.session_id,
            "executed_agents": state.executed_agents,
            "agent_outputs": state.agent_outputs,
            "step_count": state.step_count,
            "duration": (state.last_activity - state.started_at).total_seconds(),
            "errors": state.errors,
            "warnings": state.warnings
        }
        
        state.add_message(AIMessage(content=f"Workflow completed. Executed {len(state.executed_agents)} agents."))
        
        # Store final results
        if self.shared_services.is_feature_available("memory"):
            asyncio.create_task(self.shared_services.memory.store_memory(
                content=f"Workflow {state.workflow_id} completed successfully",
                memory_type="episodic",
                metadata=final_results
            ))
        
        return state
    
    def _build_sequential_workflow(self, workflow: StateGraph, agents: List[Dict[str, Any]]):
        """Build a sequential workflow."""
        workflow.add_edge(START, "initialize_context")
        
        prev_node = "initialize_context"
        for agent_config in agents:
            agent_id = agent_config["agent_id"]
            workflow.add_edge(prev_node, agent_id)
            prev_node = agent_id
        
        workflow.add_edge(prev_node, "finalize_results")
        workflow.add_edge("finalize_results", END)
    
    def _build_parallel_workflow(self, workflow: StateGraph, agents: List[Dict[str, Any]]):
        """Build a parallel workflow."""
        workflow.add_edge(START, "initialize_context")
        
        # All agents run in parallel after initialization
        for agent_config in agents:
            agent_id = agent_config["agent_id"]
            workflow.add_edge("initialize_context", agent_id)
            workflow.add_edge(agent_id, "finalize_results")
        
        workflow.add_edge("finalize_results", END)
    
    def _build_conditional_workflow(self, workflow: StateGraph, agents: List[Dict[str, Any]]):
        """Build a conditional workflow with routing logic."""
        workflow.add_edge(START, "initialize_context")
        
        # Add routing function
        def route_next_agent(state: LangGraphState) -> str:
            # Simple routing logic - can be made more sophisticated
            if not state.executed_agents:
                return agents[0]["agent_id"]
            elif len(state.executed_agents) < len(agents):
                return agents[len(state.executed_agents)]["agent_id"]
            else:
                return "finalize_results"
        
        # Add conditional edges
        agent_ids = [agent["agent_id"] for agent in agents]
        routing_map = {agent_id: agent_id for agent_id in agent_ids}
        routing_map["finalize_results"] = "finalize_results"
        
        workflow.add_conditional_edges(
            "initialize_context",
            route_next_agent,
            routing_map
        )
        
        for agent_config in agents:
            agent_id = agent_config["agent_id"]
            workflow.add_conditional_edges(
                agent_id,
                route_next_agent,
                routing_map
            )
        
        workflow.add_edge("finalize_results", END)
    
    async def execute_workflow(
        self,
        workflow_name: str,
        initial_message: str,
        user_id: str = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Execute a workflow with the given input."""
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]
        
        # Create initial state
        initial_state = LangGraphState(
            messages=[HumanMessage(content=initial_message)],
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4())
        )
        
        try:
            # Execute the workflow
            result = await workflow.ainvoke(initial_state)
            
            return {
                "workflow_id": result.workflow_id,
                "session_id": result.session_id,
                "messages": [{"type": type(msg).__name__, "content": msg.content} for msg in result.messages],
                "agent_outputs": result.agent_outputs,
                "executed_agents": result.executed_agents,
                "step_count": result.step_count,
                "duration": (result.last_activity - result.started_at).total_seconds(),
                "errors": result.errors,
                "warnings": result.warnings,
                "success": len(result.errors) == 0
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "workflow_id": initial_state.workflow_id,
                "session_id": initial_state.session_id,
                "error": str(e),
                "success": False
            }
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflows."""
        return list(self.workflows.keys())


class LangChainIntegrationManager:
    """Main manager for LangChain/LangGraph integration with shared services."""
    
    def __init__(self, shared_services: SharedServices):
        self.shared_services = shared_services
        self.toolkit = SharedServicesToolkit(shared_services)
        self.workflow_builder = LangGraphWorkflowBuilder(shared_services)
        self.callback_handler = SharedServicesCallback(shared_services)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the integration manager."""
        if self._initialized:
            return
        
        # Ensure shared services are initialized
        if not self.shared_services.is_initialized():
            await self.shared_services.initialize()
        
        # Initialize toolkit
        self.toolkit._initialize_tools()
        
        # Create default workflows
        await self._create_default_workflows()
        
        self._initialized = True
        logger.info("LangChain integration manager initialized")
    
    async def _create_default_workflows(self):
        """Create default workflows for common use cases."""
        
        # Analytics workflow
        analytics_agents = [
            {"agent_id": "data_collector", "type": "data_collection"},
            {"agent_id": "analytics_processor", "type": "analytics"},
            {"agent_id": "insights_generator", "type": "insights"},
            {"agent_id": "report_generator", "type": "reporting"}
        ]
        
        self.workflow_builder.create_collaborative_workflow(
            "analytics_pipeline",
            analytics_agents,
            "sequential"
        )
        
        # Compliance check workflow
        compliance_agents = [
            {"agent_id": "data_scanner", "type": "data_quality"},
            {"agent_id": "compliance_checker", "type": "compliance"},
            {"agent_id": "remediation_planner", "type": "remediation"}
        ]
        
        self.workflow_builder.create_collaborative_workflow(
            "compliance_check",
            compliance_agents,
            "sequential"
        )
        
        # Parallel analysis workflow
        parallel_agents = [
            {"agent_id": "quality_assessor", "type": "data_quality"},
            {"agent_id": "security_scanner", "type": "security"},
            {"agent_id": "performance_analyzer", "type": "performance"}
        ]
        
        self.workflow_builder.create_collaborative_workflow(
            "parallel_analysis",
            parallel_agents,
            "parallel"
        )
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """Get LangChain tools for use in agents."""
        return self.toolkit.get_tools()
    
    def get_callback_handler(self) -> BaseCallbackHandler:
        """Get callback handler for LangChain integration."""
        return self.callback_handler
    
    async def execute_workflow(
        self,
        workflow_name: str,
        message: str,
        user_id: str = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Execute a LangGraph workflow."""
        return await self.workflow_builder.execute_workflow(
            workflow_name=workflow_name,
            initial_message=message,
            user_id=user_id,
            session_id=session_id
        )
    
    def create_custom_workflow(
        self,
        workflow_name: str,
        agents: List[Dict[str, Any]],
        workflow_type: str = "sequential"
    ):
        """Create a custom workflow."""
        return self.workflow_builder.create_collaborative_workflow(
            workflow_name=workflow_name,
            agents=agents,
            workflow_type=workflow_type
        )
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflows."""
        return self.workflow_builder.get_available_workflows()
    
    async def shutdown(self):
        """Shutdown the integration manager."""
        logger.info("Shutting down LangChain integration manager")
        self._initialized = False


# Factory function for creating integration manager
async def create_langchain_integration(
    shared_services: SharedServices = None,
    config: Dict[str, Any] = None
) -> LangChainIntegrationManager:
    """
    Create and initialize LangChain integration manager.
    
    Args:
        shared_services: Shared services instance
        config: Configuration for integration
        
    Returns:
        Initialized LangChainIntegrationManager
    """
    if shared_services is None:
        from core.shared import create_shared_services
        shared_services = await create_shared_services(config)
    
    manager = LangChainIntegrationManager(shared_services)
    await manager.initialize()
    
    return manager
