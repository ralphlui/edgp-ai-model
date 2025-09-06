"""
Tool Management System

Central management system for AI agent tools including registration,
discovery, execution, workflow orchestration, and analytics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
import uuid

from .types import (
    ToolDefinition, ToolExecution, ToolResult, ToolWorkflow, ToolRegistration,
    ToolConfig, ToolAnalytics, ToolStatus, ToolType, BaseToolHandler,
    ToolSecurityLevel, ToolExecutionMode
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tool definitions and registrations."""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.registrations: Dict[str, ToolRegistration] = {}
        self.handlers: Dict[str, BaseToolHandler] = {}
        self.tool_index: Dict[str, Set[str]] = defaultdict(set)  # Tag -> tool_ids
        
    def register_tool(
        self,
        tool_definition: ToolDefinition,
        handler: Optional[BaseToolHandler] = None,
        agent_id: Optional[str] = None
    ) -> str:
        """Register a new tool."""
        tool_id = tool_definition.tool_id
        
        # Store tool definition
        self.tools[tool_id] = tool_definition
        
        # Store handler if provided
        if handler:
            self.handlers[tool_id] = handler
        
        # Create registration record
        registration = ToolRegistration(
            tool_id=tool_id,
            agent_id=agent_id,
            handler=handler.__class__.__name__ if handler else None
        )
        self.registrations[registration.registration_id] = registration
        
        # Update index
        self._update_tool_index(tool_definition)
        
        logger.info("Registered tool: %s (ID: %s)", tool_definition.name, tool_id)
        return tool_id
    
    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool."""
        if tool_id not in self.tools:
            return False
        
        tool_definition = self.tools[tool_id]
        
        # Remove from registry
        del self.tools[tool_id]
        self.handlers.pop(tool_id, None)
        
        # Remove registrations
        to_remove = []
        for reg_id, registration in self.registrations.items():
            if registration.tool_id == tool_id:
                to_remove.append(reg_id)
        
        for reg_id in to_remove:
            del self.registrations[reg_id]
        
        # Update index
        self._remove_from_index(tool_definition)
        
        logger.info("Unregistered tool: %s (ID: %s)", tool_definition.name, tool_id)
        return True
    
    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool definition by ID."""
        return self.tools.get(tool_id)
    
    def get_handler(self, tool_id: str) -> Optional[BaseToolHandler]:
        """Get tool handler by ID."""
        return self.handlers.get(tool_id)
    
    def list_tools(
        self,
        tool_type: Optional[ToolType] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> List[ToolDefinition]:
        """List tools with optional filtering."""
        tools = list(self.tools.values())
        
        # Filter by type
        if tool_type:
            tools = [t for t in tools if t.tool_type == tool_type]
        
        # Filter by category
        if category:
            tools = [t for t in tools if t.category == category]
        
        # Filter by tags
        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]
        
        return tools
    
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by name, description, or tags."""
        query = query.lower()
        results = []
        
        for tool in self.tools.values():
            if (query in tool.name.lower() or
                query in tool.description.lower() or
                any(query in tag.lower() for tag in tool.tags)):
                results.append(tool)
        
        return results
    
    def get_tools_by_tag(self, tag: str) -> List[ToolDefinition]:
        """Get tools by tag."""
        tool_ids = self.tool_index.get(tag, set())
        return [self.tools[tool_id] for tool_id in tool_ids if tool_id in self.tools]
    
    def _update_tool_index(self, tool: ToolDefinition):
        """Update tool index with tags."""
        for tag in tool.tags:
            self.tool_index[tag].add(tool.tool_id)
        
        # Index by type and category
        self.tool_index[tool.tool_type.value].add(tool.tool_id)
        if tool.category:
            self.tool_index[tool.category].add(tool.tool_id)
    
    def _remove_from_index(self, tool: ToolDefinition):
        """Remove tool from index."""
        for tag in tool.tags:
            self.tool_index[tag].discard(tool.tool_id)
        
        self.tool_index[tool.tool_type.value].discard(tool.tool_id)
        if tool.category:
            self.tool_index[tool.category].discard(tool.tool_id)


class ToolExecutor:
    """Manages tool execution and lifecycle."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.executions: Dict[str, ToolExecution] = {}
        self.execution_queue: deque = deque()
        self.running_executions: Set[str] = set()
        self.execution_history: List[ToolExecution] = []
        
        # Background tasks
        self._executor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the executor."""
        # Start background tasks
        self._executor_task = asyncio.create_task(self._execution_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("ToolExecutor initialized")
    
    async def execute_tool(
        self,
        tool_definition: ToolDefinition,
        parameters: Dict[str, Any],
        handler: Optional[BaseToolHandler] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> ToolExecution:
        """Execute a tool."""
        # Create execution record
        execution = ToolExecution(
            tool_id=tool_definition.tool_id,
            tool_name=tool_definition.name,
            parameters=parameters,
            priority=priority,
            agent_id=context.get("agent_id") if context else None,
            session_id=context.get("session_id") if context else None,
            metadata=context or {}
        )
        
        # Validate parameters
        validation_errors = tool_definition.validate_parameters(parameters)
        if validation_errors:
            execution.update_status(
                ToolStatus.FAILED,
                ToolResult(
                    success=False,
                    error=f"Parameter validation failed: {'; '.join(validation_errors)}"
                )
            )
            return execution
        
        # Store execution
        self.executions[execution.execution_id] = execution
        
        # Queue for execution or execute immediately
        if tool_definition.execution_mode == ToolExecutionMode.SYNCHRONOUS:
            await self._execute_immediately(execution, tool_definition, handler)
        else:
            self._queue_execution(execution, tool_definition, handler)
        
        return execution
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False
        
        if execution.status == ToolStatus.RUNNING:
            execution.update_status(ToolStatus.CANCELLED)
            self.running_executions.discard(execution_id)
            return True
        
        return False
    
    def get_execution(self, execution_id: str) -> Optional[ToolExecution]:
        """Get execution by ID."""
        return self.executions.get(execution_id)
    
    def list_executions(
        self,
        status: Optional[ToolStatus] = None,
        tool_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[ToolExecution]:
        """List executions with optional filtering."""
        executions = list(self.executions.values())
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        if tool_id:
            executions = [e for e in executions if e.tool_id == tool_id]
        
        if agent_id:
            executions = [e for e in executions if e.agent_id == agent_id]
        
        return executions
    
    async def _execute_immediately(
        self,
        execution: ToolExecution,
        tool_definition: ToolDefinition,
        handler: Optional[BaseToolHandler]
    ):
        """Execute tool immediately."""
        if not handler:
            execution.update_status(
                ToolStatus.FAILED,
                ToolResult(success=False, error="No handler available for tool")
            )
            return
        
        execution.update_status(ToolStatus.RUNNING)
        self.running_executions.add(execution.execution_id)
        
        try:
            # Set timeout
            if tool_definition.timeout_seconds > 0:
                execution.timeout_at = datetime.now() + timedelta(seconds=tool_definition.timeout_seconds)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                handler.execute(execution.parameters, execution.metadata),
                timeout=tool_definition.timeout_seconds if tool_definition.timeout_seconds > 0 else None
            )
            
            execution.update_status(ToolStatus.COMPLETED, result)
            
            # Update tool statistics
            if result:
                tool_definition.update_usage_stats(
                    result.execution_time,
                    result.success
                )
        
        except asyncio.TimeoutError:
            execution.update_status(
                ToolStatus.TIMEOUT,
                ToolResult(success=False, error="Execution timed out")
            )
        
        except Exception as e:
            execution.update_status(
                ToolStatus.FAILED,
                ToolResult(success=False, error=str(e))
            )
        
        finally:
            self.running_executions.discard(execution.execution_id)
            
            # Move to history if enabled
            if self.config.enable_execution_history:
                self.execution_history.append(execution)
    
    def _queue_execution(
        self,
        execution: ToolExecution,
        tool_definition: ToolDefinition,
        handler: Optional[BaseToolHandler]
    ):
        """Queue execution for background processing."""
        self.execution_queue.append((execution, tool_definition, handler))
    
    async def _execution_loop(self):
        """Background execution loop."""
        while True:
            try:
                if (len(self.running_executions) < self.config.max_concurrent_executions and
                    self.execution_queue):
                    
                    execution, tool_definition, handler = self.execution_queue.popleft()
                    
                    # Execute in background
                    asyncio.create_task(
                        self._execute_immediately(execution, tool_definition, handler)
                    )
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in execution loop: %s", str(e))
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old executions
                cutoff = datetime.now() - timedelta(days=self.config.execution_history_retention_days)
                
                # Remove from main storage
                to_remove = []
                for exec_id, execution in self.executions.items():
                    if execution.created_at < cutoff and execution.is_completed():
                        to_remove.append(exec_id)
                
                for exec_id in to_remove:
                    del self.executions[exec_id]
                
                # Clean up history
                self.execution_history = [
                    e for e in self.execution_history 
                    if e.created_at >= cutoff
                ]
                
                logger.info("Cleaned up %d old executions", len(to_remove))
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop: %s", str(e))
    
    async def shutdown(self):
        """Shutdown the executor."""
        # Cancel background tasks
        for task in [self._executor_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel running executions
        for execution_id in list(self.running_executions):
            await self.cancel_execution(execution_id)
        
        logger.info("ToolExecutor shutdown complete")


class WorkflowEngine:
    """Orchestrates tool workflows."""
    
    def __init__(self, tool_manager):
        self.tool_manager = tool_manager
        self.workflows: Dict[str, ToolWorkflow] = {}
        self.workflow_executions: Dict[str, Dict[str, Any]] = {}
    
    def register_workflow(self, workflow: ToolWorkflow) -> str:
        """Register a workflow."""
        self.workflows[workflow.workflow_id] = workflow
        logger.info("Registered workflow: %s (ID: %s)", workflow.name, workflow.workflow_id)
        return workflow.workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[ToolWorkflow]:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution_id = str(uuid.uuid4())
        
        # Initialize workflow execution state
        execution_state = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "running",
            "current_step": None,
            "completed_steps": [],
            "step_results": {},
            "parameters": initial_parameters,
            "context": context or {},
            "started_at": datetime.now(),
            "error": None
        }
        
        self.workflow_executions[execution_id] = execution_state
        
        # Start workflow execution
        asyncio.create_task(self._execute_workflow_steps(workflow, execution_state))
        
        return execution_id
    
    async def _execute_workflow_steps(
        self,
        workflow: ToolWorkflow,
        execution_state: Dict[str, Any]
    ):
        """Execute workflow steps."""
        try:
            current_step = None
            
            while True:
                # Get next steps
                next_steps = workflow.get_next_steps(current_step)
                
                if not next_steps:
                    # Workflow completed
                    execution_state["status"] = "completed"
                    execution_state["completed_at"] = datetime.now()
                    break
                
                # Execute steps
                if workflow.parallel_execution and len(next_steps) > 1:
                    # Execute steps in parallel
                    tasks = []
                    for step in next_steps:
                        task = asyncio.create_task(
                            self._execute_workflow_step(step, execution_state)
                        )
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Check results
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            if workflow.stop_on_error:
                                raise result
                            else:
                                logger.error("Step failed but continuing: %s", str(result))
                else:
                    # Execute steps sequentially
                    for step in next_steps:
                        result = await self._execute_workflow_step(step, execution_state)
                        if not result and workflow.stop_on_error:
                            execution_state["status"] = "failed"
                            execution_state["error"] = f"Step {step['step_id']} failed"
                            return
                
                # Update current step
                if next_steps:
                    current_step = next_steps[-1]["step_id"]
                    execution_state["current_step"] = current_step
                    execution_state["completed_steps"].extend([s["step_id"] for s in next_steps])
        
        except Exception as e:
            execution_state["status"] = "failed"
            execution_state["error"] = str(e)
            execution_state["completed_at"] = datetime.now()
            logger.error("Workflow execution failed: %s", str(e))
    
    async def _execute_workflow_step(
        self,
        step: Dict[str, Any],
        execution_state: Dict[str, Any]
    ) -> bool:
        """Execute a single workflow step."""
        try:
            tool_id = step["tool_id"]
            parameters = step["parameters"]
            
            # Resolve parameter references
            resolved_parameters = self._resolve_parameters(parameters, execution_state)
            
            # Execute tool
            execution = await self.tool_manager.execute_tool(
                tool_id=tool_id,
                parameters=resolved_parameters,
                context=execution_state["context"]
            )
            
            # Wait for completion
            while not execution.is_completed():
                await asyncio.sleep(0.1)
            
            # Store result
            execution_state["step_results"][step["step_id"]] = {
                "execution_id": execution.execution_id,
                "success": execution.is_successful(),
                "result": execution.result.dict() if execution.result else None
            }
            
            return execution.is_successful()
        
        except Exception as e:
            execution_state["step_results"][step["step_id"]] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        execution_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter references in workflow step."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${"):
                # Parameter reference
                ref = value[2:-1]  # Remove ${ and }
                
                if ref.startswith("step."):
                    # Reference to previous step result
                    parts = ref.split(".")
                    step_id = parts[1]
                    result_path = ".".join(parts[2:]) if len(parts) > 2 else "result"
                    
                    step_result = execution_state["step_results"].get(step_id)
                    if step_result and step_result.get("success"):
                        resolved[key] = self._get_nested_value(
                            step_result.get("result", {}),
                            result_path
                        )
                    else:
                        resolved[key] = None
                
                elif ref.startswith("input."):
                    # Reference to initial parameters
                    param_name = ref[6:]  # Remove "input."
                    resolved[key] = execution_state["parameters"].get(param_name)
                
                else:
                    # Direct reference to execution state
                    resolved[key] = execution_state.get(ref)
            else:
                resolved[key] = value
        
        return resolved
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        if not path or path == "result":
            return data
        
        keys = path.split(".")
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current


class ToolManager:
    """Central tool management system."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(config)
        self.workflow_engine = WorkflowEngine(self) if config.enable_workflow_engine else None
        self.analytics = ToolAnalytics()
        
        # Background tasks
        self._analytics_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the tool manager."""
        await self.executor.initialize()
        
        # Start analytics collection
        if self.config.enable_metrics:
            self._analytics_task = asyncio.create_task(self._analytics_loop())
        
        logger.info("ToolManager initialized")
    
    # Tool registration methods
    def register_tool(
        self,
        tool_definition: ToolDefinition,
        handler: Optional[BaseToolHandler] = None,
        agent_id: Optional[str] = None
    ) -> str:
        """Register a tool."""
        return self.registry.register_tool(tool_definition, handler, agent_id)
    
    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool."""
        return self.registry.unregister_tool(tool_id)
    
    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool definition."""
        return self.registry.get_tool(tool_id)
    
    def list_tools(self, **filters) -> List[ToolDefinition]:
        """List tools with optional filtering."""
        return self.registry.list_tools(**filters)
    
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools."""
        return self.registry.search_tools(query)
    
    # Tool execution methods
    async def execute_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> ToolExecution:
        """Execute a tool."""
        tool_definition = self.registry.get_tool(tool_id)
        if not tool_definition:
            raise ValueError(f"Tool {tool_id} not found")
        
        handler = self.registry.get_handler(tool_id)
        
        return await self.executor.execute_tool(
            tool_definition, parameters, handler, context, priority
        )
    
    def get_execution(self, execution_id: str) -> Optional[ToolExecution]:
        """Get execution status."""
        return self.executor.get_execution(execution_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution."""
        return await self.executor.cancel_execution(execution_id)
    
    # Workflow methods
    def register_workflow(self, workflow: ToolWorkflow) -> str:
        """Register a workflow."""
        if not self.workflow_engine:
            raise ValueError("Workflow engine not enabled")
        return self.workflow_engine.register_workflow(workflow)
    
    async def execute_workflow(
        self,
        workflow_id: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a workflow."""
        if not self.workflow_engine:
            raise ValueError("Workflow engine not enabled")
        return await self.workflow_engine.execute_workflow(workflow_id, parameters, context)
    
    # Analytics methods
    async def get_analytics(self) -> ToolAnalytics:
        """Get tool analytics."""
        return self.analytics
    
    async def _analytics_loop(self):
        """Background analytics collection loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                await self._update_analytics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in analytics loop: %s", str(e))
    
    async def _update_analytics(self):
        """Update analytics data."""
        # Basic counts
        self.analytics.total_tools = len(self.registry.tools)
        
        # Execution metrics
        all_executions = list(self.executor.executions.values()) + self.executor.execution_history
        self.analytics.total_executions = len(all_executions)
        
        successful = [e for e in all_executions if e.is_successful()]
        failed = [e for e in all_executions if e.is_completed() and not e.is_successful()]
        
        self.analytics.successful_executions = len(successful)
        self.analytics.failed_executions = len(failed)
        
        # Success rate
        if self.analytics.total_executions > 0:
            self.analytics.overall_success_rate = (
                self.analytics.successful_executions / self.analytics.total_executions
            )
        
        # Average execution time
        completed_with_time = [e for e in all_executions if e.get_execution_time() is not None]
        if completed_with_time:
            self.analytics.average_execution_time = (
                sum(e.get_execution_time() for e in completed_with_time) / len(completed_with_time)
            )
        
        # Tool type distribution
        type_counts = defaultdict(int)
        for tool in self.registry.tools.values():
            type_counts[tool.tool_type.value] += 1
        self.analytics.tool_type_distribution = dict(type_counts)
        
        # Most used tools
        tool_usage = defaultdict(int)
        for execution in all_executions:
            tool_usage[execution.tool_id] += 1
        
        most_used = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        self.analytics.most_used_tools = [
            {"tool_id": tool_id, "usage_count": count}
            for tool_id, count in most_used
        ]
        
        # Time-based metrics
        now = datetime.now()
        today = now.date()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        self.analytics.executions_today = len([
            e for e in all_executions if e.created_at.date() == today
        ])
        
        self.analytics.executions_this_week = len([
            e for e in all_executions if e.created_at >= week_ago
        ])
        
        self.analytics.executions_this_month = len([
            e for e in all_executions if e.created_at >= month_ago
        ])
        
        self.analytics.last_updated = datetime.now()
    
    async def shutdown(self):
        """Shutdown the tool manager."""
        # Cancel analytics task
        if self._analytics_task and not self._analytics_task.done():
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown executor
        await self.executor.shutdown()
        
        logger.info("ToolManager shutdown complete")
