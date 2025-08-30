"""
Orchestration layer for managing agent workflows and interactions.
Uses LangGraph for complex multi-agent workflows and state management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime
from enum import Enum
import logging

from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor

from .agent_base import BaseAgent, AgentMessage, AgentTask, AgentStatus, agent_registry
from .config import settings

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowState:
    """State management for agent workflows."""
    
    def __init__(self, workflow_id: str, initial_data: Dict[str, Any]):
        self.workflow_id = workflow_id
        self.data = initial_data
        self.messages: List[AgentMessage] = []
        self.tasks: List[AgentTask] = []
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        
    def update_data(self, updates: Dict[str, Any]):
        """Update workflow data."""
        self.data.update(updates)
        self.updated_at = datetime.utcnow()
    
    def add_message(self, message: AgentMessage):
        """Add a message to the workflow."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def add_task(self, task: AgentTask):
        """Add a task to the workflow."""
        self.tasks.append(task)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "data": self.data,
            "messages": [msg.to_dict() for msg in self.messages],
            "tasks": [{"id": task.id, "status": task.status.value} for task in self.tasks],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error": self.error,
            "result": self.result
        }


class AgentOrchestrator:
    """
    Orchestrates multi-agent workflows using LangGraph.
    Manages agent communication, task distribution, and workflow execution.
    """
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowState] = {}
        self.active_workflows: Set[str] = set()
        self.workflow_graphs: Dict[str, StateGraph] = {}
        
        # Initialize predefined workflows
        self._initialize_workflows()
    
    def _initialize_workflows(self):
        """Initialize predefined workflow templates."""
        # Data Quality Assessment Workflow
        self.register_workflow(
            "data_quality_assessment",
            self._create_data_quality_workflow()
        )
        
        # Compliance Check Workflow
        self.register_workflow(
            "compliance_check",
            self._create_compliance_workflow()
        )
        
        # Data Remediation Workflow
        self.register_workflow(
            "data_remediation",
            self._create_remediation_workflow()
        )
        
        # Policy Suggestion Workflow
        self.register_workflow(
            "policy_suggestion",
            self._create_policy_workflow()
        )
    
    def _create_data_quality_workflow(self) -> StateGraph:
        """Create data quality assessment workflow."""
        def quality_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Data quality check node."""
            data_quality_agent = agent_registry.get_agent("data_quality")
            if not data_quality_agent:
                raise ValueError("Data Quality Agent not found")
            
            # Create and execute quality check task
            task = AgentTask(
                task_type="quality_assessment",
                data=state.get("input_data", {}),
                priority=2
            )
            
            # This would be executed asynchronously in real implementation
            state["quality_results"] = {
                "issues_found": [],
                "metrics": {},
                "recommendations": []
            }
            
            return state
        
        def analytics_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Analytics generation node."""
            analytics_agent = agent_registry.get_agent("analytics")
            if not analytics_agent:
                raise ValueError("Analytics Agent not found")
            
            # Generate quality metrics
            state["analytics"] = {
                "charts": [],
                "reports": [],
                "dashboard_data": {}
            }
            
            return state
        
        def remediation_router(state: Dict[str, Any]) -> str:
            """Route to remediation if issues found."""
            quality_results = state.get("quality_results", {})
            issues = quality_results.get("issues_found", [])
            
            if issues:
                return "remediation"
            else:
                return "complete"
        
        # Create workflow graph
        workflow = StateGraph(dict)
        
        workflow.add_node("quality_check", quality_check_node)
        workflow.add_node("analytics", analytics_node)
        workflow.add_node("remediation", self._remediation_node)
        workflow.add_node("complete", self._completion_node)
        
        # Define edges
        workflow.add_edge("quality_check", "analytics")
        workflow.add_conditional_edges(
            "analytics",
            remediation_router,
            {"remediation": "remediation", "complete": "complete"}
        )
        workflow.add_edge("remediation", "complete")
        
        # Set entry point
        workflow.set_entry_point("quality_check")
        
        return workflow.compile()
    
    def _create_compliance_workflow(self) -> StateGraph:
        """Create compliance check workflow."""
        def compliance_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Compliance check node."""
            compliance_agent = agent_registry.get_agent("data_privacy_compliance")
            if not compliance_agent:
                raise ValueError("Compliance Agent not found")
            
            state["compliance_results"] = {
                "violations": [],
                "risk_level": "low",
                "recommendations": []
            }
            
            return state
        
        def violation_router(state: Dict[str, Any]) -> str:
            """Route based on violations found."""
            violations = state.get("compliance_results", {}).get("violations", [])
            
            if violations:
                return "remediation"
            else:
                return "complete"
        
        workflow = StateGraph(dict)
        
        workflow.add_node("compliance_check", compliance_check_node)
        workflow.add_node("remediation", self._remediation_node)
        workflow.add_node("complete", self._completion_node)
        
        workflow.add_conditional_edges(
            "compliance_check",
            violation_router,
            {"remediation": "remediation", "complete": "complete"}
        )
        workflow.add_edge("remediation", "complete")
        
        workflow.set_entry_point("compliance_check")
        
        return workflow.compile()
    
    def _create_remediation_workflow(self) -> StateGraph:
        """Create data remediation workflow."""
        def remediation_planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Plan remediation actions."""
            remediation_agent = agent_registry.get_agent("data_remediation")
            if not remediation_agent:
                raise ValueError("Remediation Agent not found")
            
            state["remediation_plan"] = {
                "actions": [],
                "priority": "high",
                "estimated_time": "2 hours"
            }
            
            return state
        
        def execute_remediation_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute remediation actions."""
            state["remediation_results"] = {
                "actions_completed": [],
                "success_rate": 0.95,
                "remaining_issues": []
            }
            
            return state
        
        workflow = StateGraph(dict)
        
        workflow.add_node("plan_remediation", remediation_planning_node)
        workflow.add_node("execute_remediation", execute_remediation_node)
        workflow.add_node("analytics", self._analytics_node)
        workflow.add_node("complete", self._completion_node)
        
        workflow.add_edge("plan_remediation", "execute_remediation")
        workflow.add_edge("execute_remediation", "analytics")
        workflow.add_edge("analytics", "complete")
        
        workflow.set_entry_point("plan_remediation")
        
        return workflow.compile()
    
    def _create_policy_workflow(self) -> StateGraph:
        """Create policy suggestion workflow."""
        def policy_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze data for policy suggestions."""
            policy_agent = agent_registry.get_agent("policy_suggestion")
            if not policy_agent:
                raise ValueError("Policy Suggestion Agent not found")
            
            state["policy_suggestions"] = {
                "validation_rules": [],
                "data_policies": [],
                "governance_recommendations": []
            }
            
            return state
        
        workflow = StateGraph(dict)
        
        workflow.add_node("policy_analysis", policy_analysis_node)
        workflow.add_node("complete", self._completion_node)
        
        workflow.add_edge("policy_analysis", "complete")
        workflow.set_entry_point("policy_analysis")
        
        return workflow.compile()
    
    def _remediation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generic remediation node."""
        remediation_agent = agent_registry.get_agent("data_remediation")
        if remediation_agent:
            # Execute remediation
            pass
        
        return state
    
    def _analytics_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generic analytics node."""
        analytics_agent = agent_registry.get_agent("analytics")
        if analytics_agent:
            # Generate analytics
            pass
        
        return state
    
    def _completion_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generic completion node."""
        state["completed"] = True
        state["completion_time"] = datetime.utcnow().isoformat()
        return state
    
    def register_workflow(self, workflow_name: str, workflow_graph: StateGraph):
        """Register a new workflow template."""
        self.workflow_graphs[workflow_name] = workflow_graph
        logger.info(f"Registered workflow: {workflow_name}")
    
    async def start_workflow(
        self,
        workflow_name: str,
        input_data: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> str:
        """Start a new workflow execution."""
        if workflow_name not in self.workflow_graphs:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        # Create workflow state
        workflow_id = workflow_id or f"{workflow_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        state = WorkflowState(workflow_id, input_data)
        state.status = WorkflowStatus.RUNNING
        
        self.workflows[workflow_id] = state
        self.active_workflows.add(workflow_id)
        
        # Execute workflow asynchronously
        asyncio.create_task(self._execute_workflow(workflow_name, workflow_id))
        
        logger.info(f"Started workflow {workflow_name} with ID: {workflow_id}")
        return workflow_id
    
    async def _execute_workflow(self, workflow_name: str, workflow_id: str):
        """Execute a workflow asynchronously."""
        try:
            workflow_graph = self.workflow_graphs[workflow_name]
            state = self.workflows[workflow_id]
            
            # Execute the workflow
            result = await workflow_graph.ainvoke(state.data)
            
            # Update state
            state.result = result
            state.status = WorkflowStatus.COMPLETED
            self.active_workflows.discard(workflow_id)
            
            logger.info(f"Completed workflow {workflow_id}")
            
        except Exception as e:
            state = self.workflows[workflow_id]
            state.status = WorkflowStatus.FAILED
            state.error = str(e)
            self.active_workflows.discard(workflow_id)
            
            logger.error(f"Workflow {workflow_id} failed: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status."""
        workflow = self.workflows.get(workflow_id)
        return workflow.to_dict() if workflow else None
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        return [workflow.to_dict() for workflow in self.workflows.values()]
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.workflows.get(workflow_id)
            if workflow:
                workflow.status = WorkflowStatus.CANCELLED
                self.active_workflows.discard(workflow_id)
                logger.info(f"Cancelled workflow {workflow_id}")
                return True
        
        return False
    
    async def send_agent_message(
        self,
        sender_name: str,
        recipient_name: str,
        message_type: str,
        content: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> bool:
        """Send a message between agents."""
        sender = agent_registry.get_agent(sender_name)
        recipient = agent_registry.get_agent(recipient_name)
        
        if not sender or not recipient:
            logger.error(f"Agent not found: sender={sender_name}, recipient={recipient_name}")
            return False
        
        message = AgentMessage(
            sender=sender_name,
            recipient=recipient_name,
            message_type=message_type,
            content=content
        )
        
        # Add to workflow if specified
        if workflow_id and workflow_id in self.workflows:
            self.workflows[workflow_id].add_message(message)
        
        # Deliver message
        recipient.add_message(message)
        
        logger.info(f"Delivered message from {sender_name} to {recipient_name}")
        return True


# Global orchestrator instance
orchestrator = AgentOrchestrator()
