"""
Remediation Workflow Orchestrator

Central orchestrator for managing different remediation workflows
based on data characteristics, risk levels, and processing requirements.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Our imports
from .base_workflow import RemediationState, DataRemediationGraph
from .automated_workflow import AutomatedRemediationGraph, BatchRemediationGraph
from core.shared import SharedServices
from core.integrations.langchain_integration import SharedServicesToolkit

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Available remediation workflow types."""
    STANDARD = "standard"
    AUTOMATED = "automated"
    BATCH = "batch"
    CUSTOM = "custom"


class RiskLevel(Enum):
    """Risk levels for remediation operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkflowSelection:
    """Workflow selection criteria and configuration."""
    workflow_type: WorkflowType
    risk_level: RiskLevel
    reasons: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[int] = None  # minutes
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


class RemediationOrchestrator:
    """Central orchestrator for remediation workflows."""
    
    def __init__(self, shared_services: SharedServices):
        self.shared_services = shared_services
        self.available_workflows = {}
        self.workflow_history = []
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize the orchestrator with available workflows."""
        try:
            # Initialize all workflow types
            self.available_workflows = {
                WorkflowType.STANDARD: await self._create_standard_workflow(),
                WorkflowType.AUTOMATED: await self._create_automated_workflow(),
                WorkflowType.BATCH: await self._create_batch_workflow()
            }
            
            logger.info("Remediation orchestrator initialized with all workflow types")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _create_standard_workflow(self) -> DataRemediationGraph:
        """Create standard remediation workflow."""
        graph = DataRemediationGraph(self.shared_services)
        await graph.initialize()
        return graph
    
    async def _create_automated_workflow(self) -> AutomatedRemediationGraph:
        """Create automated remediation workflow."""
        graph = AutomatedRemediationGraph(self.shared_services)
        await graph.initialize()
        return graph
    
    async def _create_batch_workflow(self) -> BatchRemediationGraph:
        """Create batch remediation workflow."""
        graph = BatchRemediationGraph(self.shared_services)
        await graph.initialize()
        return graph
    
    async def analyze_and_select_workflow(
        self, 
        data_context: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> WorkflowSelection:
        """Analyze data context and select the most appropriate workflow."""
        try:
            # Analyze data characteristics
            data_analysis = await self._analyze_data_context(data_context)
            
            # Determine risk level
            risk_level = self._assess_risk_level(data_analysis)
            
            # Select workflow type
            workflow_type = self._select_workflow_type(data_analysis, risk_level, user_preferences)
            
            # Generate workflow configuration
            configuration = self._generate_workflow_config(workflow_type, data_analysis, risk_level)
            
            # Estimate resource requirements
            resource_requirements = self._estimate_resources(workflow_type, data_analysis)
            
            # Create selection object
            selection = WorkflowSelection(
                workflow_type=workflow_type,
                risk_level=risk_level,
                reasons=self._generate_selection_reasons(workflow_type, data_analysis, risk_level),
                configuration=configuration,
                estimated_duration=self._estimate_duration(workflow_type, data_analysis),
                resource_requirements=resource_requirements
            )
            
            logger.info(f"Selected {workflow_type.value} workflow for {risk_level.value} risk scenario")
            
            return selection
            
        except Exception as e:
            logger.error(f"Workflow selection failed: {e}")
            # Fallback to standard workflow
            return WorkflowSelection(
                workflow_type=WorkflowType.STANDARD,
                risk_level=RiskLevel.MEDIUM,
                reasons=["Fallback due to analysis error"],
                configuration={"fallback_mode": True}
            )
    
    async def _analyze_data_context(self, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data context to understand characteristics."""
        analysis = {
            "data_size": data_context.get("size_mb", 100),
            "record_count": data_context.get("record_count", 10000),
            "complexity": data_context.get("complexity", "medium"),
            "data_types": data_context.get("data_types", ["text", "numeric"]),
            "quality_score": data_context.get("current_quality", 75),
            "business_criticality": data_context.get("business_criticality", "medium"),
            "has_pii": data_context.get("contains_pii", False),
            "change_frequency": data_context.get("change_frequency", "daily"),
            "downstream_dependencies": data_context.get("dependencies", []),
            "compliance_requirements": data_context.get("compliance", [])
        }
        
        # Add derived metrics
        analysis["size_category"] = self._categorize_size(analysis["data_size"])
        analysis["complexity_score"] = self._calculate_complexity_score(analysis)
        analysis["impact_score"] = self._calculate_impact_score(analysis)
        
        return analysis
    
    def _assess_risk_level(self, analysis: Dict[str, Any]) -> RiskLevel:
        """Assess risk level based on data analysis."""
        risk_factors = []
        
        # Size-based risk
        if analysis["data_size"] > 1000:  # > 1GB
            risk_factors.append("large_dataset")
        
        # PII risk
        if analysis["has_pii"]:
            risk_factors.append("contains_pii")
        
        # Business criticality
        if analysis["business_criticality"] == "high":
            risk_factors.append("business_critical")
        
        # Compliance requirements
        if analysis["compliance_requirements"]:
            risk_factors.append("compliance_required")
        
        # Downstream dependencies
        if len(analysis["downstream_dependencies"]) > 3:
            risk_factors.append("many_dependencies")
        
        # Quality score
        if analysis["quality_score"] < 60:
            risk_factors.append("poor_quality")
        
        # Determine risk level
        risk_count = len(risk_factors)
        
        if risk_count >= 4:
            return RiskLevel.CRITICAL
        elif risk_count >= 3:
            return RiskLevel.HIGH
        elif risk_count >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _select_workflow_type(
        self, 
        analysis: Dict[str, Any], 
        risk_level: RiskLevel,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> WorkflowType:
        """Select the most appropriate workflow type."""
        
        # User preference override
        if user_preferences and "preferred_workflow" in user_preferences:
            return WorkflowType(user_preferences["preferred_workflow"])
        
        # Risk-based selection
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            return WorkflowType.STANDARD  # Full human oversight
        
        # Size-based selection
        if analysis["record_count"] > 100000:  # Large datasets
            return WorkflowType.BATCH
        
        # Complexity-based selection
        if (analysis["complexity_score"] < 30 and 
            risk_level == RiskLevel.LOW and 
            not analysis["has_pii"]):
            return WorkflowType.AUTOMATED
        
        # Default to standard workflow
        return WorkflowType.STANDARD
    
    def _generate_workflow_config(
        self, 
        workflow_type: WorkflowType, 
        analysis: Dict[str, Any], 
        risk_level: RiskLevel
    ) -> Dict[str, Any]:
        """Generate workflow-specific configuration."""
        base_config = {
            "risk_level": risk_level.value,
            "backup_required": risk_level != RiskLevel.LOW,
            "validation_level": "strict" if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else "standard",
            "approval_required": risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            "rollback_enabled": True,
            "monitoring_level": "detailed" if risk_level != RiskLevel.LOW else "basic"
        }
        
        if workflow_type == WorkflowType.AUTOMATED:
            base_config.update({
                "auto_execution": True,
                "human_intervention": False,
                "batch_size": min(1000, analysis["record_count"] // 10),
                "timeout_minutes": 30
            })
        
        elif workflow_type == WorkflowType.BATCH:
            base_config.update({
                "batch_size": 10000,
                "parallel_processing": analysis["data_size"] > 100,
                "checkpoint_frequency": "every_batch",
                "resource_scaling": "auto"
            })
        
        elif workflow_type == WorkflowType.STANDARD:
            base_config.update({
                "interactive_mode": True,
                "step_by_step_approval": risk_level == RiskLevel.CRITICAL,
                "detailed_logging": True,
                "user_notifications": True
            })
        
        return base_config
    
    def _generate_selection_reasons(
        self, 
        workflow_type: WorkflowType, 
        analysis: Dict[str, Any], 
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate reasons for workflow selection."""
        reasons = []
        
        if workflow_type == WorkflowType.AUTOMATED:
            reasons.append(f"Low risk level ({risk_level.value}) suitable for automation")
            reasons.append(f"Simple data structure (complexity score: {analysis['complexity_score']})")
            if not analysis["has_pii"]:
                reasons.append("No PII detected - safe for automated processing")
        
        elif workflow_type == WorkflowType.BATCH:
            reasons.append(f"Large dataset ({analysis['record_count']:,} records) benefits from batch processing")
            if analysis["data_size"] > 100:
                reasons.append(f"Dataset size ({analysis['data_size']} MB) requires batch optimization")
        
        elif workflow_type == WorkflowType.STANDARD:
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                reasons.append(f"High risk level ({risk_level.value}) requires careful oversight")
            if analysis["has_pii"]:
                reasons.append("PII detected - requires standard workflow with safeguards")
            if analysis["business_criticality"] == "high":
                reasons.append("Business critical data requires standard workflow")
        
        return reasons
    
    def _estimate_duration(self, workflow_type: WorkflowType, analysis: Dict[str, Any]) -> int:
        """Estimate workflow duration in minutes."""
        base_time = 15  # Base processing time
        
        # Size factor
        size_factor = min(5, analysis["data_size"] / 100)  # 1 minute per 100MB, max 5 minutes
        
        # Complexity factor
        complexity_factor = analysis["complexity_score"] / 20  # Up to 5 minutes for high complexity
        
        # Workflow type factor
        type_factors = {
            WorkflowType.AUTOMATED: 0.3,
            WorkflowType.BATCH: 1.5,
            WorkflowType.STANDARD: 1.0
        }
        
        estimated = (base_time + size_factor + complexity_factor) * type_factors[workflow_type]
        
        return int(estimated)
    
    def _estimate_resources(self, workflow_type: WorkflowType, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements."""
        return {
            "cpu_cores": 2 if workflow_type == WorkflowType.BATCH else 1,
            "memory_mb": min(2048, analysis["data_size"] * 2),
            "storage_mb": analysis["data_size"] * 3,  # Original + backup + working space
            "network_bandwidth": "standard",
            "priority": "high" if analysis["business_criticality"] == "high" else "normal"
        }
    
    def _categorize_size(self, size_mb: float) -> str:
        """Categorize dataset size."""
        if size_mb < 10:
            return "small"
        elif size_mb < 100:
            return "medium"
        elif size_mb < 1000:
            return "large"
        else:
            return "very_large"
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate complexity score (0-100)."""
        score = 20  # Base score
        
        # Data type complexity
        if len(analysis["data_types"]) > 3:
            score += 20
        
        # Record count complexity
        if analysis["record_count"] > 100000:
            score += 20
        
        # Business complexity
        if analysis["business_criticality"] == "high":
            score += 15
        
        # Compliance complexity
        if analysis["compliance_requirements"]:
            score += 15
        
        # PII complexity
        if analysis["has_pii"]:
            score += 10
        
        return min(100, score)
    
    def _calculate_impact_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate potential impact score (0-100)."""
        score = 10  # Base score
        
        # Dependency impact
        score += len(analysis["downstream_dependencies"]) * 10
        
        # Business impact
        if analysis["business_criticality"] == "high":
            score += 30
        elif analysis["business_criticality"] == "medium":
            score += 15
        
        # Compliance impact
        if analysis["compliance_requirements"]:
            score += 25
        
        # Quality impact
        if analysis["quality_score"] < 50:
            score += 20
        
        return min(100, score)
    
    async def execute_workflow(
        self, 
        selection: WorkflowSelection,
        input_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the selected workflow."""
        try:
            # Get the appropriate workflow
            workflow = self.available_workflows[selection.workflow_type]
            
            # Create initial state
            initial_state = RemediationState(
                user_id=user_id or "system",
                dataset_id=input_data.get("dataset_id", "unknown"),
                shared_context={"workflow_selection": selection, **selection.configuration}
            )
            
            # Add initial message
            initial_state.add_message(HumanMessage(content=f"Execute {selection.workflow_type.value} remediation workflow"))
            
            # Execute workflow
            start_time = datetime.now()
            result = await workflow.process(initial_state)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record execution metrics
            execution_record = {
                "workflow_type": selection.workflow_type.value,
                "risk_level": selection.risk_level.value,
                "execution_time": execution_time,
                "estimated_time": selection.estimated_duration * 60 if selection.estimated_duration else None,
                "success": not result.errors,
                "issues_resolved": result.resolved_issues,
                "total_issues": result.total_issues,
                "timestamp": start_time.isoformat()
            }
            
            self.workflow_history.append(execution_record)
            
            # Update performance metrics
            await self._update_performance_metrics(execution_record)
            
            logger.info(f"Workflow execution completed in {execution_time:.2f} seconds")
            
            return {
                "state": result,
                "execution_record": execution_record,
                "performance_summary": self._get_performance_summary()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _update_performance_metrics(self, execution_record: Dict[str, Any]):
        """Update orchestrator performance metrics."""
        workflow_type = execution_record["workflow_type"]
        
        if workflow_type not in self.performance_metrics:
            self.performance_metrics[workflow_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_execution_time": 0,
                "total_issues_resolved": 0,
                "average_success_rate": 0
            }
        
        metrics = self.performance_metrics[workflow_type]
        metrics["total_executions"] += 1
        metrics["total_execution_time"] += execution_record["execution_time"]
        metrics["total_issues_resolved"] += execution_record["issues_resolved"]
        
        if execution_record["success"]:
            metrics["successful_executions"] += 1
        
        metrics["average_success_rate"] = (
            metrics["successful_executions"] / metrics["total_executions"] * 100
        )
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get orchestrator performance summary."""
        return {
            "total_workflows_executed": len(self.workflow_history),
            "workflow_performance": self.performance_metrics,
            "recent_executions": self.workflow_history[-10:] if self.workflow_history else []
        }
    
    async def get_workflow_recommendations(
        self, 
        data_context: Dict[str, Any]
    ) -> List[WorkflowSelection]:
        """Get multiple workflow recommendations ranked by suitability."""
        recommendations = []
        
        # Analyze data once
        analysis = await self._analyze_data_context(data_context)
        risk_level = self._assess_risk_level(analysis)
        
        # Generate recommendations for each workflow type
        for workflow_type in WorkflowType:
            if workflow_type in self.available_workflows:
                config = self._generate_workflow_config(workflow_type, analysis, risk_level)
                selection = WorkflowSelection(
                    workflow_type=workflow_type,
                    risk_level=risk_level,
                    reasons=self._generate_selection_reasons(workflow_type, analysis, risk_level),
                    configuration=config,
                    estimated_duration=self._estimate_duration(workflow_type, analysis),
                    resource_requirements=self._estimate_resources(workflow_type, analysis)
                )
                recommendations.append(selection)
        
        # Sort by suitability (this could be enhanced with ML scoring)
        recommendations.sort(key=lambda x: self._calculate_suitability_score(x, analysis))
        
        return recommendations
    
    def _calculate_suitability_score(self, selection: WorkflowSelection, analysis: Dict[str, Any]) -> float:
        """Calculate suitability score for ranking recommendations."""
        score = 50.0  # Base score
        
        # Risk alignment
        if selection.risk_level.value == analysis.get("recommended_risk", "medium"):
            score += 20
        
        # Efficiency factors
        if selection.workflow_type == WorkflowType.AUTOMATED and analysis["complexity_score"] < 30:
            score += 15
        
        if selection.workflow_type == WorkflowType.BATCH and analysis["record_count"] > 50000:
            score += 15
        
        # Resource efficiency
        if selection.estimated_duration and selection.estimated_duration < 60:
            score += 10
        
        return score


# Factory function
async def create_remediation_orchestrator(shared_services: SharedServices) -> RemediationOrchestrator:
    """Create and initialize a remediation orchestrator."""
    orchestrator = RemediationOrchestrator(shared_services)
    await orchestrator.initialize()
    return orchestrator
