"""
Specific types for Data Remediation Agent operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from ..base import Priority, TaskStatus, RemediationType
from ..data import DataIssue, RemediationTask


class RemediationStrategy(str, Enum):
    """Strategies for data remediation."""
    AUTOMATED = "automated"
    SEMI_AUTOMATED = "semi_automated"
    MANUAL = "manual"
    HYBRID = "hybrid"


class RemediationAction(BaseModel):
    """Individual remediation action."""
    action_id: str
    action_type: RemediationType
    description: str
    target_fields: List[str]
    target_records: List[str]
    
    # Execution details
    execution_method: RemediationStrategy
    automation_script: Optional[str] = None
    manual_instructions: List[str] = Field(default_factory=list)
    
    # Validation
    validation_rules: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    
    # Risk management
    risk_level: str = "low"
    backup_required: bool = True
    rollback_procedure: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RemediationPlan(BaseModel):
    """Comprehensive remediation plan."""
    plan_id: str
    plan_name: str
    description: str
    
    # Planning context
    issues_addressed: List[str]  # DataIssue IDs
    business_priority: Priority
    estimated_duration: str
    estimated_cost: Optional[float] = None
    
    # Execution plan
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Risk management
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    mitigation_strategies: List[str] = Field(default_factory=list)
    rollback_plan: Optional[str] = None
    
    # Success criteria
    success_metrics: Dict[str, float] = Field(default_factory=dict)
    acceptance_criteria: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RemediationOutcome(BaseModel):
    """Results of remediation execution."""
    outcome_id: str
    task_id: str
    execution_start: datetime
    execution_end: Optional[datetime] = None
    
    # Results
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    success_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Quality impact
    quality_before: Dict[str, float] = Field(default_factory=dict)
    quality_after: Dict[str, float] = Field(default_factory=dict)
    improvement_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Issues and lessons
    issues_encountered: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RemediationGuidance(BaseModel):
    """Step-by-step remediation guidance for users."""
    guidance_id: str
    target_user_role: str
    issue_type: str
    
    # Guidance steps
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    tools_required: List[str] = Field(default_factory=list)
    
    # Safety measures
    safety_checks: List[str] = Field(default_factory=list)
    backup_procedures: List[str] = Field(default_factory=list)
    
    # Validation
    quality_checks: List[str] = Field(default_factory=list)
    completion_criteria: List[str] = Field(default_factory=list)
    
    # Support
    escalation_criteria: List[str] = Field(default_factory=list)
    support_contacts: List[Dict[str, str]] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
