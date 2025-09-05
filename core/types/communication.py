"""
Standardized Agent Communication Types
Following agentic AI best practices for consistent input/output patterns.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from pydantic import BaseModel, Field, validator

from .base import (
    AgentType, Priority, Severity, ConfidenceLevel, TaskStatus,
    ComplianceRegulation, QualityDimension, RemediationType,
    VisualizationType, SensitivityLevel
)

# Generic type for typed responses
T = TypeVar('T')


# ==================== CORE COMMUNICATION ENUMS ====================

class MessageType(str, Enum):
    """Types of messages in agent communication."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    EVENT = "event"
    ACKNOWLEDGMENT = "acknowledgment"
    ERROR = "error"


class ExecutionMode(str, Enum):
    """Execution modes for agent operations."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BACKGROUND = "background"
    STREAMING = "streaming"


class ProcessingStage(str, Enum):
    """Stages in agent processing pipeline."""
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    PROCESSING = "processing"
    POSTPROCESSING = "postprocessing"
    FINALIZATION = "finalization"


class OperationType(str, Enum):
    """Types of operations agents can perform."""
    ANALYSIS = "analysis"
    ASSESSMENT = "assessment"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    RECOMMENDATION = "recommendation"
    REMEDIATION = "remediation"
    MONITORING = "monitoring"


# ==================== STANDARDIZED BASE TYPES ====================

class AgentContext(BaseModel):
    """Standardized context information for agent operations."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Execution context
    execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Tracing and debugging
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    debug_mode: bool = False
    
    # Workflow context
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    previous_step_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Timeout must be positive')
        return v


class AgentCapabilityInfo(BaseModel):
    """Information about an agent capability."""
    name: str
    description: str
    version: str = "1.0.0"
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    required_permissions: List[str] = Field(default_factory=list)
    estimated_duration_seconds: Optional[float] = None
    cost_estimate: Optional[float] = None


class AgentMetadata(BaseModel):
    """Metadata about the agent and operation."""
    agent_id: str
    agent_version: str = "1.0.0"
    capability_used: str
    operation_type: OperationType
    processing_stage: ProcessingStage = ProcessingStage.PROCESSING
    
    # Performance metrics
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    tokens_consumed: Optional[int] = None
    api_calls_made: Optional[int] = None
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Resource usage
    llm_provider_used: Optional[str] = None
    model_name: Optional[str] = None
    external_services_called: List[str] = Field(default_factory=list)


# ==================== STANDARDIZED INPUT TYPES ====================

class StandardAgentInput(BaseModel, Generic[T]):
    """Standardized input format for all agent operations."""
    
    # Message identification
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.REQUEST
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Agent routing
    source_agent_id: str
    target_agent_id: str
    capability_name: str
    
    # Context and metadata
    context: AgentContext
    priority: Priority = Priority.MEDIUM
    
    # Payload data (typed)
    data: T
    
    # Validation and constraints
    schema_version: str = "1.0.0"
    expected_output_format: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing preferences
    processing_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class DataPayload(BaseModel):
    """Standardized data payload for agent operations."""
    
    # Data identification
    data_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_type: str
    data_format: str  # json, csv, parquet, etc.
    data_size_bytes: Optional[int] = None
    
    # Data content
    content: Union[Dict[str, Any], List[Dict[str, Any]], str]
    schema_info: Optional[Dict[str, Any]] = None
    
    # Data lineage and provenance
    source_system: Optional[str] = None
    data_lineage: List[str] = Field(default_factory=list)
    last_modified: Optional[datetime] = None
    version: str = "1.0"
    
    # Data classification
    sensitivity_level: SensitivityLevel = SensitivityLevel.INTERNAL
    retention_policy: Optional[str] = None
    compliance_tags: List[ComplianceRegulation] = Field(default_factory=list)
    
    # Quality indicators
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    freshness_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class OperationParameters(BaseModel):
    """Standardized operation parameters."""
    
    # Operation configuration
    operation_mode: str = "standard"
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0)
    include_recommendations: bool = True
    include_explanations: bool = True
    
    # Output preferences
    output_format: str = "detailed"  # summary, detailed, raw
    include_metadata: bool = True
    include_confidence_scores: bool = True
    
    # Processing options
    parallel_processing: bool = False
    max_concurrent_operations: int = Field(5, ge=1, le=50)
    enable_caching: bool = True
    cache_ttl_seconds: int = Field(3600, ge=0)
    
    # External integrations
    external_validations: List[str] = Field(default_factory=list)
    notification_targets: List[str] = Field(default_factory=list)
    
    # Custom parameters
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)


# ==================== STANDARDIZED OUTPUT TYPES ====================

class StandardAgentOutput(BaseModel, Generic[T]):
    """Standardized output format for all agent operations."""
    
    # Response identification
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str  # Reference to the original request
    message_type: MessageType = MessageType.RESPONSE
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Agent information
    source_agent_id: str
    capability_used: str
    
    # Execution status
    status: TaskStatus
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Results (typed)
    result: Optional[T] = None
    
    # Metadata and metrics
    metadata: AgentMetadata
    
    # Quality and confidence
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    quality_indicators: Dict[str, float] = Field(default_factory=dict)
    
    # Recommendations and next steps
    recommendations: List[str] = Field(default_factory=list)
    suggested_next_actions: List[str] = Field(default_factory=list)
    
    # Validation and compliance
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    compliance_status: Dict[str, Any] = Field(default_factory=dict)
    
    # Additional outputs
    artifacts: List[str] = Field(default_factory=list)  # URLs or references
    side_effects: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class ProcessingResult(BaseModel):
    """Standardized processing result structure."""
    
    # Result identification
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType
    
    # Main results
    primary_output: Any
    secondary_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis and insights
    key_findings: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    patterns_detected: List[str] = Field(default_factory=list)
    
    # Issues and alerts
    issues_found: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metrics and scores
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    quality_scores: Dict[QualityDimension, float] = Field(default_factory=dict)
    
    # Statistical information
    statistics: Dict[str, Any] = Field(default_factory=dict)
    distributions: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation results
    validation_passed: bool = True
    validation_details: Dict[str, Any] = Field(default_factory=dict)


class AgentError(BaseModel):
    """Standardized error information."""
    
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    error_code: str
    error_type: str
    error_message: str
    
    # Error context
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    processing_stage: ProcessingStage
    agent_id: str
    capability_name: str
    
    # Error details
    stack_trace: Optional[str] = None
    input_data_snapshot: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    
    # Recovery information
    is_recoverable: bool = True
    suggested_retry_delay_seconds: Optional[int] = None
    recovery_suggestions: List[str] = Field(default_factory=list)
    
    # Impact assessment
    severity: Severity = Severity.MEDIUM
    impact_scope: List[str] = Field(default_factory=list)
    affected_operations: List[str] = Field(default_factory=list)


# ==================== DOMAIN-SPECIFIC INPUT TYPES ====================

class DataQualityInput(BaseModel):
    """Input for data quality operations."""
    
    data_payload: DataPayload
    operation_parameters: OperationParameters
    
    # Data quality specific parameters
    quality_dimensions: List[QualityDimension] = Field(default_factory=list)
    baseline_metrics: Optional[Dict[str, float]] = None
    comparison_dataset: Optional[DataPayload] = None
    
    # Analysis preferences
    include_profiling: bool = True
    include_anomaly_detection: bool = True
    include_pattern_analysis: bool = True
    
    # Thresholds and rules
    quality_thresholds: Dict[QualityDimension, float] = Field(default_factory=dict)
    business_rules: List[str] = Field(default_factory=list)
    custom_validators: List[str] = Field(default_factory=list)


class ComplianceInput(BaseModel):
    """Input for compliance operations."""
    
    data_payload: DataPayload
    operation_parameters: OperationParameters
    
    # Compliance specific parameters
    regulations: List[ComplianceRegulation]
    compliance_policies: List[str] = Field(default_factory=list)
    audit_scope: List[str] = Field(default_factory=list)
    
    # Assessment preferences
    include_risk_assessment: bool = True
    include_gap_analysis: bool = True
    include_remediation_plan: bool = True
    
    # Regulatory context
    jurisdiction: Optional[str] = None
    industry_sector: Optional[str] = None
    data_subject_rights: List[str] = Field(default_factory=list)


class RemediationInput(BaseModel):
    """Input for remediation operations."""
    
    data_payload: DataPayload
    operation_parameters: OperationParameters
    
    # Issues to remediate
    identified_issues: List[Dict[str, Any]]
    quality_violations: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_violations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Remediation preferences
    remediation_types: List[RemediationType] = Field(default_factory=list)
    auto_fix_enabled: bool = False
    backup_required: bool = True
    
    # Constraints and approval
    requires_approval: bool = True
    approver_ids: List[str] = Field(default_factory=list)
    max_changes_percent: float = Field(10.0, ge=0.0, le=100.0)


class AnalyticsInput(BaseModel):
    """Input for analytics operations."""
    
    data_payload: DataPayload
    operation_parameters: OperationParameters
    
    # Analytics specific parameters
    analysis_type: str  # descriptive, diagnostic, predictive, prescriptive
    visualization_types: List[VisualizationType] = Field(default_factory=list)
    target_audience: str = "technical"  # technical, business, executive
    
    # Time series analysis
    time_dimension: Optional[str] = None
    time_granularity: Optional[str] = None  # day, week, month, quarter, year
    trend_analysis: bool = False
    
    # Statistical analysis
    statistical_tests: List[str] = Field(default_factory=list)
    correlation_analysis: bool = False
    outlier_detection: bool = False
    
    # Business context
    kpis: List[str] = Field(default_factory=list)
    business_objectives: List[str] = Field(default_factory=list)
    comparative_periods: List[str] = Field(default_factory=list)


class PolicyInput(BaseModel):
    """Input for policy operations."""
    
    operation_parameters: OperationParameters
    
    # Context information
    business_context: str
    industry_sector: Optional[str] = None
    regulatory_requirements: List[ComplianceRegulation] = Field(default_factory=list)
    
    # Current state
    existing_policies: List[Dict[str, Any]] = Field(default_factory=list)
    current_issues: List[Dict[str, Any]] = Field(default_factory=list)
    stakeholder_requirements: List[str] = Field(default_factory=list)
    
    # Policy preferences
    policy_types: List[str] = Field(default_factory=list)
    implementation_complexity: str = "medium"  # low, medium, high
    enforcement_level: str = "moderate"  # strict, moderate, lenient
    
    # Templates and examples
    template_preferences: List[str] = Field(default_factory=list)
    reference_frameworks: List[str] = Field(default_factory=list)


# ==================== DOMAIN-SPECIFIC OUTPUT TYPES ====================

class DataQualityOutput(BaseModel):
    """Output for data quality operations."""
    
    processing_result: ProcessingResult
    
    # Quality assessment results
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: Dict[QualityDimension, float]
    
    # Detailed findings
    quality_issues: List[Dict[str, Any]]
    anomalies_detected: List[Dict[str, Any]]
    data_profile: Dict[str, Any]
    
    # Recommendations
    improvement_recommendations: List[str]
    priority_actions: List[Dict[str, Any]]
    
    # Comparative analysis (if baseline provided)
    quality_trend: Optional[str] = None  # improving, stable, declining
    comparison_summary: Optional[Dict[str, Any]] = None


class ComplianceOutput(BaseModel):
    """Output for compliance operations."""
    
    processing_result: ProcessingResult
    
    # Compliance assessment results
    overall_compliance_score: float = Field(..., ge=0.0, le=1.0)
    regulation_scores: Dict[ComplianceRegulation, float]
    
    # Violations and risks
    violations_found: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    
    # Remediation guidance
    remediation_priorities: List[Dict[str, Any]]
    compliance_roadmap: List[Dict[str, Any]]
    
    # Audit information
    audit_trail: List[Dict[str, Any]]
    evidence_collected: List[str]


class RemediationOutput(BaseModel):
    """Output for remediation operations."""
    
    processing_result: ProcessingResult
    
    # Remediation plan
    remediation_actions: List[Dict[str, Any]]
    estimated_effort: Dict[str, Any]  # time, cost, resources
    
    # Execution results (if auto-fix enabled)
    fixes_applied: List[Dict[str, Any]]
    fixes_pending_approval: List[Dict[str, Any]]
    
    # Impact assessment
    before_after_comparison: Optional[Dict[str, Any]] = None
    quality_improvement: Optional[Dict[str, float]] = None
    
    # Backup and rollback
    backup_created: bool = False
    rollback_procedure: Optional[str] = None


class AnalyticsOutput(BaseModel):
    """Output for analytics operations."""
    
    processing_result: ProcessingResult
    
    # Analysis results
    key_insights: List[str]
    statistical_summary: Dict[str, Any]
    
    # Visualizations
    visualizations: List[Dict[str, Any]]  # Chart data and config
    dashboard_url: Optional[str] = None
    
    # Predictions and forecasts (if applicable)
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    forecasts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Business insights
    business_recommendations: List[str]
    action_items: List[Dict[str, Any]]


class PolicyOutput(BaseModel):
    """Output for policy operations."""
    
    processing_result: ProcessingResult
    
    # Generated policies
    recommended_policies: List[Dict[str, Any]]
    policy_templates: List[Dict[str, Any]]
    
    # Implementation guidance
    implementation_plan: Dict[str, Any]
    change_management_guide: List[str]
    
    # Impact assessment
    expected_benefits: List[str]
    potential_challenges: List[str]
    success_metrics: List[Dict[str, Any]]
    
    # Governance
    approval_workflow: List[Dict[str, Any]]
    monitoring_requirements: List[str]


# ==================== TYPE HELPERS AND UNIONS ====================

# Union types for generic handling
AgentInputUnion = Union[
    DataQualityInput,
    ComplianceInput,
    RemediationInput,
    AnalyticsInput,
    PolicyInput
]

AgentOutputUnion = Union[
    DataQualityOutput,
    ComplianceOutput,
    RemediationOutput,
    AnalyticsOutput,
    PolicyOutput
]

# Type mappings for dynamic usage
INPUT_TYPE_MAPPING = {
    "data_quality": DataQualityInput,
    "compliance": ComplianceInput,
    "remediation": RemediationInput,
    "analytics": AnalyticsInput,
    "policy": PolicyInput
}

OUTPUT_TYPE_MAPPING = {
    "data_quality": DataQualityOutput,
    "compliance": ComplianceOutput,
    "remediation": RemediationOutput,
    "analytics": AnalyticsOutput,
    "policy": PolicyOutput
}


# ==================== UTILITY FUNCTIONS ====================

def create_standard_input(
    source_agent_id: str,
    target_agent_id: str,
    capability_name: str,
    data: Any,
    context: Optional[AgentContext] = None,
    priority: Priority = Priority.MEDIUM
) -> StandardAgentInput:
    """Create a standardized agent input."""
    return StandardAgentInput(
        source_agent_id=source_agent_id,
        target_agent_id=target_agent_id,
        capability_name=capability_name,
        data=data,
        context=context or AgentContext(),
        priority=priority
    )


def create_standard_output(
    request_id: str,
    source_agent_id: str,
    capability_used: str,
    status: TaskStatus,
    success: bool,
    result: Any = None,
    metadata: Optional[AgentMetadata] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None
) -> StandardAgentOutput:
    """Create a standardized agent output."""
    return StandardAgentOutput(
        request_id=request_id,
        source_agent_id=source_agent_id,
        capability_used=capability_used,
        status=status,
        success=success,
        result=result,
        metadata=metadata or AgentMetadata(
            agent_id=source_agent_id,
            capability_used=capability_used,
            operation_type=OperationType.ANALYSIS
        ),
        error_code=error_code,
        error_message=error_message
    )


def create_agent_error(
    error_code: str,
    error_message: str,
    agent_id: str,
    capability_name: str,
    processing_stage: ProcessingStage = ProcessingStage.PROCESSING,
    severity: Severity = Severity.MEDIUM
) -> AgentError:
    """Create a standardized agent error."""
    return AgentError(
        error_code=error_code,
        error_type=error_code.split('_')[0] if '_' in error_code else "GENERAL",
        error_message=error_message,
        processing_stage=processing_stage,
        agent_id=agent_id,
        capability_name=capability_name,
        severity=severity
    )
