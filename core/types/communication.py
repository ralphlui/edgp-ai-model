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
    ANALYZE = "analysis"  # Alias for backward compatibility
    ASSESSMENT = "assessment"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    RECOMMENDATION = "recommendation"
    REMEDIATION = "remediation"
    MONITORING = "monitoring"
    PROCESS = "process"
    PROCESSING = "processing"


class ErrorType(str, Enum):
    """Types of errors that can occur in agent operations."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_ERROR = "data_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RESOURCE_ERROR = "resource_error"
    SYSTEM_ERROR = "system_error"


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
    capability_used: Optional[str] = None
    capability_name: Optional[str] = None  # Alias for backward compatibility
    operation_type: Optional[OperationType] = None
    processing_stage: ProcessingStage = ProcessingStage.PROCESSING
    
    # Performance metrics
    execution_time: Optional[float] = None  # Added for test compatibility
    execution_time_ms: Optional[float] = None
    memory_usage: Optional[float] = None  # Added for test compatibility
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
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Alias for message_id
    message_type: MessageType = MessageType.REQUEST
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Operation details
    operation: Optional[OperationType] = None
    data: T
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent routing (required for backward compatibility)
    source_agent_id: str
    target_agent_id: str
    capability_name: str
    
    # Context and metadata
    context: AgentContext
    priority: Priority = Priority.MEDIUM
    
    # Validation and constraints
    schema_version: str = "1.0.0"
    expected_output_format: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing preferences
    processing_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Ensure request_id and message_id are synchronized
        if 'request_id' in data and 'message_id' not in data:
            data['message_id'] = data['request_id']
        elif 'message_id' in data and 'request_id' not in data:
            data['request_id'] = data['message_id']
        super().__init__(**data)
    
    class Config:
        arbitrary_types_allowed = True


class StandardAgentOutput(BaseModel, Generic[T]):
    """Standardized output format for all agent operations."""
    
    # Response identification
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str  # Reference to the original request
    message_type: MessageType = MessageType.RESPONSE
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Agent information
    source_agent_id: Optional[str] = None
    capability_used: Optional[str] = None
    
    # Agent metadata
    agent_metadata: AgentMetadata
    
    # Execution status
    status: Optional[TaskStatus] = None
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Results (typed)
    result: Optional[T] = None
    
    # Context (optional)
    context: Optional[AgentContext] = None
    
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


class ProcessingResult(BaseModel):
    """Standardized processing result structure."""
    
    # Result identification
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: Optional[OperationType] = None
    
    # Status and basic result info
    status: TaskStatus
    result_data: Dict[str, Any] = Field(default_factory=dict)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    # Main results (for backward compatibility)
    primary_output: Optional[Any] = None
    secondary_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis and insights
    key_findings: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    patterns_detected: List[str] = Field(default_factory=list)
    
    # Issues and alerts
    issues_found: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    # Metrics and scores
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    quality_scores: Dict[QualityDimension, float] = Field(default_factory=dict)
    
    # Statistical information
    statistics: Dict[str, Any] = Field(default_factory=dict)
    distributions: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing information
    processing_time: Optional[float] = None
    
    # Validation results
    validation_passed: bool = True
    validation_details: Dict[str, Any] = Field(default_factory=dict)


class CommunicationProtocol(BaseModel):
    """Communication protocol configuration."""
    
    protocol_version: str = "1.0"
    message_format: str = "json"
    encoding: str = "utf-8"
    compression: Optional[str] = None
    encryption: Optional[str] = None
    
    # Message size limits
    max_message_size_bytes: int = Field(10 * 1024 * 1024, ge=1)  # 10MB default
    
    # Timeout settings
    connection_timeout_seconds: int = Field(30, ge=1)
    read_timeout_seconds: int = Field(300, ge=1)
    
    # Retry configuration
    max_retries: int = Field(3, ge=0)
    retry_delay_seconds: float = Field(1.0, ge=0.0)
    
    # Quality of service
    delivery_guarantee: str = "at_least_once"  # exactly_once, at_most_once, at_least_once
    ordering_guarantee: bool = False
    
    # Security settings
    require_authentication: bool = True
    require_encryption: bool = False


class AgentError(BaseModel):
    """Standardized error information."""
    
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    error_code: Optional[str] = None
    error_type: Union[ErrorType, str]
    error_message: str
    message: str  # Alias for error_message for backward compatibility
    
    # Error context
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    timestamp: datetime = Field(default_factory=datetime.utcnow)  # Alias for occurred_at
    processing_stage: ProcessingStage = ProcessingStage.PROCESSING
    agent_id: str
    capability_name: Optional[str] = None
    
    # Error details
    stack_trace: Optional[str] = None
    input_data_snapshot: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    
    # Recovery information
    is_recoverable: bool = True
    suggested_retry_delay_seconds: Optional[int] = None
    recovery_suggestions: List[str] = Field(default_factory=list)
    
    # Impact assessment
    severity: Severity = Severity.MEDIUM
    impact_scope: List[str] = Field(default_factory=list)
    affected_operations: List[str] = Field(default_factory=list)
    
    def __init__(self, **data):
        # Ensure message and error_message are synchronized
        if 'message' in data and 'error_message' not in data:
            data['error_message'] = data['message']
        elif 'error_message' in data and 'message' not in data:
            data['message'] = data['error_message']
        
        # Auto-derive error_type from error_code if not provided
        if 'error_code' in data and 'error_type' not in data and data['error_code']:
            data['error_type'] = data['error_code'].split('_')[0] if '_' in data['error_code'] else "GENERAL"
            
        super().__init__(**data)


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
    agent_id: Optional[str] = None,
    capability_name: Optional[str] = None,
    result: Any = None,
    request_id: Optional[str] = None,
    status: Optional[TaskStatus] = None,
    # Old signature support
    source_agent_id: Optional[str] = None,
    capability_used: Optional[str] = None,
    success: Optional[bool] = None,
    metadata: Optional[AgentMetadata] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None
) -> StandardAgentOutput:
    """Create a standardized agent output."""
    # Handle backward compatibility
    final_agent_id = agent_id or source_agent_id or "unknown"
    final_capability = capability_name or capability_used or "unknown"
    final_status = status or (TaskStatus.COMPLETED if success else TaskStatus.FAILED) if success is not None else TaskStatus.COMPLETED
    
    # Create processing result
    # For backward compatibility, if result is a dict, use it directly
    # This maintains compatibility with tests that expect result.result to be a dict
    if isinstance(result, dict):
        processing_result = result
    elif result is None:
        processing_result = ProcessingResult(
            status=final_status,
            result_data={},
            confidence=ConfidenceLevel.HIGH
        )
    else:
        processing_result = result
        
    # Create metadata
    if metadata is None:
        metadata = AgentMetadata(
            agent_id=final_agent_id,
            capability_used=final_capability,
            capability_name=final_capability,
            operation_type=OperationType.ANALYSIS
        )
    
    return StandardAgentOutput(
        request_id=request_id or str(uuid.uuid4()),
        agent_metadata=metadata,
        result=processing_result,
        status=final_status,
        source_agent_id=final_agent_id,
        capability_used=final_capability,
        success=success if success is not None else True,
        error_code=error_code,
        error_message=error_message
    )


def create_agent_error(
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    agent_id: Optional[str] = None,
    capability_name: Optional[str] = None,
    error_type: Optional[ErrorType] = None,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> AgentError:
    """Create a standardized agent error."""
    # Handle different signatures for backward compatibility
    if error_type is not None and message is not None:
        # New signature: error_type, message, agent_id
        return AgentError(
            error_type=error_type,
            error_message=message,
            message=message,
            agent_id=agent_id or "unknown",
            capability_name=capability_name,
            error_code=error_code,
            details=details or {}
        )
    else:
        # Old signature: error_code, error_message, agent_id, capability_name
        derived_error_type = error_code.split('_')[0] if error_code and '_' in error_code else "GENERAL"
        return AgentError(
            error_type=derived_error_type,
            error_message=error_message or "Unknown error",
            message=error_message or "Unknown error",
            agent_id=agent_id or "unknown",
            capability_name=capability_name,
            error_code=error_code,
            details=details or {}
        )


def create_agent_context(
    session_id: str,
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentContext:
    """Create a standardized agent context."""
    return AgentContext(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata or {}
    )
