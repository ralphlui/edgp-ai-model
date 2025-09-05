"""
Agent-specific request and response types.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

from .base import (
    AgentType, Priority, Severity, ConfidenceLevel, TaskStatus,
    ComplianceRegulation, VisualizationType, TimeRange
)
from .data import (
    DataSchema, DataSource, DataIssue, QualityMetrics,
    ComplianceViolation, RemediationTask, PolicyRecommendation,
    AnalyticsVisualization
)


# ==================== AGENT CAPABILITIES ====================

class AgentStatus(str, Enum):
    """Status of an agent."""
    IDLE = "idle"
    INITIALIZING = "initializing"  # Added for BaseAgent compatibility
    PROCESSING = "processing"
    RUNNING = "running"  # Added for BaseAgent compatibility
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
    FAILED = "failed"  # Added for BaseAgent compatibility
    TIMEOUT = "timeout"  # Added for BaseAgent compatibility


class AgentCapability(str, Enum):
    """MCP agent capabilities for message-driven architecture."""
    
    # Data Quality Capabilities
    DATA_QUALITY_ASSESSMENT = "data_quality_assessment"
    ANOMALY_DETECTION = "anomaly_detection"
    DATA_PROFILING = "data_profiling"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    
        # Compliance Capabilities  
    COMPLIANCE_CHECKING = "compliance_checking"
    COMPLIANCE_CHECK = "compliance_check"  # Alias for backward compatibility
    RISK_ASSESSMENT = "risk_assessment"
    POLICY_VALIDATION = "policy_validation"
    REGULATORY_SCANNING = "regulatory_scanning"
    
    # Remediation Capabilities
    DATA_REMEDIATION = "data_remediation" 
    REMEDIATION_SUGGESTION = "remediation_suggestion"  # Added for compatibility
    ISSUE_RESOLUTION = "issue_resolution"
    DATA_TRANSFORMATION = "data_transformation"
    AUTOMATED_FIXES = "automated_fixes"
    
    # Analytics Capabilities
    PERFORMANCE_ANALYTICS = "performance_analytics"
    ANALYTICS_REPORTING = "analytics_reporting"  # Added for compatibility
    TREND_ANALYSIS = "trend_analysis"
    REPORTING = "reporting"
    VISUALIZATION = "visualization"
    
    # Policy Capabilities
    POLICY_SUGGESTION = "policy_suggestion"
    POLICY_GENERATION = "policy_generation"  # Added for compatibility
    BEST_PRACTICES = "best_practices"
    GOVERNANCE_RECOMMENDATIONS = "governance_recommendations"


# ==================== BASE AGENT TYPES ====================

class AgentRequest(BaseModel):
    """Base request for all agents."""
    request_id: str
    agent_type: AgentType
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Base response from all agents."""
    request_id: str
    agent_type: AgentType
    timestamp: str
    success: bool
    processing_time_ms: Optional[float] = None
    confidence: Optional[ConfidenceLevel] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== POLICY SUGGESTION AGENT ====================

class PolicySuggestionRequest(AgentRequest):
    """Request for Policy Suggestion Agent."""
    agent_type: AgentType = AgentType.POLICY_SUGGESTION
    
    # Input data
    data_schema: Optional[DataSchema] = None
    business_context: str
    compliance_requirements: List[ComplianceRegulation] = Field(default_factory=list)
    existing_policies: List[Dict[str, Any]] = Field(default_factory=list)
    risk_assessment: Optional[Dict[str, Any]] = None
    
    # Request type
    suggestion_type: str = "validation_policies"  # validation_policies, governance_policies, gap_analysis


class PolicySuggestionResponse(AgentResponse):
    """Response from Policy Suggestion Agent."""
    agent_type: AgentType = AgentType.POLICY_SUGGESTION
    
    # Results
    suggested_policies: List[PolicyRecommendation] = Field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_mappings: Dict[str, List[str]] = Field(default_factory=dict)
    implementation_guidance: List[str] = Field(default_factory=list)
    
    # Analysis
    policy_gaps: List[str] = Field(default_factory=list)
    risk_mitigation: List[str] = Field(default_factory=list)
    business_impact: Optional[str] = None


# ==================== DATA PRIVACY & COMPLIANCE AGENT ====================

class ComplianceRequest(AgentRequest):
    """Request for Data Privacy & Compliance Agent."""
    agent_type: AgentType = AgentType.DATA_PRIVACY_COMPLIANCE
    
    # Input data
    data_sources: List[DataSource] = Field(default_factory=list)
    data_schema: Optional[DataSchema] = None
    processing_context: str
    applicable_regulations: List[ComplianceRegulation] = Field(default_factory=list)
    current_policies: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Scan configuration
    scan_type: str = "privacy_risks"  # privacy_risks, compliance_violations, usage_monitoring
    include_historical: bool = False
    time_range: Optional[TimeRange] = None


class ComplianceResponse(AgentResponse):
    """Response from Data Privacy & Compliance Agent."""
    agent_type: AgentType = AgentType.DATA_PRIVACY_COMPLIANCE
    
    # Privacy risks
    privacy_risks: List[Dict[str, Any]] = Field(default_factory=list)
    pii_detected: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Compliance violations
    violations: List[ComplianceViolation] = Field(default_factory=list)
    compliance_score: float = Field(..., ge=0.0, le=1.0)
    
    # Remediation tasks
    remediation_tasks: List[RemediationTask] = Field(default_factory=list)
    
    # Risk assessment
    overall_risk_level: Severity
    recommendations: List[str] = Field(default_factory=list)
    next_review_date: Optional[str] = None


# ==================== DATA QUALITY AGENT ====================

class DataQualityRequest(AgentRequest):
    """Request for Data Quality Agent."""
    agent_type: AgentType = AgentType.DATA_QUALITY
    
    # Input data
    dataset_id: str
    data_schema: Optional[DataSchema] = None
    quality_rules: List[Dict[str, Any]] = Field(default_factory=list)
    quality_dimensions: List[str] = Field(default_factory=list)  # Add this field for test compatibility
    
    # Analysis configuration
    analysis_type: str = "comprehensive"  # anomalies, duplicates, metrics, comprehensive
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    include_field_analysis: bool = True
    generate_report: bool = False
    anomaly_detection: bool = True  # Add this field for test compatibility


class DataQualityResponse(AgentResponse):
    """Response from Data Quality Agent."""
    agent_type: AgentType = AgentType.DATA_QUALITY
    
    # Quality assessment
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)
    quality_metrics: QualityMetrics
    
    # Issues detected
    anomalies: List[DataIssue] = Field(default_factory=list)
    duplicates: List[Dict[str, Any]] = Field(default_factory=list)
    quality_issues: List[DataIssue] = Field(default_factory=list)
    
    # Statistics
    total_records_analyzed: int
    issues_by_severity: Dict[Severity, int] = Field(default_factory=dict)
    
    # Recommendations
    recommended_actions: List[str] = Field(default_factory=list)
    remediation_priority: List[str] = Field(default_factory=list)


# ==================== DATA REMEDIATION AGENT ====================

class RemediationRequest(AgentRequest):
    """Request for Data Remediation Agent."""
    agent_type: AgentType = AgentType.DATA_REMEDIATION
    
    # Input data
    issues: List[DataIssue] = Field(default_factory=list)
    remediation_tasks: List[str] = Field(default_factory=list)  # Task IDs
    business_rules: Dict[str, Any] = Field(default_factory=dict)
    resource_constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Request configuration
    operation_type: str = "generate_plan"  # generate_plan, execute_task, provide_guidance, track_outcomes
    user_role: Optional[str] = None
    available_tools: List[str] = Field(default_factory=list)
    approval_required: bool = True


class RemediationResponse(AgentResponse):
    """Response from Data Remediation Agent."""
    agent_type: AgentType = AgentType.DATA_REMEDIATION
    
    # Remediation plan
    remediation_plan: Optional[Dict[str, Any]] = None
    estimated_duration: Optional[str] = None
    required_resources: List[str] = Field(default_factory=list)
    
    # Task results
    processed_tasks: List[RemediationTask] = Field(default_factory=list)
    success_rate: Optional[float] = None
    
    # Guidance
    step_by_step_guidance: List[Dict[str, Any]] = Field(default_factory=list)
    safety_measures: List[str] = Field(default_factory=list)
    
    # Outcomes
    quality_improvement: Optional[Dict[str, float]] = None
    business_impact: Optional[str] = None
    lessons_learned: List[str] = Field(default_factory=list)


# ==================== ANALYTICS AGENT ====================

class AnalyticsRequest(AgentRequest):
    """Request for Analytics Agent."""
    agent_type: AgentType = AgentType.ANALYTICS
    
    # Data sources
    datasets: List[str] = Field(default_factory=list)
    metrics_sources: List[str] = Field(default_factory=list)  # Other agents
    
    # Report configuration
    report_type: str = "dashboard"  # dashboard, report, chart, table
    visualization_types: List[VisualizationType] = Field(default_factory=list)
    time_range: Optional[TimeRange] = None
    
    # Filters and grouping
    filters: Dict[str, Any] = Field(default_factory=dict)
    group_by: List[str] = Field(default_factory=list)
    aggregations: List[str] = Field(default_factory=list)
    
    # Output preferences
    include_trends: bool = True
    include_recommendations: bool = True
    export_format: str = "json"  # json, csv, excel, pdf


class AnalyticsResponse(AgentResponse):
    """Response from Analytics Agent."""
    agent_type: AgentType = AgentType.ANALYTICS
    
    # Generated content
    visualizations: List[AnalyticsVisualization] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Summary statistics
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    key_insights: List[str] = Field(default_factory=list)
    trends: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Export data
    export_url: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


# ==================== INTER-AGENT COMMUNICATION ====================

class AgentMessage(BaseModel):
    """Message for inter-agent communication."""
    message_id: str
    from_agent: AgentType
    to_agent: AgentType
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    priority: Priority = Priority.MEDIUM


class WorkflowRequest(BaseModel):
    """Request for orchestrated multi-agent workflows."""
    workflow_id: str
    workflow_type: str
    involved_agents: List[AgentType]
    input_data: Dict[str, Any]
    configuration: Dict[str, Any] = Field(default_factory=dict)
    user_context: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Response from orchestrated workflows."""
    workflow_id: str
    workflow_type: str
    status: TaskStatus
    results: Dict[AgentType, Any] = Field(default_factory=dict)
    execution_summary: Dict[str, Any] = Field(default_factory=dict)
    total_processing_time_ms: Optional[float] = None
