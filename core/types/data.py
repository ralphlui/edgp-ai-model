"""
Data-related types and schemas.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .base import (
    BaseTimestamped, DataType, SensitivityLevel, 
    Priority, Severity, ConfidenceLevel, TimeRange
)


class DataField(BaseModel):
    """Schema definition for a data field."""
    name: str
    data_type: DataType
    is_required: bool = False
    is_primary_key: bool = False
    is_foreign_key: bool = False
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern for validation
    allowed_values: Optional[List[str]] = None
    description: Optional[str] = None
    sensitivity_level: SensitivityLevel = SensitivityLevel.PUBLIC
    tags: List[str] = Field(default_factory=list)


class DataSchema(BaseModel):
    """Complete schema definition for a dataset."""
    name: str
    version: str = "1.0"
    description: Optional[str] = None
    fields: List[DataField]
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: Dict[str, str] = Field(default_factory=dict)  # field -> referenced_table.field
    indexes: List[str] = Field(default_factory=list)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataSource(BaseTimestamped):
    """Data source definition."""
    id: str
    name: str
    type: str  # database, file, api, stream, etc.
    connection_string: Optional[str] = None
    data_schema: Optional[DataSchema] = Field(None, alias="schema")
    is_active: bool = True
    last_updated: Optional[datetime] = None
    record_count: Optional[int] = None
    size_bytes: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataRecord(BaseModel):
    """Individual data record."""
    id: str
    source_id: str
    data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    version: int = 1
    is_deleted: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataIssue(BaseTimestamped):
    """Data quality or compliance issue."""
    id: str
    issue_type: str
    title: str
    description: str
    severity: Severity
    priority: Priority
    confidence: ConfidenceLevel
    affected_records: List[str] = Field(default_factory=list)
    affected_fields: List[str] = Field(default_factory=list)
    source_id: str
    detection_method: str
    suggested_actions: List[str] = Field(default_factory=list)
    status: str = "open"
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityMetrics(BaseModel):
    """Data quality metrics for a dataset."""
    dataset_id: str
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    
    # Quality dimensions
    completeness: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    timeliness: float = Field(..., ge=0.0, le=1.0)
    validity: float = Field(..., ge=0.0, le=1.0)
    uniqueness: float = Field(..., ge=0.0, le=1.0)
    
    # Detailed metrics
    total_records: int
    complete_records: int
    duplicate_records: int
    invalid_records: int
    outdated_records: int
    
    # Field-level metrics
    field_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Trends
    trend_direction: str = "stable"  # improving, declining, stable
    trend_percentage: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceViolation(BaseTimestamped):
    """Compliance violation details."""
    id: str
    regulation: str  # GDPR, CCPA, HIPAA, etc.
    violation_type: str
    description: str
    severity: Severity
    confidence: ConfidenceLevel
    affected_data: List[str] = Field(default_factory=list)
    legal_basis: Optional[str] = None
    potential_penalty: Optional[str] = None
    recommended_actions: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None
    status: str = "open"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RemediationTask(BaseTimestamped):
    """Data remediation task."""
    id: str
    title: str
    description: str
    task_type: str
    priority: Priority
    status: str = "pending"
    estimated_effort_hours: Optional[float] = None
    actual_effort_hours: Optional[float] = None
    assigned_to: Optional[str] = None
    assigned_team: Optional[str] = None
    
    # Input data
    affected_records: List[str] = Field(default_factory=list)
    remediation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Progress tracking
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    records_processed: int = 0
    records_fixed: int = 0
    records_failed: int = 0
    
    # Results
    quality_improvement: Optional[Dict[str, float]] = None
    completion_time: Optional[datetime] = None
    success_rate: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicyRecommendation(BaseTimestamped):
    """Policy recommendation from Policy Suggestion Agent."""
    id: str
    title: str
    description: str
    policy_type: str  # access, retention, validation, etc.
    confidence: ConfidenceLevel
    priority: Priority
    
    # Policy details
    scope: List[str] = Field(default_factory=list)  # data sources, fields
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    enforcement_level: str = "advisory"  # advisory, mandatory, automatic
    
    # Business context
    business_justification: str
    compliance_requirements: List[str] = Field(default_factory=list)
    risk_mitigation: List[str] = Field(default_factory=list)
    
    # Implementation
    implementation_steps: List[str] = Field(default_factory=list)
    estimated_implementation_time: Optional[str] = None
    required_resources: List[str] = Field(default_factory=list)
    
    # Approval workflow
    status: str = "pending_review"
    reviewed_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# MCP Processing Request for message-driven architecture
class ProcessingRequest(BaseModel):
    """Generic processing request for MCP messaging."""
    data_source: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: str = "normal"
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsVisualization(BaseModel):
    """Analytics visualization configuration."""
    id: str
    title: str
    visualization_type: str  # bar_chart, line_chart, table, etc.
    data_source: str
    
    # Chart configuration
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    grouping: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Styling
    colors: List[str] = Field(default_factory=list)
    theme: str = "default"
    
    # Data
    data: List[Dict[str, Any]] = Field(default_factory=list)
    summary_stats: Dict[str, Any] = Field(default_factory=dict)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
