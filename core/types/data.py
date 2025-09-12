"""
Data-related types and schemas.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .base import (
    BaseTimestamped, DataType, SensitivityLevel, 
    Priority, Severity, ConfidenceLevel, TimeRange,
    QualityDimension, DataSourceType, DataFormat, 
    PolicyType, ComplianceStatus, DataClassification
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


# Additional models for comprehensive data type testing
class DataQualityIssue(BaseTimestamped):
    """Data quality issue model."""
    issue_id: str = Field(default_factory=lambda: f"issue_{datetime.utcnow().timestamp()}")
    dimension: QualityDimension
    severity: Severity
    description: str
    column_name: str
    affected_rows: int
    recommendation: Optional[str] = None
    additional_context: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class DataQualityMetrics(BaseModel):
    """Enhanced data quality metrics model."""
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    validity_score: float = Field(..., ge=0.0, le=1.0)
    uniqueness_score: float = Field(..., ge=0.0, le=1.0)
    overall_score: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.overall_score is None:
            self.overall_score = (
                self.completeness_score + self.accuracy_score + 
                self.consistency_score + self.validity_score + 
                self.uniqueness_score
            ) / 5.0


class ColumnMetadata(BaseModel):
    """Column metadata model."""
    name: str
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    unique_values: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean_value: Optional[float] = None
    null_count: Optional[int] = None
    constraints: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class DatasetMetadata(BaseTimestamped):
    """Dataset metadata model."""
    dataset_id: str = Field(default_factory=lambda: f"dataset_{datetime.utcnow().timestamp()}")
    name: str
    source: DataSourceType
    format: DataFormat
    columns: List[ColumnMetadata]
    row_count: Optional[int] = None
    total_rows: Optional[int] = None
    database_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    file_path: Optional[str] = None
    file_format: Optional[DataFormat] = None
    file_size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class DataProfile(BaseTimestamped):
    """Data profile model."""
    dataset_metadata: DatasetMetadata
    quality_metrics: DataQualityMetrics
    issues: List[DataQualityIssue] = Field(default_factory=list)
    quality_issues: List[DataQualityIssue] = Field(default_factory=list)  # Alias for backward compatibility
    profile_id: str = Field(default_factory=lambda: f"profile_{datetime.utcnow().timestamp()}")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    profiled_at: datetime = Field(default_factory=datetime.utcnow)


class DataValidationRule(BaseTimestamped):
    """Data validation rule model."""
    rule_id: str = Field(default_factory=lambda: f"rule_{datetime.utcnow().timestamp()}")
    name: str
    description: str
    rule_type: str
    column_name: str
    condition: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    severity: Severity = Severity.MEDIUM
    is_active: bool = True
    error_message: Optional[str] = None


class DataTransformation(BaseTimestamped):
    """Data transformation model."""
    transformation_id: str = Field(default_factory=lambda: f"transform_{datetime.utcnow().timestamp()}")
    name: str
    description: str
    transformation_type: str
    source_columns: List[str]
    target_column: Optional[str] = None
    target_columns: List[str] = Field(default_factory=list)
    transformation_logic: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class DataLineage(BaseTimestamped):
    """Data lineage model."""
    lineage_id: str = Field(default_factory=lambda: f"lineage_{datetime.utcnow().timestamp()}")
    source_dataset: str
    target_dataset: str
    transformation_applied: Optional[str] = None
    transformation_steps: List[DataTransformation] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    upstream_lineage: List[str] = Field(default_factory=list)
    downstream_lineage: List[str] = Field(default_factory=list)
    impact_analysis: Dict[str, Any] = Field(default_factory=dict)
    lineage_type: str = "direct"
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicyRule(BaseTimestamped):
    """Policy rule model."""
    rule_id: str = Field(default_factory=lambda: f"policy_{datetime.utcnow().timestamp()}")
    name: str
    description: str
    policy_type: PolicyType
    rule_expression: Optional[str] = None
    severity: Severity = Severity.MEDIUM
    scope: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    is_active: bool = True
    enforcement_level: str = "advisory"
    parameters: Dict[str, Any] = Field(default_factory=dict)


# Add aliases for backward compatibility with enum-style access
# End of file
