"""
Specific types for Data Quality Agent operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..base import Severity, ConfidenceLevel, QualityDimension
from ..data import DataIssue, QualityMetrics


class AnomalyType(str, Enum):
    """Types of data anomalies."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_VIOLATION = "pattern_violation"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    COMPLETENESS_ISSUE = "completeness_issue"
    CONSISTENCY_ISSUE = "consistency_issue"
    TIMELINESS_ISSUE = "timeliness_issue"


class DuplicateType(str, Enum):
    """Types of duplicate records."""
    EXACT_DUPLICATE = "exact_duplicate"
    FUZZY_DUPLICATE = "fuzzy_duplicate"
    PARTIAL_DUPLICATE = "partial_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"


class QualityRuleType(str, Enum):
    """Types of data quality rules."""
    FORMAT_VALIDATION = "format_validation"
    RANGE_VALIDATION = "range_validation"
    PATTERN_MATCHING = "pattern_matching"
    BUSINESS_LOGIC = "business_logic"
    CROSS_FIELD_VALIDATION = "cross_field_validation"
    REFERENCE_INTEGRITY = "reference_integrity"


class Anomaly(BaseModel):
    """Data anomaly details."""
    anomaly_id: str
    anomaly_type: AnomalyType
    field_name: str
    record_ids: List[str]
    description: str
    severity: Severity
    confidence: ConfidenceLevel
    detected_value: Optional[Any] = None
    expected_range: Optional[Dict[str, Any]] = None
    statistical_metrics: Dict[str, float] = Field(default_factory=dict)
    suggested_action: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DuplicateGroup(BaseModel):
    """Group of duplicate records."""
    group_id: str
    duplicate_type: DuplicateType
    record_ids: List[str]
    matching_fields: List[str]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    master_record_id: Optional[str] = None
    suggested_action: str  # merge, delete, review
    merge_strategy: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityRule(BaseModel):
    """Data quality rule definition."""
    rule_id: str
    rule_name: str
    rule_type: QualityRuleType
    target_fields: List[str]
    rule_logic: str
    expected_outcome: str
    severity_on_failure: Severity
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FieldMetrics(BaseModel):
    """Quality metrics for a specific field."""
    field_name: str
    completeness: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    validity: float = Field(..., ge=0.0, le=1.0)
    uniqueness: float = Field(..., ge=0.0, le=1.0)
    
    # Statistics
    total_values: int
    null_count: int
    unique_count: int
    invalid_count: int
    
    # Patterns
    common_patterns: List[str] = Field(default_factory=list)
    outlier_values: List[Any] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
