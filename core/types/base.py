"""
Base types and enums used across all agents.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Types of agents in the system."""
    POLICY_SUGGESTION = "policy_suggestion"
    DATA_PRIVACY_COMPLIANCE = "data_privacy_compliance"
    DATA_QUALITY = "data_quality"
    DATA_REMEDIATION = "data_remediation"
    ANALYTICS = "analytics"


class TaskStatus(str, Enum):
    """Status of tasks and operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(str, Enum):
    """Priority levels for tasks and issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Severity(str, Enum):
    """Severity levels for issues and violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConfidenceLevel(str, Enum):
    """Confidence levels for AI predictions and suggestions."""
    VERY_HIGH = "very_high"  # 95-100%
    HIGH = "high"            # 80-95%
    MEDIUM = "medium"        # 60-80%
    LOW = "low"              # 40-60%
    VERY_LOW = "very_low"    # 0-40%


class DataType(str, Enum):
    """Data types for field classification."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    JSON = "json"
    BINARY = "binary"


class SensitivityLevel(str, Enum):
    """Data sensitivity classification."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceRegulation(str, Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    FERPA = "ferpa"
    GLBA = "glba"


class QualityDimension(str, Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


class RemediationType(str, Enum):
    """Types of data remediation actions."""
    DEDUPLICATION = "deduplication"
    FORMAT_CORRECTION = "format_correction"
    VALUE_CORRECTION = "value_correction"
    ENRICHMENT = "enrichment"
    DELETION = "deletion"
    MASKING = "masking"
    ENCRYPTION = "encryption"


class VisualizationType(str, Enum):
    """Types of visualizations for analytics."""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TABLE = "table"
    DASHBOARD = "dashboard"


class BaseTimestamped(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class Coordinates(BaseModel):
    """Geographic coordinates."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class TimeRange(BaseModel):
    """Time range specification."""
    start_date: datetime
    end_date: datetime
    timezone: str = Field(default="UTC")


class Pagination(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    total_pages: Optional[int] = None
    total_items: Optional[int] = None


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class SortBy(BaseModel):
    """Sort specification."""
    field: str
    order: SortOrder = SortOrder.ASC
