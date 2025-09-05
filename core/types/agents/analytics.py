"""
Specific types for Analytics Agent operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from ..base import VisualizationType, TimeRange, Pagination, SortBy, ConfidenceLevel


class ChartType(str, Enum):
    """Types of charts for analytics."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    GAUGE = "gauge"
    FUNNEL = "funnel"


class AggregationType(str, Enum):
    """Types of data aggregations."""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"


class DashboardLayout(str, Enum):
    """Dashboard layout options."""
    GRID = "grid"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    TABBED = "tabbed"
    CUSTOM = "custom"


class ChartConfiguration(BaseModel):
    """Configuration for individual charts."""
    chart_id: str
    title: str
    chart_type: ChartType
    
    # Data configuration
    data_source: str
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    group_by: Optional[str] = None
    aggregation: AggregationType = AggregationType.COUNT
    
    # Filtering
    filters: Dict[str, Any] = Field(default_factory=dict)
    time_range: Optional[TimeRange] = None
    
    # Styling
    color_scheme: List[str] = Field(default_factory=list)
    theme: str = "default"
    width: Optional[int] = None
    height: Optional[int] = None
    
    # Interactivity
    is_interactive: bool = True
    drill_down_enabled: bool = False
    export_enabled: bool = True
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dashboard(BaseModel):
    """Analytics dashboard definition."""
    dashboard_id: str
    title: str
    description: Optional[str] = None
    
    # Layout
    layout: DashboardLayout = DashboardLayout.GRID
    grid_columns: int = 3
    
    # Components
    charts: List[ChartConfiguration] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    kpi_widgets: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Data refresh
    auto_refresh: bool = True
    refresh_interval_minutes: int = 15
    last_updated: Optional[datetime] = None
    
    # Access control
    visibility: str = "private"  # private, team, public
    authorized_users: List[str] = Field(default_factory=list)
    authorized_roles: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReportTemplate(BaseModel):
    """Template for generating reports."""
    template_id: str
    template_name: str
    report_type: str  # quality, compliance, remediation, executive
    
    # Structure
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    default_visualizations: List[str] = Field(default_factory=list)
    
    # Configuration
    auto_generate: bool = False
    schedule: Optional[str] = None  # cron expression
    output_formats: List[str] = Field(default=["pdf", "html"])
    
    # Recipients
    default_recipients: List[str] = Field(default_factory=list)
    distribution_list: List[Dict[str, str]] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KPIDefinition(BaseModel):
    """Key Performance Indicator definition."""
    kpi_id: str
    name: str
    description: str
    category: str  # quality, compliance, performance, business
    
    # Calculation
    calculation_method: str
    data_sources: List[str]
    formula: Optional[str] = None
    
    # Thresholds
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Display
    unit: Optional[str] = None
    format_string: Optional[str] = None
    visualization_type: VisualizationType = VisualizationType.GAUGE
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsInsight(BaseModel):
    """AI-generated insight from analytics data."""
    insight_id: str
    title: str
    description: str
    insight_type: str  # trend, anomaly, correlation, recommendation
    confidence: ConfidenceLevel
    
    # Supporting data
    supporting_charts: List[str] = Field(default_factory=list)
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Business context
    business_impact: str
    recommended_actions: List[str] = Field(default_factory=list)
    
    # Validation
    data_sources: List[str] = Field(default_factory=list)
    analysis_period: Optional[TimeRange] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
