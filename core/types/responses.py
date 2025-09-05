"""
Standard response types for the API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .base import AgentType, Severity, TaskStatus


class StandardResponse(BaseModel):
    """Standard API response format."""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: str
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response format."""
    success: bool = False
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: Optional[str] = None
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # healthy, degraded, unhealthy
    version: str
    timestamp: str
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    uptime_seconds: Optional[float] = None


class MetricsResponse(BaseModel):
    """Metrics response format."""
    metrics: Dict[str, Any]
    timestamp: str
    collection_period: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationError(BaseModel):
    """Validation error details."""
    field: str
    message: str
    value: Any
    code: Optional[str] = None


# New MCP and message-driven architecture response types

class MessageResponse(BaseModel):
    """MCP message send response."""
    success: bool
    message_id: str
    recipient: str
    timestamp: str
    error: Optional[str] = None


class AgentStatusResponse(BaseModel):
    """MCP Agent status response."""
    agent_id: str
    status: str  # active, inactive, error
    capabilities: List[str]
    message_queue_size: int
    last_activity: Optional[str] = None
    error: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Workflow execution response."""
    execution_id: str
    workflow_name: str
    status: str  # started, running, completed, failed
    timestamp: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Update HealthResponse for MCP architecture
class HealthResponse(BaseModel):
    """Health check response with MCP support."""
    status: str  # healthy, degraded, unhealthy
    version: str
    timestamp: str
    services: Dict[str, str] = Field(default_factory=dict)
    agents: List[str] = Field(default_factory=list)
    message_driven_architecture: bool = False
    uptime_seconds: Optional[float] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Response for batch operations."""
    total_items: int
    successful_items: int
    failed_items: int
    errors: List[ValidationError] = Field(default_factory=list)
    results: List[Any] = Field(default_factory=list)
    processing_time_ms: float


class PaginatedResponse(BaseModel):
    """Paginated response format."""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class AgentStatusResponse(BaseModel):
    """Agent status information."""
    agent_name: str
    agent_type: AgentType
    status: TaskStatus
    last_activity: str
    processed_requests: int
    success_rate: float
    average_response_time_ms: float
    current_load: float
    capabilities: List[str] = Field(default_factory=list)


class WorkflowStatusResponse(BaseModel):
    """Workflow execution status."""
    workflow_id: str
    workflow_type: str
    status: TaskStatus
    progress_percentage: float
    current_step: str
    involved_agents: List[AgentType]
    start_time: str
    estimated_completion: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
