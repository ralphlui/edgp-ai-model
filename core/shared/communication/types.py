"""
Standardized Agent Communication Types

Provides standardized input/output types for agent communication
following industry best practices for agentic AI systems.
"""

import uuid
from datetime import datetime
from typing import TypeVar, Generic, Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# Type variables for generic input/output
T = TypeVar('T')
U = TypeVar('U')


class Priority(Enum):
    """Request priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class MessageType(Enum):
    """Message type classification."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    NOTIFICATION = "notification"
    ERROR = "error"


class Status(Enum):
    """Processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TraceInfo:
    """Distributed tracing information."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Execution context information."""
    environment: str = "development"
    version: str = "1.0.0"
    deployment_id: Optional[str] = None
    region: Optional[str] = None
    instance_id: Optional[str] = None


@dataclass
class SecurityContext:
    """Security context for requests."""
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class StandardAgentInput(Generic[T]):
    """
    Standardized input format for agent communication.
    
    Provides a consistent interface for all agent interactions with
    proper tracing, security, and metadata handling.
    """
    
    # Core identification
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Message data
    data: T = None
    message_type: MessageType = MessageType.REQUEST
    priority: Priority = Priority.MEDIUM
    
    # Temporal information
    timestamp: datetime = field(default_factory=datetime.now)
    timeout_seconds: Optional[int] = None
    expires_at: Optional[datetime] = None
    
    # Context and routing
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    
    # Security and authorization
    security_context: Optional[SecurityContext] = None
    
    # Distributed tracing
    trace: TraceInfo = field(default_factory=TraceInfo)
    
    # Execution context
    execution_context: Optional[ExecutionContext] = None
    
    # Metadata and configuration
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Request-specific options
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "data": self.data,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "security_context": self.security_context.__dict__ if self.security_context else None,
            "trace": self.trace.__dict__,
            "execution_context": self.execution_context.__dict__ if self.execution_context else None,
            "metadata": self.metadata,
            "headers": self.headers,
            "configuration": self.configuration,
            "options": self.options
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)
    
    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def has_permission(self, required_permission: str) -> bool:
        """Check if the request has required permission."""
        if not self.security_context:
            return False
        return required_permission in self.security_context.permissions
    
    def has_role(self, required_role: str) -> bool:
        """Check if the request has required role."""
        if not self.security_context:
            return False
        return required_role in self.security_context.roles


@dataclass
class StandardAgentOutput(Generic[U]):
    """
    Standardized output format for agent responses.
    
    Provides consistent response format with proper error handling,
    metrics, and tracing information.
    """
    
    # Core identification
    request_id: str
    agent_id: str
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Response data
    data: Optional[U] = None
    success: bool = True
    status: Status = Status.COMPLETED
    message: Optional[str] = None
    
    # Error information
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    error_trace: Optional[str] = None
    
    # Temporal information
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Distributed tracing
    trace: Optional[TraceInfo] = None
    
    # Performance metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Resource usage
    resources_used: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata and context
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Validation and quality
    confidence_score: Optional[float] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Follow-up information
    suggestions: List[str] = field(default_factory=list)
    next_actions: List[Dict[str, Any]] = field(default_factory=list)
    related_resources: List[str] = field(default_factory=list)
    
    # Caching information
    cache_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "response_id": self.response_id,
            "data": self.data,
            "success": self.success,
            "status": self.status.value,
            "message": self.message,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "error_trace": self.error_trace,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "trace": self.trace.__dict__ if self.trace else None,
            "metrics": self.metrics,
            "resources_used": self.resources_used,
            "metadata": self.metadata,
            "headers": self.headers,
            "confidence_score": self.confidence_score,
            "quality_metrics": self.quality_metrics,
            "suggestions": self.suggestions,
            "next_actions": self.next_actions,
            "related_resources": self.related_resources,
            "cache_info": self.cache_info
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)
    
    def set_processing_time(self):
        """Calculate and set processing time from start_time to now."""
        if self.start_time:
            self.end_time = datetime.now()
            delta = self.end_time - self.start_time
            self.processing_time_ms = delta.total_seconds() * 1000
    
    def add_metric(self, name: str, value: Any):
        """Add a performance metric."""
        self.metrics[name] = value
    
    def add_resource_usage(self, resource: str, amount: Any):
        """Add resource usage information."""
        self.resources_used[resource] = amount
    
    def add_suggestion(self, suggestion: str):
        """Add a suggestion for follow-up actions."""
        self.suggestions.append(suggestion)
    
    def add_next_action(self, action: str, parameters: Dict[str, Any] = None):
        """Add a next action recommendation."""
        self.next_actions.append({
            "action": action,
            "parameters": parameters or {}
        })


class ValidationError(Exception):
    """Validation error for agent communication."""
    
    def __init__(self, message: str, field: str = None, code: str = None):
        super().__init__(message)
        self.field = field
        self.code = code
        self.message = message


# Validation functions
def validate_input(agent_input: StandardAgentInput) -> bool:
    """
    Validate standardized agent input.
    
    Args:
        agent_input: The input to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not agent_input.request_id:
        raise ValidationError("request_id is required", "request_id", "MISSING_REQUIRED_FIELD")
    
    if agent_input.is_expired():
        raise ValidationError("Request has expired", "expires_at", "REQUEST_EXPIRED")
    
    if agent_input.data is None:
        raise ValidationError("data field cannot be None", "data", "MISSING_DATA")
    
    # Validate timeout
    if agent_input.timeout_seconds is not None and agent_input.timeout_seconds <= 0:
        raise ValidationError("timeout_seconds must be positive", "timeout_seconds", "INVALID_TIMEOUT")
    
    return True


def validate_output(agent_output: StandardAgentOutput) -> bool:
    """
    Validate standardized agent output.
    
    Args:
        agent_output: The output to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not agent_output.request_id:
        raise ValidationError("request_id is required", "request_id", "MISSING_REQUIRED_FIELD")
    
    if not agent_output.agent_id:
        raise ValidationError("agent_id is required", "agent_id", "MISSING_REQUIRED_FIELD")
    
    if not agent_output.response_id:
        raise ValidationError("response_id is required", "response_id", "MISSING_REQUIRED_FIELD")
    
    # Validate error information
    if not agent_output.success and not agent_output.error_code:
        raise ValidationError("error_code required when success=False", "error_code", "MISSING_ERROR_CODE")
    
    # Validate confidence score
    if agent_output.confidence_score is not None:
        if not 0.0 <= agent_output.confidence_score <= 1.0:
            raise ValidationError("confidence_score must be between 0.0 and 1.0", "confidence_score", "INVALID_CONFIDENCE")
    
    return True


# Convenience functions for creating standardized messages
def create_standard_input(
    data: T,
    user_id: str = None,
    session_id: str = None,
    agent_id: str = None,
    priority: Priority = Priority.MEDIUM,
    **kwargs
) -> StandardAgentInput[T]:
    """
    Create a standardized agent input with common fields.
    
    Args:
        data: The input data
        user_id: User identifier
        session_id: Session identifier
        agent_id: Target agent identifier
        priority: Request priority
        **kwargs: Additional fields
        
    Returns:
        StandardAgentInput instance
    """
    return StandardAgentInput(
        data=data,
        user_id=user_id,
        session_id=session_id,
        agent_id=agent_id,
        priority=priority,
        **kwargs
    )


def create_standard_output(
    request_id: str,
    agent_id: str,
    data: U = None,
    success: bool = True,
    message: str = None,
    **kwargs
) -> StandardAgentOutput[U]:
    """
    Create a standardized agent output with common fields.
    
    Args:
        request_id: The original request ID
        agent_id: The responding agent ID
        data: The response data
        success: Whether the operation was successful
        message: Optional status message
        **kwargs: Additional fields
        
    Returns:
        StandardAgentOutput instance
    """
    return StandardAgentOutput(
        request_id=request_id,
        agent_id=agent_id,
        data=data,
        success=success,
        message=message,
        **kwargs
    )


def create_error_output(
    request_id: str,
    agent_id: str,
    error_code: str,
    error_message: str,
    error_details: Dict[str, Any] = None,
    **kwargs
) -> StandardAgentOutput[None]:
    """
    Create a standardized error output.
    
    Args:
        request_id: The original request ID
        agent_id: The responding agent ID
        error_code: Error code
        error_message: Error message
        error_details: Additional error details
        **kwargs: Additional fields
        
    Returns:
        StandardAgentOutput instance with error information
    """
    return StandardAgentOutput(
        request_id=request_id,
        agent_id=agent_id,
        data=None,
        success=False,
        status=Status.FAILED,
        message=error_message,
        error_code=error_code,
        error_details=error_details or {},
        **kwargs
    )


# Context managers for tracing
class TracingContext:
    """Context manager for distributed tracing."""
    
    def __init__(self, operation_name: str, parent_trace: TraceInfo = None):
        self.operation_name = operation_name
        self.parent_trace = parent_trace
        self.trace_info = None
    
    def __enter__(self) -> TraceInfo:
        self.trace_info = TraceInfo(
            operation_name=self.operation_name,
            parent_span_id=self.parent_trace.span_id if self.parent_trace else None
        )
        return self.trace_info
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace_info:
            # Calculate duration
            duration = datetime.now() - self.trace_info.start_time
            self.trace_info.tags["duration_ms"] = duration.total_seconds() * 1000
            
            # Add error information if exception occurred
            if exc_type:
                self.trace_info.tags["error"] = "true"
                self.trace_info.tags["error_type"] = exc_type.__name__
                self.trace_info.tags["error_message"] = str(exc_val) if exc_val else ""


# Performance monitoring utilities
def measure_performance(func):
    """Decorator to measure function performance."""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            
            # Add performance metrics to output if it's a StandardAgentOutput
            if isinstance(result, StandardAgentOutput):
                result.start_time = start_time
                result.set_processing_time()
            
            return result
            
        except Exception as e:
            # Add error metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            # If we have an output object in args, add error info
            for arg in args:
                if isinstance(arg, StandardAgentOutput):
                    arg.success = False
                    arg.status = Status.FAILED
                    arg.error_code = "PROCESSING_ERROR"
                    arg.error_details = {"exception": str(e)}
                    arg.processing_time_ms = duration
                    break
            
            raise
    
    return wrapper
