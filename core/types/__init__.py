"""
Common types and schemas for EDGP AI Model service.
Provides standardized input/output types for all agents.
"""

from .base import *
from .data import *
from .agent_types import *
from .responses import *

__all__ = [
    # Base types
    "AgentType",
    "TaskStatus", 
    "Priority",
    "Severity",
    "ConfidenceLevel",
    
    # Data types
    "DataSchema",
    "DataSource",
    "DataRecord",
    "DataIssue",
    "QualityMetrics",
    "ComplianceViolation",
    
    # Agent request/response types
    "AgentRequest",
    "AgentResponse",
    "PolicySuggestionRequest",
    "PolicySuggestionResponse",
    "DataQualityRequest", 
    "DataQualityResponse",
    "ComplianceRequest",
    "ComplianceResponse",
    "RemediationRequest",
    "RemediationResponse",
    "AnalyticsRequest",
    "AnalyticsResponse",
    
    # Common response types
    "StandardResponse",
    "ErrorResponse",
    "HealthResponse",
    "MetricsResponse"
]
