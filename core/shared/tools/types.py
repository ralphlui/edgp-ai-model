"""
Tool Management Types

Comprehensive type system for managing AI agent tools including function
definitions, parameter validation, execution tracking, and tool chaining.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from pydantic import BaseModel, Field, validator
import json


class ToolType(str, Enum):
    """Types of tools available to agents."""
    FUNCTION = "function"
    API_CALL = "api_call"
    SCRIPT = "script"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    UTILITY = "utility"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    NOTIFICATION = "notification"


class ParameterType(str, Enum):
    """Parameter data types for tools."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    JSON = "json"


class ToolStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ToolSecurityLevel(str, Enum):
    """Security levels for tool access."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    RESTRICTED = "restricted"
    ADMIN_ONLY = "admin_only"


class ToolExecutionMode(str, Enum):
    """Tool execution modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"
    BATCH = "batch"


class ToolParameter(BaseModel):
    """Tool parameter definition."""
    name: str = Field(..., description="Parameter name")
    parameter_type: ParameterType = Field(..., description="Parameter data type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default_value: Optional[Any] = Field(default=None, description="Default value")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Validation rules")
    examples: List[Any] = Field(default_factory=list, description="Example values")
    enum_values: Optional[List[Any]] = Field(default=None, description="Allowed enum values")
    min_value: Optional[Union[int, float]] = Field(default=None, description="Minimum value")
    max_value: Optional[Union[int, float]] = Field(default=None, description="Maximum value")
    min_length: Optional[int] = Field(default=None, description="Minimum string/array length")
    max_length: Optional[int] = Field(default=None, description="Maximum string/array length")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for validation")
    
    def validate_value(self, value: Any) -> bool:
        """Validate a parameter value."""
        if self.required and value is None:
            return False
        
        if value is None:
            return True
        
        # Type validation
        if self.parameter_type == ParameterType.STRING and not isinstance(value, str):
            return False
        elif self.parameter_type == ParameterType.INTEGER and not isinstance(value, int):
            return False
        elif self.parameter_type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.parameter_type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.parameter_type == ParameterType.ARRAY and not isinstance(value, list):
            return False
        elif self.parameter_type == ParameterType.OBJECT and not isinstance(value, dict):
            return False
        
        # Enum validation
        if self.enum_values and value not in self.enum_values:
            return False
        
        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        
        # Length validation
        if hasattr(value, '__len__'):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                return False
            if self.max_length is not None and length > self.max_length:
                return False
        
        # Pattern validation
        if self.pattern and isinstance(value, str):
            import re
            if not re.match(self.pattern, value):
                return False
        
        return True


class ToolResult(BaseModel):
    """Result from tool execution."""
    success: bool = Field(..., description="Whether execution was successful")
    result: Optional[Any] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    output_type: Optional[str] = Field(default=None, description="Type of output")
    output_size: Optional[int] = Field(default=None, description="Size of output in bytes")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "result": {"data": "processed"},
                "error": None,
                "metadata": {"source": "api", "version": "1.0"},
                "execution_time": 1.25,
                "output_type": "json",
                "output_size": 256,
                "warnings": []
            }
        }


class ToolDefinition(BaseModel):
    """Definition of an AI agent tool."""
    tool_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique tool ID")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    tool_type: ToolType = Field(..., description="Type of tool")
    version: str = Field(default="1.0.0", description="Tool version")
    
    # Parameters
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    
    # Configuration
    security_level: ToolSecurityLevel = Field(default=ToolSecurityLevel.AUTHENTICATED, description="Security level")
    execution_mode: ToolExecutionMode = Field(default=ToolExecutionMode.SYNCHRONOUS, description="Execution mode")
    timeout_seconds: int = Field(default=30, description="Execution timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    category: Optional[str] = Field(default=None, description="Tool category")
    author: Optional[str] = Field(default=None, description="Tool author")
    documentation_url: Optional[str] = Field(default=None, description="Documentation URL")
    
    # Usage tracking
    usage_count: int = Field(default=0, description="Number of times used")
    success_rate: float = Field(default=1.0, description="Success rate (0-1)")
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    last_used_at: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    
    # Capabilities
    supports_streaming: bool = Field(default=False, description="Supports streaming output")
    supports_cancellation: bool = Field(default=True, description="Supports cancellation")
    is_stateful: bool = Field(default=False, description="Maintains state between calls")
    requires_context: bool = Field(default=False, description="Requires execution context")
    
    def get_parameter(self, name: str) -> Optional[ToolParameter]:
        """Get parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters against definition."""
        errors = []
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Required parameter '{param.name}' is missing")
            elif param.name in params:
                if not param.validate_value(params[param.name]):
                    errors.append(f"Parameter '{param.name}' has invalid value")
        
        # Check for unknown parameters
        defined_params = {param.name for param in self.parameters}
        for param_name in params:
            if param_name not in defined_params:
                errors.append(f"Unknown parameter '{param_name}'")
        
        return errors
    
    def update_usage_stats(self, execution_time: float, success: bool):
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Update average execution time
        if self.usage_count == 1:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (
                (self.average_execution_time * (self.usage_count - 1) + execution_time) / 
                self.usage_count
            )
        
        # Update success rate
        if success:
            new_success_count = self.success_rate * (self.usage_count - 1) + 1
        else:
            new_success_count = self.success_rate * (self.usage_count - 1)
        
        self.success_rate = new_success_count / self.usage_count
    
    class Config:
        schema_extra = {
            "example": {
                "name": "data_processor",
                "description": "Processes and transforms data",
                "tool_type": "function",
                "parameters": [
                    {
                        "name": "data",
                        "parameter_type": "object",
                        "description": "Input data to process",
                        "required": True
                    }
                ],
                "security_level": "authenticated",
                "execution_mode": "synchronous",
                "timeout_seconds": 30,
                "tags": ["data", "processing"],
                "category": "transformation"
            }
        }


class ToolExecution(BaseModel):
    """Tool execution instance."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    tool_id: str = Field(..., description="Tool being executed")
    tool_name: str = Field(..., description="Tool name")
    
    # Execution details
    status: ToolStatus = Field(default=ToolStatus.PENDING, description="Execution status")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    result: Optional[ToolResult] = Field(default=None, description="Execution result")
    
    # Context
    agent_id: Optional[str] = Field(default=None, description="Executing agent ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    workflow_id: Optional[str] = Field(default=None, description="Workflow ID if part of workflow")
    parent_execution_id: Optional[str] = Field(default=None, description="Parent execution if chained")
    
    # Timing
    started_at: Optional[datetime] = Field(default=None, description="Execution start time")
    completed_at: Optional[datetime] = Field(default=None, description="Execution completion time")
    timeout_at: Optional[datetime] = Field(default=None, description="Timeout deadline")
    
    # Metadata
    retry_count: int = Field(default=0, description="Number of retries")
    priority: int = Field(default=0, description="Execution priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Execution tags")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.status in [ToolStatus.COMPLETED, ToolStatus.FAILED, ToolStatus.CANCELLED, ToolStatus.TIMEOUT]
    
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.COMPLETED and self.result and self.result.success
    
    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def update_status(self, status: ToolStatus, result: Optional[ToolResult] = None):
        """Update execution status."""
        self.status = status
        self.updated_at = datetime.now()
        
        if status == ToolStatus.RUNNING and not self.started_at:
            self.started_at = datetime.now()
        
        if status in [ToolStatus.COMPLETED, ToolStatus.FAILED, ToolStatus.CANCELLED, ToolStatus.TIMEOUT]:
            self.completed_at = datetime.now()
            if result:
                self.result = result


class ToolWorkflow(BaseModel):
    """Workflow definition for chaining multiple tools."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    
    # Workflow steps
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow steps")
    
    # Configuration
    parallel_execution: bool = Field(default=False, description="Allow parallel step execution")
    stop_on_error: bool = Field(default=True, description="Stop workflow on step error")
    max_execution_time: int = Field(default=300, description="Maximum workflow execution time")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    category: Optional[str] = Field(default=None, description="Workflow category")
    author: Optional[str] = Field(default=None, description="Workflow author")
    
    # Usage tracking
    execution_count: int = Field(default=0, description="Number of executions")
    success_rate: float = Field(default=1.0, description="Success rate")
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def add_step(self, tool_id: str, parameters: Dict[str, Any], condition: Optional[str] = None):
        """Add a step to the workflow."""
        step = {
            "step_id": str(uuid.uuid4()),
            "tool_id": tool_id,
            "parameters": parameters,
            "condition": condition,
            "order": len(self.steps)
        }
        self.steps.append(step)
    
    def get_next_steps(self, current_step: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get next steps to execute."""
        if not current_step:
            # Return first step(s)
            if self.parallel_execution:
                return [step for step in self.steps if step.get("order", 0) == 0]
            else:
                first_step = min(self.steps, key=lambda s: s.get("order", 0), default=None)
                return [first_step] if first_step else []
        
        # Find current step and return next
        current_order = None
        for step in self.steps:
            if step["step_id"] == current_step:
                current_order = step.get("order", 0)
                break
        
        if current_order is None:
            return []
        
        next_order = current_order + 1
        next_steps = [step for step in self.steps if step.get("order", 0) == next_order]
        
        return next_steps


class ToolRegistration(BaseModel):
    """Tool registration information."""
    registration_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Registration ID")
    tool_id: str = Field(..., description="Tool ID")
    agent_id: Optional[str] = Field(default=None, description="Registering agent ID")
    handler: Optional[str] = Field(default=None, description="Handler function/class")
    endpoint: Optional[str] = Field(default=None, description="API endpoint if applicable")
    
    # Registration metadata
    registered_at: datetime = Field(default_factory=datetime.now, description="Registration timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Registration expiry")
    is_active: bool = Field(default=True, description="Registration is active")
    health_check_url: Optional[str] = Field(default=None, description="Health check endpoint")
    last_health_check: Optional[datetime] = Field(default=None, description="Last health check")
    health_status: str = Field(default="unknown", description="Health status")


class ToolConfig(BaseModel):
    """Tool management configuration."""
    # Storage settings
    storage_path: str = Field(default="./data/tools", description="Tool storage path")
    cache_size: int = Field(default=1000, description="Tool cache size")
    
    # Execution settings
    default_timeout: int = Field(default=30, description="Default execution timeout")
    max_concurrent_executions: int = Field(default=100, description="Max concurrent executions")
    enable_execution_history: bool = Field(default=True, description="Track execution history")
    execution_history_retention_days: int = Field(default=30, description="Execution history retention")
    
    # Security settings
    enable_security_validation: bool = Field(default=True, description="Enable security validation")
    allowed_tool_types: List[ToolType] = Field(default_factory=lambda: list(ToolType), description="Allowed tool types")
    require_authentication: bool = Field(default=True, description="Require authentication")
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    
    # Integration settings
    enable_workflow_engine: bool = Field(default=True, description="Enable workflow engine")
    enable_tool_discovery: bool = Field(default=True, description="Enable tool discovery")
    enable_auto_registration: bool = Field(default=False, description="Enable auto-registration")


class BaseToolHandler(ABC):
    """Base class for tool handlers."""
    
    def __init__(self, tool_definition: ToolDefinition):
        self.tool_definition = tool_definition
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters before execution."""
        return self.tool_definition.validate_parameters(parameters)
    
    async def health_check(self) -> bool:
        """Check if tool is healthy and ready for execution."""
        return True
    
    async def cleanup(self):
        """Cleanup resources after execution."""
        pass


class ToolAnalytics(BaseModel):
    """Analytics data for tool usage."""
    total_tools: int = Field(default=0, description="Total number of tools")
    total_executions: int = Field(default=0, description="Total executions")
    successful_executions: int = Field(default=0, description="Successful executions")
    failed_executions: int = Field(default=0, description="Failed executions")
    average_execution_time: float = Field(default=0.0, description="Average execution time")
    most_used_tools: List[Dict[str, Any]] = Field(default_factory=list, description="Most used tools")
    tool_type_distribution: Dict[str, int] = Field(default_factory=dict, description="Tool type distribution")
    execution_trends: Dict[str, List[int]] = Field(default_factory=dict, description="Execution trends over time")
    error_rates: Dict[str, float] = Field(default_factory=dict, description="Error rates by tool")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    
    # Time-based metrics
    executions_today: int = Field(default=0, description="Executions today")
    executions_this_week: int = Field(default=0, description="Executions this week")
    executions_this_month: int = Field(default=0, description="Executions this month")
    
    # Quality metrics
    overall_success_rate: float = Field(default=1.0, description="Overall success rate")
    reliability_score: float = Field(default=1.0, description="Tool reliability score")
    user_satisfaction: float = Field(default=0.0, description="User satisfaction rating")
    
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
