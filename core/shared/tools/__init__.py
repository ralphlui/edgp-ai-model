"""
Tool Management Package

Comprehensive tool management system for AI agents including registration,
execution, workflow orchestration, and analytics.
"""

from .types import (
    # Core data models
    ToolDefinition,
    ToolExecution,
    ToolResult,
    ToolWorkflow,
    ToolRegistration,
    ToolParameter,
    
    # Enums
    ToolType,
    ParameterType,
    ToolStatus,
    ToolSecurityLevel,
    ToolExecutionMode,
    
    # Configuration
    ToolConfig,
    ToolAnalytics,
    
    # Base classes
    BaseToolHandler
)

from .manager import ToolManager, ToolRegistry, ToolExecutor, WorkflowEngine

__all__ = [
    # Main manager
    "ToolManager",
    
    # Core components
    "ToolRegistry",
    "ToolExecutor", 
    "WorkflowEngine",
    
    # Data models
    "ToolDefinition",
    "ToolExecution",
    "ToolResult",
    "ToolWorkflow",
    "ToolRegistration",
    "ToolParameter",
    
    # Enums
    "ToolType",
    "ParameterType",
    "ToolStatus",
    "ToolSecurityLevel",
    "ToolExecutionMode",
    
    # Configuration
    "ToolConfig",
    "ToolAnalytics",
    
    # Base classes
    "BaseToolHandler"
]


def create_tool_manager(
    storage_path: str = "./data/tools",
    default_timeout: int = 30,
    max_concurrent_executions: int = 100,
    enable_workflows: bool = True,
    enable_metrics: bool = True,
    **kwargs
) -> ToolManager:
    """
    Create a tool manager with default configuration.
    
    Args:
        storage_path: Path for tool storage
        default_timeout: Default execution timeout in seconds
        max_concurrent_executions: Maximum concurrent executions
        enable_workflows: Enable workflow engine
        enable_metrics: Enable performance metrics
        **kwargs: Additional configuration options
        
    Returns:
        Configured ToolManager instance
    """
    config = ToolConfig(
        storage_path=storage_path,
        default_timeout=default_timeout,
        max_concurrent_executions=max_concurrent_executions,
        enable_workflow_engine=enable_workflows,
        enable_metrics=enable_metrics,
        **kwargs
    )
    
    return ToolManager(config)


def create_default_config() -> ToolConfig:
    """Create default tool management configuration."""
    return ToolConfig(
        storage_path="./data/tools",
        cache_size=1000,
        default_timeout=30,
        max_concurrent_executions=100,
        enable_execution_history=True,
        execution_history_retention_days=30,
        enable_security_validation=True,
        require_authentication=True,
        enable_caching=True,
        cache_ttl_seconds=3600,
        enable_metrics=True,
        enable_workflow_engine=True,
        enable_tool_discovery=True,
        enable_auto_registration=False
    )
