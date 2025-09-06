"""
LangGraph Workflows for Collaborative AI Platform

This package contains specialized LangGraph workflows for different agent types,
providing sophisticated state-based orchestration and automation capabilities.
"""

# Import remediation workflows
from .remediation import (
    RemediationState,
    DataRemediationGraph,
    AutomatedRemediationGraph,
    BatchRemediationGraph,
    RemediationOrchestrator,
    WorkflowType,
    RiskLevel,
    WorkflowSelection,
    create_remediation_graph,
    create_automated_remediation_graph,
    create_batch_remediation_graph,
    create_remediation_orchestrator
)

__all__ = [
    # Remediation workflows
    'RemediationState',
    'DataRemediationGraph',
    'AutomatedRemediationGraph',
    'BatchRemediationGraph',
    'RemediationOrchestrator',
    'WorkflowType',
    'RiskLevel',
    'WorkflowSelection',
    'create_remediation_graph',
    'create_automated_remediation_graph',
    'create_batch_remediation_graph',
    'create_remediation_orchestrator'
]

# Package metadata
__version__ = "1.0.0"
__description__ = "LangGraph workflows for collaborative AI platform"
__author__ = "EDGP AI Model Team"

# Workflow registry for dynamic discovery
AVAILABLE_WORKFLOWS = {
    "remediation": {
        "description": "Data remediation workflows with risk assessment and automation",
        "types": ["standard", "automated", "batch"],
        "module": "graphs.remediation"
    }
    # Future workflows can be added here:
    # "compliance": {
    #     "description": "Data privacy compliance workflows",
    #     "types": ["audit", "remediation", "reporting"],
    #     "module": "graphs.compliance"
    # },
    # "analytics": {
    #     "description": "Data analytics and insights workflows",
    #     "types": ["exploration", "modeling", "reporting"],
    #     "module": "graphs.analytics"
    # }
}

def get_available_workflows():
    """Get information about all available workflow types."""
    return AVAILABLE_WORKFLOWS

def get_workflow_info(workflow_name: str):
    """Get detailed information about a specific workflow."""
    return AVAILABLE_WORKFLOWS.get(workflow_name, {})

# Quick usage examples
USAGE_EXAMPLES = {
    "basic_remediation": """
from core.shared import SharedServices
from graphs import create_remediation_orchestrator

# Initialize and use remediation orchestrator
shared_services = SharedServices()
orchestrator = await create_remediation_orchestrator(shared_services)

# Analyze data and get workflow recommendation
data_context = {"size_mb": 100, "record_count": 10000, "contains_pii": False}
selection = await orchestrator.analyze_and_select_workflow(data_context)

# Execute the recommended workflow
result = await orchestrator.execute_workflow(selection, {"dataset_id": "my_data"})
print(f"Resolved {result['state'].resolved_issues} issues")
    """,
    
    "automated_workflow": """
from graphs import create_automated_remediation_graph

# Use automated workflow directly for low-risk scenarios
shared_services = SharedServices()
auto_graph = await create_automated_remediation_graph(shared_services)

# Create initial state and process
from graphs.remediation import RemediationState
state = RemediationState(user_id="user123", dataset_id="quick_fix")
result = await auto_graph.process(state)
    """,
    
    "batch_processing": """
from graphs import create_batch_remediation_graph

# Use batch workflow for large datasets
shared_services = SharedServices()
batch_graph = await create_batch_remediation_graph(shared_services)

# Process multiple datasets
state = RemediationState(user_id="user123", dataset_id="batch_job")
result = await batch_graph.process(state)
    """
}

def get_usage_examples():
    """Get usage examples for different workflow types."""
    return USAGE_EXAMPLES

def get_quick_start():
    """Get quick start guide for the graphs package."""
    return """
# LangGraph Workflows Quick Start

## 1. Import the package
```python
from core.shared import SharedServices
from graphs import create_remediation_orchestrator
```

## 2. Initialize shared services
```python
shared_services = SharedServices()
```

## 3. Create an orchestrator
```python
orchestrator = await create_remediation_orchestrator(shared_services)
```

## 4. Analyze your data
```python
data_context = {
    "size_mb": 150,
    "record_count": 25000,
    "contains_pii": False,
    "business_criticality": "medium"
}
selection = await orchestrator.analyze_and_select_workflow(data_context)
```

## 5. Execute the workflow
```python
result = await orchestrator.execute_workflow(selection, {"dataset_id": "my_dataset"})
```

## Available Workflow Types
- **Standard**: Full oversight with human validation
- **Automated**: Minimal intervention for low-risk scenarios  
- **Batch**: Optimized for large dataset processing

## Risk Levels
- **Low**: Simple data, no PII, minimal impact
- **Medium**: Moderate complexity and business impact
- **High**: Contains PII or business-critical data
- **Critical**: Regulatory compliance, high-value data

The orchestrator automatically selects the best workflow based on your data characteristics and risk assessment.
"""
