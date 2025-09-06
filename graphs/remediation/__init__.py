"""
Remediation Graphs Package

Comprehensive LangGraph-based workflows for data remediation processes.
Provides multiple workflow types with intelligent orchestration and automation.
"""

from .base_workflow import (
    RemediationState,
    DataRemediationGraph,
    create_remediation_graph
)

from .automated_workflow import (
    AutomatedRemediationGraph,
    BatchRemediationGraph,
    create_automated_remediation_graph,
    create_batch_remediation_graph
)

from .orchestrator import (
    RemediationOrchestrator,
    WorkflowType,
    RiskLevel,
    WorkflowSelection,
    create_remediation_orchestrator
)

__all__ = [
    # Base workflow
    'RemediationState',
    'DataRemediationGraph',
    'create_remediation_graph',
    
    # Specialized workflows
    'AutomatedRemediationGraph',
    'BatchRemediationGraph',
    'create_automated_remediation_graph',
    'create_batch_remediation_graph',
    
    # Orchestration
    'RemediationOrchestrator',
    'WorkflowType',
    'RiskLevel',
    'WorkflowSelection',
    'create_remediation_orchestrator'
]

# Version information
__version__ = "1.0.0"
__author__ = "EDGP AI Model Team"
__description__ = "LangGraph-based remediation workflows for collaborative AI platform"

# Quick start guide
QUICK_START = """
# Remediation Graphs Quick Start

## Basic Usage

```python
from core.shared import SharedServices
from graphs.remediation import create_remediation_orchestrator, RemediationState

# Initialize shared services
shared_services = SharedServices()

# Create orchestrator
orchestrator = await create_remediation_orchestrator(shared_services)

# Analyze and select workflow
data_context = {
    "size_mb": 150,
    "record_count": 25000,
    "contains_pii": False,
    "business_criticality": "medium"
}

selection = await orchestrator.analyze_and_select_workflow(data_context)
print(f"Selected: {selection.workflow_type.value} workflow")

# Execute workflow
input_data = {"dataset_id": "my_dataset"}
result = await orchestrator.execute_workflow(selection, input_data)
print(f"Resolved {result['state'].resolved_issues} issues")
```

## Available Workflows

1. **Standard Workflow** (`WorkflowType.STANDARD`)
   - Full human oversight with step-by-step validation
   - Risk assessment, backup creation, manual approval
   - Best for: High-risk data, PII, business-critical systems

2. **Automated Workflow** (`WorkflowType.AUTOMATED`)
   - Minimal human intervention for low-risk scenarios
   - Quick assessment and automated fixing
   - Best for: Simple data quality issues, non-critical data

3. **Batch Workflow** (`WorkflowType.BATCH`)
   - Optimized for large datasets with parallel processing
   - Aggregated reporting and performance optimization
   - Best for: Large datasets, bulk operations

## Workflow Components

### RemediationState
Central state management for all workflow types:
- Message history and error tracking
- Issue detection and resolution metrics
- Quality improvement measurements
- Shared context for cross-workflow data

### Orchestrator Intelligence
- Automatic workflow selection based on data characteristics
- Risk assessment and safety validation
- Resource estimation and performance tracking
- Multiple workflow recommendations with ranking

## Configuration Options

```python
# Custom workflow configuration
user_preferences = {
    "preferred_workflow": "automated",
    "backup_policy": "always",
    "validation_level": "strict"
}

selection = await orchestrator.analyze_and_select_workflow(
    data_context, 
    user_preferences
)
```

## Integration with Shared Services

All workflows integrate seamlessly with:
- LLM Gateway for intelligent processing
- RAG System for knowledge-based decisions
- Authentication and monitoring services
- External integrations and APIs

## Performance Monitoring

```python
# Get performance summary
summary = orchestrator._get_performance_summary()
print(f"Total workflows executed: {summary['total_workflows_executed']}")

# View workflow history
for record in orchestrator.workflow_history:
    print(f"{record['workflow_type']}: {record['execution_time']:.2f}s")
```
"""

def get_quick_start_guide() -> str:
    """Get the quick start guide for remediation graphs."""
    return QUICK_START

def get_available_workflows() -> dict:
    """Get information about available workflow types."""
    return {
        "standard": {
            "description": "Full oversight workflow with human validation",
            "best_for": ["High-risk data", "PII handling", "Business-critical systems"],
            "features": ["Risk assessment", "Backup creation", "Step-by-step validation", "Rollback capability"]
        },
        "automated": {
            "description": "Minimal intervention workflow for low-risk scenarios",
            "best_for": ["Simple quality issues", "Non-critical data", "Routine maintenance"],
            "features": ["Quick assessment", "Automated fixing", "Batch processing", "Fast execution"]
        },
        "batch": {
            "description": "Optimized workflow for large dataset processing",
            "best_for": ["Large datasets", "Bulk operations", "Performance-critical tasks"],
            "features": ["Parallel processing", "Checkpoint management", "Aggregated reporting", "Resource scaling"]
        }
    }

def get_risk_levels() -> dict:
    """Get information about risk assessment levels."""
    return {
        "low": {
            "description": "Minimal impact with simple data structures",
            "characteristics": ["No PII", "Small datasets", "Low business impact"],
            "recommended_workflow": "automated"
        },
        "medium": {
            "description": "Moderate impact requiring careful handling",
            "characteristics": ["Some business impact", "Medium complexity", "Standard compliance"],
            "recommended_workflow": "standard"
        },
        "high": {
            "description": "Significant impact requiring oversight",
            "characteristics": ["Contains PII", "Business-critical", "Complex dependencies"],
            "recommended_workflow": "standard"
        },
        "critical": {
            "description": "Maximum impact requiring strict controls",
            "characteristics": ["Regulatory compliance", "High-value data", "Many dependencies"],
            "recommended_workflow": "standard"
        }
    }

def get_integration_examples() -> dict:
    """Get examples of integrating with other system components."""
    return {
        "agent_integration": """
# Integrate with existing agents
from agents.data_remediation.agent import DataRemediationAgent
from graphs.remediation import create_remediation_orchestrator

class EnhancedRemediationAgent(DataRemediationAgent):
    async def initialize(self):
        await super().initialize()
        self.orchestrator = await create_remediation_orchestrator(self.shared_services)
    
    async def process_data_issues(self, data_context):
        selection = await self.orchestrator.analyze_and_select_workflow(data_context)
        return await self.orchestrator.execute_workflow(selection, data_context)
        """,
        
        "api_integration": """
# API endpoint integration
from fastapi import APIRouter
from graphs.remediation import create_remediation_orchestrator

router = APIRouter()

@router.post("/remediation/analyze")
async def analyze_workflow(data_context: dict):
    orchestrator = await create_remediation_orchestrator(shared_services)
    recommendations = await orchestrator.get_workflow_recommendations(data_context)
    return {"recommendations": recommendations}

@router.post("/remediation/execute")
async def execute_workflow(selection: dict, input_data: dict):
    orchestrator = await create_remediation_orchestrator(shared_services)
    result = await orchestrator.execute_workflow(selection, input_data)
    return result
        """,
        
        "monitoring_integration": """
# Integration with monitoring system
from core.infrastructure.monitoring import MetricsCollector

class RemediationMetrics(MetricsCollector):
    async def track_workflow_execution(self, execution_record):
        await self.record_metric("remediation.execution_time", execution_record["execution_time"])
        await self.record_metric("remediation.issues_resolved", execution_record["issues_resolved"])
        await self.record_metric("remediation.success_rate", execution_record["success"])
        """
    }
