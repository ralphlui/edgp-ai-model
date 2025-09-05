# Type System Documentation

## Overview

The EDGP AI Model service implements a comprehensive typing system to ensure type safety and standardized communication between agents. This document outlines the type hierarchy and usage patterns.

## Type Hierarchy

```
core/types/
├── __init__.py                 # Main exports
├── base.py                    # Base types and enums
├── data.py                    # Data-related types
├── agent_types.py             # Agent request/response types
├── responses.py               # Standard API responses
├── validation.py              # Type validation utilities
└── agents/                    # Agent-specific types
    ├── __init__.py
    ├── policy_suggestion.py
    ├── data_quality.py
    ├── compliance.py
    ├── remediation.py
    └── analytics.py
```

## Core Type Categories

### 1. Base Types (`base.py`)

**Enums for standardization**:
- `AgentType`: Types of agents in the system
- `TaskStatus`: Status of tasks and operations  
- `Priority`: Priority levels (CRITICAL, HIGH, MEDIUM, LOW)
- `Severity`: Severity levels for issues
- `ConfidenceLevel`: AI prediction confidence levels
- `DataType`: Data field types (STRING, INTEGER, EMAIL, etc.)
- `SensitivityLevel`: Data sensitivity classification
- `ComplianceRegulation`: Supported regulations (GDPR, CCPA, HIPAA, etc.)

**Base Models**:
- `BaseTimestamped`: Base model with created_at/updated_at
- `TimeRange`: Time period specification
- `Pagination`: Pagination parameters
- `SortBy`: Sorting specification

### 2. Data Types (`data.py`)

**Core Data Models**:
- `DataField`: Schema definition for individual fields
- `DataSchema`: Complete dataset schema
- `DataSource`: Data source definition with metadata
- `DataRecord`: Individual data record
- `DataIssue`: Quality or compliance issue
- `QualityMetrics`: Comprehensive quality metrics
- `ComplianceViolation`: Compliance violation details
- `RemediationTask`: Data remediation task
- `PolicyRecommendation`: Policy suggestion details

### 3. Agent Communication (`agent_types.py`)

**Base Communication**:
- `AgentRequest`: Base request for all agents
- `AgentResponse`: Base response from all agents
- `AgentMessage`: Inter-agent communication
- `WorkflowRequest`: Multi-agent workflow request
- `WorkflowResponse`: Multi-agent workflow response

**Agent-Specific Types**:
- `PolicySuggestionRequest/Response`
- `ComplianceRequest/Response`
- `DataQualityRequest/Response`
- `RemediationRequest/Response`
- `AnalyticsRequest/Response`

### 4. API Responses (`responses.py`)

**Standard Formats**:
- `StandardResponse`: Standard API response format
- `ErrorResponse`: Error response format
- `HealthResponse`: Health check response
- `MetricsResponse`: Metrics response format
- `BatchResponse`: Batch operation response
- `PaginatedResponse`: Paginated response format

## Agent-Specific Types

### Policy Suggestion Agent (`agents/policy_suggestion.py`)

```python
class PolicyType(str, Enum):
    ACCESS_CONTROL = "access_control"
    DATA_RETENTION = "data_retention"
    DATA_VALIDATION = "data_validation"
    # ... more types

class ValidationRule(BaseModel):
    rule_id: str
    rule_name: str
    rule_type: str
    target_fields: List[str]
    validation_logic: str
    # ... more fields
```

### Data Quality Agent (`agents/data_quality.py`)

```python
class AnomalyType(str, Enum):
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_VIOLATION = "pattern_violation"
    # ... more types

class Anomaly(BaseModel):
    anomaly_id: str
    anomaly_type: AnomalyType
    field_name: str
    # ... more fields
```

### Compliance Agent (`agents/compliance.py`)

```python
class PrivacyRiskType(str, Enum):
    PII_EXPOSURE = "pii_exposure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    # ... more types

class PrivacyRisk(BaseModel):
    risk_id: str
    risk_type: PrivacyRiskType
    severity: Severity
    # ... more fields
```

## Type Validation System

### TypeValidator Class

The `TypeValidator` class provides utilities for:

1. **Input Validation**:
```python
validated_request = TypeValidator.validate_input(raw_data, PolicySuggestionRequest)
```

2. **Output Serialization**:
```python
response_dict = TypeValidator.validate_output(response_model)
```

3. **Legacy Format Conversion**:
```python
new_format = TypeValidator.convert_legacy_format(old_data, NewModel, field_mapping)
```

4. **Agent Compatibility**:
```python
compatible_input = TypeValidator.ensure_agent_compatibility(
    source_output, target_input_type
)
```

## Usage Examples

### 1. Policy Suggestion Request

```python
from core.types.agent_types import PolicySuggestionRequest
from core.types.base import ComplianceRegulation

request = PolicySuggestionRequest(
    request_id="req-123",
    agent_type=AgentType.POLICY_SUGGESTION,
    timestamp="2024-01-15T10:30:00Z",
    business_context="Financial services company",
    compliance_requirements=[ComplianceRegulation.GDPR, ComplianceRegulation.PCI_DSS],
    suggestion_type="validation_policies"
)
```

### 2. Data Quality Response

```python
from core.types.agent_types import DataQualityResponse
from core.types.data import QualityMetrics

response = DataQualityResponse(
    request_id="req-456",
    agent_type=AgentType.DATA_QUALITY,
    timestamp="2024-01-15T10:35:00Z",
    success=True,
    overall_quality_score=0.88,
    quality_metrics=QualityMetrics(
        dataset_id="customers",
        overall_score=0.88,
        completeness=0.92,
        accuracy=0.85,
        # ... other metrics
    )
)
```

### 3. Inter-Agent Communication

```python
from core.types.agent_types import AgentMessage

message = AgentMessage(
    message_id="msg-789",
    from_agent=AgentType.DATA_QUALITY,
    to_agent=AgentType.DATA_REMEDIATION,
    message_type="quality_issues_detected",
    payload={
        "issues": [...],
        "priority": "HIGH",
        "requires_immediate_action": True
    }
)
```

## Type Safety Benefits

1. **Compile-time Validation**: Catch type errors during development
2. **API Documentation**: Automatic OpenAPI schema generation
3. **IDE Support**: Better autocomplete and error detection
4. **Version Compatibility**: Easier API evolution and backward compatibility
5. **Agent Interoperability**: Guaranteed compatibility between agent inputs/outputs

## Best Practices

### 1. Always Use Typed Requests/Responses
```python
# Good ✅
async def process_request(self, request: PolicySuggestionRequest) -> PolicySuggestionResponse:
    pass

# Avoid ❌
async def process_request(self, request: dict) -> dict:
    pass
```

### 2. Validate Inputs Early
```python
def process_data(self, raw_data: dict):
    # Validate immediately
    typed_data = TypeValidator.validate_input(raw_data, DataSchema)
    # Continue with typed_data
```

### 3. Use Enums for Standardization
```python
# Good ✅
priority = Priority.HIGH

# Avoid ❌  
priority = "high"  # String literals can have typos
```

### 4. Leverage Pydantic Features
```python
class CustomModel(BaseModel):
    # Use validators
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    
    # Use computed fields
    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
```

## Migration Guide

### From Untyped to Typed Implementation

1. **Update Agent Methods**:
   ```python
   # Before
   async def detect_anomalies(self, dataset: str, rules: list) -> dict:
       pass
   
   # After  
   async def detect_anomalies(self, request: DataQualityRequest) -> DataQualityResponse:
       pass
   ```

2. **Update API Endpoints**:
   ```python
   # Before
   @app.post("/detect-anomalies")
   async def detect_anomalies(request: dict):
       pass
   
   # After
   @app.post("/detect-anomalies", response_model=DataQualityResponse)
   async def detect_anomalies(request: DataQualityRequest) -> DataQualityResponse:
       pass
   ```

3. **Use Type Validation**:
   ```python
   # Validate inputs
   validated_request = TypeValidator.validate_input(raw_data, RequestType)
   
   # Process with typed data
   response = await agent.process(validated_request)
   
   # Validate outputs
   response_dict = TypeValidator.validate_output(response)
   ```

## Future Extensions

The type system is designed to be extensible:

1. **New Agent Types**: Add new enums to `AgentType`
2. **New Data Types**: Extend `DataType` enum
3. **New Regulations**: Add to `ComplianceRegulation`
4. **Custom Validators**: Extend `TypeValidator` class
5. **New Response Formats**: Add to `responses.py`

This type system provides a solid foundation for building a robust, maintainable, and interoperable agentic AI system.
