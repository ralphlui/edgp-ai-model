# API Documentation

## Overview

The EDGP AI Model exposes a RESTful API built with FastAPI that provides access to specialized AI agents for master data management. All endpoints use standardized request/response types for consistency and type safety.

## Base URL

```
Development: http://localhost:8000
Production: https://api.edgp-ai.com
```

## Authentication

Currently using API key authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.edgp-ai.com/api/v1/agents/policy-suggestion
```

## Common Response Format

All endpoints return responses in a standardized format:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "metadata": {
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z",
    "processing_time_ms": 1500
  }
}
```

## Error Response Format

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {...}
  },
  "metadata": {
    "request_id": "req-124",
    "timestamp": "2024-01-15T10:31:00Z"
  }
}
```

## System Endpoints

### Health Check

**GET** `/health`

Check system health and availability.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agents": {
    "policy_suggestion": "healthy",
    "data_quality": "healthy",
    "compliance": "healthy",
    "remediation": "healthy",
    "analytics": "healthy"
  },
  "dependencies": {
    "database": "connected",
    "redis": "connected",
    "llm_gateway": "operational"
  }
}
```

### Metrics

**GET** `/metrics`

Get system metrics and performance data.

**Response**:
```json
{
  "requests_total": 15420,
  "requests_success_rate": 0.987,
  "average_response_time_ms": 1247,
  "agents": {
    "policy_suggestion": {
      "requests": 3240,
      "success_rate": 0.991,
      "avg_response_time_ms": 1560
    }
  },
  "llm_usage": {
    "total_tokens": 2450000,
    "cost_usd": 245.50
  }
}
```

## Agent Endpoints

### 1. Policy Suggestion Agent

**POST** `/api/v1/agents/policy-suggestion`

Generate data governance policies and validation rules.

**Request Body**:
```json
{
  "request_id": "req-001",
  "agent_type": "policy_suggestion",
  "timestamp": "2024-01-15T10:30:00Z",
  "business_context": "Financial services company processing customer data",
  "compliance_requirements": ["GDPR", "PCI_DSS"],
  "data_schemas": [
    {
      "schema_id": "customer_data",
      "schema_name": "Customer Information",
      "fields": [
        {
          "field_name": "email",
          "data_type": "EMAIL",
          "is_required": true,
          "sensitivity_level": "PII"
        }
      ]
    }
  ],
  "suggestion_type": "validation_policies"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "request_id": "req-001",
    "agent_type": "policy_suggestion",
    "timestamp": "2024-01-15T10:30:15Z",
    "success": true,
    "policy_recommendations": [
      {
        "policy_id": "pol-001",
        "policy_name": "Email Validation Policy",
        "policy_type": "validation",
        "description": "Ensures all email fields contain valid email addresses",
        "target_fields": ["email"],
        "validation_rules": [
          {
            "rule_id": "rule-001",
            "rule_name": "Email Format Validation",
            "rule_type": "regex",
            "target_fields": ["email"],
            "validation_logic": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
          }
        ],
        "confidence_score": 0.95,
        "rationale": "Email validation is critical for data quality and user communication"
      }
    ],
    "processing_time_ms": 1247
  }
}
```

### 2. Data Quality Agent

**POST** `/api/v1/agents/data-quality`

Assess data quality and detect anomalies.

**Request Body**:
```json
{
  "request_id": "req-002",
  "agent_type": "data_quality",
  "timestamp": "2024-01-15T10:35:00Z",
  "data_source": {
    "source_id": "src-001",
    "source_name": "Customer Database",
    "source_type": "database",
    "connection_string": "postgresql://localhost:5432/customers"
  },
  "dataset_id": "customers",
  "quality_dimensions": ["completeness", "accuracy", "consistency"],
  "anomaly_detection": true,
  "quality_rules": [
    {
      "rule_id": "rule-002",
      "rule_name": "Email Completeness",
      "rule_type": "completeness",
      "target_fields": ["email"],
      "threshold": 0.95
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "request_id": "req-002",
    "agent_type": "data_quality",
    "timestamp": "2024-01-15T10:35:12Z",
    "success": true,
    "overall_quality_score": 0.88,
    "quality_metrics": {
      "dataset_id": "customers",
      "overall_score": 0.88,
      "completeness": 0.92,
      "accuracy": 0.85,
      "consistency": 0.89,
      "validity": 0.94,
      "uniqueness": 0.87,
      "timeliness": 0.91
    },
    "anomalies": [
      {
        "anomaly_id": "anom-001",
        "anomaly_type": "statistical_outlier",
        "field_name": "age",
        "description": "Age values >150 detected",
        "affected_records": 5,
        "severity": "MEDIUM"
      }
    ],
    "data_issues": [
      {
        "issue_id": "issue-001",
        "issue_type": "missing_values",
        "field_name": "phone_number",
        "description": "8% of phone numbers are missing",
        "severity": "LOW"
      }
    ]
  }
}
```

### 3. Data Privacy & Compliance Agent

**POST** `/api/v1/agents/compliance`

Assess compliance with data privacy regulations.

**Request Body**:
```json
{
  "request_id": "req-003",
  "agent_type": "data_privacy_compliance",
  "timestamp": "2024-01-15T10:40:00Z",
  "data_source": {
    "source_id": "src-001",
    "source_name": "Customer Database",
    "source_type": "database"
  },
  "regulations": ["GDPR", "CCPA"],
  "assessment_scope": "full_compliance_audit",
  "data_schemas": [...]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "request_id": "req-003",
    "agent_type": "data_privacy_compliance",
    "timestamp": "2024-01-15T10:40:18Z",
    "success": true,
    "overall_compliance_score": 0.87,
    "compliance_status": {
      "GDPR": {
        "status": "PARTIALLY_COMPLIANT",
        "score": 0.85,
        "violations": [
          {
            "violation_id": "viol-001",
            "regulation": "GDPR",
            "article": "Article 17",
            "description": "Right to erasure not implemented",
            "severity": "HIGH",
            "affected_fields": ["all_personal_data"]
          }
        ]
      }
    },
    "privacy_risks": [
      {
        "risk_id": "risk-001",
        "risk_type": "PII_EXPOSURE",
        "description": "Email addresses stored in plain text",
        "severity": "MEDIUM",
        "mitigation_suggestions": ["Implement field-level encryption"]
      }
    ]
  }
}
```

### 4. Data Remediation Agent

**POST** `/api/v1/agents/remediation`

Get recommendations for fixing data quality issues.

**Request Body**:
```json
{
  "request_id": "req-004",
  "agent_type": "data_remediation",
  "timestamp": "2024-01-15T10:45:00Z",
  "data_issues": [
    {
      "issue_id": "issue-001",
      "issue_type": "missing_values",
      "field_name": "phone_number",
      "description": "8% of phone numbers are missing",
      "severity": "LOW",
      "affected_records": 240
    }
  ],
  "business_constraints": {
    "budget": 10000,
    "timeline_days": 30,
    "automation_preference": "high"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "request_id": "req-004",
    "agent_type": "data_remediation",
    "timestamp": "2024-01-15T10:45:14Z",
    "success": true,
    "remediation_tasks": [
      {
        "task_id": "task-001",
        "task_name": "Phone Number Data Collection",
        "task_type": "data_enrichment",
        "description": "Implement phone number collection in user registration",
        "priority": "MEDIUM",
        "estimated_effort_hours": 16,
        "automation_level": "HIGH",
        "steps": [
          "Add phone validation to registration form",
          "Send SMS verification for existing users",
          "Update database schema with phone constraints"
        ]
      }
    ],
    "estimated_impact": {
      "quality_improvement": 0.12,
      "compliance_improvement": 0.08,
      "cost_estimate": 2400
    }
  }
}
```

### 5. Analytics Agent

**POST** `/api/v1/agents/analytics`

Generate insights and reports from data governance activities.

**Request Body**:
```json
{
  "request_id": "req-005",
  "agent_type": "analytics",
  "timestamp": "2024-01-15T10:50:00Z",
  "analysis_type": "data_governance_dashboard",
  "time_range": {
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-15T23:59:59Z"
  },
  "metrics_requested": ["quality_trends", "compliance_status", "remediation_progress"],
  "data_sources": ["src-001", "src-002"]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "request_id": "req-005",
    "agent_type": "analytics",
    "timestamp": "2024-01-15T10:50:22Z",
    "success": true,
    "insights": [
      {
        "insight_id": "insight-001",
        "insight_type": "trend",
        "title": "Data Quality Improvement Trend",
        "description": "Overall data quality has improved by 15% over the past 2 weeks",
        "confidence": 0.92,
        "supporting_data": {...}
      }
    ],
    "reports": [
      {
        "report_id": "report-001",
        "report_type": "quality_summary",
        "title": "Data Quality Summary - January 2024",
        "summary": "Quality metrics show consistent improvement...",
        "visualizations": [...]
      }
    ]
  }
}
```

## Workflow Endpoints

### Multi-Agent Workflow

**POST** `/api/v1/workflows/execute`

Execute a multi-agent workflow for comprehensive data governance.

**Request Body**:
```json
{
  "workflow_id": "data-governance-assessment",
  "workflow_name": "Complete Data Governance Assessment",
  "agents": ["data_quality", "compliance", "remediation"],
  "data_source": {
    "source_id": "src-001",
    "source_name": "Customer Database"
  },
  "parameters": {
    "include_remediation": true,
    "compliance_regulations": ["GDPR"],
    "quality_threshold": 0.85
  },
  "execution_mode": "sequential"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "workflow_id": "data-governance-assessment",
    "execution_id": "exec-001",
    "status": "completed",
    "results": {
      "data_quality": {...},
      "compliance": {...},
      "remediation": {...}
    },
    "summary": {
      "overall_score": 0.82,
      "recommendations_count": 12,
      "critical_issues": 2
    },
    "execution_time_ms": 4500
  }
}
```

## Pagination

For endpoints that return large datasets, use pagination:

**Query Parameters**:
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 20, max: 100)
- `sort_by`: Sort field
- `sort_order`: "asc" or "desc"

**Example**:
```
GET /api/v1/agents/analytics/reports?page=1&page_size=20&sort_by=created_at&sort_order=desc
```

**Paginated Response**:
```json
{
  "success": true,
  "data": {
    "items": [...],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total_items": 156,
      "total_pages": 8,
      "has_next": true,
      "has_previous": false
    }
  }
}
```

## Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour  
- **Enterprise**: Custom limits

Rate limit headers included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642248000
```

## WebSocket Endpoints

For real-time updates and streaming responses:

### Workflow Progress

**WebSocket** `/ws/workflows/{workflow_id}`

Subscribe to workflow execution updates.

**Message Format**:
```json
{
  "type": "workflow_update",
  "workflow_id": "data-governance-assessment",
  "execution_id": "exec-001",
  "status": "running",
  "current_agent": "data_quality",
  "progress": 0.33,
  "message": "Analyzing data quality metrics..."
}
```

## SDK Examples

### Python SDK

```python
from edgp_ai_client import EDGPClient
from edgp_ai_client.types import PolicySuggestionRequest

client = EDGPClient(api_key="your_api_key")

# Policy suggestion
request = PolicySuggestionRequest(
    business_context="E-commerce platform",
    compliance_requirements=["GDPR"],
    suggestion_type="validation_policies"
)

response = await client.policy_suggestion.create(request)
print(f"Generated {len(response.policy_recommendations)} policies")
```

### JavaScript SDK

```javascript
import { EDGPClient } from '@edgp/ai-client';

const client = new EDGPClient({ apiKey: 'your_api_key' });

// Data quality assessment
const qualityResponse = await client.dataQuality.assess({
  dataSource: {
    sourceId: 'customers',
    sourceName: 'Customer Database'
  },
  qualityDimensions: ['completeness', 'accuracy']
});

console.log(`Quality score: ${qualityResponse.overallQualityScore}`);
```

### cURL Examples

#### Policy Suggestion

```bash
curl -X POST "http://localhost:8000/api/v1/agents/policy-suggestion" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "request_id": "req-001",
    "agent_type": "policy_suggestion", 
    "business_context": "Healthcare provider",
    "compliance_requirements": ["HIPAA"],
    "suggestion_type": "access_control_policies"
  }'
```

#### Data Quality Assessment

```bash
curl -X POST "http://localhost:8000/api/v1/agents/data-quality" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "request_id": "req-002",
    "agent_type": "data_quality",
    "dataset_id": "patient_records",
    "quality_dimensions": ["completeness", "accuracy", "consistency"],
    "anomaly_detection": true
  }'
```

## Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Invalid request data | 400 |
| `AUTHENTICATION_ERROR` | Invalid or missing API key | 401 |
| `AUTHORIZATION_ERROR` | Insufficient permissions | 403 |
| `AGENT_NOT_FOUND` | Requested agent not available | 404 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `AGENT_ERROR` | Agent processing error | 500 |
| `LLM_ERROR` | LLM service error | 502 |
| `TIMEOUT_ERROR` | Request timeout | 504 |

## Testing Endpoints

### Test Data Endpoints

**POST** `/api/v1/test/generate-sample-data`

Generate sample data for testing agents.

**GET** `/api/v1/test/reset-demo-data`

Reset demo environment to initial state.

### Mock Responses

**GET** `/api/v1/test/mock/{agent_type}`

Get mock responses for testing agent integrations.

## OpenAPI Specification

The complete OpenAPI specification is available at:

**GET** `/docs` - Interactive Swagger UI
**GET** `/redoc` - Alternative API documentation
**GET** `/openapi.json` - Raw OpenAPI specification

## Monitoring Endpoints

### Agent Status

**GET** `/api/v1/agents/status`

Get detailed status of all agents.

### System Logs

**GET** `/api/v1/system/logs`

Access system logs (requires admin privileges).

**Query Parameters**:
- `level`: Log level filter (DEBUG, INFO, WARN, ERROR)
- `agent`: Filter by agent type
- `start_time`: Start timestamp
- `end_time`: End timestamp

## Best Practices

### 1. Request Optimization
- Include only necessary data in requests
- Use appropriate `page_size` for pagination
- Cache responses when possible
- Use WebSocket for real-time updates

### 2. Error Handling
- Always check the `success` field in responses
- Handle rate limiting with exponential backoff
- Implement retry logic for transient errors
- Log request IDs for debugging

### 3. Security
- Never expose API keys in client-side code
- Use HTTPS in production
- Implement proper authentication
- Validate all inputs on client side

This API provides a comprehensive interface for interacting with the EDGP AI Model's specialized agents and workflows.
