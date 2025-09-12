"""
Unit tests for core.types.responses module.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime
from typing import Dict, Any, List
from pydantic import ValidationError

from core.types.responses import (
    StandardResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ValidationError as ResponseValidationError,
    MessageResponse,
    AgentStatusResponse,
    WorkflowResponse,
    BatchResponse,
    PaginatedResponse,
    WorkflowStatusResponse
)
from core.types.base import AgentType, TaskStatus


class TestStandardResponse:
    """Test StandardResponse model."""
    
    def test_standard_response_creation(self):
        """Test creating StandardResponse."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = StandardResponse(
            success=True,
            message="Operation completed successfully",
            timestamp=timestamp,
            request_id="test_request_123"
        )
        
        assert response.success is True
        assert response.message == "Operation completed successfully"
        assert response.request_id == "test_request_123"
        assert response.timestamp == timestamp
        assert response.data is None
        assert response.errors == []
        assert response.warnings == []
    
    def test_standard_response_with_data(self):
        """Test StandardResponse with data."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        data = {"result": "test_result", "count": 42}
        response = StandardResponse(
            success=True,
            message="Data retrieved",
            timestamp=timestamp,
            request_id="test_request_456",
            data=data
        )
        
        assert response.data == data
        assert response.success is True
        assert response.message == "Data retrieved"
    
    def test_standard_response_with_errors(self):
        """Test StandardResponse with errors."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]
        response = StandardResponse(
            success=False,
            message="Operation failed",
            timestamp=timestamp,
            request_id="test_request_789",
            errors=errors,
            warnings=warnings
        )
        
        assert response.success is False
        assert response.errors == errors
        assert response.warnings == warnings
        assert response.message == "Operation failed"

    def test_standard_response_minimal(self):
        """Test StandardResponse with minimal required fields."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = StandardResponse(
            success=True,
            message="Minimal response",
            timestamp=timestamp
        )
        
        assert response.success is True
        assert response.message == "Minimal response"
        assert response.timestamp == timestamp
        assert response.request_id is None
        assert response.data is None
        assert response.errors == []
        assert response.warnings == []


class TestErrorResponse:
    """Test ErrorResponse model."""
    
    def test_error_response_creation(self):
        """Test creating ErrorResponse."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message="Invalid input data",
            timestamp=timestamp,
            request_id="test_request_error_123"
        )
        
        assert response.success is False  # default value
        assert response.error_code == "VALIDATION_ERROR"
        assert response.error_message == "Invalid input data"
        assert response.timestamp == timestamp
        assert response.request_id == "test_request_error_123"
        assert response.error_details is None
        assert response.trace_id is None
    
    def test_error_response_with_details(self):
        """Test ErrorResponse with error details."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        error_details = {
            "field": "username",
            "issue": "too_short",
            "min_length": 3
        }
        response = ErrorResponse(
            error_code="FIELD_VALIDATION",
            error_message="Field validation failed",
            timestamp=timestamp,
            error_details=error_details,
            trace_id="trace_12345"
        )
        
        assert response.error_details == error_details
        assert response.trace_id == "trace_12345"
        assert response.error_code == "FIELD_VALIDATION"
    
    def test_error_response_with_stack_trace(self):
        """Test ErrorResponse with trace ID for debugging."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="An unexpected error occurred",
            timestamp=timestamp,
            trace_id="stack_trace_abc123",
            request_id="req_internal_error"
        )
        
        assert response.trace_id == "stack_trace_abc123"
        assert response.error_code == "INTERNAL_ERROR"
        assert response.request_id == "req_internal_error"

    def test_error_response_minimal(self):
        """Test ErrorResponse with minimal required fields."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = ErrorResponse(
            error_code="GENERIC_ERROR",
            error_message="An error occurred",
            timestamp=timestamp
        )
        
        assert response.success is False
        assert response.error_code == "GENERIC_ERROR"
        assert response.error_message == "An error occurred"
        assert response.timestamp == timestamp
        assert response.request_id is None
        assert response.trace_id is None
        assert response.error_details is None


class TestHealthResponse:
    """Test HealthResponse model."""
    
    def test_health_response_creation(self):
        """Test creating HealthResponse."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=timestamp,
            uptime_seconds=3600.5
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.timestamp == timestamp
        assert response.uptime_seconds == 3600.5
        assert response.services == {}
        assert response.agents == []
        assert response.message_driven_architecture is False
        assert response.error is None

    def test_health_response_with_services(self):
        """Test HealthResponse with services information."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        services = {
            "database": "healthy",
            "redis": "degraded",
            "message_queue": "healthy"
        }
        agents = ["policy_suggestion", "data_quality", "analytics"]
        
        response = HealthResponse(
            status="degraded",
            version="1.2.0",
            timestamp=timestamp,
            services=services,
            agents=agents,
            message_driven_architecture=True
        )
        
        assert response.status == "degraded"
        assert response.services == services
        assert response.agents == agents
        assert response.message_driven_architecture is True

    def test_health_response_with_error(self):
        """Test HealthResponse with error information."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = HealthResponse(
            status="unhealthy",
            version="1.0.0",
            timestamp=timestamp,
            error="Database connection failed"
        )
        
        assert response.status == "unhealthy"
        assert response.error == "Database connection failed"


class TestMetricsResponse:
    """Test MetricsResponse model."""
    
    def test_metrics_response_creation(self):
        """Test creating MetricsResponse."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        metrics = {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "active_connections": 150,
            "response_time_avg": 250.5
        }
        
        response = MetricsResponse(
            metrics=metrics,
            timestamp=timestamp,
            collection_period="5m"
        )
        
        assert response.metrics == metrics
        assert response.timestamp == timestamp
        assert response.collection_period == "5m"
        assert response.metadata == {}

    def test_metrics_response_with_metadata(self):
        """Test MetricsResponse with metadata."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        metrics = {"requests_per_second": 125.6}
        metadata = {
            "collector": "prometheus",
            "region": "us-east-1",
            "environment": "production"
        }
        
        response = MetricsResponse(
            metrics=metrics,
            timestamp=timestamp,
            collection_period="1h",
            metadata=metadata
        )
        
        assert response.metadata == metadata
        assert response.collection_period == "1h"


class TestValidationError:
    """Test ValidationError model."""
    
    def test_validation_error_creation(self):
        """Test creating ValidationError."""
        error = ResponseValidationError(
            field="email",
            message="Invalid email format",
            value="invalid-email",
            code="EMAIL_FORMAT"
        )
        
        assert error.field == "email"
        assert error.message == "Invalid email format"
        assert error.value == "invalid-email"
        assert error.code == "EMAIL_FORMAT"

    def test_validation_error_without_code(self):
        """Test ValidationError without error code."""
        error = ResponseValidationError(
            field="age",
            message="Value must be positive",
            value=-5
        )
        
        assert error.field == "age"
        assert error.message == "Value must be positive"
        assert error.value == -5
        assert error.code is None


class TestMessageResponse:
    """Test MessageResponse model."""
    
    def test_message_response_creation(self):
        """Test creating MessageResponse."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = MessageResponse(
            success=True,
            message_id="msg_12345",
            recipient="agent_policy_suggestion",
            timestamp=timestamp
        )
        
        assert response.success is True
        assert response.message_id == "msg_12345"
        assert response.recipient == "agent_policy_suggestion"
        assert response.timestamp == timestamp
        assert response.error is None

    def test_message_response_with_error(self):
        """Test MessageResponse with error."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = MessageResponse(
            success=False,
            message_id="msg_failed_67890",
            recipient="agent_data_quality",
            timestamp=timestamp,
            error="Recipient not available"
        )
        
        assert response.success is False
        assert response.error == "Recipient not available"


class TestAgentStatusResponse:
    """Test AgentStatusResponse model."""
    
    def test_agent_status_response_creation(self):
        """Test creating AgentStatusResponse."""
        capabilities = ["data_analysis", "quality_assessment", "reporting"]
        response = AgentStatusResponse(
            agent_name="Data Quality Agent",
            agent_type=AgentType.DATA_QUALITY,
            status=TaskStatus.COMPLETED,
            last_activity="2024-12-15T10:30:00Z",
            processed_requests=150,
            success_rate=0.95,
            average_response_time_ms=250.5,
            current_load=0.65,
            capabilities=capabilities
        )
        
        assert response.agent_name == "Data Quality Agent"
        assert response.agent_type == AgentType.DATA_QUALITY
        assert response.status == TaskStatus.COMPLETED
        assert response.capabilities == capabilities
        assert response.last_activity == "2024-12-15T10:30:00Z"
        assert response.processed_requests == 150
        assert response.success_rate == 0.95
        assert response.average_response_time_ms == 250.5
        assert response.current_load == 0.65

    def test_agent_status_response_with_minimal_data(self):
        """Test AgentStatusResponse with minimal required fields."""
        response = AgentStatusResponse(
            agent_name="Analytics Agent",
            agent_type=AgentType.ANALYTICS,
            status=TaskStatus.FAILED,
            last_activity="2024-12-15T08:00:00Z",
            processed_requests=0,
            success_rate=0.0,
            average_response_time_ms=0.0,
            current_load=0.0
        )
        
        assert response.agent_name == "Analytics Agent"
        assert response.agent_type == AgentType.ANALYTICS
        assert response.status == TaskStatus.FAILED
        assert response.capabilities == []
        assert response.processed_requests == 0
        assert response.success_rate == 0.0


class TestWorkflowResponse:
    """Test WorkflowResponse model."""
    
    def test_workflow_response_creation(self):
        """Test creating WorkflowResponse."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        result = {
            "processed_records": 1000,
            "quality_score": 0.85,
            "issues_found": 23
        }
        
        response = WorkflowResponse(
            execution_id="exec_12345",
            workflow_name="data_quality_assessment",
            status="completed",
            timestamp=timestamp,
            result=result
        )
        
        assert response.execution_id == "exec_12345"
        assert response.workflow_name == "data_quality_assessment"
        assert response.status == "completed"
        assert response.timestamp == timestamp
        assert response.result == result
        assert response.error is None

    def test_workflow_response_with_timing(self):
        """Test WorkflowResponse for timing tracking."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = WorkflowResponse(
            execution_id="exec_timing_001",
            workflow_name="compliance_check",
            status="running",
            timestamp=timestamp
        )
        
        assert response.status == "running"
        assert response.result is None
        assert response.error is None

    def test_workflow_response_with_result(self):
        """Test WorkflowResponse with detailed results."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        detailed_result = {
            "analysis_complete": True,
            "recommendations": ["Improve data validation", "Add monitoring"],
            "metrics": {"accuracy": 0.92, "completeness": 0.88}
        }
        
        response = WorkflowResponse(
            execution_id="exec_detailed_002",
            workflow_name="policy_recommendation",
            status="completed",
            timestamp=timestamp,
            result=detailed_result
        )
        
        assert response.result == detailed_result
        assert response.status == "completed"

    def test_workflow_response_failed(self):
        """Test WorkflowResponse for failed execution."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        response = WorkflowResponse(
            execution_id="exec_failed_003",
            workflow_name="data_remediation",
            status="failed",
            timestamp=timestamp,
            error="Insufficient permissions for data modification"
        )
        
        assert response.status == "failed"
        assert response.error == "Insufficient permissions for data modification"
        assert response.result is None


class TestBatchResponse:
    """Test BatchResponse model."""
    
    def test_batch_response_creation(self):
        """Test creating BatchResponse."""
        errors = [
            ResponseValidationError(field="email", message="Invalid format", value="bad-email"),
            ResponseValidationError(field="age", message="Too young", value=10)
        ]
        results = [{"id": 1, "status": "success"}, {"id": 2, "status": "success"}]
        
        response = BatchResponse(
            total_items=4,
            successful_items=2,
            failed_items=2,
            errors=errors,
            results=results,
            processing_time_ms=1250.5
        )
        
        assert response.total_items == 4
        assert response.successful_items == 2
        assert response.failed_items == 2
        assert len(response.errors) == 2
        assert len(response.results) == 2
        assert response.processing_time_ms == 1250.5

    def test_batch_response_all_successful(self):
        """Test BatchResponse with all successful items."""
        results = [{"id": i, "status": "success"} for i in range(1, 6)]
        
        response = BatchResponse(
            total_items=5,
            successful_items=5,
            failed_items=0,
            results=results,
            processing_time_ms=800.0
        )
        
        assert response.total_items == 5
        assert response.successful_items == 5
        assert response.failed_items == 0
        assert response.errors == []
        assert len(response.results) == 5


class TestPaginatedResponse:
    """Test PaginatedResponse model."""
    
    def test_paginated_response_creation(self):
        """Test creating PaginatedResponse."""
        items = [{"id": i, "name": f"item_{i}"} for i in range(1, 11)]
        
        response = PaginatedResponse(
            items=items,
            total_count=100,
            page=1,
            page_size=10,
            total_pages=10,
            has_next=True,
            has_previous=False
        )
        
        assert len(response.items) == 10
        assert response.total_count == 100
        assert response.page == 1
        assert response.page_size == 10
        assert response.total_pages == 10
        assert response.has_next is True
        assert response.has_previous is False

    def test_paginated_response_middle_page(self):
        """Test PaginatedResponse for middle page."""
        items = [{"id": i} for i in range(21, 31)]
        
        response = PaginatedResponse(
            items=items,
            total_count=50,
            page=3,
            page_size=10,
            total_pages=5,
            has_next=True,
            has_previous=True
        )
        
        assert response.page == 3
        assert response.has_next is True
        assert response.has_previous is True

    def test_paginated_response_last_page(self):
        """Test PaginatedResponse for last page."""
        items = [{"id": i} for i in range(91, 101)]
        
        response = PaginatedResponse(
            items=items,
            total_count=100,
            page=10,
            page_size=10,
            total_pages=10,
            has_next=False,
            has_previous=True
        )
        
        assert response.page == 10
        assert response.has_next is False
        assert response.has_previous is True


class TestWorkflowStatusResponse:
    """Test WorkflowStatusResponse model."""
    
    def test_workflow_status_response_creation(self):
        """Test creating WorkflowStatusResponse."""
        involved_agents = [AgentType.DATA_QUALITY, AgentType.POLICY_SUGGESTION]
        
        response = WorkflowStatusResponse(
            workflow_id="workflow_12345",
            workflow_type="data_assessment",
            status=TaskStatus.IN_PROGRESS,
            progress_percentage=65.5,
            current_step="quality_analysis",
            involved_agents=involved_agents,
            start_time="2024-12-15T09:00:00Z",
            estimated_completion="2024-12-15T12:00:00Z"
        )
        
        assert response.workflow_id == "workflow_12345"
        assert response.workflow_type == "data_assessment"
        assert response.status == TaskStatus.IN_PROGRESS
        assert response.progress_percentage == 65.5
        assert response.current_step == "quality_analysis"
        assert response.involved_agents == involved_agents
        assert response.start_time == "2024-12-15T09:00:00Z"
        assert response.estimated_completion == "2024-12-15T12:00:00Z"
        assert response.errors == []

    def test_workflow_status_response_with_errors(self):
        """Test WorkflowStatusResponse with errors."""
        errors = ["Data source unavailable", "Permission denied for table X"]
        
        response = WorkflowStatusResponse(
            workflow_id="workflow_error_001",
            workflow_type="compliance_check",
            status=TaskStatus.FAILED,
            progress_percentage=30.0,
            current_step="data_validation",
            involved_agents=[AgentType.DATA_PRIVACY_COMPLIANCE],
            start_time="2024-12-15T08:00:00Z",
            errors=errors
        )
        
        assert response.status == TaskStatus.FAILED
        assert response.errors == errors
        assert response.estimated_completion is None

    def test_workflow_status_response_completed(self):
        """Test WorkflowStatusResponse for completed workflow."""
        response = WorkflowStatusResponse(
            workflow_id="workflow_complete_001",
            workflow_type="remediation",
            status=TaskStatus.COMPLETED,
            progress_percentage=100.0,
            current_step="finalization",
            involved_agents=[AgentType.DATA_REMEDIATION, AgentType.DATA_QUALITY],
            start_time="2024-12-15T07:00:00Z"
        )
        
        assert response.status == TaskStatus.COMPLETED
        assert response.progress_percentage == 100.0
        assert response.current_step == "finalization"


class TestResponseIntegration:
    """Test integration scenarios between different response types."""
    
    def test_error_to_standard_response_conversion(self):
        """Test converting error scenarios to standard response format."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Create an error response first
        error_response = ErrorResponse(
            error_code="VALIDATION_FAILED",
            error_message="Input validation failed",
            timestamp=timestamp,
            request_id="req_integration_001"
        )
        
        # Create corresponding standard response
        standard_response = StandardResponse(
            success=False,
            message="Request failed due to validation errors",
            timestamp=timestamp,
            request_id="req_integration_001",
            errors=["Input validation failed"]
        )
        
        assert error_response.error_message in standard_response.errors
        assert error_response.request_id == standard_response.request_id
        assert standard_response.success is False

    def test_workflow_status_aggregation(self):
        """Test aggregating multiple workflow statuses."""
        # Multiple workflow status responses
        workflow_1 = WorkflowStatusResponse(
            workflow_id="wf_001",
            workflow_type="data_quality",
            status=TaskStatus.COMPLETED,
            progress_percentage=100.0,
            current_step="finished",
            involved_agents=[AgentType.DATA_QUALITY],
            start_time="2024-12-15T08:00:00Z"
        )
        
        workflow_2 = WorkflowStatusResponse(
            workflow_id="wf_002",
            workflow_type="compliance",
            status=TaskStatus.IN_PROGRESS,
            progress_percentage=45.0,
            current_step="regulation_check",
            involved_agents=[AgentType.DATA_PRIVACY_COMPLIANCE],
            start_time="2024-12-15T09:00:00Z"
        )
        
        # Verify individual workflow statuses
        assert workflow_1.status == TaskStatus.COMPLETED
        assert workflow_2.status == TaskStatus.IN_PROGRESS
        
        # Could aggregate into summary
        all_workflows = [workflow_1, workflow_2]
        completed_count = sum(1 for wf in all_workflows if wf.status == TaskStatus.COMPLETED)
        in_progress_count = sum(1 for wf in all_workflows if wf.status == TaskStatus.IN_PROGRESS)
        
        assert completed_count == 1
        assert in_progress_count == 1

    def test_batch_with_validation_errors(self):
        """Test BatchResponse with validation errors."""
        validation_errors = [
            ResponseValidationError(field="email", message="Invalid email", value="bad@", code="EMAIL_INVALID"),
            ResponseValidationError(field="phone", message="Invalid format", value="123", code="PHONE_INVALID")
        ]
        
        batch_response = BatchResponse(
            total_items=10,
            successful_items=8,
            failed_items=2,
            errors=validation_errors,
            results=[{"id": i, "status": "success"} for i in range(8)],
            processing_time_ms=2500.0
        )
        
        assert len(batch_response.errors) == 2
        assert batch_response.failed_items == len(batch_response.errors)
        assert batch_response.successful_items + batch_response.failed_items == batch_response.total_items

    def test_health_with_agent_status_integration(self):
        """Test HealthResponse integration with agent status."""
        # Health response showing agents
        health_response = HealthResponse(
            status="healthy",
            version="2.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            agents=["policy_suggestion", "data_quality", "analytics"],
            message_driven_architecture=True
        )
        
        # Individual agent status
        agent_status = AgentStatusResponse(
            agent_name="Data Quality Agent",
            agent_type=AgentType.DATA_QUALITY,
            status=TaskStatus.COMPLETED,
            last_activity="2024-12-15T10:30:00Z",
            processed_requests=100,
            success_rate=0.98,
            average_response_time_ms=180.5,
            current_load=0.45,
            capabilities=["quality_assessment", "data_profiling"]
        )
        
        # Verify integration
        assert "data_quality" in health_response.agents
        assert agent_status.status == TaskStatus.COMPLETED
        assert health_response.message_driven_architecture is True
