import pytest
from datetime import datetime
from typing import Dict, Any, List
from core.types.responses import (
    StandardResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ValidationError
)


class TestStandardResponse:
    """Test the StandardResponse class."""
    
    def test_standard_response_creation(self):
        """Test creation of StandardResponse with all fields."""
        response = StandardResponse(
            status="success",
            message="Operation completed",
            data={"key": "value"},
            metadata={"source": "test"}
        )
        
        assert response.status == "success"
        assert response.message == "Operation completed"
        assert response.data == {"key": "value"}
        assert response.metadata == {"source": "test"}
        assert response.timestamp is not None
    
    def test_standard_response_minimal(self):
        """Test creation of StandardResponse with minimal fields."""
        response = StandardResponse(
            status="success",
            message="OK"
        )
        
        assert response.status == "success"
        assert response.message == "OK"
        assert response.data is None
        assert response.metadata is None
        assert response.timestamp is not None
    
    def test_standard_response_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated."""
        response = StandardResponse(
            status="success",
            message="Test"
        )
        
        assert isinstance(response.timestamp, datetime)
        
    def test_standard_response_serialization(self):
        """Test that StandardResponse can be serialized to dict."""
        response = StandardResponse(
            status="success",
            message="Test",
            data={"test": True}
        )
        
        response_dict = response.model_dump()
        assert "status" in response_dict
        assert "message" in response_dict
        assert "data" in response_dict
        assert "timestamp" in response_dict


class TestErrorResponse:
    """Test the ErrorResponse class."""
    
    def test_error_response_creation(self):
        """Test creation of ErrorResponse."""
        response = ErrorResponse(
            status="error",
            message="Something went wrong",
            error_code="ERR_001",
            details={"field": "value"}
        )
        
        assert response.status == "error"
        assert response.message == "Something went wrong"
        assert response.error_code == "ERR_001"
        assert response.details == {"field": "value"}
        assert response.timestamp is not None
    
    def test_error_response_minimal(self):
        """Test creation of ErrorResponse with minimal fields."""
        response = ErrorResponse(
            status="error",
            message="Error occurred"
        )
        
        assert response.status == "error"
        assert response.message == "Error occurred"
        assert response.error_code is None
        assert response.details is None
    
    def test_error_response_serialization(self):
        """Test ErrorResponse serialization."""
        response = ErrorResponse(
            status="error",
            message="Test error",
            error_code="TEST_001"
        )
        
        response_dict = response.model_dump()
        assert response_dict["status"] == "error"
        assert response_dict["message"] == "Test error"
        assert response_dict["error_code"] == "TEST_001"


class TestHealthResponse:
    """Test the HealthResponse class."""
    
    def test_health_response_creation(self):
        """Test creation of HealthResponse."""
        response = HealthResponse(
            status="healthy",
            message="Service is running",
            checks={"database": "ok", "cache": "ok"},
            uptime=3600
        )
        
        assert response.status == "healthy"
        assert response.message == "Service is running"
        assert response.checks == {"database": "ok", "cache": "ok"}
        assert response.uptime == 3600
    
    def test_health_response_unhealthy(self):
        """Test creation of unhealthy HealthResponse."""
        response = HealthResponse(
            status="unhealthy",
            message="Database connection failed",
            checks={"database": "failed", "cache": "ok"}
        )
        
        assert response.status == "unhealthy"
        assert response.message == "Database connection failed"
        assert "database" in response.checks
        assert response.checks["database"] == "failed"
    
    def test_health_response_serialization(self):
        """Test HealthResponse serialization."""
        response = HealthResponse(
            status="healthy",
            message="OK",
            checks={"service": "running"}
        )
        
        response_dict = response.model_dump()
        assert "status" in response_dict
        assert "checks" in response_dict
        assert "uptime" in response_dict


class TestMetricsResponse:
    """Test the MetricsResponse class."""
    
    def test_metrics_response_creation(self):
        """Test creation of MetricsResponse."""
        response = MetricsResponse(
            status="success",
            message="Metrics collected",
            metrics={"cpu_usage": 45.2, "memory_usage": 67.8},
            period="1h"
        )
        
        assert response.status == "success"
        assert response.message == "Metrics collected"
        assert response.metrics["cpu_usage"] == 45.2
        assert response.metrics["memory_usage"] == 67.8
        assert response.period == "1h"
    
    def test_metrics_response_empty_metrics(self):
        """Test MetricsResponse with empty metrics."""
        response = MetricsResponse(
            status="success",
            message="No metrics available",
            metrics={}
        )
        
        assert response.metrics == {}
        assert response.period is None
    
    def test_metrics_response_serialization(self):
        """Test MetricsResponse serialization."""
        response = MetricsResponse(
            status="success",
            message="Test metrics",
            metrics={"test": 100}
        )
        
        response_dict = response.model_dump()
        assert "metrics" in response_dict
        assert "period" in response_dict


class TestValidationError:
    """Test the ValidationError class."""
    
    def test_validation_error_creation(self):
        """Test creation of ValidationError."""
        error = ValidationError(
            field="email",
            message="Invalid email format",
            code="INVALID_EMAIL"
        )
        
        assert error.field == "email"
        assert error.message == "Invalid email format"
        assert error.code == "INVALID_EMAIL"
    
    def test_validation_error_minimal(self):
        """Test ValidationError with minimal fields."""
        error = ValidationError(
            field="username",
            message="Required field"
        )
        
        assert error.field == "username"
        assert error.message == "Required field"
        assert error.code is None
    
    def test_validation_error_serialization(self):
        """Test ValidationError serialization."""
        error = ValidationError(
            field="password",
            message="Too short",
            code="MIN_LENGTH"
        )
        
        error_dict = error.model_dump()
        assert "field" in error_dict
        assert "message" in error_dict
        assert "code" in error_dict


class TestResponseIntegration:
    """Test response classes integration and common patterns."""
    
    def test_response_timestamps_consistent(self):
        """Test that all response types include timestamps."""
        standard = StandardResponse(status="success", message="OK")
        error = ErrorResponse(status="error", message="Failed")
        health = HealthResponse(status="healthy", message="Running")
        metrics = MetricsResponse(status="success", message="Collected", metrics={})
        
        # All should have timestamps
        assert hasattr(standard, 'timestamp')
        assert hasattr(error, 'timestamp')
        assert hasattr(health, 'timestamp')
        assert hasattr(metrics, 'timestamp')
    
    def test_error_response_with_validation_errors(self):
        """Test ErrorResponse can include validation errors."""
        validation_errors = [
            ValidationError(field="email", message="Invalid format"),
            ValidationError(field="age", message="Must be positive")
        ]
        
        response = ErrorResponse(
            status="validation_error",
            message="Validation failed",
            details={"validation_errors": [error.model_dump() for error in validation_errors]}
        )
        
        assert response.status == "validation_error"
        assert "validation_errors" in response.details
        assert len(response.details["validation_errors"]) == 2
    
    def test_all_response_types_serializable(self):
        """Test that all response types can be serialized."""
        responses = [
            StandardResponse(status="success", message="OK"),
            ErrorResponse(status="error", message="Failed"),
            HealthResponse(status="healthy", message="Running"),
            MetricsResponse(status="success", message="Metrics", metrics={"test": 1})
        ]
        
        for response in responses:
            response_dict = response.model_dump()
            assert isinstance(response_dict, dict)
            assert "status" in response_dict
            assert "message" in response_dict
            assert "timestamp" in response_dict
    
    def test_response_status_patterns(self):
        """Test common status patterns."""
        # Success patterns
        success_response = StandardResponse(status="success", message="OK")
        assert success_response.status == "success"
        
        # Error patterns
        error_response = ErrorResponse(status="error", message="Failed")
        assert error_response.status == "error"
        
        # Health patterns
        healthy_response = HealthResponse(status="healthy", message="OK")
        unhealthy_response = HealthResponse(status="unhealthy", message="Failed")
        assert healthy_response.status == "healthy"
        assert unhealthy_response.status == "unhealthy"
    
    def test_response_with_complex_data(self):
        """Test responses with complex nested data structures."""
        complex_data = {
            "users": [
                {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                {"id": 2, "name": "Bob", "roles": ["user"]}
            ],
            "metadata": {
                "total_count": 2,
                "filters_applied": ["active", "verified"],
                "query_time": 0.125
            }
        }
        
        response = StandardResponse(
            status="success",
            message="Complex data retrieved",
            data=complex_data
        )
        
        assert response.data["users"][0]["name"] == "Alice"
        assert len(response.data["users"][0]["roles"]) == 2
        assert response.data["metadata"]["total_count"] == 2
    
    def test_error_response_with_stack_trace(self):
        """Test ErrorResponse with detailed error information."""
        response = ErrorResponse(
            status="internal_error",
            message="Database connection failed",
            error_code="DB_CONNECTION_ERROR",
            details={
                "connection_string": "postgresql://***:***@localhost:5432/db",
                "error_type": "ConnectionTimeout",
                "retry_count": 3,
                "stack_trace": ["line1", "line2", "line3"]
            }
        )
        
        assert response.error_code == "DB_CONNECTION_ERROR"
        assert response.details["retry_count"] == 3
        assert len(response.details["stack_trace"]) == 3
    
    def test_health_response_comprehensive_checks(self):
        """Test HealthResponse with comprehensive health checks."""
        response = HealthResponse(
            status="degraded",
            message="Some services experiencing issues",
            checks={
                "database": {"status": "healthy", "response_time": 0.05},
                "cache": {"status": "degraded", "response_time": 1.2, "warning": "High latency"},
                "message_queue": {"status": "unhealthy", "error": "Connection refused"},
                "external_api": {"status": "healthy", "response_time": 0.3}
            },
            uptime=86400
        )
        
        assert response.status == "degraded"
        assert response.checks["database"]["status"] == "healthy"
        assert response.checks["cache"]["status"] == "degraded"
        assert response.checks["message_queue"]["status"] == "unhealthy"
        assert response.uptime == 86400
    
    def test_metrics_response_time_series_data(self):
        """Test MetricsResponse with time series metrics."""
        time_series_metrics = {
            "cpu_usage": [
                {"timestamp": "2024-01-01T00:00:00Z", "value": 45.2},
                {"timestamp": "2024-01-01T00:01:00Z", "value": 47.8},
                {"timestamp": "2024-01-01T00:02:00Z", "value": 43.1}
            ],
            "memory_usage": [
                {"timestamp": "2024-01-01T00:00:00Z", "value": 67.8},
                {"timestamp": "2024-01-01T00:01:00Z", "value": 69.2},
                {"timestamp": "2024-01-01T00:02:00Z", "value": 65.4}
            ],
            "summary": {
                "avg_cpu": 45.37,
                "avg_memory": 67.47,
                "peak_cpu": 47.8,
                "peak_memory": 69.2
            }
        }
        
        response = MetricsResponse(
            status="success",
            message="Time series metrics collected",
            metrics=time_series_metrics,
            period="3m"
        )
        
        assert response.period == "3m"
        assert len(response.metrics["cpu_usage"]) == 3
        assert response.metrics["summary"]["avg_cpu"] == 45.37
        assert response.metrics["summary"]["peak_memory"] == 69.2
