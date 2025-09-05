"""
Unit tests for core.types.agent_types module and agent type validation.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from core.types.agent_types import (
    AgentCapability, 
    AgentStatus,
    PolicySuggestionRequest, 
    DataQualityRequest,
    ComplianceRequest,
    RemediationRequest,
    AnalyticsRequest
)
from core.types.base import AgentType, Severity, ConfidenceLevel


class TestAgentCapability:
    """Test the AgentCapability enum."""
    
    def test_all_capabilities_exist(self):
        """Test that all expected capabilities are defined."""
        expected_capabilities = [
            "DATA_QUALITY_ASSESSMENT",
            "ANOMALY_DETECTION", 
            "DATA_PROFILING",
            "COMPLIANCE_CHECK",
            "REMEDIATION_SUGGESTION",
            "POLICY_GENERATION",
            "ANALYTICS_REPORTING"
        ]
        
        for capability_name in expected_capabilities:
            assert hasattr(AgentCapability, capability_name)
    
    def test_capability_values(self):
        """Test that capability enum values are correct."""
        assert AgentCapability.DATA_QUALITY_ASSESSMENT.value == "data_quality_assessment"
        assert AgentCapability.ANOMALY_DETECTION.value == "anomaly_detection"
        assert AgentCapability.DATA_PROFILING.value == "data_profiling"
        assert AgentCapability.COMPLIANCE_CHECK.value == "compliance_check"
        assert AgentCapability.REMEDIATION_SUGGESTION.value == "remediation_suggestion"
        assert AgentCapability.POLICY_GENERATION.value == "policy_generation"
        assert AgentCapability.ANALYTICS_REPORTING.value == "analytics_reporting"
    
    def test_capability_from_value(self):
        """Test creating capability from string value."""
        capability = AgentCapability("data_quality_assessment")
        assert capability == AgentCapability.DATA_QUALITY_ASSESSMENT
        
        capability = AgentCapability("anomaly_detection")
        assert capability == AgentCapability.ANOMALY_DETECTION
    
    def test_invalid_capability_value(self):
        """Test that invalid capability values raise ValueError."""
        with pytest.raises(ValueError):
            AgentCapability("invalid_capability")


class TestAgentStatus:
    """Test the AgentStatus enum."""
    
    def test_all_statuses_exist(self):
        """Test that all expected statuses are defined."""
        expected_statuses = [
            "IDLE",
            "PROCESSING", 
            "WAITING",
            "ERROR",
            "COMPLETED"
        ]
        
        for status_name in expected_statuses:
            assert hasattr(AgentStatus, status_name)
    
    def test_status_values(self):
        """Test that status enum values are correct."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.PROCESSING.value == "processing"
        assert AgentStatus.WAITING.value == "waiting"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.COMPLETED.value == "completed"
    
    def test_status_from_value(self):
        """Test creating status from string value."""
        status = AgentStatus("idle")
        assert status == AgentStatus.IDLE
        
        status = AgentStatus("processing")
        assert status == AgentStatus.PROCESSING


class TestAgentTypes:
    """Test agent type validation and serialization."""
    
    def test_policy_suggestion_request_validation(self):
        """Test PolicySuggestionRequest validation."""
        # Valid request
        request_data = {
            "request_id": "test-001",
            "agent_type": "policy_suggestion",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "business_context": "E-commerce data validation",
            "suggestion_type": "validation_policies"
        }
        
        request = PolicySuggestionRequest(**request_data)
        assert request.agent_type == AgentType.POLICY_SUGGESTION
        assert request.business_context == "E-commerce data validation"
        
    def test_policy_suggestion_request_invalid_agent_type(self):
        """Test PolicySuggestionRequest with invalid agent type."""
        request_data = {
            "request_id": "test-001",
            "agent_type": "invalid_agent",  # Invalid agent type
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "business_context": "Test context",
            "suggestion_type": "validation_policies"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PolicySuggestionRequest(**request_data)
        
        assert "agent_type" in str(exc_info.value)
    
    def test_data_quality_request_validation(self):
        """Test DataQualityRequest validation."""
        request_data = {
            "request_id": "test-dq-001",
            "agent_type": "data_quality",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_source": {
                "id": "src-001",
                "name": "Test Database",
                "type": "database"
            },
            "dataset_id": "test_dataset",
            "quality_dimensions": ["completeness", "accuracy"],
            "anomaly_detection": True
        }
        
        request = DataQualityRequest(**request_data)
        assert request.agent_type == AgentType.DATA_QUALITY
        assert request.dataset_id == "test_dataset"
        assert len(request.quality_dimensions) == 2
    
    def test_compliance_request_validation(self):
        """Test ComplianceRequest validation."""
        request_data = {
            "request_id": "test-comp-001",
            "agent_type": "data_privacy_compliance",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_sources": [{
                "id": "src-001",
                "name": "Test Database",
                "type": "database"
            }],
            "processing_context": "Test processing context",
            "applicable_regulations": ["gdpr"],
            "scan_type": "privacy_risks"
        }
        
        request = ComplianceRequest(**request_data)
        assert request.agent_type == AgentType.DATA_PRIVACY_COMPLIANCE
        assert request.processing_context == "Test processing context"
        assert len(request.data_sources) == 1
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        # Missing request_id
        request_data = {
            "agent_type": "policy_suggestion",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "business_context": "Test context",
            "suggestion_type": "validation_policies"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PolicySuggestionRequest(**request_data)
        
        assert "request_id" in str(exc_info.value)
    
    def test_severity_enum_validation(self):
        """Test Severity enum validation."""
        # Valid severity values
        valid_severities = ["critical", "high", "medium", "low", "info"]
        
        for severity in valid_severities:
            severity_obj = Severity(severity)
            assert severity_obj.value == severity
        
        # Invalid severity
        with pytest.raises(ValueError):
            Severity("invalid_severity")
    
    def test_confidence_level_validation(self):
        """Test ConfidenceLevel enum validation."""
        # Valid confidence levels
        valid_levels = ["very_high", "high", "medium", "low", "very_low"]
        
        for level in valid_levels:
            confidence_obj = ConfidenceLevel(level)
            assert confidence_obj.value == level
        
        # Invalid confidence level
        with pytest.raises(ValueError):
            ConfidenceLevel("invalid_confidence")
