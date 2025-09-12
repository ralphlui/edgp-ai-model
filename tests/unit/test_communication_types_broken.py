"""
Unit tests for core.types.communication module.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any
from core.types.communication import (
    MessageType, OperationType, ProcessingStage, ExecutionMode, ErrorType,
    StandardAgentInput, StandardAgentOutput, AgentContext, AgentMetadata,
    ProcessingResult, AgentError, DataPayload, OperationParameters,
    CommunicationProtocol, create_standard_input, create_standard_output, 
    create_agent_error, create_agent_context
)
from core.types.base import (
    TaskStatus, Priority, ConfidenceLevel, Severity
)


class TestEnums:
    """Test enum classes."""
    
    def test_message_type_enum(self):
        """Test MessageType enum values."""
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.NOTIFICATION.value == "notification"
        assert MessageType.ERROR.value == "error"
    
    def test_operation_type_enum(self):
        """Test OperationType enum values."""
        assert OperationType.ANALYSIS.value == "analysis"
        assert OperationType.ASSESSMENT.value == "assessment"
        assert OperationType.VALIDATION.value == "validation"
        assert OperationType.GENERATION.value == "generation"
    
    def test_processing_stage_enum(self):
        """Test ProcessingStage enum values."""
        assert ProcessingStage.PREPROCESSING.value == "preprocessing"
        assert ProcessingStage.PROCESSING.value == "processing"
        assert ProcessingStage.POSTPROCESSING.value == "postprocessing"
        assert ProcessingStage.VALIDATION.value == "validation"
    
    def test_task_status_enum(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
    
    def test_priority_enum(self):
        """Test Priority enum values."""
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"
    
    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"


class TestAgentContext:
    """Test AgentContext model."""
    
    def test_agent_context_creation(self):
        """Test creating AgentContext with basic data."""
        context = AgentContext(
            session_id="test_session",
            user_id="test_user",
            tenant_id="test_tenant"
        )
        
        assert context.session_id == "test_session"
        assert context.user_id == "test_user"
        assert context.tenant_id == "test_tenant"
        assert context.correlation_id is not None
        assert context.execution_mode == ExecutionMode.SYNCHRONOUS
        assert context.metadata == {}
    
    def test_agent_context_defaults(self):
        """Test AgentContext with minimal required fields."""
        context = AgentContext()
        
        assert context.session_id is not None
        assert context.correlation_id is not None
        assert context.user_id is None
        assert context.tenant_id is None


class TestAgentMetadata:
    """Test AgentMetadata model."""
    
    def test_agent_metadata_creation(self):
        """Test creating AgentMetadata."""
        metadata = AgentMetadata(
            agent_id="test_agent",
            agent_version="1.0.0",
            capability_name="data_quality"
        )
        
        assert metadata.agent_id == "test_agent"
        assert metadata.agent_version == "1.0.0"
        assert metadata.capability_name == "data_quality"
        assert metadata.execution_time is None
        assert metadata.memory_usage is None
    
    def test_agent_metadata_with_performance(self):
        """Test AgentMetadata with performance metrics."""
        metadata = AgentMetadata(
            agent_id="test_agent",
            agent_version="1.0.0",
            capability_name="data_quality",
            execution_time=1.5,
            memory_usage=1024
        )
        
        assert metadata.execution_time == 1.5
        assert metadata.memory_usage == 1024


class TestProcessingResult:
    """Test ProcessingResult model."""
    
    def test_processing_result_creation(self):
        """Test creating ProcessingResult."""
        result = ProcessingResult(
            status=TaskStatus.COMPLETED,
            result_data={"key": "value"},
            confidence=ConfidenceLevel.HIGH
        )
        
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data == {"key": "value"}
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.processing_time is None
        assert result.errors == []
    
    def test_processing_result_with_errors(self):
        """Test ProcessingResult with errors."""
        errors = ["Error 1", "Error 2"]
        result = ProcessingResult(
            status=TaskStatus.FAILED,
            result_data={},
            confidence=ConfidenceLevel.LOW,
            errors=errors
        )
        
        assert result.status == TaskStatus.FAILED
        assert result.errors == errors
    
    def test_processing_result_with_timing(self):
        """Test ProcessingResult with processing time."""
        result = ProcessingResult(
            status=TaskStatus.COMPLETED,
            result_data={"key": "value"},
            confidence=ConfidenceLevel.HIGH,
            processing_time=2.5
        )
        
        assert result.processing_time == 2.5


class TestAgentError:
    """Test AgentError model."""
    
    def test_agent_error_creation(self):
        """Test creating AgentError."""
        error = AgentError(
            error_type=ErrorType.VALIDATION_ERROR,
            message="Test error message",
            agent_id="test_agent"
        )
        
        assert error.error_type == ErrorType.VALIDATION_ERROR
        assert error.message == "Test error message"
        assert error.agent_id == "test_agent"
        assert error.error_code is None
        assert error.details == {}
        assert error.timestamp is not None
    
    def test_agent_error_with_details(self):
        """Test AgentError with details."""
        details = {"field": "value", "line": 42}
        error = AgentError(
            error_type=ErrorType.PROCESSING_ERROR,
            message="Processing failed",
            agent_id="test_agent",
            error_code="PROC_001",
            details=details
        )
        
        assert error.error_code == "PROC_001"
        assert error.details == details


class TestStandardAgentInput:
    """Test StandardAgentInput model."""
    
    def test_standard_agent_input_creation(self):
        """Test creating StandardAgentInput."""
        context = AgentContext(
            session_id="test_session",
            user_id="test_user"
        )
        
        agent_input = StandardAgentInput(
            operation=OperationType.ANALYZE,
            data={"input": "test data"},
            context=context
        )
        
        assert agent_input.operation == OperationType.ANALYZE
        assert agent_input.data == {"input": "test data"}
        assert agent_input.context == context
        assert agent_input.request_id is not None
        assert agent_input.timestamp is not None
    
    def test_standard_agent_input_with_parameters(self):
        """Test StandardAgentInput with parameters."""
        parameters = {"param1": "value1", "param2": 42}
        context = AgentContext(
            session_id="test_session",
            user_id="test_user"
        )
        
        agent_input = StandardAgentInput(
            operation=OperationType.PROCESS,
            data={"input": "test data"},
            context=context,
            parameters=parameters
        )
        
        assert agent_input.parameters == parameters


class TestStandardAgentOutput:
    """Test StandardAgentOutput model."""
    
    def test_standard_agent_output_creation(self):
        """Test creating StandardAgentOutput."""
        metadata = AgentMetadata(
            agent_id="test_agent",
            agent_version="1.0.0",
            capability_name="test_capability"
        )
        
        result = ProcessingResult(
            status=TaskStatus.COMPLETED,
            result_data={"output": "test result"},
            confidence=ConfidenceLevel.HIGH
        )
        
        agent_output = StandardAgentOutput(
            request_id="test_request",
            agent_metadata=metadata,
            result=result
        )
        
        assert agent_output.request_id == "test_request"
        assert agent_output.agent_metadata == metadata
        assert agent_output.result == result
        assert agent_output.timestamp is not None
    
    def test_standard_agent_output_with_context(self):
        """Test StandardAgentOutput with context."""
        metadata = AgentMetadata(
            agent_id="test_agent",
            agent_version="1.0.0",
            capability_name="test_capability"
        )
        
        result = ProcessingResult(
            status=TaskStatus.COMPLETED,
            result_data={"output": "test result"},
            confidence=ConfidenceLevel.HIGH
        )
        
        context = AgentContext(
            session_id="test_session",
            user_id="test_user"
        )
        
        agent_output = StandardAgentOutput(
            request_id="test_request",
            agent_metadata=metadata,
            result=result,
            context=context
        )
        
        assert agent_output.context == context


class TestCommunicationProtocol:
    """Test CommunicationProtocol model."""
    
    def test_communication_protocol_creation(self):
        """Test creating CommunicationProtocol."""
        protocol = CommunicationProtocol(
            protocol_version="1.0",
            message_format="json",
            encoding="utf-8"
        )
        
        assert protocol.protocol_version == "1.0"
        assert protocol.message_format == "json"
        assert protocol.encoding == "utf-8"
        assert protocol.compression is None
    
    def test_communication_protocol_with_compression(self):
        """Test CommunicationProtocol with compression."""
        protocol = CommunicationProtocol(
            protocol_version="1.0",
            message_format="json",
            encoding="utf-8",
            compression="gzip"
        )
        
        assert protocol.compression == "gzip"


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_standard_output(self):
        """Test create_standard_output helper function."""
        output = create_standard_output(
            agent_id="test_agent",
            capability_name="test_capability",
            result={"key": "value"}
        )
        
        assert isinstance(output, StandardAgentOutput)
        assert output.agent_metadata.agent_id == "test_agent"
        assert output.agent_metadata.capability_name == "test_capability"
        assert output.result.result_data == {"key": "value"}
        assert output.result.status == TaskStatus.COMPLETED
    
    def test_create_standard_output_with_request_id(self):
        """Test create_standard_output with request_id."""
        output = create_standard_output(
            agent_id="test_agent",
            capability_name="test_capability",
            result={"key": "value"},
            request_id="custom_request_id"
        )
        
        assert output.request_id == "custom_request_id"
    
    def test_create_agent_error(self):
        """Test create_agent_error helper function."""
        error = create_agent_error(
            error_type=ErrorType.VALIDATION_ERROR,
            message="Test error",
            agent_id="test_agent"
        )
        
        assert isinstance(error, AgentError)
        assert error.error_type == ErrorType.VALIDATION_ERROR
        assert error.message == "Test error"
        assert error.agent_id == "test_agent"
    
    def test_create_agent_error_with_details(self):
        """Test create_agent_error with details."""
        details = {"field": "invalid_field"}
        error = create_agent_error(
            error_type=ErrorType.VALIDATION_ERROR,
            message="Validation failed",
            agent_id="test_agent",
            error_code="VAL_001",
            details=details
        )
        
        assert error.error_code == "VAL_001"
        assert error.details == details
    
    def test_create_agent_context(self):
        """Test create_agent_context helper function."""
        context = create_agent_context(
            session_id="test_session",
            user_id="test_user"
        )
        
        assert isinstance(context, AgentContext)
        assert context.session_id == "test_session"
        assert context.user_id == "test_user"
    
    def test_create_agent_context_with_metadata(self):
        """Test create_agent_context with metadata."""
        metadata = {"custom": "value"}
        context = create_agent_context(
            session_id="test_session",
            user_id="test_user",
            metadata=metadata
        )
        
        assert context.metadata == metadata


class TestValidation:
    """Test model validation."""
    
    def test_standard_agent_input_validation(self):
        """Test StandardAgentInput validation."""
        # Valid input should not raise
        context = AgentContext(
            session_id="test_session",
            user_id="test_user"
        )
        
        agent_input = StandardAgentInput(
            operation=OperationType.ANALYZE,
            data={"input": "test"},
            context=context
        )
        
        assert agent_input.operation == OperationType.ANALYZE
    
    def test_processing_result_validation(self):
        """Test ProcessingResult validation."""
        # Valid result should not raise
        result = ProcessingResult(
            status=TaskStatus.COMPLETED,
            result_data={"result": "success"},
            confidence=ConfidenceLevel.HIGH
        )
        
        assert result.status == TaskStatus.COMPLETED
    
    def test_agent_error_validation(self):
        """Test AgentError validation."""
        # Valid error should not raise
        error = AgentError(
            error_type=ErrorType.PROCESSING_ERROR,
            message="Something went wrong",
            agent_id="test_agent"
        )
        
        assert error.error_type == ErrorType.PROCESSING_ERROR


class TestIntegration:
    """Test integration scenarios."""
    
    def test_complete_communication_flow(self):
        """Test complete communication flow with all components."""
        # Create context
        context = create_agent_context(
            session_id="integration_session",
            user_id="integration_user"
        )
        
        # Create input
        agent_input = StandardAgentInput(
            operation=OperationType.ANALYZE,
            data={"dataset": "test_data.csv"},
            context=context,
            parameters={"threshold": 0.8}
        )
        
        # Create output
        output = create_standard_output(
            agent_id="data_quality_agent",
            capability_name="quality_assessment",
            result={
                "quality_score": 0.95,
                "issues_found": [],
                "recommendations": ["None"]
            },
            request_id=agent_input.request_id
        )
        
        # Verify complete flow
        assert output.request_id == agent_input.request_id
        assert output.agent_metadata.agent_id == "data_quality_agent"
        assert output.result.status == TaskStatus.COMPLETED
        assert output.result.result_data["quality_score"] == 0.95
    
    def test_error_handling_flow(self):
        """Test error handling in communication flow."""
        # Create error
        error = create_agent_error(
            error_type=ErrorType.VALIDATION_ERROR,
            message="Invalid dataset format",
            agent_id="data_quality_agent",
            error_code="DQ_001",
            details={"expected": "CSV", "received": "XLS"}
        )
        
        # Create failed result
        result = ProcessingResult(
            status=TaskStatus.FAILED,
            result_data={},
            confidence=ConfidenceLevel.LOW,
            errors=[error.message]
        )
        
        # Create output with error
        output = create_standard_output(
            agent_id="data_quality_agent",
            capability_name="quality_assessment",
            result=result.result_data,
            status=TaskStatus.FAILED
        )
        
        assert output.result.status == TaskStatus.FAILED
