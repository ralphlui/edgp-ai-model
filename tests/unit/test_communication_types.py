"""
Simple unit tests for core.types.communication module that only test existing functionality.
"""
import pytest
from datetime import datetime
from typing import Dict, Any
from core.types.communication import (
    MessageType, OperationType, ProcessingStage, ExecutionMode,
    StandardAgentInput, AgentContext, AgentError, DataPayload,
    create_standard_input, create_agent_error
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
    
    def test_execution_mode_enum(self):
        """Test ExecutionMode enum values."""
        assert ExecutionMode.SYNCHRONOUS.value == "synchronous"
        assert ExecutionMode.ASYNCHRONOUS.value == "asynchronous"
        assert ExecutionMode.BACKGROUND.value == "background"


class TestAgentContext:
    """Test AgentContext model."""
    
    def test_agent_context_creation(self):
        """Test AgentContext creation with basic fields."""
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
        """Test AgentContext with default values."""
        context = AgentContext()
        
        assert context.session_id is not None
        assert context.correlation_id is not None
        assert context.user_id is None
        assert context.tenant_id is None
        assert context.execution_mode == ExecutionMode.SYNCHRONOUS
        assert context.retry_count == 0
        assert context.max_retries == 3


class TestDataPayload:
    """Test DataPayload model."""
    
    def test_data_payload_creation(self):
        """Test DataPayload creation."""
        content = {"key": "value", "number": 42}
        payload = DataPayload(
            data_type="json",
            data_format="application/json",
            content=content
        )
        
        assert payload.data_type == "json"
        assert payload.data_format == "application/json"
        assert payload.content == content
        assert payload.data_id is not None


class TestAgentError:
    """Test AgentError model."""
    
    def test_agent_error_creation(self):
        """Test AgentError creation."""
        error = AgentError(
            error_code="VALIDATION_001",
            error_type="VALIDATION",
            error_message="Invalid input data",
            processing_stage=ProcessingStage.VALIDATION,
            agent_id="test_agent",
            capability_name="test_capability"
        )
        
        assert error.error_code == "VALIDATION_001"
        assert error.error_type == "VALIDATION"
        assert error.error_message == "Invalid input data"
        assert error.processing_stage == ProcessingStage.VALIDATION
        assert error.agent_id == "test_agent"
        assert error.capability_name == "test_capability"
        assert error.is_recoverable is True
        assert error.severity == Severity.MEDIUM


class TestStandardAgentInput:
    """Test StandardAgentInput model."""
    
    def test_standard_agent_input_creation(self):
        """Test StandardAgentInput creation."""
        context = AgentContext()
        data = {"test": "data"}
        
        agent_input = StandardAgentInput(
            source_agent_id="source_agent",
            target_agent_id="target_agent",
            capability_name="test_capability",
            context=context,
            data=data
        )
        
        assert agent_input.source_agent_id == "source_agent"
        assert agent_input.target_agent_id == "target_agent"
        assert agent_input.capability_name == "test_capability"
        assert agent_input.context == context
        assert agent_input.data == data
        assert agent_input.message_type == MessageType.REQUEST
        assert agent_input.priority == Priority.MEDIUM


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_standard_input(self):
        """Test create_standard_input function."""
        data = {"test": "data"}
        
        agent_input = create_standard_input(
            source_agent_id="source_agent",
            target_agent_id="target_agent",
            capability_name="test_capability",
            data=data
        )
        
        assert agent_input.source_agent_id == "source_agent"
        assert agent_input.target_agent_id == "target_agent"
        assert agent_input.capability_name == "test_capability"
        assert agent_input.data == data
        assert isinstance(agent_input.context, AgentContext)
    
    def test_create_agent_error(self):
        """Test create_agent_error function."""
        error = create_agent_error(
            error_code="TEST_001",
            error_message="Test error message",
            agent_id="test_agent",
            capability_name="test_capability"
        )
        
        assert error.error_code == "TEST_001"
        assert error.error_message == "Test error message"
        assert error.agent_id == "test_agent"
        assert error.capability_name == "test_capability"
        assert error.error_type == "TEST"  # Derived from error_code
        assert error.processing_stage == ProcessingStage.PROCESSING
        assert error.severity == Severity.MEDIUM


class TestIntegration:
    """Test integration scenarios."""
    
    def test_complete_communication_flow(self):
        """Test a complete communication flow."""
        # Create context
        context = AgentContext(
            session_id="test_session",
            user_id="test_user"
        )
        
        # Create input data
        data = {"operation": "test", "parameters": {"value": 42}}
        
        # Create standardized input
        agent_input = create_standard_input(
            source_agent_id="source_agent",
            target_agent_id="target_agent",
            capability_name="test_capability",
            data=data
        )
        
        # Verify the flow
        assert agent_input.source_agent_id == "source_agent"
        assert agent_input.target_agent_id == "target_agent"
        assert agent_input.data == data
        assert agent_input.message_id is not None
        assert isinstance(agent_input.timestamp, datetime)
