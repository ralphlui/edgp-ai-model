"""
Unit tests for core.services.message_queue module.
"""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from core.services.message_queue import (
    MessageQueueService, 
    QueueMessage, 
    MessageType, 
    QueueConfig,
    MessageContext
)


class TestQueueMessage:
    """Test the QueueMessage model."""
    
    def test_queue_message_creation(self):
        """Test creating a basic queue message."""
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_REQUEST,
            source="test-agent",
            payload={"test": "data"}
        )
        
        assert message.message_type == MessageType.WORKFLOW_REQUEST
        assert message.source == "test-agent"
        assert message.payload == {"test": "data"}
        assert message.message_id is not None
        assert message.timestamp is not None
    
    def test_queue_message_with_optional_fields(self):
        """Test creating a queue message with optional fields."""
        message = QueueMessage(
            message_id="custom-id",
            message_type=MessageType.COMPLIANCE_CHECK,
            source="compliance-agent",
            destination="data-agent",
            payload={"compliance": "check"},
            correlation_id="corr-123",
            reply_to="response-queue",
            ttl_seconds=300
        )
        
        assert message.message_id == "custom-id"
        assert message.destination == "data-agent"
        assert message.correlation_id == "corr-123"
        assert message.reply_to == "response-queue"
        assert message.ttl_seconds == 300
    
    def test_queue_message_auto_generated_id(self):
        """Test that message ID is auto-generated when not provided."""
        message1 = QueueMessage(
            message_type=MessageType.ANALYTICS_REQUEST,
            source="analytics-agent",
            payload={}
        )
        
        message2 = QueueMessage(
            message_type=MessageType.ANALYTICS_REQUEST,
            source="analytics-agent",
            payload={}
        )
        
        assert message1.message_id != message2.message_id


class TestQueueConfig:
    """Test the QueueConfig dataclass."""
    
    def test_queue_config_creation(self):
        """Test creating a queue config."""
        config = QueueConfig(
            name="test-queue",
            url="https://sqs.region.amazonaws.com/account/queue",
            message_type=MessageType.DATA_QUALITY_ASSESSMENT
        )
        
        assert config.name == "test-queue"
        assert config.url == "https://sqs.region.amazonaws.com/account/queue"
        assert config.message_type == MessageType.DATA_QUALITY_ASSESSMENT
        assert config.visibility_timeout == 30  # default
        assert config.wait_time_seconds == 20   # default
        assert config.max_messages == 10        # default
    
    def test_queue_config_with_custom_values(self):
        """Test creating queue config with custom values."""
        def dummy_handler():
            pass
            
        config = QueueConfig(
            name="custom-queue",
            url="https://sqs.region.amazonaws.com/account/custom",
            message_type=MessageType.REMEDIATION_TASK,
            visibility_timeout=60,
            wait_time_seconds=10,
            max_messages=5,
            handler=dummy_handler
        )
        
        assert config.visibility_timeout == 60
        assert config.wait_time_seconds == 10
        assert config.max_messages == 5
        assert config.handler == dummy_handler


class TestMessageQueueService:
    """Test the MessageQueueService class."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings fixture."""
        with patch('core.services.message_queue.get_settings') as mock:
            mock_obj = Mock()
            mock_obj.aws_region = 'us-east-1'
            mock_obj.aws_access_key_id = 'test-key'
            mock_obj.aws_secret_access_key = 'test-secret'
            mock.return_value = mock_obj
            yield mock
    
    @pytest.fixture 
    def mock_boto3_client(self):
        """Mock boto3 SQS client."""
        with patch('core.services.message_queue.boto3') as mock_boto3:
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            yield mock_client
    
    def test_init_with_credentials(self, mock_settings, mock_boto3_client):
        """Test initialization with AWS credentials."""
        service = MessageQueueService()
        
        assert service.settings is not None
        assert service.sqs_client is not None
        assert service.queues == {}
        assert service.handlers == {}
        assert service.stats == {'messages_sent': 0, 'messages_received': 0, 'errors': 0}
        assert service.is_processing is False
    
    def test_register_queue(self, mock_settings, mock_boto3_client):
        """Test registering a queue."""
        service = MessageQueueService()
        
        service.register_queue(
            name="test-queue",
            url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
            message_type=MessageType.WORKFLOW_REQUEST
        )
        
        assert "test-queue" in service.queues
        queue_config = service.queues["test-queue"]
        assert queue_config.name == "test-queue"
        assert queue_config.message_type == MessageType.WORKFLOW_REQUEST
    
    def test_register_message_handler(self, mock_settings, mock_boto3_client):
        """Test registering a message handler."""
        service = MessageQueueService()
        
        async def test_handler(message, context):
            return True
        
        service.register_message_handler(MessageType.ANALYTICS_REQUEST, test_handler)
        
        assert MessageType.ANALYTICS_REQUEST in service.handlers
        assert service.handlers[MessageType.ANALYTICS_REQUEST] == test_handler
    
    @pytest.mark.asyncio
    async def test_send_message_with_mocked_enum(self, mock_settings, mock_boto3_client):
        """Test successful message sending with proper enum handling."""
        service = MessageQueueService()
        
        # Setup queue
        service.register_queue(
            name="test-queue",
            url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
            message_type=MessageType.DATA_QUALITY_ASSESSMENT
        )
        
        # Setup mock response
        mock_boto3_client.send_message.return_value = {
            'MessageId': 'test-message-id'
        }
        
        message = QueueMessage(
            message_type=MessageType.DATA_QUALITY_ASSESSMENT,
            source="test-agent",
            payload={"test": "data"}
        )
        
        # Mock the enum value conversion
        with patch.object(message.message_type, 'value', 'data_quality_assessment'):
            success = await service.send_message("test-queue", message)
        
        # Check if the send_message call was made to boto3
        mock_boto3_client.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_message_queue_not_found(self, mock_settings, mock_boto3_client):
        """Test sending message to non-registered queue."""
        service = MessageQueueService()
        
        message = QueueMessage(
            message_type=MessageType.DATA_QUALITY_ASSESSMENT,
            source="test-agent",
            payload={"test": "data"}
        )
        
        success = await service.send_message("non-existent-queue", message)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_process_queue_messages_empty_queue(self, mock_settings, mock_boto3_client):
        """Test processing messages from empty queue."""
        service = MessageQueueService()
        
        # Setup queue
        service.register_queue(
            name="test-queue",
            url="https://sqs.us-east-1.amazonaws.com/123456789012/test-queue",
            message_type=MessageType.DATA_QUALITY_ASSESSMENT
        )
        
        # Setup empty response
        mock_boto3_client.receive_message.return_value = {}
        
        # This should complete without error even with empty queue
        await service.process_queue_messages("test-queue")
        
        # Verify receive_message was called
        mock_boto3_client.receive_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_start_stop_processing(self, mock_settings, mock_boto3_client):
        """Test starting and stopping message processing."""
        service = MessageQueueService()
        
        # Mock asyncio.create_task to avoid actual background processing
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            
            await service.start_processing()
            assert service.is_processing is True
            
            # Add at least one queue for processing to start
            service.register_queue(
                name="test-queue",
                url="https://test-url",
                message_type=MessageType.WORKFLOW_REQUEST
            )
            
            await service.stop_processing()
            assert service.is_processing is False
    
    def test_get_queue_stats(self, mock_settings, mock_boto3_client):
        """Test getting queue statistics."""
        service = MessageQueueService()
        
        # Register some queues
        service.register_queue(
            name="queue1",
            url="https://test-url1",
            message_type=MessageType.WORKFLOW_REQUEST
        )
        service.register_queue(
            name="queue2", 
            url="https://test-url2",
            message_type=MessageType.COMPLIANCE_CHECK
        )
        
        stats = service.get_queue_stats()
        
        assert 'total_queues' in stats
        assert 'messages_sent' in stats
        assert 'messages_received' in stats
        assert 'errors' in stats
        assert 'queues' in stats
        assert stats['total_queues'] == 2
        assert len(stats['queues']) == 2
    
    def test_message_type_enum_values(self):
        """Test that all MessageType enum values are valid."""
        expected_types = [
            "workflow_request",
            "workflow_response", 
            "agent_communication",
            "remediation_task",
            "compliance_check",
            "data_quality_assessment",
            "analytics_request"
        ]
        
        for msg_type in MessageType:
            assert msg_type.value in expected_types


class TestMessageContext:
    """Test the MessageContext dataclass."""
    
    def test_message_context_creation(self):
        """Test creating a message context."""
        context = MessageContext(
            queue_name="test-queue",
            message_id="msg-123",
            receipt_handle="receipt-456"
        )
        
        assert context.queue_name == "test-queue"
        assert context.message_id == "msg-123" 
        assert context.receipt_handle == "receipt-456"
        assert context.attempt_count == 1
        assert isinstance(context.received_at, datetime)
        assert context.attributes == {}
    
    def test_message_context_with_custom_values(self):
        """Test creating message context with custom values."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        custom_attrs = {"retries": 3, "priority": "high"}
        
        context = MessageContext(
            queue_name="priority-queue",
            message_id="priority-msg-789",
            receipt_handle="priority-receipt-101",
            received_at=custom_time,
            attempt_count=2,
            attributes=custom_attrs
        )
        
        assert context.received_at == custom_time
        assert context.attempt_count == 2
        assert context.attributes == custom_attrs


class TestMessageHandlers:
    """Test the message handler functions."""
    
    @pytest.mark.asyncio
    async def test_handle_workflow_request(self):
        """Test workflow request handler."""
        from core.services.message_queue import handle_workflow_request
        
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_REQUEST,
            source="workflow-engine",
            payload={"workflow_id": "wf-123", "action": "start"}
        )
        
        context = MessageContext(
            queue_name="workflow-request-queue",
            message_id="msg-456",
            receipt_handle="receipt-789"
        )
        
        # Mock any dependencies that might be called
        with patch('core.services.message_queue.logger') as mock_logger:
            result = await handle_workflow_request(message, context)
            
            # The handler should return True and log the message
            assert result is True
            mock_logger.info.assert_called()
    
    @pytest.mark.asyncio 
    async def test_handle_workflow_response(self):
        """Test workflow response handler."""
        from core.services.message_queue import handle_workflow_response
        
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_RESPONSE,
            source="data-agent",
            payload={"workflow_id": "wf-123", "status": "completed", "result": {"data": "processed"}}
        )
        
        context = MessageContext(
            queue_name="workflow-response-queue",
            message_id="msg-789",
            receipt_handle="receipt-101"
        )
        
        with patch('core.services.message_queue.logger') as mock_logger:
            result = await handle_workflow_response(message, context)
            
            assert result is True
            mock_logger.info.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_agent_communication(self):
        """Test agent communication handler."""
        from core.services.message_queue import handle_agent_communication
        
        message = QueueMessage(
            message_type=MessageType.AGENT_COMMUNICATION,
            source="analytics-agent",
            destination="compliance-agent",
            payload={"type": "data_request", "data_id": "dataset-456"}
        )
        
        context = MessageContext(
            queue_name="agent-communication-queue",
            message_id="msg-101",
            receipt_handle="receipt-202"
        )
        
        with patch('core.services.message_queue.logger') as mock_logger:
            result = await handle_agent_communication(message, context)
            
            assert result is True
            mock_logger.info.assert_called()


class TestServiceFunctions:
    """Test module-level service functions."""
    
    def test_initialize_default_queues(self):
        """Test initializing default queues."""
        from core.services.message_queue import initialize_default_queues
        
        with patch('core.services.message_queue.message_queue_service') as mock_service:
            initialize_default_queues()
            
            # Should register the default queues
            assert mock_service.register_queue.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_start_message_queue_service(self):
        """Test starting the message queue service."""
        from core.services.message_queue import start_message_queue_service
        
        with patch('core.services.message_queue.message_queue_service') as mock_service:
            await start_message_queue_service()
            
            mock_service.start_processing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_message_queue_service(self):
        """Test stopping the message queue service."""
        from core.services.message_queue import stop_message_queue_service
        
        with patch('core.services.message_queue.message_queue_service') as mock_service:
            await stop_message_queue_service()
            
            mock_service.stop_processing.assert_called_once()
