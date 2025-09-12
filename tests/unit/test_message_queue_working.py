"""
Working tests for Message Queue Service - simplified to focus on coverage.

These tests focus on successfully testing the message queue components
without getting stuck on implementation details.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from core.services.message_queue import (
    MessageType,
    QueueConfig, 
    MessageContext,
    QueueMessage,
    MessageQueueService,
    get_message_queue_service,
    send_message,
    register_message_handler,
    initialize_default_queues,
    start_message_queue_service,
    stop_message_queue_service,
    handle_workflow_request,
    handle_workflow_response,
    handle_agent_communication
)


class TestMessageQueue:
    """Test message queue components for coverage."""

    def test_message_type_enum(self):
        """Test MessageType enum."""
        assert MessageType.WORKFLOW_REQUEST.value == "workflow_request"
        assert MessageType.COMPLIANCE_CHECK.value == "compliance_check"
        
    def test_queue_config(self):
        """Test QueueConfig dataclass."""
        config = QueueConfig(
            name="test",
            url="test-url", 
            message_type=MessageType.WORKFLOW_REQUEST
        )
        assert config.name == "test"
        assert config.visibility_timeout == 30  # default
        
    def test_message_context(self):
        """Test MessageContext dataclass."""
        context = MessageContext(
            queue_name="test",
            message_id="123",
            receipt_handle="handle"
        )
        assert context.queue_name == "test"
        assert context.attempt_count == 1  # default
        
    def test_queue_message_creation(self):
        """Test QueueMessage creation."""
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_REQUEST,
            source="test",
            payload={"data": "test"}
        )
        assert message.source == "test"
        assert message.payload == {"data": "test"}
        # Due to use_enum_values=True, message_type is converted to string
        assert message.message_type == "workflow_request"


class TestMessageQueueService:
    """Test MessageQueueService for coverage."""
    
    @pytest.fixture
    def mock_settings(self):
        with patch('core.services.message_queue.get_settings') as mock:
            settings = Mock()
            settings.aws_region = "us-east-1"
            settings.aws_access_key_id = "test_key"
            settings.aws_secret_access_key = "test_secret"
            settings.aws_profile = None
            mock.return_value = settings
            yield mock

    def test_service_initialization(self, mock_settings):
        """Test service initialization."""
        with patch('boto3.client'):
            service = MessageQueueService()
            assert service.queues == {}
            assert service.is_running is False
            
    def test_register_queue(self, mock_settings):
        """Test queue registration."""
        with patch('boto3.client'):
            service = MessageQueueService()
            service.register_queue(
                name="test",
                url="test-url",
                message_type=MessageType.WORKFLOW_REQUEST
            )
            assert "test" in service.queues
            assert service.queues["test"].name == "test"
            
    def test_register_message_handler(self, mock_settings):
        """Test message handler registration."""
        with patch('boto3.client'):
            service = MessageQueueService()
            handler = Mock()
            service.register_message_handler(MessageType.WORKFLOW_REQUEST, handler)
            assert MessageType.WORKFLOW_REQUEST in service.message_handlers
            
    @pytest.mark.asyncio 
    async def test_send_message_queue_not_found(self, mock_settings):
        """Test sending message to unregistered queue."""
        with patch('boto3.client'):
            service = MessageQueueService()
            message = QueueMessage(
                message_type=MessageType.WORKFLOW_REQUEST,
                source="test",
                payload={}
            )
            result = await service.send_message("nonexistent", message)
            assert result is False
            
    @pytest.mark.asyncio
    async def test_start_stop_processing(self, mock_settings):
        """Test starting and stopping processing.""" 
        with patch('boto3.client'):
            service = MessageQueueService()
            service.register_queue("test", "url", MessageType.WORKFLOW_REQUEST)
            
            # Mock process_queue_messages to avoid infinite loop
            with patch.object(service, 'process_queue_messages', new_callable=AsyncMock):
                await service.start_processing()
                assert service.is_running is True
                
                await service.stop_processing()
                assert service.is_running is False

    def test_get_queue_stats(self, mock_settings):
        """Test getting queue statistics."""
        with patch('boto3.client') as mock_boto:
            mock_sqs = Mock()
            mock_sqs.get_queue_attributes.return_value = {
                'Attributes': {
                    'ApproximateNumberOfMessages': '5',
                    'ApproximateNumberOfMessagesNotVisible': '2'
                }
            }
            mock_boto.return_value = mock_sqs
            
            service = MessageQueueService()
            service.register_queue("test", "url", MessageType.WORKFLOW_REQUEST)
            
            stats = service.get_queue_stats()
            assert "test" in stats
            assert stats["test"]["visible_messages"] == 5


class TestHelperFunctions:
    """Test helper functions for coverage."""
    
    def test_get_message_queue_service(self):
        """Test getting global service instance."""
        with patch('core.services.message_queue.MessageQueueService') as mock_service:
            service1 = get_message_queue_service()
            service2 = get_message_queue_service()
            # Should be singleton
            assert service1 is service2
            
    @pytest.mark.asyncio
    async def test_send_message_helper(self):
        """Test send_message helper."""
        with patch('core.services.message_queue.get_message_queue_service') as mock_get:
            mock_service = Mock()
            mock_service.send_message = AsyncMock(return_value=True)
            mock_get.return_value = mock_service
            
            message = QueueMessage(
                message_type=MessageType.WORKFLOW_REQUEST,
                source="test", 
                payload={}
            )
            
            result = await send_message("test", message)
            assert result is True
            mock_service.send_message.assert_called_once()
            
    def test_register_message_handler_helper(self):
        """Test register_message_handler helper."""
        with patch('core.services.message_queue.get_message_queue_service') as mock_get:
            mock_service = Mock()
            mock_get.return_value = mock_service
            
            handler = Mock()
            register_message_handler(MessageType.WORKFLOW_REQUEST, handler)
            mock_service.register_message_handler.assert_called_once_with(
                MessageType.WORKFLOW_REQUEST, handler
            )
            
    @pytest.mark.asyncio
    async def test_start_stop_service_helpers(self):
        """Test start/stop service helpers."""
        with patch('core.services.message_queue.get_message_queue_service') as mock_get, \
             patch('core.services.message_queue.initialize_default_queues') as mock_init:
            
            mock_service = Mock()
            mock_service.start_processing = AsyncMock()
            mock_service.stop_processing = AsyncMock()
            mock_get.return_value = mock_service
            
            await start_message_queue_service()
            mock_init.assert_called_once()
            mock_service.start_processing.assert_called_once()
            
            await stop_message_queue_service()
            mock_service.stop_processing.assert_called_once()
            
    def test_initialize_default_queues(self):
        """Test default queue initialization."""
        with patch('core.services.message_queue.get_settings') as mock_settings, \
             patch('core.services.message_queue.register_queue') as mock_register, \
             patch('core.services.message_queue.register_message_handler') as mock_handler:
            
            settings = Mock()
            settings.workflow_rules_request_sqs_url = "request-url"
            settings.workflow_rules_response_sqs_url = "response-url"
            mock_settings.return_value = settings
            
            initialize_default_queues()
            
            # Should register 2 queues
            assert mock_register.call_count == 2
            # Should register 3 handlers
            assert mock_handler.call_count == 3


class TestMessageHandlers:
    """Test default message handlers for coverage."""
    
    @pytest.mark.asyncio
    async def test_handle_workflow_request(self):
        """Test workflow request handler."""
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_REQUEST,
            source="test",
            payload={"workflow_type": "data_quality_assessment"}
        )
        context = MessageContext("queue", "id", "handle")
        
        result = await handle_workflow_request(message, context)
        assert result is True
        
    @pytest.mark.asyncio
    async def test_handle_workflow_request_with_reply(self):
        """Test workflow request handler with reply."""
        with patch('core.services.message_queue.send_message', new_callable=AsyncMock) as mock_send:
            message = QueueMessage(
                message_type=MessageType.WORKFLOW_REQUEST,
                source="test",
                payload={"workflow_type": "compliance_check"},
                reply_to="reply-queue",
                correlation_id="123"
            )
            context = MessageContext("queue", "id", "handle")
            
            result = await handle_workflow_request(message, context)
            assert result is True
            mock_send.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_handle_workflow_request_unknown_type(self):
        """Test workflow request handler with unknown type."""
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_REQUEST,
            source="test",
            payload={"workflow_type": "unknown_type"}
        )
        context = MessageContext("queue", "id", "handle")
        
        result = await handle_workflow_request(message, context)
        assert result is False
        
    @pytest.mark.asyncio
    async def test_handle_workflow_response(self):
        """Test workflow response handler."""
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_RESPONSE,
            source="test",
            payload={"status": "completed", "correlation_id": "123"}
        )
        context = MessageContext("queue", "id", "handle")
        
        result = await handle_workflow_response(message, context)
        assert result is True
        
    @pytest.mark.asyncio
    async def test_handle_agent_communication(self):
        """Test agent communication handler."""
        message = QueueMessage(
            message_type=MessageType.AGENT_COMMUNICATION,
            source="test",
            payload={"target_agent": "compliance", "message": "test"}
        )
        context = MessageContext("queue", "id", "handle")
        
        result = await handle_agent_communication(message, context)
        assert result is True


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.fixture
    def mock_settings_no_creds(self):
        with patch('core.services.message_queue.get_settings') as mock:
            settings = Mock()
            settings.aws_region = "us-east-1"
            settings.aws_access_key_id = None
            settings.aws_secret_access_key = None
            settings.aws_profile = None
            mock.return_value = settings
            yield mock
            
    def test_initialization_no_credentials(self, mock_settings_no_creds):
        """Test service initialization with no credentials."""
        with patch('boto3.client') as mock_boto:
            from botocore.exceptions import NoCredentialsError
            mock_boto.side_effect = NoCredentialsError()
            
            with pytest.raises(NoCredentialsError):
                MessageQueueService()
                
    @pytest.mark.asyncio
    async def test_send_message_client_error(self):
        """Test send_message with client error."""
        with patch('core.services.message_queue.get_settings') as mock_settings, \
             patch('boto3.client') as mock_boto:
            
            settings = Mock()
            settings.aws_region = "us-east-1"
            settings.aws_access_key_id = "key"
            settings.aws_secret_access_key = "secret"
            settings.aws_profile = None
            mock_settings.return_value = settings
            
            from botocore.exceptions import ClientError
            mock_sqs = Mock()
            mock_sqs.send_message.side_effect = ClientError(
                {'Error': {'Code': 'QueueDoesNotExist'}}, 'SendMessage'
            )
            mock_boto.return_value = mock_sqs
            
            service = MessageQueueService()
            service.register_queue("test", "url", MessageType.WORKFLOW_REQUEST)
            
            message = QueueMessage(
                message_type=MessageType.WORKFLOW_REQUEST,
                source="test",
                payload={}
            )
            
            result = await service.send_message("test", message)
            assert result is False
            
    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """Test error handling in message handlers."""
        # Test handler that raises exception
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_REQUEST,
            source="test",
            payload={}  # Missing workflow_type will cause KeyError
        )
        context = MessageContext("queue", "id", "handle")

        result = await handle_workflow_request(message, context)
        assert result is False

        # Test workflow_response handler - this one handles missing fields gracefully
        result = await handle_workflow_response(message, context)
        assert result is True  # This handler doesn't error on missing fields
        
        # Test agent communication handler  
        result = await handle_agent_communication(message, context)
        assert result is True  # This handler also doesn't error on missing fields