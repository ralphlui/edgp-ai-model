"""
Message Queue Service for EDGP AI Model

Handles sending and processing messages from multiple AWS SQS queues.
Similar to the EDGP rules engine implementation with runtime AWS credentials support.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import BaseModel, Field

from core.infrastructure.config import get_settings

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be processed."""
    WORKFLOW_REQUEST = "workflow_request"
    WORKFLOW_RESPONSE = "workflow_response"
    AGENT_COMMUNICATION = "agent_communication"
    REMEDIATION_TASK = "remediation_task"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_QUALITY_ASSESSMENT = "data_quality_assessment"
    ANALYTICS_REQUEST = "analytics_request"


@dataclass
class QueueConfig:
    """Configuration for a specific queue."""
    name: str
    url: str
    message_type: MessageType
    visibility_timeout: int = 30
    wait_time_seconds: int = 20
    max_messages: int = 10
    handler: Optional[Callable] = None


@dataclass
class MessageContext:
    """Context information for processing messages."""
    queue_name: str
    message_id: str
    receipt_handle: str
    received_at: datetime = field(default_factory=datetime.utcnow)
    attempt_count: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)


class QueueMessage(BaseModel):
    """Standard message format for queue operations."""
    
    message_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    message_type: MessageType
    source: str
    destination: Optional[str] = None
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: Optional[int] = None
    
    class Config:
        use_enum_values = False


class MessageQueueService:
    """Service for managing multiple SQS queues and message processing."""
    
    def __init__(self):
        self.settings = get_settings()
        self.sqs_client = None
        self.queues: Dict[str, QueueConfig] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.handlers = self.message_handlers  # Alias for backward compatibility
        self.processing_tasks: List[asyncio.Task] = []
        self.is_running = False
        self.is_processing = False  # Alias for backward compatibility
        self.stats = {'messages_sent': 0, 'messages_received': 0, 'errors': 0}
        
        # Initialize AWS client with runtime credentials
        self._initialize_aws_client()
        
    def _initialize_aws_client(self):
        """Initialize AWS SQS client with runtime credentials."""
        try:
            # Use runtime credentials if provided
            if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
                self.sqs_client = boto3.client(
                    'sqs',
                    aws_access_key_id=self.settings.aws_access_key_id,
                    aws_secret_access_key=self.settings.aws_secret_access_key,
                    region_name=self.settings.aws_region
                )
                logger.info(f"Initialized SQS client with runtime credentials for region: {self.settings.aws_region}")
            elif self.settings.aws_profile:
                # Use named profile
                session = boto3.Session(profile_name=self.settings.aws_profile)
                self.sqs_client = session.client('sqs', region_name=self.settings.aws_region)
                logger.info(f"Initialized SQS client with profile: {self.settings.aws_profile}")
            else:
                # Use default credentials (IAM role, environment variables, etc.)
                self.sqs_client = boto3.client('sqs', region_name=self.settings.aws_region)
                logger.info("Initialized SQS client with default credentials")
                
        except NoCredentialsError:
            logger.error("No AWS credentials found. Please provide credentials via environment variables or IAM role.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS SQS client: {e}")
            raise
    
    def register_queue(
        self, 
        config_or_name=None, 
        url: str = None, 
        message_type: MessageType = None,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 20,
        max_messages: int = 10,
        name: str = None  # Support keyword argument
    ):
        """Register a queue for processing. Can accept a QueueConfig object or individual parameters."""
        if isinstance(config_or_name, QueueConfig):
            # Called with a QueueConfig object
            queue_config = config_or_name
            queue_name = queue_config.name
        else:
            # Called with individual parameters - support both positional and keyword
            queue_name = name if name is not None else config_or_name
            queue_config = QueueConfig(
                name=queue_name,
                url=url,
                message_type=message_type,
                visibility_timeout=visibility_timeout,
                wait_time_seconds=wait_time_seconds,
                max_messages=max_messages
            )
        
        self.queues[queue_name] = queue_config
        logger.info(f"Registered queue: {queue_name} -> {queue_config.url} (type: {queue_config.message_type if isinstance(queue_config.message_type, str) else queue_config.message_type.value})")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type if isinstance(message_type, str) else message_type.value}")
    
    async def send_message(
        self, 
        queue_name: str, 
        message: QueueMessage,
        delay_seconds: int = 0
    ) -> bool:
        """Send a message to a specific queue."""
        try:
            if queue_name not in self.queues:
                logger.error(f"Queue not registered: {queue_name}")
                return False
            
            queue_config = self.queues[queue_name]
            
            # Prepare message body
            message_body = json.dumps(message.dict(), default=str)
            
            # Prepare message attributes
            message_attributes = {
                'MessageType': {
                    'StringValue': message.message_type if isinstance(message.message_type, str) else message.message_type.value,
                    'DataType': 'String'
                },
                'Source': {
                    'StringValue': message.source,
                    'DataType': 'String'
                },
                'Timestamp': {
                    'StringValue': message.timestamp.isoformat(),
                    'DataType': 'String'
                }
            }
            
            if message.correlation_id:
                message_attributes['CorrelationId'] = {
                    'StringValue': message.correlation_id,
                    'DataType': 'String'
                }
            
            # Send message
            response = self.sqs_client.send_message(
                QueueUrl=queue_config.url,
                MessageBody=message_body,
                MessageAttributes=message_attributes,
                DelaySeconds=delay_seconds
            )
            
            logger.info(f"Message sent to {queue_name}: {response['MessageId']}")
            self.stats['messages_sent'] += 1
            return True
            
        except ClientError as e:
            logger.error(f"Failed to send message to {queue_name}: {e}")
            self.stats['errors'] += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to {queue_name}: {e}")
            return False
    
    async def process_queue_messages(self, queue_name: str):
        """Process messages from a specific queue."""
        if queue_name not in self.queues:
            logger.error(f"Queue not registered: {queue_name}")
            return
        
        queue_config = self.queues[queue_name]
        logger.info(f"Starting message processing for queue: {queue_name}")
        
        while self.is_running:
            try:
                # Receive messages from queue
                response = self.sqs_client.receive_message(
                    QueueUrl=queue_config.url,
                    MaxNumberOfMessages=queue_config.max_messages,
                    WaitTimeSeconds=queue_config.wait_time_seconds,
                    VisibilityTimeout=queue_config.visibility_timeout,
                    MessageAttributeNames=['All']
                )
                
                messages = response.get('Messages', [])
                if not messages:
                    continue
                
                # Process each message
                for message in messages:
                    await self._process_single_message(queue_config, message)
                    
            except ClientError as e:
                logger.error(f"Error receiving messages from {queue_name}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
            except Exception as e:
                logger.error(f"Unexpected error processing {queue_name}: {e}")
                await asyncio.sleep(5)
    
    async def _process_single_message(self, queue_config: QueueConfig, raw_message: Dict[str, Any]):
        """Process a single message from the queue."""
        try:
            # Parse message
            message_body = json.loads(raw_message['Body'])
            message = QueueMessage(**message_body)
            
            # Create message context
            context = MessageContext(
                queue_name=queue_config.name,
                message_id=raw_message['MessageId'],
                receipt_handle=raw_message['ReceiptHandle'],
                attributes=raw_message.get('MessageAttributes', {})
            )
            
            # Get appropriate handler - normalize the lookup key
            # Handle both enum and string message types
            handler_key = message.message_type
            if isinstance(message.message_type, str):
                # Find the corresponding enum value
                handler_key = next((mt for mt in MessageType if mt.value == message.message_type), message.message_type)
                
            handler = self.message_handlers.get(handler_key)
            if not handler:
                logger.warning(f"No handler registered for message type: {message.message_type if isinstance(message.message_type, str) else message.message_type.value}")
                await self._delete_message(queue_config.url, context.receipt_handle)
                return
            
            # Process message
            logger.info(f"Processing message {context.message_id} of type {message.message_type if isinstance(message.message_type, str) else message.message_type.value}")
            
            success = await handler(message, context)
            
            if success:
                # Delete message from queue on successful processing
                await self._delete_message(queue_config.url, context.receipt_handle)
                logger.info(f"Successfully processed and deleted message {context.message_id}")
            else:
                logger.warning(f"Failed to process message {context.message_id}, leaving in queue")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message body: {e}")
            # Delete malformed messages
            await self._delete_message(queue_config.url, raw_message['ReceiptHandle'])
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _delete_message(self, queue_url: str, receipt_handle: str):
        """Delete a processed message from the queue."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
        except ClientError as e:
            logger.error(f"Failed to delete message: {e}")
    
    async def receive_messages(self, queue_name: str, max_messages: int = 10, wait_time_seconds: int = 20) -> List[QueueMessage]:
        """Receive messages from a specific queue."""
        try:
            if queue_name not in self.queues:
                logger.error(f"Queue not registered: {queue_name}")
                return []
            
            queue_config = self.queues[queue_name]
            
            response = self.sqs_client.receive_message(
                QueueUrl=queue_config.url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_seconds,
                MessageAttributeNames=['All']
            )
            
            messages = []
            for raw_message in response.get('Messages', []):
                try:
                    message_body = json.loads(raw_message['Body'])
                    message = QueueMessage(**message_body)
                    # Add receipt handle for deletion
                    message.receipt_handle = raw_message['ReceiptHandle']
                    message.message_id = raw_message['MessageId']
                    messages.append(message)
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Failed to parse message: {e}")
                    continue
            
            return messages
            
        except ClientError as e:
            logger.error(f"Failed to receive messages from {queue_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error receiving messages from {queue_name}: {e}")
            return []
    
    async def delete_message(self, queue_name: str, receipt_handle: str) -> bool:
        """Delete a message from a specific queue using receipt handle."""
        try:
            if queue_name not in self.queues:
                logger.error(f"Queue not registered: {queue_name}")
                return False
            
            queue_config = self.queues[queue_name]
            await self._delete_message(queue_config.url, receipt_handle)
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete message from {queue_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting message from {queue_name}: {e}")
            return False
    
    def get_queue_stats(self):
        """Get queue statistics."""
        return self.stats.copy()
    
    async def _process_message(self, queue_name: str, message: QueueMessage) -> bool:
        """Process a single message (used for testing)."""
        try:
            # Get appropriate handler - normalize the lookup key
            # Handle both enum and string message types
            handler_key = message.message_type
            if isinstance(message.message_type, str):
                # Find the corresponding enum value
                handler_key = next((mt for mt in MessageType if mt.value == message.message_type), message.message_type)
            
            handler = self.message_handlers.get(handler_key)
            if not handler:
                logger.warning(f"No handler registered for message type: {message.message_type if isinstance(message.message_type, str) else message.message_type.value}")
                return False
            
            # Create a mock context for testing
            context = MessageContext(
                queue_name=queue_name,
                message_id=getattr(message, 'message_id', 'test-id'),
                receipt_handle=getattr(message, 'receipt_handle', 'test-handle'),
                attributes={}
            )
            
            # Process message
            logger.info(f"Processing message {context.message_id} of type {message.message_type if isinstance(message.message_type, str) else message.message_type.value}")
            
            success = await handler.handle_message(message)
            return success
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False
    
    async def start_processing(self):
        """Start processing all registered queues."""
        if self.is_running:
            logger.warning("Message processing is already running")
            return
        
        self.is_running = True
        self.is_processing = True
        logger.info("Starting message queue processing for all registered queues")
        
        # Start processing task for each queue
        for queue_name in self.queues:
            task = asyncio.create_task(
                self.process_queue_messages(queue_name),
                name=f"queue_processor_{queue_name}"
            )
            self.processing_tasks.append(task)
        
        logger.info(f"Started {len(self.processing_tasks)} queue processing tasks")
    
    async def stop_processing(self):
        """Stop processing all queues."""
        if not self.is_running:
            return
        
        logger.info("Stopping message queue processing")
        self.is_running = False
        self.is_processing = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        self.processing_tasks.clear()
        logger.info("Message queue processing stopped")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all registered queues."""
        stats = {}
        
        for queue_name, queue_config in self.queues.items():
            try:
                response = self.sqs_client.get_queue_attributes(
                    QueueUrl=queue_config.url,
                    AttributeNames=[
                        'ApproximateNumberOfMessages',
                        'ApproximateNumberOfMessagesNotVisible',
                        'ApproximateNumberOfMessagesDelayed'
                    ]
                )
                
                stats[queue_name] = {
                    'url': queue_config.url,
                    'message_type': queue_config.message_type if isinstance(queue_config.message_type, str) else queue_config.message_type.value,
                    'visible_messages': int(response['Attributes'].get('ApproximateNumberOfMessages', 0)),
                    'in_flight_messages': int(response['Attributes'].get('ApproximateNumberOfMessagesNotVisible', 0)),
                    'delayed_messages': int(response['Attributes'].get('ApproximateNumberOfMessagesDelayed', 0))
                }
                
            except ClientError as e:
                logger.error(f"Failed to get stats for queue {queue_name}: {e}")
                stats[queue_name] = {'error': str(e)}
        
        return stats


# Global message queue service instance
# Commented out to prevent import-time initialization during tests
# message_queue_service = MessageQueueService()

def get_message_queue_service() -> MessageQueueService:
    """Get or create the global message queue service instance."""
    global _message_queue_service
    if '_message_queue_service' not in globals():
        _message_queue_service = MessageQueueService()
    return _message_queue_service


# Helper functions that use the global service
async def send_message(queue_name: str, message: QueueMessage) -> bool:
    """Send a message using the global service."""
    return await get_message_queue_service().send_message(queue_name, message)


async def receive_messages(queue_name: str, max_messages: int = 10) -> List[QueueMessage]:
    """Receive messages using the global service."""
    return await get_message_queue_service().receive_messages(queue_name, max_messages)


def register_queue(config: QueueConfig) -> bool:
    """Register a queue using the global service."""
    return get_message_queue_service().register_queue(config)


def register_message_handler(message_type: MessageType, handler) -> None:
    """Register a message handler using the global service."""
    return get_message_queue_service().register_message_handler(message_type, handler)


# Default message handlers
async def handle_workflow_request(message: QueueMessage, context: MessageContext) -> bool:
    """Handle workflow request messages."""
    try:
        logger.info(f"Processing workflow request: {message.payload}")
        
        # Extract workflow details
        workflow_type = message.payload.get('workflow_type')
        workflow_data = message.payload.get('data', {})
        
        # Process based on workflow type
        if workflow_type == 'data_quality_assessment':
            # Trigger data quality agent
            logger.info("Triggering data quality assessment workflow")
            # Add your data quality processing logic here
            
        elif workflow_type == 'compliance_check':
            # Trigger compliance agent
            logger.info("Triggering compliance check workflow")
            # Add your compliance processing logic here
            
        elif workflow_type == 'remediation_task':
            # Trigger remediation agent
            logger.info("Triggering remediation task workflow")
            # Add your remediation processing logic here
            
        else:
            logger.warning(f"Unknown workflow type: {workflow_type}")
            return False
        
        # Send response if reply_to is specified
        if message.reply_to:
            response_message = QueueMessage(
                message_type=MessageType.WORKFLOW_RESPONSE,
                source="edgp_ai_model",
                destination=message.source,
                payload={
                    'status': 'completed',
                    'correlation_id': message.correlation_id,
                    'original_workflow_type': workflow_type,
                    'result': 'success'
                },
                correlation_id=message.correlation_id
            )
            
            await send_message(message.reply_to, response_message)
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling workflow request: {e}")
        return False


async def handle_workflow_response(message: QueueMessage, context: MessageContext) -> bool:
    """Handle workflow response messages."""
    try:
        logger.info(f"Processing workflow response: {message.payload}")
        
        # Process workflow completion
        status = message.payload.get('status')
        correlation_id = message.payload.get('correlation_id')
        
        logger.info(f"Workflow {correlation_id} completed with status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling workflow response: {e}")
        return False


async def handle_agent_communication(message: QueueMessage, context: MessageContext) -> bool:
    """Handle inter-agent communication messages."""
    try:
        logger.info(f"Processing agent communication: {message.payload}")
        
        # Route message to appropriate agent
        target_agent = message.payload.get('target_agent')
        agent_message = message.payload.get('message')
        
        logger.info(f"Routing message to agent: {target_agent}")
        
        # Add your agent routing logic here
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling agent communication: {e}")
        return False


# Global service instance
message_queue_service = None


def get_message_queue_service() -> MessageQueueService:
    """Get or create the global message queue service instance."""
    global message_queue_service
    if message_queue_service is None:
        message_queue_service = MessageQueueService()
    return message_queue_service


def register_queue(config: QueueConfig):
    """Module-level function to register a queue."""
    service = get_message_queue_service()
    service.register_queue(config)


def register_message_handler(message_type: MessageType, handler: Callable):
    """Module-level function to register a message handler."""
    service = get_message_queue_service()
    service.register_message_handler(message_type, handler)


def initialize_default_queues():
    """Initialize default queues based on configuration."""
    global message_queue_service
    settings = get_settings()
    
    # Always register at least default queues for testing
    # Register workflow request queue if configured or use default for testing
    if settings.workflow_rules_request_sqs_url:
        message_queue_service.register_queue(QueueConfig(
            name="workflow_request",
            url=settings.workflow_rules_request_sqs_url,
            message_type=MessageType.WORKFLOW_REQUEST
        ))
    else:
        # Default queue for testing
        message_queue_service.register_queue(QueueConfig(
            name="workflow_request",
            url="https://sqs.us-east-1.amazonaws.com/123456789012/workflow-request",
            message_type=MessageType.WORKFLOW_REQUEST
        ))
    
    # Register workflow response queue if configured or use default for testing
    if settings.workflow_rules_response_sqs_url:
        message_queue_service.register_queue(QueueConfig(
            name="workflow_response", 
            url=settings.workflow_rules_response_sqs_url,
            message_type=MessageType.WORKFLOW_RESPONSE
        ))
    else:
        # Default queue for testing
        message_queue_service.register_queue(QueueConfig(
            name="workflow_response", 
            url="https://sqs.us-east-1.amazonaws.com/123456789012/workflow-response",
            message_type=MessageType.WORKFLOW_RESPONSE
        ))
    
    # Register default message handlers
    message_queue_service.register_message_handler(MessageType.WORKFLOW_REQUEST, handle_workflow_request)
    message_queue_service.register_message_handler(MessageType.WORKFLOW_RESPONSE, handle_workflow_response)
    message_queue_service.register_message_handler(MessageType.AGENT_COMMUNICATION, handle_agent_communication)


async def start_message_queue_service():
    """Start the message queue service with default configuration."""
    initialize_default_queues()
    await get_message_queue_service().start_processing()


async def stop_message_queue_service():
    """Stop the message queue service."""
    await get_message_queue_service().stop_processing()
    await get_message_queue_service().stop_processing()
