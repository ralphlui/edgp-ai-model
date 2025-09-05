"""
AWS SQS/SNS based external microservice communication system.
"""

import json
import boto3
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from botocore.exceptions import ClientError, BotoCoreError

from ..infrastructure.config import get_settings

logger = logging.getLogger(__name__)


class MessagePriority(str, Enum):
    """Message priority levels for external communication."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ExternalServiceType(str, Enum):
    """External service types for routing."""
    DATA_INGESTION = "data_ingestion"
    NOTIFICATION_SERVICE = "notification_service"
    AUDIT_SERVICE = "audit_service"
    REPORTING_SERVICE = "reporting_service"
    AUTHENTICATION_SERVICE = "authentication_service"
    STORAGE_SERVICE = "storage_service"
    ANALYTICS_PLATFORM = "analytics_platform"


@dataclass
class ExternalMessage:
    """External message structure for microservice communication."""
    message_id: str
    service_type: ExternalServiceType
    action: str
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: str
    correlation_id: Optional[str] = None
    reply_queue: Optional[str] = None
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExternalMessage':
        return cls(**data)


class SQSMessageProducer:
    """AWS SQS message producer for external communication."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.settings = get_settings()
        self.region_name = region_name
        self.sqs_client = boto3.client(
            'sqs',
            region_name=region_name,
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key
        )
        self.queue_urls: Dict[str, str] = {}
    
    async def initialize_queues(self):
        """Initialize SQS queues for different service types."""
        queue_configs = {
            ExternalServiceType.DATA_INGESTION: {
                'queue_name': 'edgp-data-ingestion',
                'visibility_timeout': 300,
                'message_retention_period': 1209600  # 14 days
            },
            ExternalServiceType.NOTIFICATION_SERVICE: {
                'queue_name': 'edgp-notifications',
                'visibility_timeout': 60,
                'message_retention_period': 345600  # 4 days
            },
            ExternalServiceType.AUDIT_SERVICE: {
                'queue_name': 'edgp-audit-logs',
                'visibility_timeout': 120,
                'message_retention_period': 2592000  # 30 days
            },
            ExternalServiceType.REPORTING_SERVICE: {
                'queue_name': 'edgp-reports',
                'visibility_timeout': 600,
                'message_retention_period': 604800  # 7 days
            }
        }
        
        for service_type, config in queue_configs.items():
            try:
                queue_url = await self._create_or_get_queue(
                    config['queue_name'],
                    config['visibility_timeout'],
                    config['message_retention_period']
                )
                self.queue_urls[service_type] = queue_url
                logger.info(f"Initialized queue for {service_type}: {queue_url}")
                
            except Exception as e:
                logger.error(f"Failed to initialize queue for {service_type}: {e}")
    
    async def _create_or_get_queue(self, queue_name: str, visibility_timeout: int, retention_period: int) -> str:
        """Create or get existing SQS queue."""
        try:
            # Try to get existing queue
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            return response['QueueUrl']
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'AWS.SimpleQueueService.NonExistentQueue':
                # Create new queue
                response = self.sqs_client.create_queue(
                    QueueName=queue_name,
                    Attributes={
                        'VisibilityTimeoutSeconds': str(visibility_timeout),
                        'MessageRetentionPeriod': str(retention_period),
                        'ReceiveMessageWaitTimeSeconds': '20',  # Long polling
                        'DelaySeconds': '0'
                    }
                )
                return response['QueueUrl']
            else:
                raise
    
    async def send_message(self, message: ExternalMessage) -> str:
        """Send message to appropriate SQS queue."""
        if message.service_type not in self.queue_urls:
            raise ValueError(f"No queue configured for service type: {message.service_type}")
        
        queue_url = self.queue_urls[message.service_type]
        
        # Prepare message attributes
        message_attributes = {
            'Priority': {
                'StringValue': message.priority.value,
                'DataType': 'String'
            },
            'ServiceType': {
                'StringValue': message.service_type.value,
                'DataType': 'String'
            },
            'Action': {
                'StringValue': message.action,
                'DataType': 'String'
            }
        }
        
        if message.correlation_id:
            message_attributes['CorrelationId'] = {
                'StringValue': message.correlation_id,
                'DataType': 'String'
            }
        
        try:
            # Send message
            response = self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message.to_dict()),
                MessageAttributes=message_attributes,
                DelaySeconds=0 if message.priority in [MessagePriority.HIGH, MessagePriority.CRITICAL] else 5
            )
            
            logger.info(f"Sent message {message.message_id} to {message.service_type}")
            return response['MessageId']
            
        except ClientError as e:
            logger.error(f"Failed to send message {message.message_id}: {e}")
            raise
    
    async def send_batch_messages(self, messages: List[ExternalMessage]) -> Dict[str, Any]:
        """Send multiple messages in batches for efficiency."""
        results = {
            'successful': [],
            'failed': []
        }
        
        # Group messages by service type
        grouped_messages = {}
        for message in messages:
            if message.service_type not in grouped_messages:
                grouped_messages[message.service_type] = []
            grouped_messages[message.service_type].append(message)
        
        # Send batches for each service type
        for service_type, service_messages in grouped_messages.items():
            if service_type not in self.queue_urls:
                for msg in service_messages:
                    results['failed'].append({
                        'message_id': msg.message_id,
                        'error': f'No queue for service type: {service_type}'
                    })
                continue
            
            queue_url = self.queue_urls[service_type]
            
            # Process in batches of 10 (SQS limit)
            for i in range(0, len(service_messages), 10):
                batch = service_messages[i:i + 10]
                
                try:
                    # Prepare batch entries
                    entries = []
                    for msg in batch:
                        entries.append({
                            'Id': msg.message_id,
                            'MessageBody': json.dumps(msg.to_dict()),
                            'MessageAttributes': {
                                'Priority': {
                                    'StringValue': msg.priority.value,
                                    'DataType': 'String'
                                },
                                'ServiceType': {
                                    'StringValue': msg.service_type.value,
                                    'DataType': 'String'
                                },
                                'Action': {
                                    'StringValue': msg.action,
                                    'DataType': 'String'
                                }
                            }
                        })
                    
                    # Send batch
                    response = self.sqs_client.send_message_batch(
                        QueueUrl=queue_url,
                        Entries=entries
                    )
                    
                    # Process results
                    for success in response.get('Successful', []):
                        results['successful'].append({
                            'message_id': success['Id'],
                            'sqs_message_id': success['MessageId']
                        })
                    
                    for failure in response.get('Failed', []):
                        results['failed'].append({
                            'message_id': failure['Id'],
                            'error': failure['Message']
                        })
                        
                except ClientError as e:
                    logger.error(f"Batch send failed for {service_type}: {e}")
                    for msg in batch:
                        results['failed'].append({
                            'message_id': msg.message_id,
                            'error': str(e)
                        })
        
        return results


class SNSMessageBroadcaster:
    """AWS SNS message broadcaster for event notifications."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.settings = get_settings()
        self.region_name = region_name
        self.sns_client = boto3.client(
            'sns',
            region_name=region_name,
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key
        )
        self.topic_arns: Dict[str, str] = {}
    
    async def initialize_topics(self):
        """Initialize SNS topics for event broadcasting."""
        topic_configs = {
            'data_quality_events': {
                'name': 'edgp-data-quality-events',
                'display_name': 'EDGP Data Quality Events'
            },
            'compliance_events': {
                'name': 'edgp-compliance-events',
                'display_name': 'EDGP Compliance Events'
            },
            'system_events': {
                'name': 'edgp-system-events',
                'display_name': 'EDGP System Events'
            },
            'alert_events': {
                'name': 'edgp-alert-events',
                'display_name': 'EDGP Alert Events'
            }
        }
        
        for topic_key, config in topic_configs.items():
            try:
                topic_arn = await self._create_or_get_topic(
                    config['name'],
                    config['display_name']
                )
                self.topic_arns[topic_key] = topic_arn
                logger.info(f"Initialized SNS topic {topic_key}: {topic_arn}")
                
            except Exception as e:
                logger.error(f"Failed to initialize topic {topic_key}: {e}")
    
    async def _create_or_get_topic(self, topic_name: str, display_name: str) -> str:
        """Create or get existing SNS topic."""
        try:
            # Try to create topic (idempotent operation)
            response = self.sns_client.create_topic(
                Name=topic_name,
                Attributes={
                    'DisplayName': display_name,
                    'DeliveryPolicy': json.dumps({
                        'http': {
                            'defaultHealthyRetryPolicy': {
                                'minDelayTarget': 20,
                                'maxDelayTarget': 20,
                                'numRetries': 3,
                                'numMaxDelayRetries': 0,
                                'numMinDelayRetries': 0,
                                'numNoDelayRetries': 0,
                                'backoffFunction': 'linear'
                            }
                        }
                    })
                }
            )
            return response['TopicArn']
            
        except ClientError as e:
            logger.error(f"Failed to create/get topic {topic_name}: {e}")
            raise
    
    async def publish_event(
        self, 
        topic_key: str, 
        subject: str, 
        message: Dict[str, Any],
        message_attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """Publish event to SNS topic."""
        if topic_key not in self.topic_arns:
            raise ValueError(f"Topic {topic_key} not configured")
        
        topic_arn = self.topic_arns[topic_key]
        
        # Prepare message attributes
        sns_attributes = {}
        if message_attributes:
            for key, value in message_attributes.items():
                sns_attributes[key] = {
                    'DataType': 'String',
                    'StringValue': str(value)
                }
        
        try:
            response = self.sns_client.publish(
                TopicArn=topic_arn,
                Subject=subject,
                Message=json.dumps(message, indent=2),
                MessageAttributes=sns_attributes
            )
            
            logger.info(f"Published event to {topic_key}: {subject}")
            return response['MessageId']
            
        except ClientError as e:
            logger.error(f"Failed to publish event to {topic_key}: {e}")
            raise


class ExternalCommunicationManager:
    """Manager for external microservice communication."""
    
    def __init__(self):
        self.sqs_producer = SQSMessageProducer()
        self.sns_broadcaster = SNSMessageBroadcaster()
        self.message_handlers: Dict[str, Callable] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize all external communication components."""
        try:
            await self.sqs_producer.initialize_queues()
            await self.sns_broadcaster.initialize_topics()
            self.initialized = True
            logger.info("External communication manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize external communication: {e}")
            raise
    
    async def send_to_service(
        self, 
        service_type: ExternalServiceType, 
        action: str, 
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> str:
        """Send message to external service."""
        if not self.initialized:
            await self.initialize()
        
        message = ExternalMessage(
            message_id=str(uuid.uuid4()),
            service_type=service_type,
            action=action,
            payload=payload,
            priority=priority,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id
        )
        
        return await self.sqs_producer.send_message(message)
    
    async def broadcast_event(
        self, 
        event_type: str, 
        subject: str, 
        data: Dict[str, Any],
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """Broadcast event via SNS."""
        if not self.initialized:
            await self.initialize()
        
        # Map event types to topics
        topic_mapping = {
            'data_quality': 'data_quality_events',
            'compliance': 'compliance_events',
            'system': 'system_events',
            'alert': 'alert_events'
        }
        
        topic_key = topic_mapping.get(event_type, 'system_events')
        
        event_message = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        return await self.sns_broadcaster.publish_event(
            topic_key, subject, event_message, attributes
        )
    
    async def send_data_quality_results(self, results: Dict[str, Any]) -> str:
        """Send data quality results to external analytics platform."""
        return await self.send_to_service(
            service_type=ExternalServiceType.ANALYTICS_PLATFORM,
            action="ingest_quality_metrics",
            payload={
                "metric_type": "data_quality",
                "results": results,
                "source": "edgp_ai_model"
            },
            priority=MessagePriority.NORMAL
        )
    
    async def send_compliance_alert(self, alert_data: Dict[str, Any]) -> str:
        """Send compliance alert to notification service."""
        return await self.send_to_service(
            service_type=ExternalServiceType.NOTIFICATION_SERVICE,
            action="send_compliance_alert",
            payload=alert_data,
            priority=MessagePriority.HIGH
        )
    
    async def log_audit_event(self, event_data: Dict[str, Any]) -> str:
        """Log audit event to external audit service."""
        return await self.send_to_service(
            service_type=ExternalServiceType.AUDIT_SERVICE,
            action="log_event",
            payload=event_data,
            priority=MessagePriority.NORMAL
        )
    
    async def request_data_ingestion(self, ingestion_config: Dict[str, Any]) -> str:
        """Request data ingestion from external service."""
        return await self.send_to_service(
            service_type=ExternalServiceType.DATA_INGESTION,
            action="ingest_data",
            payload=ingestion_config,
            priority=MessagePriority.NORMAL
        )


# Global external communication manager
external_comm = ExternalCommunicationManager()
