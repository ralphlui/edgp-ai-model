"""
External Microservice Integration Patterns and Shared Functions
Provides consistent patterns for async, MQ, and API interactions with external services.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

from fastapi import HTTPException, status
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class IntegrationPattern(Enum):
    """Types of integration patterns."""
    SYNC_API = "sync_api"
    ASYNC_API = "async_api" 
    MESSAGE_QUEUE = "message_queue"
    WEBHOOK_CALLBACK = "webhook_callback"
    EVENT_STREAMING = "event_streaming"


class RequestCorrelationStatus(Enum):
    """Status of correlated requests."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class IntegrationRequest:
    """Standardized request for external service integration."""
    request_id: str
    service_name: str
    pattern: IntegrationPattern
    endpoint: str
    payload: Dict[str, Any]
    headers: Dict[str, str]
    callback_url: Optional[str] = None
    timeout_seconds: int = 300
    retry_count: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class IntegrationResponse:
    """Standardized response from external service integration."""
    request_id: str
    service_name: str
    status: RequestCorrelationStatus
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None
    completed_at: Optional[datetime] = None


class RequestCorrelationManager:
    """Manages correlation of async requests and responses."""
    
    def __init__(self):
        self.pending_requests: Dict[str, IntegrationRequest] = {}
        self.completed_responses: Dict[str, IntegrationResponse] = {}
        self.response_handlers: Dict[str, Callable] = {}
        self._cleanup_task = None
    
    async def start(self):
        """Start the correlation manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        logger.info("Request correlation manager started")
    
    async def stop(self):
        """Stop the correlation manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Request correlation manager stopped")
    
    def register_request(self, request: IntegrationRequest, response_handler: Optional[Callable] = None):
        """Register a new async request for correlation."""
        self.pending_requests[request.request_id] = request
        if response_handler:
            self.response_handlers[request.request_id] = response_handler
        logger.info(f"Registered request {request.request_id} for service {request.service_name}")
    
    async def handle_response(self, request_id: str, response_data: Dict[str, Any], status: RequestCorrelationStatus = RequestCorrelationStatus.COMPLETED):
        """Handle response for a correlated request."""
        if request_id not in self.pending_requests:
            logger.warning(f"Received response for unknown request: {request_id}")
            return
        
        request = self.pending_requests.pop(request_id)
        processing_time = int((datetime.utcnow() - request.created_at).total_seconds() * 1000)
        
        response = IntegrationResponse(
            request_id=request_id,
            service_name=request.service_name,
            status=status,
            response_data=response_data,
            processing_time_ms=processing_time,
            completed_at=datetime.utcnow()
        )
        
        self.completed_responses[request_id] = response
        
        # Call response handler if registered
        if request_id in self.response_handlers:
            try:
                handler = self.response_handlers.pop(request_id)
                await handler(response)
            except Exception as e:
                logger.error(f"Response handler failed for {request_id}: {e}")
        
        logger.info(f"Handled response for request {request_id}")
    
    def get_request_status(self, request_id: str) -> Optional[Union[RequestCorrelationStatus, IntegrationResponse]]:
        """Get status of a request."""
        if request_id in self.completed_responses:
            return self.completed_responses[request_id]
        elif request_id in self.pending_requests:
            return RequestCorrelationStatus.PENDING
        return None
    
    async def _cleanup_expired_requests(self):
        """Clean up expired requests periodically."""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_requests = []
                
                for request_id, request in self.pending_requests.items():
                    if current_time - request.created_at > timedelta(seconds=request.timeout_seconds):
                        expired_requests.append(request_id)
                
                for request_id in expired_requests:
                    request = self.pending_requests.pop(request_id)
                    response = IntegrationResponse(
                        request_id=request_id,
                        service_name=request.service_name,
                        status=RequestCorrelationStatus.TIMEOUT,
                        error_message=f"Request timed out after {request.timeout_seconds} seconds"
                    )
                    self.completed_responses[request_id] = response
                    logger.warning(f"Request {request_id} timed out")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)


class ExternalServiceAdapter(ABC):
    """Base adapter for external service integration."""
    
    def __init__(self, service_name: str, base_url: str, headers: Dict[str, str] = None):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.correlation_manager = RequestCorrelationManager()
    
    @abstractmethod
    async def make_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Make request to external service."""
        pass
    
    async def start(self):
        """Start the adapter."""
        await self.correlation_manager.start()
    
    async def stop(self):
        """Stop the adapter."""
        await self.correlation_manager.stop()


class SyncAPIAdapter(ExternalServiceAdapter):
    """Adapter for synchronous API calls."""
    
    async def make_request(self, request: IntegrationRequest) -> IntegrationResponse:
        """Make synchronous API request."""
        start_time = datetime.utcnow()
        
        try:
            async with aiohttp.ClientSession() as session:
                full_url = f"{self.base_url}{request.endpoint}"
                
                async with session.post(
                    full_url,
                    json=request.payload,
                    headers={**self.headers, **request.headers},
                    timeout=aiohttp.ClientTimeout(total=request.timeout_seconds)
                ) as response:
                    response_data = await response.json()
                    
                    if response.status >= 400:
                        return IntegrationResponse(
                            request_id=request.request_id,
                            service_name=request.service_name,
                            status=RequestCorrelationStatus.FAILED,
                            error_message=f"HTTP {response.status}: {response_data}",
                            processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                        )
                    
                    return IntegrationResponse(
                        request_id=request.request_id,
                        service_name=request.service_name,
                        status=RequestCorrelationStatus.COMPLETED,
                        response_data=response_data,
                        processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                        completed_at=datetime.utcnow()
                    )
                    
        except asyncio.TimeoutError:
            return IntegrationResponse(
                request_id=request.request_id,
                service_name=request.service_name,
                status=RequestCorrelationStatus.TIMEOUT,
                error_message="Request timed out",
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
        except Exception as e:
            return IntegrationResponse(
                request_id=request.request_id,
                service_name=request.service_name,
                status=RequestCorrelationStatus.FAILED,
                error_message=str(e),
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )


class AsyncAPIAdapter(ExternalServiceAdapter):
    """Adapter for asynchronous API calls with callback handling."""
    
    async def make_request(self, request: IntegrationRequest) -> str:
        """Make asynchronous API request, returns correlation ID."""
        start_time = datetime.utcnow()
        
        # Register for correlation tracking
        self.correlation_manager.register_request(request)
        
        try:
            async with aiohttp.ClientSession() as session:
                full_url = f"{self.base_url}{request.endpoint}"
                
                # Add callback URL to payload if provided
                payload = request.payload.copy()
                if request.callback_url:
                    payload['callback_url'] = request.callback_url
                    payload['correlation_id'] = request.request_id
                
                async with session.post(
                    full_url,
                    json=payload,
                    headers={**self.headers, **request.headers},
                    timeout=aiohttp.ClientTimeout(total=30)  # Short timeout for async initiation
                ) as response:
                    
                    if response.status >= 400:
                        await self.correlation_manager.handle_response(
                            request.request_id,
                            {"error": f"HTTP {response.status}"},
                            RequestCorrelationStatus.FAILED
                        )
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"External service request failed: {response.status}"
                        )
                    
                    # For async requests, we typically get an acknowledgment
                    ack_data = await response.json()
                    logger.info(f"Async request {request.request_id} initiated successfully")
                    
                    return request.request_id
                    
        except Exception as e:
            await self.correlation_manager.handle_response(
                request.request_id,
                {"error": str(e)},
                RequestCorrelationStatus.FAILED
            )
            raise
    
    async def handle_callback(self, correlation_id: str, callback_data: Dict[str, Any]):
        """Handle callback from external service."""
        await self.correlation_manager.handle_response(
            correlation_id,
            callback_data,
            RequestCorrelationStatus.COMPLETED
        )


class MessageQueueAdapter(ExternalServiceAdapter):
    """Adapter for message queue based communication."""
    
    def __init__(self, service_name: str, queue_manager, topic_manager):
        super().__init__(service_name, "", {})
        self.queue_manager = queue_manager
        self.topic_manager = topic_manager
    
    async def make_request(self, request: IntegrationRequest) -> str:
        """Send request via message queue."""
        # Register for correlation tracking
        self.correlation_manager.register_request(request)
        
        try:
            # Send via queue
            await self.queue_manager.send_message(
                request.service_name,
                {
                    "correlation_id": request.request_id,
                    "payload": request.payload,
                    "callback_topic": f"responses.{request.service_name}"
                }
            )
            
            logger.info(f"Message queue request {request.request_id} sent successfully")
            return request.request_id
            
        except Exception as e:
            await self.correlation_manager.handle_response(
                request.request_id,
                {"error": str(e)},
                RequestCorrelationStatus.FAILED
            )
            raise


class WebhookHandler:
    """Handles incoming webhooks from external services."""
    
    def __init__(self, correlation_manager: RequestCorrelationManager):
        self.correlation_manager = correlation_manager
        self.webhook_processors: Dict[str, Callable] = {}
    
    def register_processor(self, service_name: str, processor: Callable):
        """Register a webhook processor for a specific service."""
        self.webhook_processors[service_name] = processor
        logger.info(f"Registered webhook processor for {service_name}")
    
    async def process_webhook(self, service_name: str, webhook_data: Dict[str, Any]):
        """Process incoming webhook."""
        try:
            correlation_id = webhook_data.get('correlation_id')
            if not correlation_id:
                logger.warning(f"Webhook from {service_name} missing correlation_id")
                return
            
            # Apply service-specific processing if available
            if service_name in self.webhook_processors:
                processor = self.webhook_processors[service_name]
                processed_data = await processor(webhook_data)
            else:
                processed_data = webhook_data
            
            # Handle the correlated response
            await self.correlation_manager.handle_response(
                correlation_id,
                processed_data,
                RequestCorrelationStatus.COMPLETED
            )
            
            logger.info(f"Processed webhook for correlation {correlation_id}")
            
        except Exception as e:
            logger.error(f"Webhook processing failed: {e}")
            if correlation_id:
                await self.correlation_manager.handle_response(
                    correlation_id,
                    {"error": str(e)},
                    RequestCorrelationStatus.FAILED
                )


class IntegrationOrchestrator:
    """Orchestrates different integration patterns."""
    
    def __init__(self):
        self.adapters: Dict[str, ExternalServiceAdapter] = {}
        self.correlation_manager = RequestCorrelationManager()
        self.webhook_handler = WebhookHandler(self.correlation_manager)
        self.result_aggregator = ResultAggregator()
    
    async def start(self):
        """Start the integration orchestrator."""
        await self.correlation_manager.start()
        for adapter in self.adapters.values():
            await adapter.start()
        logger.info("Integration orchestrator started")
    
    async def stop(self):
        """Stop the integration orchestrator."""
        await self.correlation_manager.stop()
        for adapter in self.adapters.values():
            await adapter.stop()
        logger.info("Integration orchestrator stopped")
    
    def register_adapter(self, service_name: str, adapter: ExternalServiceAdapter):
        """Register an external service adapter."""
        self.adapters[service_name] = adapter
        logger.info(f"Registered adapter for {service_name}")
    
    async def make_request(
        self,
        service_name: str,
        pattern: IntegrationPattern,
        endpoint: str,
        payload: Dict[str, Any],
        **kwargs
    ) -> Union[IntegrationResponse, str]:
        """Make request using specified integration pattern."""
        request_id = kwargs.get('request_id') or str(uuid.uuid4())
        
        request = IntegrationRequest(
            request_id=request_id,
            service_name=service_name,
            pattern=pattern,
            endpoint=endpoint,
            payload=payload,
            headers=kwargs.get('headers', {}),
            callback_url=kwargs.get('callback_url'),
            timeout_seconds=kwargs.get('timeout', 300),
            retry_count=kwargs.get('retry_count', 3)
        )
        
        if service_name not in self.adapters:
            raise ValueError(f"No adapter registered for service: {service_name}")
        
        adapter = self.adapters[service_name]
        
        if pattern == IntegrationPattern.SYNC_API:
            return await adapter.make_request(request)
        else:
            # For async patterns, return correlation ID
            return await adapter.make_request(request)
    
    async def get_response(self, request_id: str, wait_timeout: int = 300) -> Optional[IntegrationResponse]:
        """Wait for and retrieve response by correlation ID."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < wait_timeout:
            response = self.correlation_manager.get_request_status(request_id)
            
            if isinstance(response, IntegrationResponse):
                return response
            
            await asyncio.sleep(1)  # Poll every second
        
        # Timeout
        return IntegrationResponse(
            request_id=request_id,
            service_name="unknown",
            status=RequestCorrelationStatus.TIMEOUT,
            error_message=f"Response timeout after {wait_timeout} seconds"
        )


class ResultAggregator:
    """Aggregates results from multiple external services."""
    
    def __init__(self):
        self.aggregation_tasks: Dict[str, Dict] = {}
    
    def create_aggregation_task(
        self,
        task_id: str,
        service_requests: List[str],
        completion_callback: Optional[Callable] = None
    ):
        """Create a new result aggregation task."""
        self.aggregation_tasks[task_id] = {
            'service_requests': set(service_requests),
            'completed_responses': {},
            'completion_callback': completion_callback,
            'created_at': datetime.utcnow()
        }
        logger.info(f"Created aggregation task {task_id} for {len(service_requests)} services")
    
    async def handle_response(self, task_id: str, request_id: str, response: IntegrationResponse):
        """Handle response for aggregation task."""
        if task_id not in self.aggregation_tasks:
            logger.warning(f"Received response for unknown aggregation task: {task_id}")
            return
        
        task = self.aggregation_tasks[task_id]
        
        if request_id in task['service_requests']:
            task['completed_responses'][request_id] = response
            
            # Check if all responses received
            if len(task['completed_responses']) == len(task['service_requests']):
                # All responses received, call completion callback
                if task['completion_callback']:
                    try:
                        await task['completion_callback'](task['completed_responses'])
                    except Exception as e:
                        logger.error(f"Aggregation completion callback failed: {e}")
                
                # Clean up task
                self.aggregation_tasks.pop(task_id)
                logger.info(f"Aggregation task {task_id} completed")
    
    def get_aggregation_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of aggregation task."""
        if task_id not in self.aggregation_tasks:
            return None
        
        task = self.aggregation_tasks[task_id]
        return {
            'task_id': task_id,
            'total_requests': len(task['service_requests']),
            'completed_responses': len(task['completed_responses']),
            'pending_requests': list(task['service_requests'] - set(task['completed_responses'].keys())),
            'created_at': task['created_at'].isoformat()
        }


class SharedIntegrationFunctions:
    """Shared utility functions for external service integration."""
    
    @staticmethod
    def create_correlation_id() -> str:
        """Create a unique correlation ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def standardize_payload(data: Dict[str, Any], service_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Standardize payload format for external service."""
        standardized = {
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0',
            'data': data
        }
        
        if service_schema:
            # Apply schema transformations if needed
            standardized['schema_version'] = service_schema.get('version', '1.0')
        
        return standardized
    
    @staticmethod
    def create_callback_url(base_url: str, service_name: str, correlation_id: str) -> str:
        """Create callback URL for async operations."""
        return f"{base_url}/webhooks/{service_name}/callback/{correlation_id}"
    
    @staticmethod
    def extract_error_details(response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize error details from response."""
        return {
            'error_code': response_data.get('error_code', 'UNKNOWN'),
            'error_message': response_data.get('message', response_data.get('error', 'Unknown error')),
            'error_details': response_data.get('details', {}),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    async def retry_with_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0
    ) -> Any:
        """Retry function with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)


class ExternalServiceRegistry:
    """Registry for external service configurations and adapters."""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.adapters: Dict[str, ExternalServiceAdapter] = {}
    
    def register_service(
        self,
        service_name: str,
        base_url: str,
        pattern: IntegrationPattern,
        config: Dict[str, Any] = None
    ):
        """Register an external service."""
        self.services[service_name] = {
            'base_url': base_url,
            'pattern': pattern,
            'config': config or {},
            'registered_at': datetime.utcnow()
        }
        
        # Create appropriate adapter
        if pattern == IntegrationPattern.SYNC_API:
            adapter = SyncAPIAdapter(service_name, base_url, config.get('headers', {}))
        elif pattern == IntegrationPattern.ASYNC_API:
            adapter = AsyncAPIAdapter(service_name, base_url, config.get('headers', {}))
        elif pattern == IntegrationPattern.MESSAGE_QUEUE:
            # Would use existing queue manager
            adapter = None  # Handled by existing external_communication.py
        else:
            adapter = SyncAPIAdapter(service_name, base_url, config.get('headers', {}))
        
        if adapter:
            self.adapters[service_name] = adapter
        
        logger.info(f"Registered service {service_name} with pattern {pattern.value}")
    
    def get_service_config(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a service."""
        return self.services.get(service_name)
    
    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self.services.keys())
    
    async def start_all_adapters(self):
        """Start all registered adapters."""
        for adapter in self.adapters.values():
            await adapter.start()
    
    async def stop_all_adapters(self):
        """Stop all registered adapters."""
        for adapter in self.adapters.values():
            await adapter.stop()


# Global instances for shared usage
integration_orchestrator = IntegrationOrchestrator()
service_registry = ExternalServiceRegistry()
shared_functions = SharedIntegrationFunctions()
