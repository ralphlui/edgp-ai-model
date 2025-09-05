"""
External Service Endpoints and Webhook Handlers
Provides standardized endpoints for external microservice interactions.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Request
from pydantic import BaseModel

from .patterns import (
    integration_orchestrator,
    service_registry,
    IntegrationPattern,
    IntegrationRequest,
    IntegrationResponse,
    RequestCorrelationStatus,
    SharedIntegrationFunctions
)
from ..services.llm_bridge import llm_bridge, cross_agent_bridge

logger = logging.getLogger(__name__)

# Create router for external service endpoints
external_router = APIRouter(prefix="/external", tags=["external-services"])


class ExternalServiceRequest(BaseModel):
    """Request model for external service calls."""
    service_name: str
    endpoint: str
    payload: Dict[str, Any]
    pattern: str = "sync_api"
    headers: Dict[str, str] = {}
    timeout: int = 300
    callback_url: Optional[str] = None


class ExternalServiceResponse(BaseModel):
    """Response model for external service calls."""
    request_id: str
    service_name: str
    status: str
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None


class WebhookData(BaseModel):
    """Model for incoming webhook data."""
    service_name: str
    correlation_id: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None


class ServiceRegistrationRequest(BaseModel):
    """Model for registering external services."""
    service_name: str
    base_url: str
    pattern: str
    config: Dict[str, Any] = {}


@external_router.post("/services/register")
async def register_external_service(request: ServiceRegistrationRequest):
    """Register a new external service for integration."""
    try:
        pattern = IntegrationPattern(request.pattern)
        service_registry.register_service(
            request.service_name,
            request.base_url,
            pattern,
            request.config
        )
        
        return {
            "status": "registered",
            "service_name": request.service_name,
            "pattern": request.pattern,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Service registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@external_router.get("/services")
async def list_external_services():
    """List all registered external services."""
    try:
        services = service_registry.list_services()
        service_details = []
        
        for service_name in services:
            config = service_registry.get_service_config(service_name)
            service_details.append({
                "name": service_name,
                "base_url": config.get('base_url'),
                "pattern": config.get('pattern').value if config.get('pattern') else None,
                "registered_at": config.get('registered_at').isoformat() if config.get('registered_at') else None
            })
        
        return {"services": service_details}
        
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@external_router.post("/request", response_model=ExternalServiceResponse)
async def make_external_request(request: ExternalServiceRequest):
    """Make request to external service using specified pattern."""
    try:
        pattern = IntegrationPattern(request.pattern)
        request_id = SharedIntegrationFunctions.create_correlation_id()
        
        result = await integration_orchestrator.make_request(
            service_name=request.service_name,
            pattern=pattern,
            endpoint=request.endpoint,
            payload=request.payload,
            request_id=request_id,
            headers=request.headers,
            timeout=request.timeout,
            callback_url=request.callback_url
        )
        
        if isinstance(result, IntegrationResponse):
            # Synchronous response
            return ExternalServiceResponse(
                request_id=result.request_id,
                service_name=result.service_name,
                status=result.status.value,
                response_data=result.response_data,
                error_message=result.error_message,
                processing_time_ms=result.processing_time_ms
            )
        else:
            # Asynchronous response (correlation ID returned)
            return ExternalServiceResponse(
                request_id=result,
                service_name=request.service_name,
                status=RequestCorrelationStatus.PENDING.value,
                response_data={"correlation_id": result}
            )
            
    except Exception as e:
        logger.error(f"External request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@external_router.get("/request/{request_id}/status", response_model=ExternalServiceResponse)
async def get_request_status(request_id: str):
    """Get status of an external request by correlation ID."""
    try:
        status_info = integration_orchestrator.correlation_manager.get_request_status(request_id)
        
        if status_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Request {request_id} not found"
            )
        
        if isinstance(status_info, IntegrationResponse):
            return ExternalServiceResponse(
                request_id=status_info.request_id,
                service_name=status_info.service_name,
                status=status_info.status.value,
                response_data=status_info.response_data,
                error_message=status_info.error_message,
                processing_time_ms=status_info.processing_time_ms
            )
        else:
            # Still pending
            return ExternalServiceResponse(
                request_id=request_id,
                service_name="unknown",
                status=status_info.value,
                response_data=None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get request status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@external_router.post("/webhooks/{service_name}/callback/{correlation_id}")
async def handle_webhook_callback(
    service_name: str,
    correlation_id: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle webhook callbacks from external services."""
    try:
        webhook_data = await request.json()
        
        # Process webhook in background
        background_tasks.add_task(
            _process_webhook_callback,
            service_name,
            correlation_id,
            webhook_data
        )
        
        return {
            "status": "callback_received",
            "correlation_id": correlation_id,
            "service_name": service_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Webhook callback handling failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


async def _process_webhook_callback(service_name: str, correlation_id: str, webhook_data: Dict[str, Any]):
    """Process webhook callback in background."""
    try:
        await integration_orchestrator.webhook_handler.process_webhook(
            service_name,
            {**webhook_data, 'correlation_id': correlation_id}
        )
        logger.info(f"Processed webhook callback for {correlation_id}")
        
    except Exception as e:
        logger.error(f"Background webhook processing failed: {e}")


@external_router.post("/webhooks/generic")
async def handle_generic_webhook(webhook: WebhookData, background_tasks: BackgroundTasks):
    """Handle generic webhooks with correlation ID."""
    try:
        background_tasks.add_task(
            _process_webhook_callback,
            webhook.service_name,
            webhook.correlation_id,
            webhook.data
        )
        
        return {
            "status": "webhook_processed",
            "correlation_id": webhook.correlation_id,
            "service_name": webhook.service_name
        }
        
    except Exception as e:
        logger.error(f"Generic webhook handling failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@external_router.post("/batch-request")
async def make_batch_external_requests(
    requests: List[ExternalServiceRequest],
    aggregation_id: Optional[str] = None
):
    """Make multiple external requests and optionally aggregate results."""
    try:
        request_ids = []
        
        for req in requests:
            pattern = IntegrationPattern(req.pattern)
            request_id = SharedIntegrationFunctions.create_correlation_id()
            
            result = await integration_orchestrator.make_request(
                service_name=req.service_name,
                pattern=pattern,
                endpoint=req.endpoint,
                payload=req.payload,
                request_id=request_id,
                headers=req.headers,
                timeout=req.timeout,
                callback_url=req.callback_url
            )
            
            if isinstance(result, str):
                request_ids.append(result)
            else:
                request_ids.append(result.request_id)
        
        # Set up aggregation if requested
        if aggregation_id:
            integration_orchestrator.result_aggregator.create_aggregation_task(
                aggregation_id,
                request_ids
            )
        
        return {
            "batch_id": aggregation_id or SharedIntegrationFunctions.create_correlation_id(),
            "request_ids": request_ids,
            "total_requests": len(request_ids),
            "aggregation_enabled": bool(aggregation_id)
        }
        
    except Exception as e:
        logger.error(f"Batch request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@external_router.get("/aggregation/{task_id}/status")
async def get_aggregation_status(task_id: str):
    """Get status of result aggregation task."""
    try:
        status_info = integration_orchestrator.result_aggregator.get_aggregation_status(task_id)
        
        if status_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Aggregation task {task_id} not found"
            )
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get aggregation status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@external_router.get("/metrics")
async def get_integration_metrics():
    """Get integration performance metrics."""
    try:
        bridge_metrics = llm_bridge.get_metrics()
        
        return {
            "llm_bridge_metrics": bridge_metrics,
            "pending_requests": len(integration_orchestrator.correlation_manager.pending_requests),
            "completed_responses": len(integration_orchestrator.correlation_manager.completed_responses),
            "registered_services": len(service_registry.services),
            "aggregation_tasks": len(integration_orchestrator.result_aggregator.aggregation_tasks)
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# LLM Gateway interaction endpoints
llm_router = APIRouter(prefix="/llm", tags=["llm-gateway"])


class LLMGatewayRequest(BaseModel):
    """Request model for LLM gateway interactions."""
    agent_id: str
    prompt: str
    template: Optional[str] = None
    response_format: str = "json"
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = False
    metadata: Dict[str, Any] = {}


@llm_router.post("/generate")
async def generate_llm_response(request: LLMGatewayRequest):
    """Generate LLM response for agent through bridge."""
    try:
        # Get or create agent interface
        agent_interface = cross_agent_bridge.register_agent(request.agent_id)
        
        # Convert template string to enum if provided
        template = None
        if request.template:
            from ..services.llm_bridge import PromptTemplate, ResponseFormat
            try:
                template = PromptTemplate(request.template.lower())
            except ValueError:
                logger.warning(f"Unknown template: {request.template}")
        
        response_format = ResponseFormat(request.response_format.lower())
        
        # Generate response
        bridge_response = await agent_interface.generate_response(
            prompt=request.prompt,
            template=template,
            response_format=response_format,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            streaming=request.streaming,
            metadata=request.metadata
        )
        
        return {
            "request_id": bridge_response.request_id,
            "agent_id": bridge_response.agent_id,
            "content": bridge_response.content,
            "format": bridge_response.format.value,
            "success": bridge_response.success,
            "error": bridge_response.error,
            "metadata": bridge_response.metadata,
            "processing_time_ms": bridge_response.processing_time_ms
        }
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.post("/analyze")
async def analyze_data_via_llm(
    agent_id: str,
    data: Dict[str, Any],
    analysis_type: str,
    context: Optional[Dict[str, Any]] = None
):
    """Analyze data using LLM through bridge."""
    try:
        agent_interface = cross_agent_bridge.register_agent(agent_id)
        
        response = await agent_interface.analyze_data(
            data=data,
            analysis_type=analysis_type,
            context=context
        )
        
        return {
            "request_id": response.request_id,
            "agent_id": response.agent_id,
            "analysis_results": response.content,
            "analysis_type": analysis_type,
            "success": response.success,
            "error": response.error,
            "processing_time_ms": response.processing_time_ms
        }
        
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.post("/collaborate")
async def facilitate_agent_collaboration(
    initiating_agent: str,
    target_agent: str,
    collaboration_prompt: str,
    context_data: Dict[str, Any] = {}
):
    """Facilitate collaboration between agents via LLM bridge."""
    try:
        results = await cross_agent_bridge.facilitate_agent_collaboration(
            initiating_agent=initiating_agent,
            target_agent=target_agent,
            collaboration_prompt=collaboration_prompt,
            context_data=context_data
        )
        
        return {
            "collaboration_id": SharedIntegrationFunctions.create_correlation_id(),
            "initiating_agent": initiating_agent,
            "target_agent": target_agent,
            "results": {
                "initiating_agent_response": {
                    "content": results['initiating_agent_response'].content,
                    "success": results['initiating_agent_response'].success,
                    "processing_time_ms": results['initiating_agent_response'].processing_time_ms
                },
                "target_agent_response": {
                    "content": results['target_agent_response'].content,
                    "success": results['target_agent_response'].success,
                    "processing_time_ms": results['target_agent_response'].processing_time_ms
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Agent collaboration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.get("/metrics")
async def get_llm_bridge_metrics():
    """Get LLM bridge performance metrics."""
    try:
        return llm_bridge.get_metrics()
        
    except Exception as e:
        logger.error(f"Failed to get LLM bridge metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@llm_router.post("/cache/clear")
async def clear_llm_cache():
    """Clear LLM bridge cache."""
    try:
        llm_bridge.clear_cache()
        return {"status": "cache_cleared", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Webhook router
webhook_router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@webhook_router.post("/{service_name}/callback/{correlation_id}")
async def service_webhook_callback(
    service_name: str,
    correlation_id: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Handle webhook callback from specific external service."""
    try:
        webhook_data = await request.json()
        
        background_tasks.add_task(
            integration_orchestrator.webhook_handler.process_webhook,
            service_name,
            {**webhook_data, 'correlation_id': correlation_id}
        )
        
        return {
            "status": "callback_received",
            "service_name": service_name,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Service webhook callback failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@webhook_router.post("/generic")
async def generic_webhook_handler(webhook: WebhookData, background_tasks: BackgroundTasks):
    """Handle generic webhook with service identification."""
    try:
        background_tasks.add_task(
            integration_orchestrator.webhook_handler.process_webhook,
            webhook.service_name,
            {**webhook.data, 'correlation_id': webhook.correlation_id}
        )
        
        return {
            "status": "webhook_processed",
            "service_name": webhook.service_name,
            "correlation_id": webhook.correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Generic webhook handling failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# Integration utilities router
integration_router = APIRouter(prefix="/integration", tags=["integration-utils"])


@integration_router.post("/correlation/create")
async def create_correlation_id():
    """Create a new correlation ID for tracking."""
    return {
        "correlation_id": SharedIntegrationFunctions.create_correlation_id(),
        "timestamp": datetime.utcnow().isoformat()
    }


@integration_router.post("/payload/standardize")
async def standardize_payload(
    data: Dict[str, Any],
    service_schema: Optional[Dict[str, Any]] = None
):
    """Standardize payload for external service."""
    try:
        standardized = SharedIntegrationFunctions.standardize_payload(data, service_schema)
        
        return {
            "standardized_payload": standardized,
            "original_size": len(str(data)),
            "standardized_size": len(str(standardized))
        }
        
    except Exception as e:
        logger.error(f"Payload standardization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@integration_router.get("/patterns")
async def list_integration_patterns():
    """List available integration patterns."""
    return {
        "patterns": [
            {
                "name": pattern.value,
                "description": _get_pattern_description(pattern)
            }
            for pattern in IntegrationPattern
        ]
    }


def _get_pattern_description(pattern: IntegrationPattern) -> str:
    """Get description for integration pattern."""
    descriptions = {
        IntegrationPattern.SYNC_API: "Synchronous API calls with immediate response",
        IntegrationPattern.ASYNC_API: "Asynchronous API calls with callback handling",
        IntegrationPattern.MESSAGE_QUEUE: "Message queue based communication",
        IntegrationPattern.WEBHOOK_CALLBACK: "Webhook-based event handling",
        IntegrationPattern.EVENT_STREAMING: "Real-time event streaming"
    }
    return descriptions.get(pattern, "Unknown pattern")


# Export routers for main app integration
__all__ = [
    'external_router',
    'webhook_router', 
    'llm_router',
    'integration_router'
]
