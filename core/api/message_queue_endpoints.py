"""
Message Queue API Endpoints

REST API endpoints for managing message queues and sending messages.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from core.services.message_queue import (
    get_message_queue_service,
    QueueMessage, 
    MessageType,
    QueueConfig
)
from core.types.responses import StandardResponse

router = APIRouter(prefix="/message-queue", tags=["message-queue"])


class SendMessageRequest(BaseModel):
    """Request to send a message to a queue."""
    queue_name: str
    message_type: MessageType
    source: str
    destination: Optional[str] = None
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    delay_seconds: int = 0


class RegisterQueueRequest(BaseModel):
    """Request to register a new queue."""
    name: str
    url: str
    message_type: MessageType
    visibility_timeout: int = 30
    wait_time_seconds: int = 20
    max_messages: int = 10


class MessageQueueStats(BaseModel):
    """Message queue statistics."""
    queue_name: str
    url: str
    message_type: str
    visible_messages: int
    in_flight_messages: int
    delayed_messages: int


class QueueStatusResponse(StandardResponse):
    """Response for queue status."""
    queues: List[Dict[str, Any]]
    total_queues: int
    processing_status: str


class SendMessageResponse(StandardResponse):
    """Response for sending a message."""
    message_id: str
    queue_name: str
    sent_at: datetime


@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get status of all registered queues."""
    try:
        mq_service = get_message_queue_service()
        stats = mq_service.get_queue_stats()
        
        queue_list = []
        for queue_name, queue_stats in stats.items():
            if 'error' not in queue_stats:
                queue_list.append({
                    'name': queue_name,
                    'url': queue_stats.get('url', ''),
                    'message_type': queue_stats.get('message_type', ''),
                    'visible_messages': queue_stats.get('visible_messages', 0),
                    'in_flight_messages': queue_stats.get('in_flight_messages', 0),
                    'delayed_messages': queue_stats.get('delayed_messages', 0)
                })
            else:
                queue_list.append({
                    'name': queue_name,
                    'error': queue_stats['error']
                })
        
        return QueueStatusResponse(
            success=True,
            message="Queue status retrieved successfully",
            queues=queue_list,
            total_queues=len(queue_list),
            processing_status="running" if mq_service.is_running else "stopped"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )


@router.post("/register-queue", response_model=StandardResponse)
async def register_queue(request: RegisterQueueRequest):
    """Register a new queue for processing."""
    try:
        mq_service = get_message_queue_service()
        mq_service.register_queue(
            name=request.name,
            url=request.url,
            message_type=request.message_type,
            visibility_timeout=request.visibility_timeout,
            wait_time_seconds=request.wait_time_seconds,
            max_messages=request.max_messages
        )
        
        return StandardResponse(
            success=True,
            message=f"Queue '{request.name}' registered successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register queue: {str(e)}"
        )


@router.post("/send-message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """Send a message to a specific queue."""
    try:
        # Create queue message
        message = QueueMessage(
            message_type=request.message_type,
            source=request.source,
            destination=request.destination,
            payload=request.payload,
            correlation_id=request.correlation_id,
            reply_to=request.reply_to
        )
        
        # Send message
        mq_service = get_message_queue_service()
        success = await mq_service.send_message(
            queue_name=request.queue_name,
            message=message,
            delay_seconds=request.delay_seconds
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to send message to queue '{request.queue_name}'"
            )
        
        return SendMessageResponse(
            success=True,
            message="Message sent successfully",
            message_id=message.message_id,
            queue_name=request.queue_name,
            sent_at=message.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}"
        )


@router.post("/send-workflow-request", response_model=SendMessageResponse)
async def send_workflow_request(
    workflow_type: str,
    workflow_data: Dict[str, Any],
    source: str = "api_client",
    correlation_id: Optional[str] = None,
    reply_to: Optional[str] = None
):
    """Send a workflow request message (convenience endpoint)."""
    try:
        # Create workflow request message
        message = QueueMessage(
            message_type=MessageType.WORKFLOW_REQUEST,
            source=source,
            payload={
                'workflow_type': workflow_type,
                'data': workflow_data
            },
            correlation_id=correlation_id,
            reply_to=reply_to
        )
        
        # Send to workflow request queue
        mq_service = get_message_queue_service()
        success = await mq_service.send_message(
            queue_name="workflow_request",
            message=message
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send workflow request"
            )
        
        return SendMessageResponse(
            success=True,
            message="Workflow request sent successfully",
            message_id=message.message_id,
            queue_name="workflow_request",
            sent_at=message.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send workflow request: {str(e)}"
        )


@router.post("/trigger-data-quality-assessment", response_model=SendMessageResponse)
async def trigger_data_quality_assessment(
    dataset_id: str,
    assessment_type: str = "full",
    source: str = "api_client",
    correlation_id: Optional[str] = None
):
    """Trigger a data quality assessment workflow."""
    workflow_data = {
        'dataset_id': dataset_id,
        'assessment_type': assessment_type,
        'requested_at': datetime.utcnow().isoformat()
    }
    
    return await send_workflow_request(
        workflow_type="data_quality_assessment",
        workflow_data=workflow_data,
        source=source,
        correlation_id=correlation_id
    )


@router.post("/trigger-compliance-check", response_model=SendMessageResponse)
async def trigger_compliance_check(
    dataset_id: str,
    compliance_type: str = "gdpr",
    source: str = "api_client",
    correlation_id: Optional[str] = None
):
    """Trigger a compliance check workflow."""
    workflow_data = {
        'dataset_id': dataset_id,
        'compliance_type': compliance_type,
        'requested_at': datetime.utcnow().isoformat()
    }
    
    return await send_workflow_request(
        workflow_type="compliance_check",
        workflow_data=workflow_data,
        source=source,
        correlation_id=correlation_id
    )


@router.post("/trigger-remediation-task", response_model=SendMessageResponse)
async def trigger_remediation_task(
    dataset_id: str,
    issues: List[Dict[str, Any]],
    remediation_type: str = "automated",
    source: str = "api_client",
    correlation_id: Optional[str] = None
):
    """Trigger a data remediation task workflow."""
    workflow_data = {
        'dataset_id': dataset_id,
        'remediation_type': remediation_type,
        'issues': issues,
        'requested_at': datetime.utcnow().isoformat()
    }
    
    return await send_workflow_request(
        workflow_type="remediation_task",
        workflow_data=workflow_data,
        source=source,
        correlation_id=correlation_id
    )


@router.get("/message-types")
async def get_supported_message_types():
    """Get list of supported message types."""
    return {
        "message_types": [msg_type.value for msg_type in MessageType],
        "descriptions": {
            MessageType.WORKFLOW_REQUEST.value: "Request to execute a workflow",
            MessageType.WORKFLOW_RESPONSE.value: "Response from workflow execution",
            MessageType.AGENT_COMMUNICATION.value: "Inter-agent communication",
            MessageType.REMEDIATION_TASK.value: "Data remediation task",
            MessageType.COMPLIANCE_CHECK.value: "Compliance validation task",
            MessageType.DATA_QUALITY_ASSESSMENT.value: "Data quality assessment task",
            MessageType.ANALYTICS_REQUEST.value: "Analytics processing request"
        }
    }


@router.get("/queues")
async def get_registered_queues():
    """Get list of registered queues."""
    try:
        mq_service = get_message_queue_service()
        queues_info = []
        for queue_name, queue_config in mq_service.queues.items():
            queues_info.append({
                "name": queue_name,
                "url": queue_config.url,
                "message_type": queue_config.message_type.value,
                "visibility_timeout": queue_config.visibility_timeout,
                "wait_time_seconds": queue_config.wait_time_seconds,
                "max_messages": queue_config.max_messages
            })
        
        return {
            "queues": queues_info,
            "total_queues": len(queues_info)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get registered queues: {str(e)}"
        )
