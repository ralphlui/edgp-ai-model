"""
Core services for the EDGP AI Model.
"""

# Import services individually to avoid circular dependencies
from .llm_gateway import llm_gateway, LLMResponse

# Import message queue service 
from .message_queue import (
    MessageQueueService, QueueMessage, MessageType, 
    get_message_queue_service, send_message, receive_messages,
    register_queue, register_message_handler
)

__all__ = [
    "llm_gateway",
    "LLMResponse", 
    "MessageQueueService",
    "QueueMessage", 
    "MessageType",
    "get_message_queue_service",
    "send_message",
    "receive_messages", 
    "register_queue",
    "register_message_handler"
]
