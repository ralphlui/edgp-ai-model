"""
Unit tests for the message queue API endpoints.
"""
import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import directly to avoid circular imports during testing
from core.api.message_queue_endpoints import router as message_queue_router
from core.services.message_queue import QueueMessage, MessageType, QueueConfig


@pytest.fixture
def mock_message_queue_service():
    """Mock message queue service."""
    with patch('core.api.message_queue_endpoints.MessageQueueService') as mock_service:
        mock_instance = AsyncMock()
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client():
    """Create test client."""
    app = FastAPI()
    app.include_router(message_queue_router, prefix="/api/v1")
    return TestClient(app)


class TestMessageQueueEndpoints:
    """Test cases for message queue API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/message-queue/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "message_queue"
        assert "timestamp" in data
    
    def test_create_queue_success(self, client, mock_message_queue_service):
        """Test successful queue creation."""
        mock_message_queue_service.create_queue.return_value = True
        
        response = client.post(
            "/api/v1/message-queue/queues",
            json={"queue_name": "test-queue", "attributes": {}}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Queue 'test-queue' created successfully"
        assert data["queue_name"] == "test-queue"
        mock_message_queue_service.create_queue.assert_called_once_with("test-queue", {})
    
    def test_create_queue_failure(self, client, mock_message_queue_service):
        """Test queue creation failure."""
        mock_message_queue_service.create_queue.side_effect = Exception("Creation failed")
        
        response = client.post(
            "/api/v1/message-queue/queues",
            json={"queue_name": "test-queue", "attributes": {}}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Failed to create queue: Creation failed"
    
    def test_send_message_success(self, client, mock_message_queue_service):
        """Test successful message sending."""
        mock_message_queue_service.send_message.return_value = True
        
        response = client.post(
            "/api/v1/message-queue/queues/test-queue/messages",
            json={
                "content": {"test": "data"},
                "priority": "high",
                "max_retries": 3
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Message sent successfully"
        assert data["queue_name"] == "test-queue"
        assert "message_id" in data
        mock_message_queue_service.send_message.assert_called_once()
    
    def test_send_message_failure(self, client, mock_message_queue_service):
        """Test message sending failure."""
        mock_message_queue_service.send_message.side_effect = Exception("Send failed")
        
        response = client.post(
            "/api/v1/message-queue/queues/test-queue/messages",
            json={
                "content": {"test": "data"},
                "priority": "normal",
                "max_retries": 3
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Failed to send message: Send failed"
    
    def test_receive_messages_success(self, client, mock_message_queue_service):
        """Test successful message receiving."""
        mock_message = QueueMessage(
            message_id="test-123",
            message_type=MessageType.WORKFLOW_REQUEST,
            source="test-source",
            payload={"test": "data"}
        )
        mock_message_queue_service.receive_messages.return_value = [mock_message]
        
        response = client.get("/api/v1/message-queue/queues/test-queue/messages?max_messages=5")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["message_id"] == "test-123"
        assert data["messages"][0]["content"] == {"test": "data"}
        mock_message_queue_service.receive_messages.assert_called_once_with("test-queue", 5, 20)
    
    def test_receive_messages_empty(self, client, mock_message_queue_service):
        """Test receiving messages from empty queue."""
        mock_message_queue_service.receive_messages.return_value = []
        
        response = client.get("/api/v1/message-queue/queues/test-queue/messages")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 0
        assert data["queue_name"] == "test-queue"
    
    def test_delete_message_success(self, client, mock_message_queue_service):
        """Test successful message deletion."""
        mock_message_queue_service.delete_message.return_value = True
        
        response = client.delete(
            "/api/v1/message-queue/queues/test-queue/messages/test-receipt-handle"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Message deleted successfully"
        mock_message_queue_service.delete_message.assert_called_once_with("test-queue", "test-receipt-handle")
    
    def test_delete_message_failure(self, client, mock_message_queue_service):
        """Test message deletion failure."""
        mock_message_queue_service.delete_message.side_effect = Exception("Delete failed")
        
        response = client.delete(
            "/api/v1/message-queue/queues/test-queue/messages/test-receipt-handle"
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Failed to delete message: Delete failed"
    
    def test_get_queue_attributes_success(self, client, mock_message_queue_service):
        """Test successful queue attributes retrieval."""
        mock_attributes = {
            "ApproximateNumberOfMessages": "5",
            "ApproximateNumberOfMessagesNotVisible": "2"
        }
        mock_message_queue_service.get_queue_attributes.return_value = mock_attributes
        
        response = client.get("/api/v1/message-queue/queues/test-queue/attributes")
        
        assert response.status_code == 200
        data = response.json()
        assert data["queue_name"] == "test-queue"
        assert data["attributes"] == mock_attributes
    
    def test_purge_queue_success(self, client, mock_message_queue_service):
        """Test successful queue purging."""
        mock_message_queue_service.purge_queue.return_value = True
        
        response = client.delete("/api/v1/message-queue/queues/test-queue/purge")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Queue 'test-queue' purged successfully"
        mock_message_queue_service.purge_queue.assert_called_once_with("test-queue")
    
    def test_trigger_data_quality_task_success(self, client, mock_message_queue_service):
        """Test successful data quality task triggering."""
        mock_message_queue_service.send_message.return_value = True
        
        response = client.post(
            "/api/v1/message-queue/triggers/data-quality",
            json={
                "dataset_id": "dataset-123",
                "quality_dimensions": ["completeness", "accuracy"]
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["message"] == "Data quality task triggered successfully"
        assert data["task_type"] == "data_quality_assessment"
        assert "task_id" in data
        mock_message_queue_service.send_message.assert_called_once()
    
    def test_trigger_analytics_task_success(self, client, mock_message_queue_service):
        """Test successful analytics task triggering."""
        mock_message_queue_service.send_message.return_value = True
        
        response = client.post(
            "/api/v1/message-queue/triggers/analytics",
            json={
                "dataset_id": "dataset-123",
                "analysis_type": "statistical_summary"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["message"] == "Analytics task triggered successfully"
        assert data["task_type"] == "analytics_processing"
        assert "task_id" in data
        mock_message_queue_service.send_message.assert_called_once()
    
    def test_trigger_compliance_check_success(self, client, mock_message_queue_service):
        """Test successful compliance check triggering."""
        mock_message_queue_service.send_message.return_value = True
        
        response = client.post(
            "/api/v1/message-queue/triggers/compliance",
            json={
                "dataset_id": "dataset-123",
                "regulations": ["GDPR", "CCPA"]
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["message"] == "Compliance check triggered successfully"
        assert data["task_type"] == "compliance_check"
        assert "task_id" in data
        mock_message_queue_service.send_message.assert_called_once()
    
    def test_trigger_policy_suggestion_success(self, client, mock_message_queue_service):
        """Test successful policy suggestion triggering."""
        mock_message_queue_service.send_message.return_value = True
        
        response = client.post(
            "/api/v1/message-queue/triggers/policy-suggestion",
            json={
                "dataset_id": "dataset-123",
                "context": "data_governance"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["message"] == "Policy suggestion task triggered successfully"
        assert data["task_type"] == "policy_suggestion"
        assert "task_id" in data
        mock_message_queue_service.send_message.assert_called_once()
    
    def test_trigger_remediation_task_success(self, client, mock_message_queue_service):
        """Test successful remediation task triggering."""
        mock_message_queue_service.send_message.return_value = True
        
        response = client.post(
            "/api/v1/message-queue/triggers/remediation",
            json={
                "dataset_id": "dataset-123",
                "issues": [
                    {"type": "missing_data", "column": "email", "severity": "high"}
                ],
                "remediation_type": "automated"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["message"] == "Remediation task triggered successfully"
        assert data["task_type"] == "data_remediation"
        assert "task_id" in data
        mock_message_queue_service.send_message.assert_called_once()
    
    def test_trigger_remediation_task_validation_error(self, client):
        """Test remediation task triggering with validation error."""
        response = client.post(
            "/api/v1/message-queue/triggers/remediation",
            json={
                "dataset_id": "dataset-123",
                "remediation_type": "automated"
                # Missing required 'issues' field
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_start_background_processing_success(self, client, mock_message_queue_service):
        """Test successful background processing start."""
        mock_message_queue_service.start_background_processing.return_value = None
        
        response = client.post(
            "/api/v1/message-queue/queues/test-queue/background-processing/start"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Background processing started for queue 'test-queue'"
        mock_message_queue_service.start_background_processing.assert_called_once()
    
    def test_stop_background_processing_success(self, client, mock_message_queue_service):
        """Test successful background processing stop."""
        mock_message_queue_service.stop_background_processing.return_value = None
        
        response = client.post(
            "/api/v1/message-queue/queues/test-queue/background-processing/stop"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Background processing stopped for queue 'test-queue'"
        mock_message_queue_service.stop_background_processing.assert_called_once()
    
    def test_list_queues_success(self, client, mock_message_queue_service):
        """Test successful queue listing."""
        mock_queue_urls = {
            "queue1": "https://sqs.us-east-1.amazonaws.com/123456789012/queue1",
            "queue2": "https://sqs.us-east-1.amazonaws.com/123456789012/queue2"
        }
        mock_message_queue_service.queue_urls = mock_queue_urls
        
        response = client.get("/api/v1/message-queue/queues")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["queues"]) == 2
        assert "queue1" in [q["name"] for q in data["queues"]]
        assert "queue2" in [q["name"] for q in data["queues"]]
    
    def test_invalid_priority_in_send_message(self, client, mock_message_queue_service):
        """Test sending message with invalid priority."""
        response = client.post(
            "/api/v1/message-queue/queues/test-queue/messages",
            json={
                "content": {"test": "data"},
                "priority": "invalid_priority",
                "max_retries": 3
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_negative_max_retries_in_send_message(self, client, mock_message_queue_service):
        """Test sending message with negative max_retries."""
        response = client.post(
            "/api/v1/message-queue/queues/test-queue/messages",
            json={
                "content": {"test": "data"},
                "priority": "normal",
                "max_retries": -1
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_receive_messages_with_invalid_parameters(self, client):
        """Test receiving messages with invalid parameters."""
        response = client.get("/api/v1/message-queue/queues/test-queue/messages?max_messages=0")
        
        assert response.status_code == 422  # Validation error
    
    def test_queue_name_validation(self, client):
        """Test queue name validation."""
        response = client.post(
            "/api/v1/message-queue/queues",
            json={"queue_name": "", "attributes": {}}
        )
        
        assert response.status_code == 422  # Validation error
