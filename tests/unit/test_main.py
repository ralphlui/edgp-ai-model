"""
Unit tests for main.py FastAPI application.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status

# Import the app
import main
from main import app


class TestFastAPIApp:
    """Test the FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_app_creation(self):
        """Test that the FastAPI app is created properly."""
        assert app.title == "EDGP AI Model Service"
        assert "Enterprise Data Governance" in app.description
        assert app.version == "1.0.0"
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert data["service"] == "EDGP AI Model"
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "EDGP AI Model Service" in data["message"]
        assert "version" in data
        assert "docs_url" in data
    
    def test_metrics_endpoint(self, client):
        """Test the metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "uptime" in data
        assert "requests_total" in data
        assert "active_agents" in data
        assert "memory_usage" in data
        assert isinstance(data["uptime"], (int, float))
    
    @patch('main.DataQualityAgent')
    def test_agents_list_endpoint(self, mock_data_quality_agent, client):
        """Test listing available agents."""
        # Mock the agent
        mock_agent = Mock()
        mock_agent.agent_id = "data_quality_agent"
        mock_agent.status.value = "idle"
        mock_agent.capabilities = ["data_quality_assessment", "anomaly_detection"]
        mock_data_quality_agent.return_value = mock_agent
        
        response = client.get("/api/v1/agents")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) > 0
        
        # Check agent structure
        agent_data = data["agents"][0]
        assert "agent_id" in agent_data
        assert "agent_type" in agent_data
        assert "status" in agent_data
        assert "capabilities" in agent_data
    
    @patch('main.DataQualityAgent')
    def test_agent_status_endpoint(self, mock_data_quality_agent, client):
        """Test getting specific agent status."""
        # Mock the agent
        mock_agent = Mock()
        mock_agent.agent_id = "data_quality_agent"
        mock_agent.status.value = "idle"
        mock_agent.capabilities = ["data_quality_assessment"]
        mock_agent.session_id = "test-session-123"
        mock_data_quality_agent.return_value = mock_agent
        
        response = client.get("/api/v1/agents/data_quality_agent/status")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["agent_id"] == "data_quality_agent"
        assert data["status"] == "idle"
        assert "capabilities" in data
        assert "session_id" in data
    
    def test_agent_status_not_found(self, client):
        """Test getting status for non-existent agent."""
        response = client.get("/api/v1/agents/nonexistent_agent/status")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('main.DataQualityAgent')
    @pytest.mark.asyncio
    async def test_process_message_endpoint(self, mock_data_quality_agent, client):
        """Test processing a message through an agent."""
        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.agent_id = "data_quality_agent"
        mock_agent.process_message.return_value = {
            "response": "Message processed",
            "analysis": "Sample analysis"
        }
        mock_data_quality_agent.return_value = mock_agent
        
        message_data = {
            "message": "Analyze data quality for dataset_123",
            "context": {"dataset_id": "dataset_123"}
        }
        
        response = client.post("/api/v1/agents/data_quality_agent/message", json=message_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert "agent_id" in data
        assert data["agent_id"] == "data_quality_agent"
    
    @patch('main.DataQualityAgent')
    @pytest.mark.asyncio
    async def test_execute_capability_endpoint(self, mock_data_quality_agent, client):
        """Test executing a capability through an agent."""
        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.agent_id = "data_quality_agent"
        mock_agent.has_capability.return_value = True
        mock_agent.execute_capability.return_value = {
            "capability": "data_quality_assessment",
            "result": "success",
            "quality_score": 85
        }
        mock_data_quality_agent.return_value = mock_agent
        
        capability_data = {
            "capability": "data_quality_assessment",
            "parameters": {"dataset_id": "test_dataset"}
        }
        
        response = client.post("/api/v1/agents/data_quality_agent/execute", json=capability_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "result" in data
        assert "agent_id" in data
        assert data["agent_id"] == "data_quality_agent"
    
    def test_execute_capability_invalid_capability(self, client):
        """Test executing an invalid capability."""
        capability_data = {
            "capability": "invalid_capability",
            "parameters": {}
        }
        
        response = client.post("/api/v1/agents/data_quality_agent/execute", json=capability_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "invalid capability" in data["detail"].lower()
    
    @patch('main.DataQualityAgent')
    def test_execute_capability_unsupported(self, mock_data_quality_agent, client):
        """Test executing a capability not supported by agent."""
        # Mock the agent
        mock_agent = Mock()
        mock_agent.agent_id = "data_quality_agent"
        mock_agent.has_capability.return_value = False
        mock_data_quality_agent.return_value = mock_agent
        
        capability_data = {
            "capability": "policy_generation",  # Not supported by data quality agent
            "parameters": {}
        }
        
        response = client.post("/api/v1/agents/data_quality_agent/execute", json=capability_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "does not support" in data["detail"].lower()
    
    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON in requests."""
        response = client.post(
            "/api/v1/agents/data_quality_agent/message",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        # Missing 'message' field
        message_data = {
            "context": {"dataset_id": "test"}
        }
        
        response = client.post("/api/v1/agents/data_quality_agent/message", json=message_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_cors_middleware(self, client):
        """Test CORS middleware configuration."""
        # OPTIONS request should be handled by CORS
        response = client.options("/api/v1/agents")
        
        # Should not return 405 Method Not Allowed if CORS is properly configured
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_api_docs_available(self, client):
        """Test that API documentation is available."""
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
        
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK
        
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
    
    @patch('main.get_settings')
    def test_settings_integration(self, mock_get_settings):
        """Test that settings are properly integrated."""
        mock_settings = Mock()
        mock_settings.api_prefix = "/api/v1"
        mock_settings.debug = True
        mock_get_settings.return_value = mock_settings
        
        # Test that the app uses settings
        # This is more of a smoke test since the app is already created
        assert "/api/v1" in str(app.routes)
    
    def test_error_handling_middleware(self, client):
        """Test that error handling middleware works."""
        # Make a request to a non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @patch('main.logger')
    def test_logging_integration(self, mock_logger):
        """Test that logging is properly integrated."""
        client = TestClient(app)
        
        # Make a request that should be logged
        client.get("/health")
        
        # Verify logging was called (at least for app startup)
        assert mock_logger.info.called or mock_logger.debug.called
    
    def test_request_validation(self, client):
        """Test request validation for different endpoints."""
        # Test with extra fields (should be allowed/ignored)
        message_data = {
            "message": "Test message",
            "context": {"key": "value"},
            "extra_field": "should be ignored"
        }
        
        response = client.post("/api/v1/agents/data_quality_agent/message", json=message_data)
        
        # Should not fail due to extra fields
        assert response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_response_format(self, client):
        """Test that responses follow expected format."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
    
    @pytest.mark.asyncio
    async def test_async_endpoint_handling(self, client):
        """Test that async endpoints are properly handled."""
        # The message processing endpoint is async
        message_data = {
            "message": "Test async processing",
            "context": {}
        }
        
        # Should not timeout or fail due to async handling
        response = client.post("/api/v1/agents/data_quality_agent/message", json=message_data)
        
        # Response should be received (regardless of success/failure of agent lookup)
        assert response.status_code in [200, 404, 500]  # Any valid HTTP response


class TestAppConfiguration:
    """Test application configuration and setup."""
    
    def test_app_metadata(self):
        """Test application metadata configuration."""
        assert app.title == "EDGP AI Model Service"
        assert app.version == "1.0.0"
        assert "Enterprise Data Governance" in app.description
    
    def test_middleware_setup(self):
        """Test that middleware is properly configured."""
        middleware_types = [type(middleware) for middleware in app.user_middleware]
        
        # Should have CORS middleware
        from fastapi.middleware.cors import CORSMiddleware
        assert any(issubclass(mw, CORSMiddleware) for mw in middleware_types)
    
    def test_exception_handlers(self):
        """Test that exception handlers are configured."""
        # The app should have exception handlers configured
        assert hasattr(app, 'exception_handlers')
    
    def test_route_configuration(self):
        """Test that routes are properly configured."""
        routes = [route.path for route in app.routes]
        
        expected_routes = [
            "/",
            "/health", 
            "/metrics",
            "/api/v1/agents",
            "/api/v1/agents/{agent_id}/status",
            "/api/v1/agents/{agent_id}/message",
            "/api/v1/agents/{agent_id}/execute"
        ]
        
        for expected_route in expected_routes:
            assert any(expected_route in route or route.endswith(expected_route.split("/")[-1]) for route in routes)
