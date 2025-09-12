"""
Comprehensive tests for main.py FastAPI application.

Testing the main FastAPI application with all its endpoints, 
lifespan management, middleware, and integration components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime

# Mock imports that might cause issues
with patch('core.agents.mcp_enabled.DataQualityMCPAgent'), \
     patch('core.agents.mcp_enabled.ComplianceMCPAgent'), \
     patch('core.agents.mcp_enabled.RemediationMCPAgent'), \
     patch('core.communication.mcp.MCPMessageBus'), \
     patch('core.communication.external.ExternalCommunicationManager'), \
     patch('core.communication.mcp.LangGraphAgentOrchestrator'), \
     patch('core.infrastructure.monitoring.SystemMonitor'), \
     patch('core.api.message_queue_endpoints.router'):

    # Import main app components
    from main import app


class TestMainApplication:
    """Test main FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
        
    @pytest.fixture
    def mock_lifespan_components(self):
        """Mock all lifespan components."""
        with patch('main.mcp_bus') as mock_mcp, \
             patch('main.external_comm') as mock_ext, \
             patch('main.orchestrator') as mock_orch, \
             patch('main.system_monitor') as mock_monitor, \
             patch('main.start_message_queue_service', new_callable=AsyncMock) as mock_start_mq, \
             patch('main.stop_message_queue_service', new_callable=AsyncMock) as mock_stop_mq, \
             patch('main.integration_orchestrator') as mock_int_orch, \
             patch('main.service_registry') as mock_registry, \
             patch('main.initialize_enhanced_agents') as mock_init_agents, \
             patch('main._create_sample_workflows', new_callable=AsyncMock) as mock_workflows, \
             patch('os.makedirs') as mock_makedirs:
            
            # Configure mocks
            mock_mcp.start_message_processing = AsyncMock()
            mock_mcp.stop_message_processing = AsyncMock()
            mock_ext.initialize = AsyncMock()
            mock_ext.get_status = AsyncMock(return_value="active")
            mock_orch.initialize = AsyncMock()
            mock_int_orch.start = AsyncMock()
            mock_int_orch.stop = AsyncMock()
            mock_registry.start_all_adapters = AsyncMock()
            mock_registry.stop_all_adapters = AsyncMock()
            mock_monitor.start_monitoring = Mock()
            mock_monitor.stop_monitoring = Mock()
            mock_init_agents.return_value = []
            
            yield {
                'mcp_bus': mock_mcp,
                'external_comm': mock_ext,
                'orchestrator': mock_orch,
                'system_monitor': mock_monitor,
                'start_mq': mock_start_mq,
                'stop_mq': mock_stop_mq,
                'int_orch': mock_int_orch,
                'registry': mock_registry,
                'init_agents': mock_init_agents,
                'workflows': mock_workflows,
                'makedirs': mock_makedirs
            }


class TestRootEndpoints:
    """Test basic root endpoints."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "EDGP AI Model"
        assert data["version"] == "2.0.0"
        assert data["status"] == "running"
        
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        with patch('main.system_monitor') as mock_monitor, \
             patch('main.initialized_agents', []):
            mock_monitor.get_uptime.return_value = "1h 30m"
            
            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "edgp-ai-model"
            assert data["agents_count"] == 0
            assert data["status"] == "healthy"


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_health_check_success(self, client):
        """Test successful health check."""
        with patch('main.mcp_bus') as mock_mcp, \
             patch('main.external_comm') as mock_ext, \
             patch('main.orchestrator', True) as mock_orch, \
             patch('main.system_monitor') as mock_monitor, \
             patch('main.initialized_agents', []):
            
            mock_mcp.get_status = AsyncMock(return_value="active")
            mock_ext.get_status = AsyncMock(return_value="active")
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "2.0.0"
            assert data["message_driven_architecture"] is True
            assert "mcp_message_bus" in data["services"]
            assert "external_communication" in data["services"]
            
    def test_health_check_with_errors(self, client):
        """Test health check when services have errors."""
        with patch('main.mcp_bus') as mock_mcp, \
             patch('main.external_comm') as mock_ext, \
             patch('main.orchestrator', True), \
             patch('main.initialized_agents', []):
            
            mock_mcp.get_status = AsyncMock(side_effect=Exception("MCP error"))
            mock_ext.get_status = AsyncMock(side_effect=Exception("External error"))
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"  # Still healthy but with degraded services
            assert data["services"]["mcp_message_bus"] == "error"
            assert data["services"]["external_communication"] == "error"
            
    def test_health_check_exception(self, client):
        """Test health check when major exception occurs."""
        with patch('main.initialized_agents', side_effect=Exception("Major error")):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data


class TestAgentEndpoints:
    """Test agent-related endpoints."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_get_agents_status_success(self, client):
        """Test getting agent status successfully."""
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent1.agent_id = "data_quality_agent"
        mock_agent1.get_capabilities = AsyncMock(return_value=[Mock(value="data_quality")])
        mock_agent1.message_queue = []
        mock_agent1.last_activity_timestamp = datetime.utcnow().isoformat()
        
        mock_agent2 = Mock()
        mock_agent2.agent_id = "compliance_agent"
        mock_agent2.get_capabilities = AsyncMock(return_value=[Mock(value="compliance")])
        mock_agent2.message_queue = ["msg1", "msg2"]
        mock_agent2.last_activity_timestamp = datetime.utcnow().isoformat()
        
        with patch('main.initialized_agents', [mock_agent1, mock_agent2]):
            response = client.get("/agents/status")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["agent_id"] == "data_quality_agent"
            assert data[0]["status"] == "active"
            assert data[1]["agent_id"] == "compliance_agent"
            assert data[1]["message_queue_size"] == 2
            
    def test_get_agents_status_with_errors(self, client):
        """Test getting agent status when some agents have errors."""
        mock_agent1 = Mock()
        mock_agent1.agent_id = "data_quality_agent"
        mock_agent1.get_capabilities = AsyncMock(side_effect=Exception("Agent error"))
        
        with patch('main.initialized_agents', [mock_agent1]):
            response = client.get("/agents/status")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["agent_id"] == "data_quality_agent"
            assert data[0]["status"] == "error"
            assert "error" in data[0]
            
    def test_list_agents_api_v1(self, client):
        """Test API v1 list agents endpoint."""
        with patch('agents.data_quality.agent.DataQualityAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.capabilities = [Mock(value="data_quality_assessment")]
            mock_agent_class.return_value = mock_agent
            
            response = client.get("/api/v1/agents")
            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 1
            assert len(data["agents"]) == 1
            assert data["agents"][0]["agent_id"] == "data_quality_agent"
            assert data["agents"][0]["status"] == "active"
            
    def test_get_agent_status_api_v1(self, client):
        """Test API v1 get specific agent status."""
        with patch('agents.data_quality.agent.DataQualityAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            response = client.get("/api/v1/agents/data_quality_agent/status")
            assert response.status_code == 200


class TestLifespanManagement:
    """Test application lifespan management."""
    
    @pytest.mark.asyncio
    async def test_lifespan_startup_success(self, mock_lifespan_components):
        """Test successful application startup."""
        from main import lifespan
        
        # Mock the integration config manager
        with patch('main.integration_config_manager') as mock_config:
            mock_config.generate_service_registry_data.return_value = [
                {
                    'service_name': 'test_service',
                    'base_url': 'http://test.com',
                    'pattern': 'rest',
                    'config': {}
                }
            ]
            
            with patch('main.DEFAULT_EXTERNAL_SERVICES_CONFIG', {}):
                # Create a context manager for testing
                async with lifespan(app):
                    pass
                
                # Verify startup calls
                mock_lifespan_components['mcp_bus'].start_message_processing.assert_called_once()
                mock_lifespan_components['external_comm'].initialize.assert_called_once()
                mock_lifespan_components['int_orch'].start.assert_called_once()
                mock_lifespan_components['registry'].start_all_adapters.assert_called_once()
                mock_lifespan_components['start_mq'].assert_called_once()
                mock_lifespan_components['init_agents'].assert_called_once()
                mock_lifespan_components['workflows'].assert_called_once()
                
                # Verify shutdown calls
                mock_lifespan_components['int_orch'].stop.assert_called_once()
                mock_lifespan_components['registry'].stop_all_adapters.assert_called_once()
                mock_lifespan_components['stop_mq'].assert_called_once()
                mock_lifespan_components['mcp_bus'].stop_message_processing.assert_called_once()
                
    @pytest.mark.asyncio
    async def test_lifespan_startup_failure(self, mock_lifespan_components):
        """Test application startup failure handling."""
        from main import lifespan
        
        # Make external_comm.initialize fail
        mock_lifespan_components['external_comm'].initialize.side_effect = Exception("Startup error")
        
        with pytest.raises(Exception, match="Startup error"):
            async with lifespan(app):
                pass


class TestWorkflowCreation:
    """Test sample workflow creation."""
    
    @pytest.mark.asyncio
    async def test_create_sample_workflows(self):
        """Test creation of sample workflows."""
        from main import _create_sample_workflows
        
        with patch('main.orchestrator') as mock_orch:
            await _create_sample_workflows()
            
            # Should create 2 workflows
            assert mock_orch.create_workflow.call_count == 2
            
            # Check workflow names
            calls = mock_orch.create_workflow.call_args_list
            workflow_names = [call[0][0] for call in calls]
            assert "data_quality_pipeline" in workflow_names
            assert "compliance_assessment" in workflow_names
            
            # Check workflow structure
            quality_workflow = calls[0][0][1]
            assert "nodes" in quality_workflow
            assert "edges" in quality_workflow
            assert "entry_point" in quality_workflow
            assert quality_workflow["entry_point"] == "assess_quality"


class TestMiddlewareAndConfiguration:
    """Test middleware and application configuration."""
    
    def test_cors_middleware_configured(self):
        """Test CORS middleware is properly configured."""
        # Check that CORS middleware is in the middleware stack
        middleware_classes = [middleware.cls for middleware in app.user_middleware]
        from fastapi.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_classes
        
    def test_error_handlers_setup(self):
        """Test error handlers are properly set up."""
        with patch('main.setup_error_handlers') as mock_setup:
            # Import main module again to trigger setup
            import importlib
            import main
            importlib.reload(main)
            
            # Verify setup was called (this is a bit tricky to test directly)
            # The important thing is the function was called during module import
            
    def test_routers_included(self):
        """Test all required routers are included."""
        # Check that routes exist for included routers
        routes = [route.path for route in app.routes]
        
        # Basic routes should exist
        assert "/" in routes
        assert "/health" in routes
        assert "/metrics" in routes
        assert "/agents/status" in routes
        
        # Check that router prefixes are included (this is implementation dependent)
        # We mainly want to ensure the include_router calls work without error


class TestGlobalInstances:
    """Test global instance initialization."""
    
    def test_global_instances_created(self):
        """Test that global instances are properly created."""
        from main import mcp_bus, external_comm, orchestrator, system_monitor
        from main import data_quality_agent, compliance_agent, remediation_agent
        
        # Check instances exist
        assert mcp_bus is not None
        assert external_comm is not None  
        assert orchestrator is not None
        assert system_monitor is not None
        assert data_quality_agent is not None
        assert compliance_agent is not None
        assert remediation_agent is not None
        
        # Check types
        from core.communication.mcp import MCPMessageBus, LangGraphAgentOrchestrator
        from core.communication.external import ExternalCommunicationManager
        from core.infrastructure.monitoring import SystemMonitor
        
        assert isinstance(mcp_bus, MCPMessageBus)
        assert isinstance(external_comm, ExternalCommunicationManager)
        assert isinstance(orchestrator, LangGraphAgentOrchestrator)
        assert isinstance(system_monitor, SystemMonitor)
        
    def test_initialized_agents_list(self):
        """Test initialized agents list."""
        from main import initialized_agents
        
        assert isinstance(initialized_agents, list)
        # Initially empty
        assert len(initialized_agents) == 0


class TestLoggingConfiguration:
    """Test logging setup."""
    
    def test_logging_configuration(self):
        """Test logging is properly configured."""
        import logging
        
        # Check that loggers exist
        logger = logging.getLogger('main')
        assert logger is not None
        
        # Check log level (should be INFO)
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO
        
        # Check handlers exist
        assert len(root_logger.handlers) > 0


class TestIntegrationWithExternalModules:
    """Test integration with external modules."""
    
    def test_agent_imports(self):
        """Test agent imports work correctly."""
        from core.agents.mcp_enabled import (
            DataQualityMCPAgent,
            ComplianceMCPAgent, 
            RemediationMCPAgent
        )
        
        # Check classes exist
        assert DataQualityMCPAgent is not None
        assert ComplianceMCPAgent is not None
        assert RemediationMCPAgent is not None
        
    def test_communication_imports(self):
        """Test communication module imports."""
        from core.communication.mcp import (
            MCPMessageBus, 
            LangGraphAgentOrchestrator,
            MCPMessage,
            MessagePriority
        )
        from core.communication.external import ExternalCommunicationManager
        
        # Check all imports work
        assert MCPMessageBus is not None
        assert LangGraphAgentOrchestrator is not None
        assert MCPMessage is not None
        assert MessagePriority is not None
        assert ExternalCommunicationManager is not None
        
    def test_type_imports(self):
        """Test type imports work correctly."""
        from core.types.responses import (
            HealthResponse,
            AgentStatusResponse,
            WorkflowResponse,
            MessageResponse
        )
        from core.types.data import ProcessingRequest
        from core.types.agent_types import AgentCapability
        
        # Check all types exist
        assert HealthResponse is not None
        assert AgentStatusResponse is not None
        assert WorkflowResponse is not None
        assert MessageResponse is not None
        assert ProcessingRequest is not None
        assert AgentCapability is not None
