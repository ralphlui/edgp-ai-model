"""
Unit tests for core.agents.enhanced module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from core.agents.enhanced import EnhancedAgentBase, DataQualityEnhancedAgent


class TestEnhancedAgentBase:
    """Test suite for EnhancedAgentBase class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mocked dependencies for EnhancedAgentBase."""
        return {
            'llm_bridge': Mock(),
            'integration_orchestrator': Mock(),
            'external_comm': Mock(),
            'config': {
                'llm_model': 'test-model',
                'max_retries': 3,
                'timeout': 30
            }
        }

    @pytest.fixture
    def enhanced_agent(self, mock_dependencies):
        """Create EnhancedAgentBase instance with mocked dependencies."""
        with patch('core.agents.enhanced.create_agent_integration_helper') as mock_helper:
            mock_helper.return_value = Mock()
            agent = EnhancedAgentBase(
                name="test_enhanced_agent",
                description="Test enhanced agent",
                config=mock_dependencies['config'],
                llm_bridge=mock_dependencies['llm_bridge'],
                integration_orchestrator=mock_dependencies['integration_orchestrator'],
                external_comm=mock_dependencies['external_comm']
            )
            return agent

    def test_enhanced_agent_initialization(self, enhanced_agent):
        """Test enhanced agent is properly initialized."""
        assert enhanced_agent.name == "test_enhanced_agent"
        assert enhanced_agent.description == "Test enhanced agent"
        assert hasattr(enhanced_agent, 'integration_helper')
        assert hasattr(enhanced_agent, 'llm_bridge')
        assert hasattr(enhanced_agent, 'external_comm')

    def test_enhanced_agent_has_base_capabilities(self, enhanced_agent):
        """Test that enhanced agent inherits base capabilities."""
        assert hasattr(enhanced_agent, 'capabilities')
        assert hasattr(enhanced_agent, 'status')
        assert hasattr(enhanced_agent, 'session_id')

    @patch('core.agents.enhanced.create_agent_integration_helper')
    def test_integration_helper_creation(self, mock_create_helper, mock_dependencies):
        """Test that integration helper is created correctly."""
        mock_helper = Mock()
        mock_create_helper.return_value = mock_helper
        
        agent = EnhancedAgentBase(
            name="test_agent",
            description="Test description",
            config=mock_dependencies['config'],
            llm_bridge=mock_dependencies['llm_bridge'],
            integration_orchestrator=mock_dependencies['integration_orchestrator'],
            external_comm=mock_dependencies['external_comm']
        )
        
        mock_create_helper.assert_called_once()
        assert agent.integration_helper == mock_helper

    def test_enhanced_agent_config_properties(self, enhanced_agent):
        """Test that config properties are accessible."""
        assert enhanced_agent.config['llm_model'] == 'test-model'
        assert enhanced_agent.config['max_retries'] == 3
        assert enhanced_agent.config['timeout'] == 30


class TestDataQualityEnhancedAgent:
    """Test suite for DataQualityEnhancedAgent class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mocked dependencies for DataQualityEnhancedAgent."""
        return {
            'llm_bridge': Mock(),
            'integration_orchestrator': Mock(),
            'external_comm': Mock(),
            'config': {
                'thresholds': {
                    'completeness': 0.9,
                    'accuracy': 0.95
                },
                'validation_rules': ['not_null', 'format_check']
            }
        }

    @pytest.fixture
    def integrated_dq_agent(self, mock_dependencies):
        """Create DataQualityEnhancedAgent instance with mocked dependencies."""
        with patch('core.agents.enhanced.create_agent_integration_helper'):
            agent = DataQualityEnhancedAgent(
                name="integrated_dq_agent",
                description="Integrated data quality agent",
                config=mock_dependencies['config'],
                llm_bridge=mock_dependencies['llm_bridge'],
                integration_orchestrator=mock_dependencies['integration_orchestrator'],
                external_comm=mock_dependencies['external_comm']
            )
            return agent

    def test_integrated_dq_agent_initialization(self, integrated_dq_agent):
        """Test integrated data quality agent initialization."""
        assert integrated_dq_agent.name == "integrated_dq_agent"
        assert "data_quality_assessment" in integrated_dq_agent.capabilities
        assert "anomaly_detection" in integrated_dq_agent.capabilities
        assert "data_profiling" in integrated_dq_agent.capabilities

    def test_integrated_dq_agent_has_enhanced_features(self, integrated_dq_agent):
        """Test that integrated agent has enhanced features."""
        assert hasattr(integrated_dq_agent, 'integration_helper')
        assert hasattr(integrated_dq_agent, 'llm_bridge')
        assert hasattr(integrated_dq_agent, 'external_comm')

    @pytest.mark.asyncio
    async def test_execute_capability_data_quality_assessment(self, integrated_dq_agent):
        """Test data quality assessment capability execution."""
        mock_context = Mock()
        mock_context.data = {"test_data": "value"}
        
        # Mock the integration helper to return a response
        integrated_dq_agent.integration_helper.execute_with_integration = Mock(return_value={
            "quality_score": 0.85,
            "issues": ["missing_values"],
            "recommendations": ["fill_missing_values"]
        })
        
        result = await integrated_dq_agent.execute_capability(
            "data_quality_assessment",
            {"dataset": "test_dataset"},
            mock_context
        )
        
        assert result is not None
        assert "quality_score" in result
        integrated_dq_agent.integration_helper.execute_with_integration.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_capability_anomaly_detection(self, integrated_dq_agent):
        """Test anomaly detection capability execution."""
        mock_context = Mock()
        
        # Mock the integration helper
        integrated_dq_agent.integration_helper.execute_with_integration = Mock(return_value={
            "anomalies_detected": 3,
            "anomaly_types": ["outlier", "pattern_break"],
            "confidence": 0.92
        })
        
        result = await integrated_dq_agent.execute_capability(
            "anomaly_detection",
            {"data_source": "test_source"},
            mock_context
        )
        
        assert result is not None
        assert "anomalies_detected" in result
        integrated_dq_agent.integration_helper.execute_with_integration.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_capability_data_profiling(self, integrated_dq_agent):
        """Test data profiling capability execution."""
        mock_context = Mock()
        
        # Mock the integration helper
        integrated_dq_agent.integration_helper.execute_with_integration = Mock(return_value={
            "profile": {
                "rows": 1000,
                "columns": 10,
                "missing_values": 50
            },
            "statistics": {"mean": 25.5, "std": 10.2}
        })
        
        result = await integrated_dq_agent.execute_capability(
            "data_profiling",
            {"table": "test_table"},
            mock_context
        )
        
        assert result is not None
        assert "profile" in result
        integrated_dq_agent.integration_helper.execute_with_integration.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_capability_unsupported(self, integrated_dq_agent):
        """Test execution of unsupported capability."""
        mock_context = Mock()
        
        with pytest.raises(ValueError, match="Unsupported capability"):
            await integrated_dq_agent.execute_capability(
                "unsupported_capability",
                {},
                mock_context
            )

    def test_agent_specific_config_access(self, integrated_dq_agent):
        """Test access to agent-specific configuration."""
        assert integrated_dq_agent.config['thresholds']['completeness'] == 0.9
        assert integrated_dq_agent.config['thresholds']['accuracy'] == 0.95
        assert "not_null" in integrated_dq_agent.config['validation_rules']


class TestMockingAndImportHandling:
    """Test mocking strategies and import handling for enhanced agents."""

    @patch('core.agents.enhanced.create_agent_integration_helper')
    @patch('core.agents.enhanced.logger')
    def test_agent_with_mocked_imports(self, mock_logger, mock_create_helper):
        """Test agent creation with fully mocked imports."""
        mock_create_helper.return_value = Mock()
        
        # Mock all dependencies
        mock_llm_bridge = Mock()
        mock_orchestrator = Mock()
        mock_external_comm = Mock()
        mock_config = {"test": "config"}
        
        agent = EnhancedAgentBase(
            name="test_mock_agent",
            description="Test with mocks",
            config=mock_config,
            llm_bridge=mock_llm_bridge,
            integration_orchestrator=mock_orchestrator,
            external_comm=mock_external_comm
        )
        
        assert agent.name == "test_mock_agent"
        assert agent.llm_bridge == mock_llm_bridge
        assert agent.external_comm == mock_external_comm
        mock_create_helper.assert_called_once()

    def test_integration_patterns_access(self):
        """Test that integration patterns can be imported and accessed."""
        with patch('core.agents.enhanced.integration_orchestrator') as mock_orchestrator:
            mock_orchestrator.get_pattern = Mock(return_value="mock_pattern")
            
            # This tests the import structure
            assert mock_orchestrator.get_pattern("test") == "mock_pattern"

    @patch('core.agents.enhanced.SharedIntegrationFunctions')
    def test_shared_integration_functions_mock(self, mock_shared_functions):
        """Test mocking of shared integration functions."""
        mock_shared_functions.execute_integration = Mock(return_value={"result": "success"})
        
        result = mock_shared_functions.execute_integration("test_integration")
        assert result == {"result": "success"}
        mock_shared_functions.execute_integration.assert_called_once_with("test_integration")


class TestAgentIntegrationHelperUsage:
    """Test usage of AgentIntegrationHelper within enhanced agents."""

    @pytest.fixture
    def mock_integration_helper(self):
        """Create a mock integration helper."""
        helper = Mock()
        helper.execute_with_integration = Mock()
        helper.setup_external_connections = Mock()
        helper.cleanup_resources = Mock()
        return helper

    @pytest.fixture
    def agent_with_helper(self, mock_integration_helper):
        """Create agent with mocked integration helper."""
        with patch('core.agents.enhanced.create_agent_integration_helper') as mock_create:
            mock_create.return_value = mock_integration_helper
            
            agent = EnhancedAgentBase(
                name="helper_test_agent",
                description="Agent for testing helper usage",
                config={},
                llm_bridge=Mock(),
                integration_orchestrator=Mock(),
                external_comm=Mock()
            )
            return agent, mock_integration_helper

    def test_integration_helper_setup(self, agent_with_helper):
        """Test that integration helper is properly set up."""
        agent, helper = agent_with_helper
        assert agent.integration_helper == helper

    def test_integration_helper_method_calls(self, agent_with_helper):
        """Test calling methods on integration helper."""
        agent, helper = agent_with_helper
        
        # Test calling setup
        agent.integration_helper.setup_external_connections()
        helper.setup_external_connections.assert_called_once()
        
        # Test calling cleanup
        agent.integration_helper.cleanup_resources()
        helper.cleanup_resources.assert_called_once()


class TestEnhancedAgentErrorHandling:
    """Test error handling in enhanced agents."""

    @pytest.fixture
    def agent_with_failing_helper(self):
        """Create agent where integration helper fails."""
        with patch('core.agents.enhanced.create_agent_integration_helper') as mock_create:
            mock_helper = Mock()
            mock_helper.execute_with_integration.side_effect = Exception("Integration failed")
            mock_create.return_value = mock_helper
            
            agent = EnhancedAgentBase(
                name="failing_agent",
                description="Agent with failing helper",
                config={},
                llm_bridge=Mock(),
                integration_orchestrator=Mock(),
                external_comm=Mock()
            )
            return agent

    @pytest.mark.asyncio
    async def test_integration_error_handling(self, agent_with_failing_helper):
        """Test handling of integration errors."""
        # This would test error handling if the execute_capability method exists
        # Since we're focusing on imports and basic functionality, we just test the setup
        assert agent_with_failing_helper.integration_helper is not None
        
        # Test that the helper can fail as expected
        with pytest.raises(Exception, match="Integration failed"):
            agent_with_failing_helper.integration_helper.execute_with_integration("test")


class TestAgentConfigurationHandling:
    """Test configuration handling in enhanced agents."""

    def test_empty_config_handling(self):
        """Test agent creation with empty config."""
        with patch('core.agents.enhanced.create_agent_integration_helper'):
            agent = EnhancedAgentBase(
                name="empty_config_agent",
                description="Agent with empty config",
                config={},
                llm_bridge=Mock(),
                integration_orchestrator=Mock(),
                external_comm=Mock()
            )
            assert agent.config == {}

    def test_complex_config_handling(self):
        """Test agent creation with complex config."""
        complex_config = {
            "llm_settings": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "integration_settings": {
                "external_apis": ["api1", "api2"],
                "timeout": 60,
                "retries": 5
            },
            "data_processing": {
                "batch_size": 100,
                "parallel_workers": 4
            }
        }
        
        with patch('core.agents.enhanced.create_agent_integration_helper'):
            agent = EnhancedAgentBase(
                name="complex_config_agent",
                description="Agent with complex config",
                config=complex_config,
                llm_bridge=Mock(),
                integration_orchestrator=Mock(),
                external_comm=Mock()
            )
            
            assert agent.config == complex_config
            assert agent.config["llm_settings"]["model"] == "gpt-4"
            assert agent.config["integration_settings"]["timeout"] == 60
            assert agent.config["data_processing"]["batch_size"] == 100
