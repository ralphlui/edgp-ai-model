"""
Unit tests for core.agents.base module.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID
from core.agents.base import BaseAgent
from core.types.agent_types import AgentCapability, AgentStatus


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing."""
    
    def __init__(self, agent_id: str = "test_agent"):
        super().__init__(
            agent_id=agent_id,
            capabilities=[
                AgentCapability.DATA_QUALITY_ASSESSMENT,
                AgentCapability.ANOMALY_DETECTION
            ]
        )
    
    async def process_message(self, message: str, context: dict = None) -> dict:
        """Test implementation of process_message."""
        return {
            "response": f"Processed: {message}",
            "context": context or {}
        }
    
    async def execute_capability(self, capability: AgentCapability, parameters: dict = None) -> dict:
        """Test implementation of execute_capability."""
        return {
            "capability": capability.value,
            "parameters": parameters or {},
            "result": "success"
        }


class TestBaseAgent:
    """Test the BaseAgent class."""
    
    def test_agent_initialization(self):
        """Test agent initialization with correct properties."""
        agent = TestAgent("test_agent_123")
        
        assert agent.agent_id == "test_agent_123"
        assert agent.status == AgentStatus.IDLE
        assert len(agent.capabilities) == 2
        assert AgentCapability.DATA_QUALITY_ASSESSMENT in agent.capabilities
        assert AgentCapability.ANOMALY_DETECTION in agent.capabilities
        assert isinstance(UUID(agent.session_id), UUID)
    
    def test_agent_initialization_auto_id(self):
        """Test agent initialization with auto-generated ID."""
        agent = TestAgent()
        
        assert agent.agent_id == "test_agent"
        assert agent.status == AgentStatus.IDLE
    
    def test_agent_id_property(self):
        """Test that agent_id property is read-only after initialization."""
        agent = TestAgent("original_id")
        
        # Try to modify agent_id (should remain unchanged)
        original_id = agent.agent_id
        with pytest.raises(AttributeError):
            agent.agent_id = "new_id"
        
        assert agent.agent_id == original_id
    
    def test_status_property(self):
        """Test status property getter and setter."""
        agent = TestAgent()
        
        # Initial status
        assert agent.status == AgentStatus.IDLE
        
        # Change status
        agent.status = AgentStatus.PROCESSING
        assert agent.status == AgentStatus.PROCESSING
        
        agent.status = AgentStatus.ERROR
        assert agent.status == AgentStatus.ERROR
    
    def test_capabilities_property(self):
        """Test capabilities property is read-only."""
        agent = TestAgent()
        original_capabilities = agent.capabilities.copy()
        
        # Capabilities should be a copy, not the original list
        agent.capabilities.append(AgentCapability.COMPLIANCE_CHECK)
        assert len(agent._capabilities) == 2  # Original unchanged
        assert agent.capabilities == original_capabilities
    
    def test_session_id_property(self):
        """Test that session_id is generated and immutable."""
        agent = TestAgent()
        
        # Should have a session ID
        assert agent.session_id is not None
        assert isinstance(UUID(agent.session_id), UUID)
        
        # Should be read-only
        original_session_id = agent.session_id
        with pytest.raises(AttributeError):
            agent.session_id = "new_session"
        
        assert agent.session_id == original_session_id
    
    def test_logger_property(self):
        """Test that logger is properly configured."""
        agent = TestAgent()
        
        assert agent.logger is not None
        assert agent.logger.name == "core.agents.base"
    
    @patch('core.services.llm_gateway.LLMGatewayBridge')
    def test_llm_gateway_property(self, mock_llm_gateway):
        """Test llm_gateway property lazy loading."""
        mock_instance = Mock()
        mock_llm_gateway.return_value = mock_instance
        
        agent = TestAgent()
        
        # First access should create the instance
        llm_gateway = agent.llm_gateway
        assert llm_gateway == mock_instance
        mock_llm_gateway.assert_called_once()
        
        # Second access should return the same instance
        llm_gateway2 = agent.llm_gateway
        assert llm_gateway2 == mock_instance
        assert mock_llm_gateway.call_count == 1  # Not called again
    
    @patch('core.services.rag_system.RAGSystem')
    def test_rag_system_property(self, mock_rag_system):
        """Test rag_system property lazy loading."""
        mock_instance = Mock()
        mock_rag_system.return_value = mock_instance
        
        agent = TestAgent()
        
        # First access should create the instance
        rag_system = agent.rag_system
        assert rag_system == mock_instance
        mock_rag_system.assert_called_once()
        
        # Second access should return the same instance
        rag_system2 = agent.rag_system
        assert rag_system2 == mock_instance
        assert mock_rag_system.call_count == 1  # Not called again
    
    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test the process_message method."""
        agent = TestAgent()
        
        result = await agent.process_message("test message", {"key": "value"})
        
        assert result["response"] == "Processed: test message"
        assert result["context"] == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_process_message_no_context(self):
        """Test process_message without context."""
        agent = TestAgent()
        
        result = await agent.process_message("test message")
        
        assert result["response"] == "Processed: test message"
        assert result["context"] == {}
    
    @pytest.mark.asyncio
    async def test_execute_capability(self):
        """Test the execute_capability method."""
        agent = TestAgent()
        
        result = await agent.execute_capability(
            AgentCapability.DATA_QUALITY_ASSESSMENT,
            {"param1": "value1"}
        )
        
        assert result["capability"] == "data_quality_assessment"
        assert result["parameters"] == {"param1": "value1"}
        assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_execute_capability_no_parameters(self):
        """Test execute_capability without parameters."""
        agent = TestAgent()
        
        result = await agent.execute_capability(AgentCapability.ANOMALY_DETECTION)
        
        assert result["capability"] == "anomaly_detection"
        assert result["parameters"] == {}
    
    def test_has_capability(self):
        """Test checking if agent has specific capabilities."""
        agent = TestAgent()
        
        assert agent.has_capability(AgentCapability.DATA_QUALITY_ASSESSMENT)
        assert agent.has_capability(AgentCapability.ANOMALY_DETECTION)
        assert not agent.has_capability(AgentCapability.COMPLIANCE_CHECK)
        assert not agent.has_capability(AgentCapability.POLICY_GENERATION)
    
    def test_add_capability(self):
        """Test adding capabilities to an agent."""
        agent = TestAgent()
        original_count = len(agent.capabilities)
        
        # Add new capability
        agent.add_capability(AgentCapability.COMPLIANCE_CHECK)
        assert len(agent.capabilities) == original_count + 1
        assert agent.has_capability(AgentCapability.COMPLIANCE_CHECK)
        
        # Adding same capability again should not duplicate
        agent.add_capability(AgentCapability.COMPLIANCE_CHECK)
        assert len(agent.capabilities) == original_count + 1
    
    def test_remove_capability(self):
        """Test removing capabilities from an agent."""
        agent = TestAgent()
        original_count = len(agent.capabilities)
        
        # Remove existing capability
        agent.remove_capability(AgentCapability.DATA_QUALITY_ASSESSMENT)
        assert len(agent.capabilities) == original_count - 1
        assert not agent.has_capability(AgentCapability.DATA_QUALITY_ASSESSMENT)
        
        # Removing non-existent capability should not error
        agent.remove_capability(AgentCapability.POLICY_GENERATION)
        assert len(agent.capabilities) == original_count - 1
    
    def test_repr(self):
        """Test string representation of agent."""
        agent = TestAgent("test_agent_repr")
        
        repr_str = repr(agent)
        assert "TestAgent" in repr_str
        assert "test_agent_repr" in repr_str
        assert str(len(agent.capabilities)) in repr_str
    
    def test_agent_equality(self):
        """Test agent equality based on agent_id."""
        agent1 = TestAgent("same_id")
        agent2 = TestAgent("same_id")
        agent3 = TestAgent("different_id")
        
        assert agent1 == agent2
        assert agent1 != agent3
        assert agent2 != agent3
    
    def test_agent_hash(self):
        """Test agent hashing for use in sets/dicts."""
        agent1 = TestAgent("agent_1")
        agent2 = TestAgent("agent_2")
        agent3 = TestAgent("agent_1")  # Same ID as agent1
        
        agent_set = {agent1, agent2, agent3}
        assert len(agent_set) == 2  # agent1 and agent3 should be the same
        
        agent_dict = {agent1: "value1", agent2: "value2"}
        assert len(agent_dict) == 2


class TestAbstractMethods:
    """Test that abstract methods are properly enforced."""
    
    def test_abstract_base_cannot_be_instantiated(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent("test", [AgentCapability.DATA_QUALITY_ASSESSMENT])
    
    def test_missing_process_message_implementation(self):
        """Test that missing process_message implementation raises TypeError."""
        class IncompleteAgent(BaseAgent):
            def __init__(self):
                super().__init__("incomplete", [AgentCapability.DATA_QUALITY_ASSESSMENT])
            
            async def execute_capability(self, capability, parameters=None):
                return {}
        
        with pytest.raises(TypeError):
            IncompleteAgent()
    
    def test_missing_execute_capability_implementation(self):
        """Test that missing execute_capability implementation raises TypeError."""
        class IncompleteAgent(BaseAgent):
            def __init__(self):
                super().__init__("incomplete", [AgentCapability.DATA_QUALITY_ASSESSMENT])
            
            async def process_message(self, message, context=None):
                return {}
        
        with pytest.raises(TypeError):
            IncompleteAgent()
