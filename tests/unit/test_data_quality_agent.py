"""
Unit tests for agents.data_quality.agent module.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from agents.data_quality.agent import DataQualityAgent
from core.types.agent_types import AgentCapability, AgentStatus


class TestDataQualityAgent:
    """Test the DataQualityAgent class."""
    
    def test_initialization(self):
        """Test DataQualityAgent initialization."""
        agent = DataQualityAgent()
        
        assert agent.agent_id == "data_quality_agent"
        assert agent.status == AgentStatus.IDLE
        assert len(agent.capabilities) == 3
        assert AgentCapability.DATA_QUALITY_ASSESSMENT in agent.capabilities
        assert AgentCapability.ANOMALY_DETECTION in agent.capabilities
        assert AgentCapability.DATA_PROFILING in agent.capabilities
    
    def test_initialization_with_custom_id(self):
        """Test DataQualityAgent initialization with custom ID."""
        agent = DataQualityAgent(agent_id="custom_dq_agent")
        
        assert agent.agent_id == "custom_dq_agent"
        assert agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test process_message method."""
        agent = DataQualityAgent()
        
        result = await agent.process_message(
            "Analyze data quality for dataset_123",
            {"dataset_id": "dataset_123"}
        )
        
        assert "response" in result
        assert "analysis" in result
        assert result["dataset_id"] == "dataset_123"
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_process_message_no_context(self):
        """Test process_message without context."""
        agent = DataQualityAgent()
        
        result = await agent.process_message("Check data quality")
        
        assert "response" in result
        assert "analysis" in result
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_capability_data_quality_assessment(self):
        """Test execute_capability for DATA_QUALITY_ASSESSMENT."""
        agent = DataQualityAgent()
        
        parameters = {
            "dataset_id": "test_dataset",
            "dimensions": ["completeness", "accuracy"]
        }
        
        result = await agent.execute_capability(
            AgentCapability.DATA_QUALITY_ASSESSMENT,
            parameters
        )
        
        assert result["capability"] == "data_quality_assessment"
        assert result["dataset_id"] == "test_dataset"
        assert "quality_score" in result
        assert "dimensions_assessed" in result
        assert "issues_found" in result
        assert isinstance(result["quality_score"], (int, float))
        assert result["quality_score"] >= 0
        assert result["quality_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_execute_capability_anomaly_detection(self):
        """Test execute_capability for ANOMALY_DETECTION."""
        agent = DataQualityAgent()
        
        parameters = {
            "dataset_id": "test_dataset",
            "threshold": 0.95
        }
        
        result = await agent.execute_capability(
            AgentCapability.ANOMALY_DETECTION,
            parameters
        )
        
        assert result["capability"] == "anomaly_detection"
        assert result["dataset_id"] == "test_dataset"
        assert "anomalies_detected" in result
        assert "threshold_used" in result
        assert "confidence_scores" in result
        assert isinstance(result["anomalies_detected"], list)
        assert result["threshold_used"] == 0.95
    
    @pytest.mark.asyncio
    async def test_execute_capability_data_profiling(self):
        """Test execute_capability for DATA_PROFILING."""
        agent = DataQualityAgent()
        
        parameters = {
            "dataset_id": "test_dataset",
            "include_statistics": True
        }
        
        result = await agent.execute_capability(
            AgentCapability.DATA_PROFILING,
            parameters
        )
        
        assert result["capability"] == "data_profiling"
        assert result["dataset_id"] == "test_dataset"
        assert "profile" in result
        assert "statistics" in result
        assert "schema_info" in result
        
        # Check profile structure
        profile = result["profile"]
        assert "total_rows" in profile
        assert "total_columns" in profile
        assert "data_types" in profile
        assert isinstance(profile["total_rows"], int)
        assert isinstance(profile["total_columns"], int)
    
    @pytest.mark.asyncio
    async def test_execute_capability_unsupported(self):
        """Test execute_capability with unsupported capability."""
        agent = DataQualityAgent()
        
        result = await agent.execute_capability(
            AgentCapability.COMPLIANCE_CHECK,  # Not supported by this agent
            {"test": "data"}
        )
        
        assert result["capability"] == "compliance_check"
        assert result["error"] == "Capability not supported by this agent"
        assert result["supported_capabilities"] == [
            "data_quality_assessment",
            "anomaly_detection", 
            "data_profiling"
        ]
    
    @pytest.mark.asyncio
    async def test_execute_capability_no_parameters(self):
        """Test execute_capability without parameters."""
        agent = DataQualityAgent()
        
        result = await agent.execute_capability(AgentCapability.DATA_QUALITY_ASSESSMENT)
        
        assert result["capability"] == "data_quality_assessment"
        assert "quality_score" in result
        # Should use default dataset_id when none provided
        assert "dataset_id" in result
    
    def test_domain_knowledge_initialization(self):
        """Test that domain knowledge is properly initialized."""
        agent = DataQualityAgent()
        
        # Check that domain knowledge was set
        assert hasattr(agent, '_domain_knowledge')
        assert agent._domain_knowledge is not None
        assert len(agent._domain_knowledge) > 0
        
        # Check expected knowledge categories
        knowledge_text = agent._domain_knowledge
        assert "data quality" in knowledge_text.lower()
        assert "anomaly detection" in knowledge_text.lower()
        assert "data profiling" in knowledge_text.lower()
    
    def test_agent_inheritance(self):
        """Test that DataQualityAgent properly inherits from BaseAgent."""
        agent = DataQualityAgent()
        
        # Should have BaseAgent methods
        assert hasattr(agent, 'add_capability')
        assert hasattr(agent, 'remove_capability')
        assert hasattr(agent, 'has_capability')
        
        # Should be able to add/remove capabilities
        initial_count = len(agent.capabilities)
        agent.add_capability(AgentCapability.COMPLIANCE_CHECK)
        assert len(agent.capabilities) == initial_count + 1
        assert agent.has_capability(AgentCapability.COMPLIANCE_CHECK)
        
        agent.remove_capability(AgentCapability.COMPLIANCE_CHECK)
        assert len(agent.capabilities) == initial_count
        assert not agent.has_capability(AgentCapability.COMPLIANCE_CHECK)
    
    def test_agent_string_representation(self):
        """Test string representation of the agent."""
        agent = DataQualityAgent()
        
        repr_str = repr(agent)
        assert "DataQualityAgent" in repr_str
        assert agent.agent_id in repr_str
        assert str(len(agent.capabilities)) in repr_str
    
    @pytest.mark.asyncio
    async def test_multiple_capability_executions(self):
        """Test executing multiple capabilities in sequence."""
        agent = DataQualityAgent()
        
        # Execute data profiling first
        profile_result = await agent.execute_capability(
            AgentCapability.DATA_PROFILING,
            {"dataset_id": "test_seq"}
        )
        assert profile_result["capability"] == "data_profiling"
        
        # Execute anomaly detection
        anomaly_result = await agent.execute_capability(
            AgentCapability.ANOMALY_DETECTION,
            {"dataset_id": "test_seq"}
        )
        assert anomaly_result["capability"] == "anomaly_detection"
        
        # Execute quality assessment
        quality_result = await agent.execute_capability(
            AgentCapability.DATA_QUALITY_ASSESSMENT,
            {"dataset_id": "test_seq"}
        )
        assert quality_result["capability"] == "data_quality_assessment"
        
        # All should have same dataset_id
        assert profile_result["dataset_id"] == "test_seq"
        assert anomaly_result["dataset_id"] == "test_seq"
        assert quality_result["dataset_id"] == "test_seq"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_capabilities(self):
        """Test error handling within capability execution."""
        agent = DataQualityAgent()
        
        # Test with invalid parameters that might cause issues
        result = await agent.execute_capability(
            AgentCapability.ANOMALY_DETECTION,
            {"threshold": "invalid_threshold"}  # Invalid type
        )
        
        # Should handle gracefully and return result
        assert "capability" in result
        assert result["capability"] == "anomaly_detection"
        # Should still provide some result even with invalid input
    
    @pytest.mark.asyncio
    async def test_concurrent_capability_execution(self):
        """Test that agent can handle concurrent capability executions."""
        import asyncio
        
        agent = DataQualityAgent()
        
        # Create multiple concurrent tasks
        tasks = [
            agent.execute_capability(AgentCapability.DATA_QUALITY_ASSESSMENT, {"dataset_id": f"dataset_{i}"})
            for i in range(3)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["capability"] == "data_quality_assessment"
            assert result["dataset_id"] == f"dataset_{i}"
    
    def test_agent_capabilities_immutable(self):
        """Test that agent capabilities list is properly protected."""
        agent = DataQualityAgent()
        
        original_capabilities = agent.capabilities.copy()
        
        # Try to modify the returned capabilities list
        capabilities_list = agent.capabilities
        capabilities_list.append(AgentCapability.COMPLIANCE_CHECK)
        
        # Original capabilities should be unchanged
        assert agent.capabilities == original_capabilities
        assert AgentCapability.COMPLIANCE_CHECK not in agent.capabilities
