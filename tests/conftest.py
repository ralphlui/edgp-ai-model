"""
Pytest configuration and fixtures for the EDGP AI Model tests.
"""
import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agents.base import BaseAgent
from core.types.agent_types import AgentCapability


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('core.infrastructure.config.get_settings') as mock_get_settings:
        mock_config = Mock()
        mock_config.environment = "test"
        mock_config.debug = True
        mock_config.api_host = "127.0.0.1"
        mock_config.api_port = 8000
        mock_config.api_prefix = "/api/v1"
        mock_config.log_level = "DEBUG"
        mock_config.vector_store_type = "chroma"
        mock_config.chroma_persist_directory = "./test_data/chroma"
        mock_config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.rag_chunk_size = 1000
        mock_config.rag_chunk_overlap = 200
        mock_config.rag_top_k = 5
        mock_config.max_concurrent_agents = 5
        mock_config.agent_timeout = 30
        mock_config.is_development = True
        mock_config.is_production = False
        mock_get_settings.return_value = mock_config
        yield mock_config


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""
    
    def __init__(self, agent_id: str = "test_agent", capabilities: list = None):
        capabilities = capabilities or [AgentCapability.DATA_QUALITY_ASSESSMENT]
        
        from core.types.base import AgentType
        super().__init__(
            agent_type=AgentType.DATA_QUALITY,
            name=agent_id,
            description="Test agent for unit testing",
            capabilities=capabilities
        )
        
        # Add agent_id property for test compatibility
        self.agent_id = agent_id
    
    async def execute_capability(self, capability: AgentCapability, parameters: dict = None):
        """Mock implementation of execute_capability."""
        return {"test": "result"}
    
    async def process_message(self, message, context=None):
        """Mock implementation of process_message."""
        return {"response": "Mock response"}
    
    async def process_task(self, task):
        """Mock implementation of process_task."""
        return {"task_result": "Mock task result"}
    
    async def handle_message(self, message):
        """Mock implementation of handle_message."""
        return {"handled": True}


@pytest.fixture
def mock_llm_gateway():
    """Mock LLM gateway for testing."""
    with patch('core.services.llm_gateway.LLMGatewayBridge') as mock_llm:
        mock_instance = Mock()
        mock_instance.generate_response.return_value = {
            "response": "Mock LLM response",
            "usage": {"tokens": 100}
        }
        mock_llm.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing."""
    with patch('core.services.rag_system.RAGSystem') as mock_rag:
        mock_instance = Mock()
        mock_instance.search.return_value = [
            {
                "document": "Mock document",
                "metadata": {"source": "test"},
                "score": 0.9
            }
        ]
        mock_rag.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    with patch('core.services.rag_system.ChromaVectorStore') as mock_store:
        mock_instance = Mock()
        mock_instance.add_documents.return_value = None
        mock_instance.search.return_value = []
        mock_store.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def setup_test_environment(mock_settings):
    """Automatically set up test environment for all tests."""
    # Set test environment variables
    os.environ['ENVIRONMENT'] = 'test'
    os.environ['DEBUG'] = 'true'
    
    yield
    
    # Cleanup
    if 'ENVIRONMENT' in os.environ:
        del os.environ['ENVIRONMENT']
    if 'DEBUG' in os.environ:
        del os.environ['DEBUG']


@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        "agent_id": "test_agent",
        "capabilities": ["data_quality_assessment", "anomaly_detection"],
        "status": "idle",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "message": "Test message for agent processing",
        "context": {
            "dataset_id": "test_dataset_123",
            "user_id": "test_user"
        }
    }


@pytest.fixture
def sample_capability_data():
    """Sample capability execution data for testing."""
    return {
        "capability": "data_quality_assessment",
        "parameters": {
            "dataset_id": "test_dataset",
            "dimensions": ["completeness", "accuracy"]
        }
    }
