"""
Unit tests for core.infrastructure.config module.
"""
import os
import pytest
from unittest.mock import patch
from core.infrastructure.config import Settings, get_settings


class TestSettings:
    """Test the Settings class."""
    
    def test_default_values(self):
        """Test that Settings class loads correctly with expected structure."""
        settings = Settings()
        
        # Environment
        assert settings.environment in ["development", "test", "production"]
        assert isinstance(settings.debug, bool)
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        # API Configuration
        assert settings.api_host is not None
        assert isinstance(settings.api_port, int)
        assert settings.api_port > 0
        assert settings.api_prefix.startswith("/")
        
        # LLM Configuration - can be None or string
        assert settings.openai_api_key is None or isinstance(settings.openai_api_key, str)
        assert settings.anthropic_api_key is None or isinstance(settings.anthropic_api_key, str)
        # Default configuration
        assert settings.default_llm_provider in ["openai", "anthropic", "bedrock"]
        assert settings.default_model is not None
        
        # AWS Configuration
        assert settings.aws_region == "us-east-1"
        assert settings.bedrock_model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        
        # Database Configuration
        assert settings.database_url == "sqlite:///./edgp.db"
        assert settings.redis_url == "redis://localhost:6379/0"
        
        # Vector Store Configuration
        assert settings.vector_store_type == "chroma"
        assert settings.chroma_persist_directory == "./data/chroma"
        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        
        # Agent Configuration
        assert settings.max_concurrent_agents == 10
        assert settings.agent_timeout == 300
        assert settings.orchestration_mode == "langgraph"
        
        # MCP Configuration
        assert settings.mcp_server_host == "localhost"
        assert settings.mcp_server_port == 9000
        assert settings.mcp_protocol_version == "1.0"
        
        # RAG Configuration
        assert settings.rag_chunk_size == 1000
        assert settings.rag_chunk_overlap == 200
        assert settings.rag_top_k == 5
        assert settings.rag_similarity_threshold == 0.7
        
        # Monitoring
        assert settings.enable_metrics is True
        assert settings.metrics_port == 9090
        assert settings.enable_tracing is True
    
    def test_environment_variables(self):
        """Test that Settings correctly reads environment variables."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'API_HOST': '127.0.0.1',
            'API_PORT': '9000',
            'OPENAI_API_KEY': 'test-key',
            'AWS_REGION': 'us-west-2',
            'MAX_CONCURRENT_AGENTS': '20'
        }):
            settings = Settings()
            
            assert settings.environment == "production"
            assert settings.debug is False
            assert settings.api_host == "127.0.0.1"
            assert settings.api_port == 9000
            assert settings.openai_api_key == "test-key"
            assert settings.aws_region == "us-west-2"
            assert settings.max_concurrent_agents == 20
    
    def test_is_development_property(self):
        """Test the is_development property."""
        # Development environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            settings = Settings()
            assert settings.is_development is True
            assert settings.is_production is False
        
        # Production environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            settings = Settings()
            assert settings.is_development is False
            assert settings.is_production is True
        
        # Test case insensitive
        with patch.dict(os.environ, {'ENVIRONMENT': 'DEVELOPMENT'}):
            settings = Settings()
            assert settings.is_development is True
    
    def test_is_production_property(self):
        """Test the is_production property."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            settings = Settings()
            assert settings.is_production is True
            assert settings.is_development is False
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'PRODUCTION'}):
            settings = Settings()
            assert settings.is_production is True
    
    def test_get_settings_caching(self):
        """Test that get_settings returns the same instance (caching)."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored due to extra='ignore'."""
        with patch.dict(os.environ, {'UNKNOWN_FIELD': 'some_value'}):
            # Should not raise an error
            settings = Settings()
            assert not hasattr(settings, 'unknown_field')
    
    def test_type_conversion(self):
        """Test that environment variables are properly converted to correct types."""
        with patch.dict(os.environ, {
            'DEBUG': 'true',
            'API_PORT': '8080',
            'RAG_SIMILARITY_THRESHOLD': '0.85',
            'ENABLE_METRICS': 'false'
        }):
            settings = Settings()
            
            assert isinstance(settings.debug, bool)
            assert settings.debug is True
            assert isinstance(settings.api_port, int)
            assert settings.api_port == 8080
            assert isinstance(settings.rag_similarity_threshold, float)
            assert settings.rag_similarity_threshold == 0.85
            assert isinstance(settings.enable_metrics, bool)
            assert settings.enable_metrics is False
    
    def test_external_service_urls(self):
        """Test external service URL configuration."""
        settings = Settings()
        
        # Check that all external service URLs have defaults
        assert settings.data_validator_url.startswith("http://")
        assert settings.regulatory_service_url.startswith("http://")
        assert settings.automation_engine_url.startswith("http://")
        assert settings.bi_service_url.startswith("http://")
        assert settings.governance_platform_url.startswith("http://")
        assert settings.notification_service_url.startswith("http://")
    
    def test_credential_fields(self):
        """Test that credential fields have default values."""
        settings = Settings()
        
        # These should have default placeholder values
        assert settings.data_validator_api_key is not None
        assert settings.regulatory_service_token is not None
        assert settings.automation_engine_api_key is not None
        assert settings.bi_service_client_id is not None
        assert settings.bi_service_client_secret is not None
        assert settings.governance_cert_path is not None
        assert settings.governance_key_path is not None
        assert settings.notification_service_api_key is not None
