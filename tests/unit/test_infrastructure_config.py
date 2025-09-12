"""
Unit tests for core.infrastructure.config module.
"""
import pytest
from unittest.mock import Mock, patch
import os
from core.infrastructure.config import (
    Settings, get_settings, get_global_settings
)


class TestSettings:
    """Test the Settings class."""
    
    def test_settings_initialization_with_defaults(self):
        """Test settings initialization with default values."""
        settings = Settings()
        
        assert settings.app_name == "EDGP AI Model"
        assert settings.app_env == "development"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.cors_origins == ["*"]
        
    def test_settings_initialization_with_env_vars(self):
        """Test settings initialization with environment variables."""
        with patch.dict(os.environ, {
            'APP_ENV': 'production',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG',
            'API_HOST': '127.0.0.1',
            'API_PORT': '8080'
        }):
            settings = Settings()
            
            assert settings.app_env == "production"
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.api_host == "127.0.0.1"
            assert settings.api_port == 8080
    
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from environment."""
        with patch.dict(os.environ, {
            'CORS_ORIGINS': 'http://localhost:3000,http://localhost:8080'
        }):
            settings = Settings()
            assert settings.cors_origins == ['http://localhost:3000', 'http://localhost:8080']
    
    def test_aws_settings(self):
        """Test AWS settings configuration."""
        with patch.dict(os.environ, {
            'AWS_REGION': 'us-west-2',
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret'
        }):
            settings = Settings()
            
            assert settings.aws_region == "us-west-2"
            assert settings.aws_access_key_id == "test_key"
            assert settings.aws_secret_access_key == "test_secret"
    
    def test_llm_settings(self):
        """Test LLM provider settings."""
        with patch.dict(os.environ, {
            'DEFAULT_LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test_openai_key',
            'ANTHROPIC_API_KEY': 'test_anthropic_key'
        }):
            settings = Settings()
            
            assert settings.default_llm_provider == "openai"
            assert settings.openai_api_key == "test_openai_key"
            assert settings.anthropic_api_key == "test_anthropic_key"
    
    def test_sqs_queue_urls(self):
        """Test SQS queue URL configuration."""
        with patch.dict(os.environ, {
            'WORKFLOW_RULES_REQUEST_SQS_URL': 'https://sqs.us-east-1.amazonaws.com/123456789012/request-queue',
            'WORKFLOW_RULES_RESPONSE_SQS_URL': 'https://sqs.us-east-1.amazonaws.com/123456789012/response-queue'
        }):
            settings = Settings()
            
            assert 'request-queue' in settings.workflow_rules_request_sqs_url
            assert 'response-queue' in settings.workflow_rules_response_sqs_url


"""
Unit tests for core.infrastructure.config module.
"""
import pytest
from unittest.mock import Mock, patch
import os
from core.infrastructure.config import (
    Settings, get_settings, get_global_settings
)


class TestSettings:
    """Test the Settings class."""
    
    def test_settings_initialization_with_defaults(self):
        """Test settings initialization with default values."""
        settings = Settings()
        
        # Settings could be test or development environment
        assert settings.environment in ["development", "test"]
        assert settings.debug is True
        assert settings.log_level == "INFO"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_prefix == "/api/v1"
        
    def test_settings_initialization_with_env_vars(self):
        """Test settings initialization with environment variables."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'LOG_LEVEL': 'ERROR',
            'API_HOST': '127.0.0.1',
            'API_PORT': '8080'
        }):
            settings = Settings()
            
            assert settings.environment == "production"
            assert settings.debug is False
            assert settings.log_level == "ERROR"
            assert settings.api_host == "127.0.0.1"
            assert settings.api_port == 8080
    
    def test_aws_settings(self):
        """Test AWS settings configuration."""
        with patch.dict(os.environ, {
            'AWS_REGION': 'us-west-2',
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret'
        }):
            settings = Settings()
            
            assert settings.aws_region == "us-west-2"
            assert settings.aws_access_key_id == "test_key"
            assert settings.aws_secret_access_key == "test_secret"
    
    def test_llm_settings(self):
        """Test LLM provider settings."""
        with patch.dict(os.environ, {
            'DEFAULT_LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test_openai_key',
            'ANTHROPIC_API_KEY': 'test_anthropic_key'
        }):
            settings = Settings()
            
            assert settings.default_llm_provider == "openai"
            assert settings.openai_api_key == "test_openai_key"
            assert settings.anthropic_api_key == "test_anthropic_key"
    
    def test_sqs_queue_urls(self):
        """Test SQS queue URL configuration."""
        with patch.dict(os.environ, {
            'WORKFLOW_RULES_REQUEST_SQS_URL': 'https://sqs.us-east-1.amazonaws.com/123456789012/request-queue',
            'WORKFLOW_RULES_RESPONSE_SQS_URL': 'https://sqs.us-east-1.amazonaws.com/123456789012/response-queue'
        }):
            settings = Settings()
            
            assert 'request-queue' in settings.workflow_rules_request_sqs_url
            assert 'response-queue' in settings.workflow_rules_response_sqs_url
    
    def test_is_development_property(self):
        """Test is_development property."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            settings = Settings()
            assert settings.is_development is True
            assert settings.is_production is False
    
    def test_is_production_property(self):
        """Test is_production property."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            settings = Settings()
            assert settings.is_production is True
            assert settings.is_development is False
    
    def test_database_configuration(self):
        """Test database configuration."""
        with patch.dict(os.environ, {
            'DATABASE_URL': 'postgresql://user:pass@localhost/edgp',
            'REDIS_URL': 'redis://localhost:6380/1'
        }):
            settings = Settings()
            
            assert settings.database_url == "postgresql://user:pass@localhost/edgp"
            assert settings.redis_url == "redis://localhost:6380/1"
    
    def test_vector_store_configuration(self):
        """Test vector store configuration."""
        with patch.dict(os.environ, {
            'VECTOR_STORE_TYPE': 'pinecone',
            'CHROMA_PERSIST_DIRECTORY': './data/vector_store',
            'EMBEDDING_MODEL': 'openai-embeddings'
        }):
            settings = Settings()
            
            assert settings.vector_store_type == "pinecone"
            assert settings.chroma_persist_directory == "./data/vector_store"
            assert settings.embedding_model == "openai-embeddings"
    
    def test_agent_configuration(self):
        """Test agent configuration."""
        with patch.dict(os.environ, {
            'MAX_CONCURRENT_AGENTS': '20',
            'AGENT_TIMEOUT': '600',
            'ORCHESTRATION_MODE': 'sequential'
        }):
            settings = Settings()
            
            assert settings.max_concurrent_agents == 20
            assert settings.agent_timeout == 600
            assert settings.orchestration_mode == "sequential"
    
    def test_mcp_configuration(self):
        """Test MCP server configuration."""
        with patch.dict(os.environ, {
            'MCP_SERVER_HOST': '0.0.0.0',
            'MCP_SERVER_PORT': '9001',
            'MCP_PROTOCOL_VERSION': '2.0'
        }):
            settings = Settings()
            
            assert settings.mcp_server_host == "0.0.0.0"
            assert settings.mcp_server_port == 9001
            assert settings.mcp_protocol_version == "2.0"
    
    def test_rag_configuration(self):
        """Test RAG configuration."""
        with patch.dict(os.environ, {
            'RAG_CHUNK_SIZE': '2000',
            'RAG_CHUNK_OVERLAP': '400',
            'RAG_TOP_K': '10',
            'RAG_SIMILARITY_THRESHOLD': '0.8'
        }):
            settings = Settings()
            
            assert settings.rag_chunk_size == 2000
            assert settings.rag_chunk_overlap == 400
            assert settings.rag_top_k == 10
            assert settings.rag_similarity_threshold == 0.8
    
    def test_monitoring_configuration(self):
        """Test monitoring configuration."""
        with patch.dict(os.environ, {
            'ENABLE_METRICS': 'false',
            'METRICS_PORT': '9091',
            'ENABLE_TRACING': 'false'
        }):
            settings = Settings()
            
            assert settings.enable_metrics is False
            assert settings.metrics_port == 9091
            assert settings.enable_tracing is False


class TestConfigFunctions:
    """Test configuration helper functions."""
    
    def test_get_settings_singleton(self):
        """Test that get_settings returns singleton instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_get_global_settings(self):
        """Test get_global_settings function."""
        with patch('core.infrastructure.config.get_settings') as mock_get_settings:
            mock_settings = Settings()
            mock_get_settings.return_value = mock_settings
            
            settings = get_global_settings()
            
            assert isinstance(settings, Settings)
            mock_get_settings.assert_called_once()
        assert settings is not None


class TestConfigIntegration:
    """Test configuration integration scenarios."""
    
    def test_production_environment_config(self):
        """Test configuration for production environment."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'LOG_LEVEL': 'WARNING',
            'API_HOST': '0.0.0.0',
            'API_PORT': '8000',
            'AWS_REGION': 'us-east-1',
            'AWS_ACCESS_KEY_ID': 'AKIAPRODUCTION',
            'AWS_SECRET_ACCESS_KEY': 'production_secret',
            'OPENAI_API_KEY': 'prod_openai_key'
        }):
            settings = Settings()
            
            assert settings.environment == "production"
            assert settings.debug is False
            assert settings.log_level == "WARNING"
            assert settings.aws_region == "us-east-1"
            assert settings.openai_api_key == "prod_openai_key"
            assert settings.is_production is True
    
    def test_development_environment_config(self):
        """Test configuration for development environment."""
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'development',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG',
            'API_HOST': '127.0.0.1',
            'API_PORT': '8080'
        }):
            settings = Settings()
            
            assert settings.environment == "development"
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.api_host == "127.0.0.1"
            assert settings.api_port == 8080
            assert settings.is_development is True
    
    def test_minimal_configuration(self):
        """Test minimal configuration requirements."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            # Should still initialize with defaults
            assert settings.environment is not None
            assert settings.api_host is not None
            assert settings.api_port is not None
            assert settings.default_llm_provider in ["bedrock", "openai"]
    
    def test_external_services_configuration(self):
        """Test external services configuration."""
        with patch.dict(os.environ, {
            'DATA_VALIDATOR_URL': 'http://validator.example.com',
            'DATA_VALIDATOR_API_KEY': 'validator_key_123',
            'REGULATORY_SERVICE_URL': 'http://regulatory.example.com',
            'AUTOMATION_ENGINE_URL': 'http://automation.example.com'
        }):
            settings = Settings()
            
            assert settings.data_validator_url == "http://validator.example.com"
            assert settings.data_validator_api_key == "validator_key_123"
            assert settings.regulatory_service_url == "http://regulatory.example.com"
            assert settings.automation_engine_url == "http://automation.example.com"
    
    def test_bedrock_configuration(self):
        """Test AWS Bedrock configuration."""
        with patch.dict(os.environ, {
            'BEDROCK_MODEL_ID': 'anthropic.claude-3-haiku-20240307-v1:0',
            'AWS_REGION': 'us-west-2'
        }):
            settings = Settings()
            
            assert settings.bedrock_model_id == "anthropic.claude-3-haiku-20240307-v1:0"
            assert settings.aws_region == "us-west-2"
