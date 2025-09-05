"""
Configuration management for the EDGP AI Model service.
Handles environment variables and application settings.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra='ignore'  # Allow extra fields to be ignored
    )
    
    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    # AWS Bedrock Configuration
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_profile: Optional[str] = Field(default=None, alias="AWS_PROFILE")
    bedrock_model_id: str = Field(default="anthropic.claude-3-sonnet-20240229-v1:0", alias="BEDROCK_MODEL_ID")
    
    default_llm_provider: str = Field(default="bedrock", alias="DEFAULT_LLM_PROVIDER")
    default_model: str = Field(default="gpt-4", alias="DEFAULT_MODEL")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./edgp.db", alias="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    
    # Vector Store Configuration
    vector_store_type: str = Field(default="chroma", alias="VECTOR_STORE_TYPE")
    chroma_persist_directory: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIRECTORY")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    
    # Agent Configuration
    max_concurrent_agents: int = Field(default=10, alias="MAX_CONCURRENT_AGENTS")
    agent_timeout: int = Field(default=300, alias="AGENT_TIMEOUT")
    orchestration_mode: str = Field(default="langgraph", alias="ORCHESTRATION_MODE")
    
    # MCP Configuration
    mcp_server_host: str = Field(default="localhost", alias="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=9000, alias="MCP_SERVER_PORT")
    mcp_protocol_version: str = Field(default="1.0", alias="MCP_PROTOCOL_VERSION")
    
    # RAG Configuration
    rag_chunk_size: int = Field(default=1000, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_similarity_threshold: float = Field(default=0.7, alias="RAG_SIMILARITY_THRESHOLD")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    enable_tracing: bool = Field(default=True, alias="ENABLE_TRACING")
    
    # External Service Integration
    data_validator_url: str = Field(default="http://localhost:8001", alias="DATA_VALIDATOR_URL")
    data_validator_api_key: str = Field(default="your_data_validator_api_key_here", alias="DATA_VALIDATOR_API_KEY")
    regulatory_service_url: str = Field(default="http://localhost:8002", alias="REGULATORY_SERVICE_URL")
    regulatory_service_token: str = Field(default="your_regulatory_service_token_here", alias="REGULATORY_SERVICE_TOKEN")
    automation_engine_url: str = Field(default="http://localhost:8003", alias="AUTOMATION_ENGINE_URL")
    automation_engine_api_key: str = Field(default="your_automation_engine_api_key_here", alias="AUTOMATION_ENGINE_API_KEY")
    bi_service_url: str = Field(default="http://localhost:8004", alias="BI_SERVICE_URL")
    bi_service_client_id: str = Field(default="your_bi_service_client_id_here", alias="BI_SERVICE_CLIENT_ID")
    bi_service_client_secret: str = Field(default="your_bi_service_client_secret_here", alias="BI_SERVICE_CLIENT_SECRET")
    governance_platform_url: str = Field(default="http://localhost:8005", alias="GOVERNANCE_PLATFORM_URL")
    governance_cert_path: str = Field(default="/path/to/client.crt", alias="GOVERNANCE_CERT_PATH")
    governance_key_path: str = Field(default="/path/to/client.key", alias="GOVERNANCE_KEY_PATH")
    notification_service_url: str = Field(default="http://localhost:8006", alias="NOTIFICATION_SERVICE_URL")
    notification_service_api_key: str = Field(default="your_notification_service_api_key_here", alias="NOTIFICATION_SERVICE_API_KEY")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance - only created when needed
def get_global_settings() -> Settings:
    """Get global settings instance."""
    return get_settings()


# For backward compatibility
settings = get_global_settings()
