"""
Production Configuration Example for LangChain/LangGraph Integration

This shows how to configure the collaborative AI platform with
LangChain/LangGraph integration for production deployment.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class LangChainConfig:
    """Configuration for LangChain integration."""
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # LangGraph Configuration
    workflow_timeout: int = 300  # seconds
    max_parallel_agents: int = 5
    workflow_retry_attempts: int = 3
    state_persistence: bool = True
    
    # Tool Configuration
    enable_shared_tools: bool = True
    tool_timeout: int = 30
    max_tool_calls_per_agent: int = 10
    
    # Callback Configuration
    enable_monitoring: bool = True
    enable_performance_tracking: bool = True
    log_level: str = "INFO"


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    
    # Environment
    environment: str = "production"
    debug: bool = False
    
    # Database Configuration
    database_url: str = "postgresql://user:pass@localhost/edgp_ai"
    redis_url: str = "redis://localhost:6379"
    
    # Shared Services Configuration
    prompt_config: Dict[str, Any] = None
    rag_config: Dict[str, Any] = None
    memory_config: Dict[str, Any] = None
    knowledge_config: Dict[str, Any] = None
    context_config: Dict[str, Any] = None
    
    # LangChain Configuration
    langchain_config: LangChainConfig = None
    
    # Authentication & Security
    api_key_required: bool = True
    jwt_secret: str = "your-secret-key"
    cors_origins: List[str] = None
    rate_limit: int = 100  # requests per minute
    
    # Monitoring & Logging
    log_level: str = "INFO"
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    
    # Performance
    max_concurrent_requests: int = 50
    request_timeout: int = 300
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.prompt_config is None:
            self.prompt_config = {
                "storage_type": "postgresql",
                "storage_path": self.database_url,
                "cache_enabled": True,
                "template_validation": True
            }
        
        if self.rag_config is None:
            self.rag_config = {
                "storage_type": "postgresql",
                "storage_path": self.database_url,
                "vector_store": "pgvector",
                "embedding_model": "text-embedding-ada-002",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "search_k": 5
            }
        
        if self.memory_config is None:
            self.memory_config = {
                "storage_type": "postgresql",
                "storage_path": self.database_url,
                "consolidation_enabled": True,
                "max_memories_per_user": 10000,
                "auto_cleanup": True
            }
        
        if self.knowledge_config is None:
            self.knowledge_config = {
                "storage_type": "postgresql", 
                "storage_path": self.database_url,
                "graph_enabled": True,
                "inference_enabled": True,
                "versioning_enabled": True
            }
        
        if self.context_config is None:
            self.context_config = {
                "storage_type": "redis",
                "storage_path": self.redis_url,
                "session_timeout": 3600,  # 1 hour
                "max_context_size": 50000,
                "compression_enabled": True
            }
        
        if self.langchain_config is None:
            self.langchain_config = LangChainConfig()
        
        if self.cors_origins is None:
            self.cors_origins = ["https://yourdomain.com"]


def get_production_config() -> ProductionConfig:
    """Get production configuration from environment variables."""
    
    return ProductionConfig(
        # Environment settings
        environment=os.getenv("ENVIRONMENT", "production"),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        
        # Database configuration
        database_url=os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/edgp_ai"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        
        # LangChain configuration
        langchain_config=LangChainConfig(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4"),
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            workflow_timeout=int(os.getenv("WORKFLOW_TIMEOUT", "300")),
            max_parallel_agents=int(os.getenv("MAX_PARALLEL_AGENTS", "5")),
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        ),
        
        # Security settings
        api_key_required=os.getenv("API_KEY_REQUIRED", "true").lower() == "true",
        jwt_secret=os.getenv("JWT_SECRET", "your-secret-key"),
        cors_origins=os.getenv("CORS_ORIGINS", "https://yourdomain.com").split(","),
        rate_limit=int(os.getenv("RATE_LIMIT", "100")),
        
        # Performance settings
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
        
        # Monitoring settings
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
        tracing_enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true"
    )


def get_development_config() -> ProductionConfig:
    """Get development configuration."""
    
    return ProductionConfig(
        # Environment settings
        environment="development",
        debug=True,
        
        # Database configuration (use local/in-memory for development)
        database_url="sqlite:///./dev_edgp_ai.db",
        redis_url="redis://localhost:6379",
        
        # Override shared services for development
        prompt_config={
            "storage_type": "sqlite",
            "storage_path": "./dev_prompts.db",
            "cache_enabled": False
        },
        rag_config={
            "storage_type": "sqlite",
            "storage_path": "./dev_rag.db",
            "vector_store": "chroma",
            "embedding_model": "text-embedding-ada-002",
            "chunk_size": 500,
            "search_k": 3
        },
        memory_config={
            "storage_type": "sqlite",
            "storage_path": "./dev_memory.db",
            "consolidation_enabled": False,
            "auto_cleanup": False
        },
        knowledge_config={
            "storage_type": "sqlite",
            "storage_path": "./dev_knowledge.db",
            "inference_enabled": False,
            "versioning_enabled": False
        },
        context_config={
            "storage_type": "memory",
            "session_timeout": 1800,  # 30 minutes
            "max_context_size": 10000
        },
        
        # LangChain configuration for development
        langchain_config=LangChainConfig(
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.5,
            llm_max_tokens=1000,
            workflow_timeout=120,
            max_parallel_agents=3,
            enable_monitoring=True
        ),
        
        # Development security (more relaxed)
        api_key_required=False,
        cors_origins=["http://localhost:3000", "http://localhost:8000"],
        rate_limit=1000,
        
        # Development performance
        max_concurrent_requests=10,
        request_timeout=120,
        
        # Development monitoring
        log_level="DEBUG",
        metrics_enabled=True,
        tracing_enabled=True
    )


def get_testing_config() -> ProductionConfig:
    """Get testing configuration."""
    
    return ProductionConfig(
        # Environment settings
        environment="testing",
        debug=True,
        
        # In-memory databases for testing
        database_url="sqlite:///:memory:",
        redis_url="redis://localhost:6379",
        
        # Override all shared services for testing
        prompt_config={
            "storage_type": "memory",
            "cache_enabled": False
        },
        rag_config={
            "storage_type": "memory",
            "vector_store": "memory",
            "search_k": 2
        },
        memory_config={
            "storage_type": "memory",
            "consolidation_enabled": False
        },
        knowledge_config={
            "storage_type": "memory",
            "inference_enabled": False
        },
        context_config={
            "storage_type": "memory",
            "session_timeout": 300  # 5 minutes
        },
        
        # Testing LangChain configuration
        langchain_config=LangChainConfig(
            llm_provider="mock",  # Use mock LLM for testing
            llm_model="mock-model",
            workflow_timeout=30,
            max_parallel_agents=2,
            enable_monitoring=False
        ),
        
        # Testing security
        api_key_required=False,
        cors_origins=["*"],
        rate_limit=10000,
        
        # Testing performance
        max_concurrent_requests=5,
        request_timeout=30,
        
        # Testing monitoring
        log_level="DEBUG",
        metrics_enabled=False,
        tracing_enabled=False
    )


# Example environment configuration files

DOCKER_COMPOSE_EXAMPLE = """
# docker-compose.yml for production deployment
version: '3.8'

services:
  edgp-ai-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/edgp_ai
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_MODEL=gpt-4
      - JWT_SECRET=${JWT_SECRET}
      - CORS_ORIGINS=https://yourdomain.com
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs

  db:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_DB=edgp_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
"""

KUBERNETES_EXAMPLE = """
# kubernetes-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edgp-ai-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: edgp-ai
  template:
    metadata:
      labels:
        app: edgp-ai
    spec:
      containers:
      - name: edgp-ai
        image: edgp-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: edgp-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: edgp-secrets
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: edgp-secrets
              key: openai-api-key
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: edgp-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: edgp-ai-service
spec:
  selector:
    app: edgp-ai
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""

ENV_EXAMPLE = """
# .env.production
ENVIRONMENT=production
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/edgp_ai
REDIS_URL=redis://localhost:6379

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your-openai-api-key
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Workflow Configuration
WORKFLOW_TIMEOUT=300
MAX_PARALLEL_AGENTS=5
ENABLE_MONITORING=true

# Security Configuration
API_KEY_REQUIRED=true
JWT_SECRET=your-very-secure-jwt-secret
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
RATE_LIMIT=100

# Performance Configuration
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=300

# Monitoring Configuration
LOG_LEVEL=INFO
METRICS_ENABLED=true
TRACING_ENABLED=true
"""

if __name__ == "__main__":
    # Example usage
    import json
    
    print("=== Production Configuration ===")
    prod_config = get_production_config()
    print(f"Environment: {prod_config.environment}")
    print(f"LLM Model: {prod_config.langchain_config.llm_model}")
    print(f"Max Parallel Agents: {prod_config.langchain_config.max_parallel_agents}")
    
    print("\n=== Development Configuration ===")
    dev_config = get_development_config()
    print(f"Environment: {dev_config.environment}")
    print(f"Debug Mode: {dev_config.debug}")
    print(f"Database: {dev_config.database_url}")
    
    print("\n=== Testing Configuration ===")
    test_config = get_testing_config()
    print(f"Environment: {test_config.environment}")
    print(f"LLM Provider: {test_config.langchain_config.llm_provider}")
    print(f"Session Timeout: {test_config.context_config['session_timeout']}")
