"""
Knowledge Base Package

Comprehensive knowledge management system for AI agents including entity
management, relationship tracking, knowledge extraction, and inference.
"""

from .types import (
    # Core data models
    Entity,
    Relationship,
    KnowledgeFact,
    KnowledgeGraph,
    OntologyClass,
    
    # Enums
    EntityType,
    RelationshipType,
    FactType,
    KnowledgeSource,
    
    # Query and result types
    KnowledgeQuery,
    KnowledgeSearchResult,
    
    # Configuration
    KnowledgeBaseConfig,
    KnowledgeBaseAnalytics,
    
    # Base classes
    BaseKnowledgeStore,
    BaseKnowledgeExtractor,
    BaseInferenceEngine
)

from .manager import KnowledgeBaseManager
from .stores import (
    KnowledgeStoreManager,
    SQLiteKnowledgeStore,
    InMemoryKnowledgeStore
)
from .extraction import (
    KnowledgeExtractionManager,
    RegexKnowledgeExtractor,
    NLPKnowledgeExtractor,
    LLMKnowledgeExtractor
)
from .inference import (
    InferenceEngineManager,
    RuleBasedInferenceEngine,
    GraphBasedInferenceEngine,
    MLBasedInferenceEngine
)

__all__ = [
    # Main manager
    "KnowledgeBaseManager",
    
    # Data models
    "Entity",
    "Relationship", 
    "KnowledgeFact",
    "KnowledgeGraph",
    "OntologyClass",
    
    # Enums
    "EntityType",
    "RelationshipType", 
    "FactType",
    "KnowledgeSource",
    
    # Query types
    "KnowledgeQuery",
    "KnowledgeSearchResult",
    
    # Configuration
    "KnowledgeBaseConfig",
    "KnowledgeBaseAnalytics",
    
    # Storage
    "KnowledgeStoreManager",
    "SQLiteKnowledgeStore",
    "InMemoryKnowledgeStore",
    
    # Extraction
    "KnowledgeExtractionManager",
    "RegexKnowledgeExtractor",
    "NLPKnowledgeExtractor", 
    "LLMKnowledgeExtractor",
    
    # Inference
    "InferenceEngineManager",
    "RuleBasedInferenceEngine",
    "GraphBasedInferenceEngine",
    "MLBasedInferenceEngine",
    
    # Base classes
    "BaseKnowledgeStore",
    "BaseKnowledgeExtractor",
    "BaseInferenceEngine"
]


def create_knowledge_manager(
    storage_path: str = "./data/knowledge",
    storage_backend: str = "sqlite",
    enable_embeddings: bool = True,
    enable_auto_extraction: bool = True,
    enable_inference: bool = True,
    enable_validation: bool = True,
    **kwargs
) -> KnowledgeBaseManager:
    """
    Create a knowledge base manager with default configuration.
    
    Args:
        storage_path: Path for knowledge storage
        storage_backend: Storage backend ("sqlite", "memory")
        enable_embeddings: Enable embedding generation
        enable_auto_extraction: Enable automatic knowledge extraction
        enable_inference: Enable knowledge inference
        enable_validation: Enable consistency validation
        **kwargs: Additional configuration options
        
    Returns:
        Configured KnowledgeBaseManager instance
    """
    config = KnowledgeBaseConfig(
        storage_path=storage_path,
        storage_backend=storage_backend,
        enable_embeddings=enable_embeddings,
        enable_auto_extraction=enable_auto_extraction,
        enable_inference=enable_inference,
        enable_validation=enable_validation,
        **kwargs
    )
    
    return KnowledgeBaseManager(config)


def create_default_config() -> KnowledgeBaseConfig:
    """Create default knowledge base configuration."""
    return KnowledgeBaseConfig(
        storage_path="./data/knowledge",
        storage_backend="sqlite",
        enable_embeddings=True,
        enable_auto_extraction=True,
        enable_inference=True,
        enable_validation=True,
        enable_rule_inference=True,
        enable_graph_inference=True,
        enable_ml_inference=False,
        enable_nlp_extraction=True,
        enable_llm_extraction=False,
        max_entities=10000,
        max_relationships=50000,
        max_facts=25000,
        inference_confidence_threshold=0.7,
        extraction_confidence_threshold=0.6
    )
