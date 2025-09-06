"""
Knowledge Base Management Types

Defines comprehensive types for knowledge base systems including
entities, relationships, ontologies, and knowledge graphs.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from pydantic import BaseModel, Field
import numpy as np


class EntityType(str, Enum):
    """Types of entities in the knowledge base."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    DOCUMENT = "document"
    PRODUCT = "product"
    SERVICE = "service"
    TECHNOLOGY = "technology"
    PROCESS = "process"
    CUSTOM = "custom"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    IS_A = "is_a"                      # Taxonomic
    PART_OF = "part_of"                # Composition
    RELATED_TO = "related_to"          # General association
    CAUSES = "causes"                  # Causal
    LOCATED_IN = "located_in"          # Spatial
    OCCURS_AT = "occurs_at"            # Temporal
    CREATED_BY = "created_by"          # Creation
    USED_BY = "used_by"                # Usage
    SIMILAR_TO = "similar_to"          # Similarity
    OPPOSITE_OF = "opposite_of"        # Opposition
    PREREQUISITE_FOR = "prerequisite_for"  # Dependency
    FOLLOWS = "follows"                # Sequence
    CONTAINS = "contains"              # Containment
    CUSTOM = "custom"


class ConfidenceLevel(float, Enum):
    """Confidence levels for knowledge assertions."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


class KnowledgeSource(str, Enum):
    """Sources of knowledge."""
    HUMAN_INPUT = "human_input"
    DOCUMENT_EXTRACTION = "document_extraction"
    API_IMPORT = "api_import"
    MACHINE_LEARNING = "machine_learning"
    INFERENCE = "inference"
    EXPERT_SYSTEM = "expert_system"
    CROWD_SOURCE = "crowd_source"
    SENSOR_DATA = "sensor_data"
    DATABASE_IMPORT = "database_import"


class Entity(BaseModel):
    """Represents an entity in the knowledge base."""
    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    
    # Properties and attributes
    properties: Dict[str, Any] = Field(default_factory=dict)
    aliases: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    # Metadata
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: KnowledgeSource = Field(default=KnowledgeSource.HUMAN_INPUT)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    # Embeddings for semantic search
    embedding: Optional[List[float]] = None
    
    # Validation and quality
    verified: bool = Field(default=False)
    verification_source: Optional[str] = None
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """Represents a relationship between entities."""
    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    
    # Relationship properties
    properties: Dict[str, Any] = Field(default_factory=dict)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Temporal aspects
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Metadata
    source: KnowledgeSource = Field(default=KnowledgeSource.HUMAN_INPUT)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    # Evidence supporting this relationship
    evidence: List[str] = Field(default_factory=list)  # References to evidence documents/facts
    
    # Quality metrics
    verified: bool = Field(default=False)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)


class KnowledgeFact(BaseModel):
    """Represents a fact or assertion in the knowledge base."""
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_entity_id: str
    predicate: str  # What is being asserted
    object_value: Union[str, int, float, bool]  # The value or object entity ID
    
    # Fact metadata
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: KnowledgeSource = Field(default=KnowledgeSource.HUMAN_INPUT)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    # Evidence and verification
    evidence: List[str] = Field(default_factory=list)
    verified: bool = Field(default=False)
    verification_source: Optional[str] = None


class OntologyClass(BaseModel):
    """Represents a class in an ontology."""
    class_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Hierarchical relationships
    parent_classes: List[str] = Field(default_factory=list)  # Parent class IDs
    child_classes: List[str] = Field(default_factory=list)   # Child class IDs
    
    # Properties and constraints
    properties: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    namespace: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class KnowledgeGraph(BaseModel):
    """Represents a knowledge graph structure."""
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Graph components
    entities: Dict[str, Entity] = Field(default_factory=dict)
    relationships: Dict[str, Relationship] = Field(default_factory=dict)
    facts: Dict[str, KnowledgeFact] = Field(default_factory=dict)
    
    # Ontology information
    ontology_classes: Dict[str, OntologyClass] = Field(default_factory=dict)
    namespaces: Dict[str, str] = Field(default_factory=dict)
    
    # Metadata
    version: str = Field(default="1.0")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    # Graph statistics
    entity_count: int = Field(default=0)
    relationship_count: int = Field(default=0)
    fact_count: int = Field(default=0)


class KnowledgeQuery(BaseModel):
    """Query for knowledge base search."""
    query_text: Optional[str] = None
    entity_types: List[EntityType] = Field(default_factory=list)
    relationship_types: List[RelationshipType] = Field(default_factory=list)
    
    # Entity filters
    entity_names: List[str] = Field(default_factory=list)
    entity_properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Relationship filters
    source_entity_id: Optional[str] = None
    target_entity_id: Optional[str] = None
    relationship_strength_min: float = Field(default=0.0)
    
    # Confidence and quality filters
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verified_only: bool = Field(default=False)
    
    # Result parameters
    limit: int = Field(default=10, ge=1)
    include_related: bool = Field(default=False)
    max_depth: int = Field(default=2)  # For traversal queries


class KnowledgeSearchResult(BaseModel):
    """Result from knowledge base search."""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    facts: List[KnowledgeFact] = Field(default_factory=list)
    
    # Result metadata
    scores: List[float] = Field(default_factory=list)
    total_found: int = Field(default=0)
    query_time: float = Field(default=0.0)
    search_strategy: str = Field(default="")


class KnowledgeExtractionRule(BaseModel):
    """Rule for extracting knowledge from text."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Rule definition
    pattern: str  # Regex or other pattern
    entity_types: List[EntityType] = Field(default_factory=list)
    relationship_types: List[RelationshipType] = Field(default_factory=list)
    
    # Rule metadata
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None


class KnowledgeValidationRule(BaseModel):
    """Rule for validating knowledge consistency."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Rule logic
    condition: str  # Logical condition to check
    severity: str = Field(default="warning")  # error, warning, info
    
    # Rule metadata
    active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)


class KnowledgeBaseConfig(BaseModel):
    """Configuration for knowledge base system."""
    # Storage settings
    storage_backend: str = Field(default="neo4j")  # neo4j, rdf, sqlite, memory
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing settings
    enable_auto_extraction: bool = Field(default=True)
    enable_inference: bool = Field(default=True)
    enable_validation: bool = Field(default=True)
    
    # Quality settings
    min_confidence_threshold: float = Field(default=0.5)
    auto_verify_threshold: float = Field(default=0.9)
    
    # Performance settings
    cache_size: int = Field(default=10000)
    batch_size: int = Field(default=100)
    
    # Embedding settings
    enable_embeddings: bool = Field(default=True)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


class KnowledgeBaseAnalytics(BaseModel):
    """Analytics for knowledge base usage and quality."""
    total_entities: int = Field(default=0)
    total_relationships: int = Field(default=0)
    total_facts: int = Field(default=0)
    
    # Quality metrics
    average_confidence: float = Field(default=0.0)
    verified_percentage: float = Field(default=0.0)
    quality_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Usage statistics
    query_count: int = Field(default=0)
    average_query_time: float = Field(default=0.0)
    most_queried_entities: List[str] = Field(default_factory=list)
    
    # Growth metrics
    entities_added_today: int = Field(default=0)
    relationships_added_today: int = Field(default=0)
    daily_growth_rate: float = Field(default=0.0)


# Base classes for extensibility
class BaseKnowledgeStore:
    """Base class for knowledge storage backends."""
    
    async def store_entity(self, entity: Entity) -> bool:
        """Store an entity."""
        raise NotImplementedError
    
    async def store_relationship(self, relationship: Relationship) -> bool:
        """Store a relationship."""
        raise NotImplementedError
    
    async def search_knowledge(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """Search knowledge base."""
        raise NotImplementedError
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        raise NotImplementedError


class BaseKnowledgeExtractor:
    """Base class for knowledge extraction."""
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        raise NotImplementedError
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from text."""
        raise NotImplementedError


class BaseInferenceEngine:
    """Base class for knowledge inference."""
    
    async def infer_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Infer new relationships."""
        raise NotImplementedError
    
    async def validate_consistency(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """Validate knowledge consistency."""
        raise NotImplementedError


# Export all types
__all__ = [
    # Enums
    'EntityType',
    'RelationshipType',
    'ConfidenceLevel',
    'KnowledgeSource',
    
    # Core Models
    'Entity',
    'Relationship',
    'KnowledgeFact',
    'OntologyClass',
    'KnowledgeGraph',
    
    # Query and Results
    'KnowledgeQuery',
    'KnowledgeSearchResult',
    
    # Rules and Validation
    'KnowledgeExtractionRule',
    'KnowledgeValidationRule',
    
    # Configuration and Analytics
    'KnowledgeBaseConfig',
    'KnowledgeBaseAnalytics',
    
    # Base Classes
    'BaseKnowledgeStore',
    'BaseKnowledgeExtractor',
    'BaseInferenceEngine'
]
