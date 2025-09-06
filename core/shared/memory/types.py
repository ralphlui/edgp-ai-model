"""
Memory Management Types

Defines comprehensive types for agent memory systems including
short-term memory, long-term memory, episodic memory, and semantic memory.
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
import numpy as np


class MemoryType(str, Enum):
    """Types of memory systems."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


class MemoryStorageType(str, Enum):
    """Memory storage backend types."""
    IN_MEMORY = "in_memory"
    REDIS = "redis"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    VECTOR_STORE = "vector_store"


class MemoryPersistenceLevel(str, Enum):
    """Memory persistence levels."""
    TEMPORARY = "temporary"      # Session only
    SESSION = "session"          # Until session ends
    PERSISTENT = "persistent"    # Across sessions
    PERMANENT = "permanent"      # Never deleted


class MemoryImportance(str, Enum):
    """Memory importance levels for retention."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class MemoryAccessPattern(str, Enum):
    """Memory access patterns for optimization."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"


class MemoryCompressionType(str, Enum):
    """Memory compression strategies."""
    NONE = "none"
    SUMMARIZATION = "summarization"
    CLUSTERING = "clustering"
    HIERARCHICAL = "hierarchical"
    EMBEDDING_BASED = "embedding_based"


class MemoryMetadata(BaseModel):
    """Metadata for memory entries."""
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0)
    importance: MemoryImportance = Field(default=MemoryImportance.MEDIUM)
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None
    
    # Memory relationships
    related_memories: List[str] = Field(default_factory=list)
    parent_memory: Optional[str] = None
    child_memories: List[str] = Field(default_factory=list)
    
    # Performance metrics
    retrieval_frequency: float = Field(default=0.0)
    last_modification: datetime = Field(default_factory=datetime.now)


class MemoryEntry(BaseModel):
    """Base memory entry."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: Any
    memory_type: MemoryType
    persistence_level: MemoryPersistenceLevel = Field(default=MemoryPersistenceLevel.SESSION)
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    
    # Optional embedding for semantic search
    embedding: Optional[List[float]] = None
    
    # Memory specific fields
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None


class EpisodicMemory(MemoryEntry):
    """Episodic memory for specific events/experiences."""
    memory_type: Literal[MemoryType.EPISODIC] = MemoryType.EPISODIC
    
    # Episode specific fields
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    temporal_context: datetime = Field(default_factory=datetime.now)
    spatial_context: Optional[str] = None
    participants: List[str] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    outcomes: List[Dict[str, Any]] = Field(default_factory=list)
    emotions: Optional[Dict[str, float]] = None


class SemanticMemory(MemoryEntry):
    """Semantic memory for facts and concepts."""
    memory_type: Literal[MemoryType.SEMANTIC] = MemoryType.SEMANTIC
    
    # Semantic specific fields
    concept: str
    definition: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    relations: Dict[str, List[str]] = Field(default_factory=dict)  # relation_type -> related_concepts
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)  # Supporting evidence memory IDs


class WorkingMemory(MemoryEntry):
    """Working memory for current processing."""
    memory_type: Literal[MemoryType.WORKING] = MemoryType.WORKING
    
    # Working memory specific fields
    task_id: Optional[str] = None
    subtasks: List[Dict[str, Any]] = Field(default_factory=list)
    current_focus: Optional[str] = None
    attention_weights: Dict[str, float] = Field(default_factory=dict)
    processing_state: Dict[str, Any] = Field(default_factory=dict)


class ProceduralMemory(MemoryEntry):
    """Procedural memory for skills and procedures."""
    memory_type: Literal[MemoryType.PROCEDURAL] = MemoryType.PROCEDURAL
    
    # Procedural specific fields
    skill_name: str
    steps: List[Dict[str, Any]]
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    usage_count: int = Field(default=0)


class MemoryQuery(BaseModel):
    """Query for memory retrieval."""
    query_text: Optional[str] = None
    memory_types: List[MemoryType] = Field(default_factory=list)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Temporal filters
    time_range: Optional[tuple[datetime, datetime]] = None
    max_age: Optional[timedelta] = None
    
    # Content filters
    tags: List[str] = Field(default_factory=list)
    importance_threshold: Optional[MemoryImportance] = None
    content_filter: Optional[Dict[str, Any]] = None
    
    # Retrieval parameters
    limit: int = Field(default=10, ge=1)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_related: bool = Field(default=False)
    access_pattern: MemoryAccessPattern = Field(default=MemoryAccessPattern.TEMPORAL)


class MemorySearchResult(BaseModel):
    """Result from memory search."""
    memories: List[MemoryEntry]
    scores: List[float] = Field(default_factory=list)
    total_found: int
    query_time: float
    search_strategy: str


class MemoryConsolidationConfig(BaseModel):
    """Configuration for memory consolidation."""
    enable_consolidation: bool = Field(default=True)
    consolidation_interval: timedelta = Field(default=timedelta(hours=1))
    importance_decay_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    max_short_term_size: int = Field(default=1000)
    max_working_memory_size: int = Field(default=100)
    
    # Consolidation strategies
    compression_type: MemoryCompressionType = Field(default=MemoryCompressionType.SUMMARIZATION)
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    cluster_size_threshold: int = Field(default=5)


class MemoryStorageConfig(BaseModel):
    """Configuration for memory storage."""
    storage_type: MemoryStorageType = Field(default=MemoryStorageType.IN_MEMORY)
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance settings
    cache_size: int = Field(default=10000)
    batch_size: int = Field(default=100)
    async_writes: bool = Field(default=True)
    
    # Persistence settings
    auto_save_interval: timedelta = Field(default=timedelta(minutes=5))
    backup_enabled: bool = Field(default=True)
    backup_interval: timedelta = Field(default=timedelta(hours=24))


class MemorySystemConfig(BaseModel):
    """Complete memory system configuration."""
    storage_config: MemoryStorageConfig = Field(default_factory=MemoryStorageConfig)
    consolidation_config: MemoryConsolidationConfig = Field(default_factory=MemoryConsolidationConfig)
    
    # System settings
    enable_embeddings: bool = Field(default=True)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    enable_compression: bool = Field(default=True)
    enable_analytics: bool = Field(default=True)
    
    # Memory limits by type
    memory_limits: Dict[MemoryType, int] = Field(default_factory=lambda: {
        MemoryType.SHORT_TERM: 1000,
        MemoryType.LONG_TERM: 10000,
        MemoryType.EPISODIC: 5000,
        MemoryType.SEMANTIC: 10000,
        MemoryType.WORKING: 100,
        MemoryType.PROCEDURAL: 1000
    })
    
    # Retention policies
    retention_policies: Dict[MemoryType, timedelta] = Field(default_factory=lambda: {
        MemoryType.SHORT_TERM: timedelta(hours=24),
        MemoryType.LONG_TERM: timedelta(days=365),
        MemoryType.EPISODIC: timedelta(days=90),
        MemoryType.SEMANTIC: timedelta(days=365),
        MemoryType.WORKING: timedelta(hours=1),
        MemoryType.PROCEDURAL: timedelta(days=365)
    })


class MemoryAnalytics(BaseModel):
    """Memory system analytics."""
    total_memories: Dict[MemoryType, int] = Field(default_factory=dict)
    memory_usage: Dict[MemoryType, float] = Field(default_factory=dict)
    access_patterns: Dict[str, int] = Field(default_factory=dict)
    
    # Performance metrics
    average_retrieval_time: float = Field(default=0.0)
    cache_hit_rate: float = Field(default=0.0)
    consolidation_efficiency: float = Field(default=0.0)
    
    # Usage statistics
    most_accessed_memories: List[str] = Field(default_factory=list)
    memory_growth_rate: float = Field(default=0.0)
    storage_efficiency: float = Field(default=0.0)
    
    # Temporal patterns
    access_frequency_by_hour: Dict[int, int] = Field(default_factory=dict)
    memory_creation_by_day: Dict[str, int] = Field(default_factory=dict)


# Base classes for extensibility
class BaseMemoryStore:
    """Base class for memory storage backends."""
    
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry."""
        raise NotImplementedError
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        raise NotImplementedError
    
    async def search_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching query."""
        raise NotImplementedError
    
    async def update_memory(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        raise NotImplementedError
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        raise NotImplementedError
    
    async def get_analytics(self) -> MemoryAnalytics:
        """Get memory analytics."""
        raise NotImplementedError


class BaseMemoryConsolidator:
    """Base class for memory consolidation strategies."""
    
    async def consolidate_memories(
        self,
        memories: List[MemoryEntry],
        config: MemoryConsolidationConfig
    ) -> List[MemoryEntry]:
        """Consolidate a list of memories."""
        raise NotImplementedError
    
    async def should_consolidate(
        self,
        memory: MemoryEntry,
        config: MemoryConsolidationConfig
    ) -> bool:
        """Determine if a memory should be consolidated."""
        raise NotImplementedError


class BaseMemoryIndex:
    """Base class for memory indexing strategies."""
    
    async def index_memory(self, memory: MemoryEntry):
        """Index a memory for fast retrieval."""
        raise NotImplementedError
    
    async def search_index(self, query: MemoryQuery) -> List[str]:
        """Search index for matching memory IDs."""
        raise NotImplementedError
    
    async def update_index(self, memory: MemoryEntry):
        """Update index for modified memory."""
        raise NotImplementedError
    
    async def remove_from_index(self, memory_id: str):
        """Remove memory from index."""
        raise NotImplementedError


# Memory patterns and strategies
class MemoryPattern(BaseModel):
    """Represents a pattern in memory access or content."""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str  # "temporal", "semantic", "behavioral", etc.
    description: str
    frequency: int = Field(default=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    associated_memories: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryAssociation(BaseModel):
    """Represents associations between memories."""
    association_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id_1: str
    memory_id_2: str
    association_type: str  # "semantic", "temporal", "causal", etc.
    strength: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    reinforcement_count: int = Field(default=1)


# Export all types
__all__ = [
    # Enums
    'MemoryType',
    'MemoryStorageType',
    'MemoryPersistenceLevel',
    'MemoryImportance',
    'MemoryAccessPattern',
    'MemoryCompressionType',
    
    # Core Models
    'MemoryMetadata',
    'MemoryEntry',
    'EpisodicMemory',
    'SemanticMemory',
    'WorkingMemory',
    'ProceduralMemory',
    
    # Query and Results
    'MemoryQuery',
    'MemorySearchResult',
    
    # Configuration
    'MemoryConsolidationConfig',
    'MemoryStorageConfig',
    'MemorySystemConfig',
    
    # Analytics
    'MemoryAnalytics',
    
    # Base Classes
    'BaseMemoryStore',
    'BaseMemoryConsolidator',
    'BaseMemoryIndex',
    
    # Patterns and Associations
    'MemoryPattern',
    'MemoryAssociation'
]
