"""
Memory Management System

This module provides a comprehensive memory management system for agents
supporting multiple memory types, storage backends, and intelligent features.

Key Features:
- Multiple memory types (episodic, semantic, working, procedural)
- Various storage backends (in-memory, SQLite, Redis, PostgreSQL)
- Automatic memory consolidation and compression
- Intelligent indexing for fast retrieval
- Memory analytics and pattern detection
- Session and conversation management

Example Usage:
    from core.shared.memory import MemorySystem, MemorySystemConfig
    
    # Create memory system
    config = MemorySystemConfig()
    memory_system = MemorySystem(config)
    await memory_system.initialize()
    
    # Store different types of memories
    episodic_id = await memory_system.create_episodic_memory(
        content="User asked about weather",
        agent_id="agent_1",
        session_id="session_123",
        participants=["user", "agent"]
    )
    
    semantic_id = await memory_system.create_semantic_memory(
        concept="weather",
        definition="Atmospheric conditions at a specific time and place",
        agent_id="agent_1"
    )
    
    # Search memories
    from core.shared.memory.types import MemoryQuery, MemoryType
    
    query = MemoryQuery(
        query_text="weather information",
        memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
        agent_id="agent_1"
    )
    
    results = await memory_system.search_memories(query)
    print(f"Found {len(results.memories)} relevant memories")
    
    # Get analytics
    analytics = await memory_system.get_analytics()
    print(f"Total memories: {sum(analytics.total_memories.values())}")
"""

from typing import Dict, List, Optional, Any

# Type definitions
from .types import (
    # Enums
    MemoryType,
    MemoryStorageType,
    MemoryPersistenceLevel,
    MemoryImportance,
    MemoryAccessPattern,
    MemoryCompressionType,
    
    # Core Models
    MemoryMetadata,
    MemoryEntry,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    ProceduralMemory,
    
    # Query and Results
    MemoryQuery,
    MemorySearchResult,
    
    # Configuration
    MemoryConsolidationConfig,
    MemoryStorageConfig,
    MemorySystemConfig,
    
    # Analytics and Patterns
    MemoryAnalytics,
    MemoryPattern,
    MemoryAssociation,
    
    # Base Classes
    BaseMemoryStore,
    BaseMemoryConsolidator,
    BaseMemoryIndex
)

# Core components
from .manager import MemoryManager
from .stores import (
    MemoryStoreManager,
    InMemoryStore,
    SQLiteStore,
    RedisStore
)
from .consolidation import (
    MemoryConsolidationManager,
    SummarizationConsolidator,
    ClusteringConsolidator,
    HierarchicalConsolidator
)
from .indexing import (
    MemoryIndexManager,
    TemporalIndex,
    SemanticIndex,
    BehavioralIndex,
    CompositeIndex
)


class MemorySystem:
    """
    Complete memory management system for agents.
    
    This is the main interface for using the memory system. It combines
    all memory management components including storage, indexing,
    consolidation, and analytics.
    """
    
    def __init__(self, config: MemorySystemConfig):
        """Initialize memory system with configuration."""
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory system."""
        if self.initialized:
            return
        
        await self.memory_manager.initialize()
        self.initialized = True
    
    # Memory storage operations
    async def store_memory(self, memory: MemoryEntry) -> str:
        """Store a memory entry."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.store_memory(memory)
    
    async def retrieve_memory(self, memory_id: str) -> MemoryEntry:
        """Retrieve a specific memory by ID."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.retrieve_memory(memory_id)
    
    async def search_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching the query."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.search_memories(query)
    
    async def update_memory(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.update_memory(memory)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.delete_memory(memory_id)
    
    # Convenience methods for creating specific memory types
    async def create_episodic_memory(
        self,
        content: Any,
        agent_id: str,
        session_id: str,
        conversation_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        outcomes: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """Create an episodic memory entry."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.create_episodic_memory(
            content, agent_id, session_id, conversation_id,
            participants, actions, outcomes, **kwargs
        )
    
    async def create_semantic_memory(
        self,
        concept: str,
        definition: str,
        agent_id: str,
        properties: Optional[Dict[str, Any]] = None,
        relations: Optional[Dict[str, List[str]]] = None,
        confidence: float = 1.0,
        evidence: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Create a semantic memory entry."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.create_semantic_memory(
            concept, definition, agent_id, properties,
            relations, confidence, evidence, **kwargs
        )
    
    async def create_working_memory(
        self,
        content: Any,
        agent_id: str,
        session_id: str,
        task_id: Optional[str] = None,
        current_focus: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create a working memory entry."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.create_working_memory(
            content, agent_id, session_id, task_id, current_focus, **kwargs
        )
    
    async def create_procedural_memory(
        self,
        skill_name: str,
        steps: List[Dict[str, Any]],
        agent_id: str,
        preconditions: Optional[List[str]] = None,
        postconditions: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Create a procedural memory entry."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.create_procedural_memory(
            skill_name, steps, agent_id, preconditions, postconditions, **kwargs
        )
    
    # Session management
    async def get_session_memories(self, session_id: str) -> List[MemoryEntry]:
        """Get all memories for a session."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.get_session_memories(session_id)
    
    async def clear_session_memories(self, session_id: str) -> int:
        """Clear all temporary memories for a session."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.clear_session_memories(session_id)
    
    # Advanced features
    async def consolidate_memories(self, memory_type: Optional[MemoryType] = None) -> Dict[str, int]:
        """Manually trigger memory consolidation."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.consolidate_memories(memory_type)
    
    async def get_memory_associations(self, memory_id: str) -> List[MemoryAssociation]:
        """Get associations for a memory."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.get_memory_associations(memory_id)
    
    async def create_memory_association(
        self,
        memory_id_1: str,
        memory_id_2: str,
        association_type: str,
        strength: float
    ) -> str:
        """Create an association between memories."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.create_memory_association(
            memory_id_1, memory_id_2, association_type, strength
        )
    
    async def find_memory_patterns(self, pattern_type: str) -> List[MemoryPattern]:
        """Find patterns in memory access or content."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.find_memory_patterns(pattern_type)
    
    # Analytics and monitoring
    async def get_analytics(self) -> MemoryAnalytics:
        """Get comprehensive memory analytics."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.get_analytics()
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.initialized:
            await self.initialize()
        return {
            "memory_manager": self.memory_manager.performance_metrics,
            "analytics": await self.get_analytics()
        }
    
    # Maintenance operations
    async def cleanup_expired_memories(self) -> int:
        """Clean up expired memories based on retention policies."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.cleanup_expired_memories()
    
    async def optimize_system(self):
        """Optimize memory system performance."""
        if not self.initialized:
            await self.initialize()
        await self.memory_manager.optimize_memory_usage()
    
    # Import/Export
    async def export_memories(
        self,
        export_format: str = "json",
        memory_types: Optional[List[MemoryType]] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export memories for backup or transfer."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.export_memories(
            export_format, memory_types, agent_id
        )
    
    async def import_memories(self, import_data: Dict[str, Any]) -> int:
        """Import memories from export data."""
        if not self.initialized:
            await self.initialize()
        return await self.memory_manager.import_memories(import_data)
    
    async def shutdown(self):
        """Shutdown the memory system."""
        if self.initialized:
            await self.memory_manager.shutdown()
            self.initialized = False


# Convenience functions for quick setup
def create_simple_memory_system(
    storage_type: MemoryStorageType = MemoryStorageType.IN_MEMORY
) -> MemorySystem:
    """Create a simple memory system with default configuration."""
    storage_config = MemoryStorageConfig(storage_type=storage_type)
    config = MemorySystemConfig(storage_config=storage_config)
    return MemorySystem(config)


def create_persistent_memory_system(
    db_path: str = "memories.db"
) -> MemorySystem:
    """Create a persistent memory system using SQLite."""
    storage_config = MemoryStorageConfig(
        storage_type=MemoryStorageType.SQLITE,
        connection_params={"db_path": db_path}
    )
    config = MemorySystemConfig(storage_config=storage_config)
    return MemorySystem(config)


def create_distributed_memory_system(
    redis_url: str = "redis://localhost:6379"
) -> MemorySystem:
    """Create a distributed memory system using Redis."""
    storage_config = MemoryStorageConfig(
        storage_type=MemoryStorageType.REDIS,
        connection_params={"redis_url": redis_url}
    )
    config = MemorySystemConfig(storage_config=storage_config)
    return MemorySystem(config)


def create_production_memory_system(
    storage_config: MemoryStorageConfig,
    enable_analytics: bool = True,
    enable_consolidation: bool = True
) -> MemorySystem:
    """Create a production-ready memory system."""
    consolidation_config = MemoryConsolidationConfig(
        enable_consolidation=enable_consolidation
    )
    
    config = MemorySystemConfig(
        storage_config=storage_config,
        consolidation_config=consolidation_config,
        enable_analytics=enable_analytics,
        enable_compression=True
    )
    
    return MemorySystem(config)


# Export all components
__all__ = [
    # Main system
    'MemorySystem',
    'create_simple_memory_system',
    'create_persistent_memory_system',
    'create_distributed_memory_system',
    'create_production_memory_system',
    
    # Types
    'MemoryType',
    'MemoryStorageType',
    'MemoryPersistenceLevel',
    'MemoryImportance',
    'MemoryAccessPattern',
    'MemoryCompressionType',
    'MemoryMetadata',
    'MemoryEntry',
    'EpisodicMemory',
    'SemanticMemory',
    'WorkingMemory',
    'ProceduralMemory',
    'MemoryQuery',
    'MemorySearchResult',
    'MemoryConsolidationConfig',
    'MemoryStorageConfig',
    'MemorySystemConfig',
    'MemoryAnalytics',
    'MemoryPattern',
    'MemoryAssociation',
    'BaseMemoryStore',
    'BaseMemoryConsolidator',
    'BaseMemoryIndex',
    
    # Managers
    'MemoryManager',
    'MemoryStoreManager',
    'MemoryConsolidationManager',
    'MemoryIndexManager',
    
    # Storage implementations
    'InMemoryStore',
    'SQLiteStore',
    'RedisStore',
    
    # Consolidation strategies
    'SummarizationConsolidator',
    'ClusteringConsolidator',
    'HierarchicalConsolidator',
    
    # Indexing strategies
    'TemporalIndex',
    'SemanticIndex',
    'BehavioralIndex',
    'CompositeIndex'
]
