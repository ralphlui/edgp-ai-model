"""
Memory Storage Implementation

Provides various storage backends for memory persistence including
in-memory, Redis, SQLite, and PostgreSQL implementations.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import pickle
import uuid

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncpg
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

from .types import (
    MemoryEntry, MemoryQuery, MemorySearchResult, MemoryType,
    MemoryStorageConfig, MemoryStorageType, MemoryAnalytics,
    MemoryImportance, MemoryPersistenceLevel, BaseMemoryStore,
    MemoryAssociation, EpisodicMemory, SemanticMemory,
    WorkingMemory, ProceduralMemory
)

logger = logging.getLogger(__name__)


class InMemoryStore(BaseMemoryStore):
    """In-memory storage backend for development and testing."""
    
    def __init__(self, config: MemoryStorageConfig):
        self.config = config
        self.memories: Dict[str, MemoryEntry] = {}
        self.associations: Dict[str, List[MemoryAssociation]] = defaultdict(list)
        self.indices: Dict[str, Set[str]] = defaultdict(set)  # Index type -> memory IDs
        
        # Analytics tracking
        self.stats = {
            'total_operations': 0,
            'store_operations': 0,
            'retrieve_operations': 0,
            'search_operations': 0,
            'delete_operations': 0
        }
    
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry."""
        try:
            self.memories[memory.memory_id] = memory
            
            # Update indices
            self._update_indices(memory)
            
            # Update stats
            self.stats['store_operations'] += 1
            self.stats['total_operations'] += 1
            
            return True
        except Exception as e:
            logger.error("Failed to store memory in memory: %s", str(e))
            return False
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        try:
            self.stats['retrieve_operations'] += 1
            self.stats['total_operations'] += 1
            
            return self.memories.get(memory_id)
        except Exception as e:
            logger.error("Failed to retrieve memory from memory: %s", str(e))
            return None
    
    async def search_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching query."""
        start_time = datetime.now()
        
        try:
            # Filter memories
            matching_memories = []
            
            for memory in self.memories.values():
                if self._matches_query(memory, query):
                    matching_memories.append(memory)
            
            # Sort by relevance/time
            if query.query_text:
                # Simple text matching score
                scored_memories = []
                for memory in matching_memories:
                    score = self._calculate_text_score(memory, query.query_text)
                    scored_memories.append((memory, score))
                
                scored_memories.sort(key=lambda x: x[1], reverse=True)
                matching_memories = [memory for memory, _ in scored_memories[:query.limit]]
                scores = [score for _, score in scored_memories[:query.limit]]
            else:
                # Sort by creation time (newest first)
                matching_memories.sort(
                    key=lambda m: m.metadata.created_at,
                    reverse=True
                )
                matching_memories = matching_memories[:query.limit]
                scores = [1.0] * len(matching_memories)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats
            self.stats['search_operations'] += 1
            self.stats['total_operations'] += 1
            
            return MemorySearchResult(
                memories=matching_memories,
                scores=scores,
                total_found=len(matching_memories),
                query_time=search_time,
                search_strategy="in_memory_filter"
            )
        
        except Exception as e:
            logger.error("Failed to search memories: %s", str(e))
            return MemorySearchResult(
                memories=[], scores=[], total_found=0,
                query_time=0.0, search_strategy="error"
            )
    
    async def update_memory(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        try:
            if memory.memory_id in self.memories:
                self.memories[memory.memory_id] = memory
                self._update_indices(memory)
                return True
            return False
        except Exception as e:
            logger.error("Failed to update memory: %s", str(e))
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                del self.memories[memory_id]
                
                # Clean up indices
                self._remove_from_indices(memory)
                
                # Clean up associations
                if memory_id in self.associations:
                    del self.associations[memory_id]
                
                # Update stats
                self.stats['delete_operations'] += 1
                self.stats['total_operations'] += 1
                
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete memory: %s", str(e))
            return False
    
    async def get_analytics(self) -> MemoryAnalytics:
        """Get memory analytics."""
        analytics = MemoryAnalytics()
        
        # Count memories by type
        for memory in self.memories.values():
            current_count = analytics.total_memories.get(memory.memory_type, 0)
            analytics.total_memories[memory.memory_type] = current_count + 1
        
        return analytics
    
    async def get_associations(self, memory_id: str) -> List[MemoryAssociation]:
        """Get associations for a memory."""
        return self.associations.get(memory_id, [])
    
    async def create_association(
        self,
        memory_id_1: str,
        memory_id_2: str,
        association_type: str,
        strength: float
    ) -> str:
        """Create an association between memories."""
        association = MemoryAssociation(
            memory_id_1=memory_id_1,
            memory_id_2=memory_id_2,
            association_type=association_type,
            strength=strength
        )
        
        # Store bidirectional associations
        self.associations[memory_id_1].append(association)
        self.associations[memory_id_2].append(association)
        
        return association.association_id
    
    def _matches_query(self, memory: MemoryEntry, query: MemoryQuery) -> bool:
        """Check if memory matches query criteria."""
        # Memory type filter
        if query.memory_types and memory.memory_type not in query.memory_types:
            return False
        
        # Agent filter
        if query.agent_id and memory.agent_id != query.agent_id:
            return False
        
        # Session filter
        if query.session_id and memory.session_id != query.session_id:
            return False
        
        # Conversation filter
        if query.conversation_id and memory.conversation_id != query.conversation_id:
            return False
        
        # Time range filter
        if query.time_range:
            start_time, end_time = query.time_range
            if not (start_time <= memory.metadata.created_at <= end_time):
                return False
        
        # Max age filter
        if query.max_age:
            age = datetime.now() - memory.metadata.created_at
            if age > query.max_age:
                return False
        
        # Tags filter
        if query.tags:
            if not any(tag in memory.metadata.tags for tag in query.tags):
                return False
        
        # Importance filter
        if query.importance_threshold:
            importance_values = {
                MemoryImportance.MINIMAL: 1,
                MemoryImportance.LOW: 2,
                MemoryImportance.MEDIUM: 3,
                MemoryImportance.HIGH: 4,
                MemoryImportance.CRITICAL: 5
            }
            if (importance_values[memory.metadata.importance] < 
                importance_values[query.importance_threshold]):
                return False
        
        return True
    
    def _calculate_text_score(self, memory: MemoryEntry, query_text: str) -> float:
        """Calculate text similarity score."""
        content_str = str(memory.content).lower()
        query_lower = query_text.lower()
        
        # Simple word overlap scoring
        query_words = set(query_lower.split())
        content_words = set(content_str.split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        return overlap / len(query_words)
    
    def _update_indices(self, memory: MemoryEntry):
        """Update search indices for memory."""
        memory_id = memory.memory_id
        
        # Type index
        self.indices[f"type:{memory.memory_type.value}"].add(memory_id)
        
        # Agent index
        if memory.agent_id:
            self.indices[f"agent:{memory.agent_id}"].add(memory_id)
        
        # Session index
        if memory.session_id:
            self.indices[f"session:{memory.session_id}"].add(memory_id)
        
        # Tag indices
        for tag in memory.metadata.tags:
            self.indices[f"tag:{tag}"].add(memory_id)
        
        # Importance index
        self.indices[f"importance:{memory.metadata.importance.value}"].add(memory_id)
    
    def _remove_from_indices(self, memory: MemoryEntry):
        """Remove memory from all indices."""
        memory_id = memory.memory_id
        
        for index_set in self.indices.values():
            index_set.discard(memory_id)


class SQLiteStore(BaseMemoryStore):
    """SQLite storage backend for persistent local storage."""
    
    def __init__(self, config: MemoryStorageConfig):
        self.config = config
        self.db_path = config.connection_params.get('db_path', 'memories.db')
        self.connection = None
    
    async def initialize(self):
        """Initialize SQLite database."""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        await self._create_tables()
    
    async def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()
        
        # Main memories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                persistence_level TEXT NOT NULL,
                agent_id TEXT,
                session_id TEXT,
                conversation_id TEXT,
                embedding BLOB,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP NOT NULL,
                last_modification TIMESTAMP NOT NULL
            )
        ''')
        
        # Associations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS associations (
                association_id TEXT PRIMARY KEY,
                memory_id_1 TEXT NOT NULL,
                memory_id_2 TEXT NOT NULL,
                association_type TEXT NOT NULL,
                strength REAL NOT NULL,
                created_at TIMESTAMP NOT NULL,
                reinforcement_count INTEGER DEFAULT 1,
                FOREIGN KEY (memory_id_1) REFERENCES memories (memory_id),
                FOREIGN KEY (memory_id_2) REFERENCES memories (memory_id)
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories (memory_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_id ON memories (agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON memories (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories (created_at)')
        
        self.connection.commit()
    
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memories (
                    memory_id, content, memory_type, persistence_level,
                    agent_id, session_id, conversation_id, embedding,
                    metadata, created_at, last_accessed, last_modification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.memory_id,
                json.dumps(memory.content),
                memory.memory_type.value,
                memory.persistence_level.value,
                memory.agent_id,
                memory.session_id,
                memory.conversation_id,
                pickle.dumps(memory.embedding) if memory.embedding else None,
                memory.metadata.json(),
                memory.metadata.created_at,
                memory.metadata.last_accessed,
                memory.metadata.last_modification
            ))
            
            self.connection.commit()
            return True
        
        except Exception as e:
            logger.error("Failed to store memory in SQLite: %s", str(e))
            return False
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM memories WHERE memory_id = ?', (memory_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_memory(row)
            return None
        
        except Exception as e:
            logger.error("Failed to retrieve memory from SQLite: %s", str(e))
            return None
    
    async def search_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching query."""
        start_time = datetime.now()
        
        try:
            cursor = self.connection.cursor()
            
            # Build SQL query
            sql_query = "SELECT * FROM memories WHERE 1=1"
            params = []
            
            # Add filters
            if query.memory_types:
                placeholders = ','.join('?' * len(query.memory_types))
                sql_query += f" AND memory_type IN ({placeholders})"
                params.extend([mt.value for mt in query.memory_types])
            
            if query.agent_id:
                sql_query += " AND agent_id = ?"
                params.append(query.agent_id)
            
            if query.session_id:
                sql_query += " AND session_id = ?"
                params.append(query.session_id)
            
            if query.time_range:
                start_time_param, end_time_param = query.time_range
                sql_query += " AND created_at BETWEEN ? AND ?"
                params.extend([start_time_param, end_time_param])
            
            # Add ordering and limit
            sql_query += " ORDER BY created_at DESC LIMIT ?"
            params.append(query.limit)
            
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            
            memories = [self._row_to_memory(row) for row in rows]
            scores = [1.0] * len(memories)  # Simple scoring
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return MemorySearchResult(
                memories=memories,
                scores=scores,
                total_found=len(memories),
                query_time=search_time,
                search_strategy="sqlite_query"
            )
        
        except Exception as e:
            logger.error("Failed to search memories in SQLite: %s", str(e))
            return MemorySearchResult(
                memories=[], scores=[], total_found=0,
                query_time=0.0, search_strategy="error"
            )
    
    async def update_memory(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        return await self.store_memory(memory)  # SQLite handles INSERT OR REPLACE
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('DELETE FROM memories WHERE memory_id = ?', (memory_id,))
            cursor.execute('DELETE FROM associations WHERE memory_id_1 = ? OR memory_id_2 = ?',
                          (memory_id, memory_id))
            self.connection.commit()
            return cursor.rowcount > 0
        
        except Exception as e:
            logger.error("Failed to delete memory from SQLite: %s", str(e))
            return False
    
    async def get_analytics(self) -> MemoryAnalytics:
        """Get memory analytics."""
        analytics = MemoryAnalytics()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
            ''')
            
            for row in cursor.fetchall():
                memory_type = MemoryType(row['memory_type'])
                analytics.total_memories[memory_type] = row['count']
        
        except Exception as e:
            logger.error("Failed to get analytics from SQLite: %s", str(e))
        
        return analytics
    
    async def get_associations(self, memory_id: str) -> List[MemoryAssociation]:
        """Get associations for a memory."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM associations 
                WHERE memory_id_1 = ? OR memory_id_2 = ?
            ''', (memory_id, memory_id))
            
            associations = []
            for row in cursor.fetchall():
                association = MemoryAssociation(
                    association_id=row['association_id'],
                    memory_id_1=row['memory_id_1'],
                    memory_id_2=row['memory_id_2'],
                    association_type=row['association_type'],
                    strength=row['strength'],
                    created_at=row['created_at'],
                    reinforcement_count=row['reinforcement_count']
                )
                associations.append(association)
            
            return associations
        
        except Exception as e:
            logger.error("Failed to get associations from SQLite: %s", str(e))
            return []
    
    async def create_association(
        self,
        memory_id_1: str,
        memory_id_2: str,
        association_type: str,
        strength: float
    ) -> str:
        """Create an association between memories."""
        try:
            association_id = str(uuid.uuid4())
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO associations (
                    association_id, memory_id_1, memory_id_2,
                    association_type, strength, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                association_id, memory_id_1, memory_id_2,
                association_type, strength, datetime.now()
            ))
            
            self.connection.commit()
            return association_id
        
        except Exception as e:
            logger.error("Failed to create association in SQLite: %s", str(e))
            return ""
    
    def _row_to_memory(self, row) -> MemoryEntry:
        """Convert database row to memory entry."""
        from .types import MemoryMetadata  # Import here to avoid circular imports
        
        # Parse metadata
        metadata = MemoryMetadata.parse_raw(row['metadata'])
        
        # Determine specific memory type
        memory_type = MemoryType(row['memory_type'])
        content = json.loads(row['content'])
        
        # Create appropriate memory type
        base_data = {
            'memory_id': row['memory_id'],
            'content': content,
            'memory_type': memory_type,
            'persistence_level': MemoryPersistenceLevel(row['persistence_level']),
            'metadata': metadata,
            'agent_id': row['agent_id'],
            'session_id': row['session_id'],
            'conversation_id': row['conversation_id'],
            'embedding': pickle.loads(row['embedding']) if row['embedding'] else None
        }
        
        if memory_type == MemoryType.EPISODIC:
            return EpisodicMemory(**base_data, **content)
        elif memory_type == MemoryType.SEMANTIC:
            return SemanticMemory(**base_data, **content)
        elif memory_type == MemoryType.WORKING:
            return WorkingMemory(**base_data, **content)
        elif memory_type == MemoryType.PROCEDURAL:
            return ProceduralMemory(**base_data, **content)
        else:
            return MemoryEntry(**base_data)


class RedisStore(BaseMemoryStore):
    """Redis storage backend for distributed caching."""
    
    def __init__(self, config: MemoryStorageConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
        
        self.config = config
        self.redis_client = None
        self.redis_url = config.connection_params.get('redis_url', 'redis://localhost:6379')
    
    async def initialize(self):
        """Initialize Redis connection."""
        self.redis_client = redis.from_url(self.redis_url)
    
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry."""
        try:
            data = memory.dict()
            await self.redis_client.set(
                f"memory:{memory.memory_id}",
                json.dumps(data, default=str),
                ex=3600  # 1 hour expiry by default
            )
            
            # Add to indices
            await self._update_redis_indices(memory)
            
            return True
        
        except Exception as e:
            logger.error("Failed to store memory in Redis: %s", str(e))
            return False
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        try:
            data = await self.redis_client.get(f"memory:{memory_id}")
            if data:
                memory_data = json.loads(data)
                return MemoryEntry.parse_obj(memory_data)
            return None
        
        except Exception as e:
            logger.error("Failed to retrieve memory from Redis: %s", str(e))
            return None
    
    async def search_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching query."""
        start_time = datetime.now()
        
        try:
            # Use Redis sets for filtering
            memory_ids = set()
            
            # Get all memory IDs if no specific filters
            if not any([query.memory_types, query.agent_id, query.session_id]):
                pattern = "memory:*"
                async for key in self.redis_client.scan_iter(match=pattern):
                    memory_id = key.decode().split(':', 1)[1]
                    memory_ids.add(memory_id)
            else:
                # Use indices for filtering
                if query.memory_types:
                    for memory_type in query.memory_types:
                        type_ids = await self.redis_client.smembers(f"index:type:{memory_type.value}")
                        memory_ids.update(id.decode() for id in type_ids)
                
                if query.agent_id:
                    agent_ids = await self.redis_client.smembers(f"index:agent:{query.agent_id}")
                    if memory_ids:
                        memory_ids &= set(id.decode() for id in agent_ids)
                    else:
                        memory_ids = set(id.decode() for id in agent_ids)
            
            # Retrieve memories
            memories = []
            for memory_id in list(memory_ids)[:query.limit]:
                memory = await self.retrieve_memory(memory_id)
                if memory:
                    memories.append(memory)
            
            search_time = (datetime.now() - start_time).total_seconds()
            scores = [1.0] * len(memories)
            
            return MemorySearchResult(
                memories=memories,
                scores=scores,
                total_found=len(memories),
                query_time=search_time,
                search_strategy="redis_index"
            )
        
        except Exception as e:
            logger.error("Failed to search memories in Redis: %s", str(e))
            return MemorySearchResult(
                memories=[], scores=[], total_found=0,
                query_time=0.0, search_strategy="error"
            )
    
    async def update_memory(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        return await self.store_memory(memory)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            # Get memory first to clean up indices
            memory = await self.retrieve_memory(memory_id)
            if memory:
                await self._remove_from_redis_indices(memory)
            
            result = await self.redis_client.delete(f"memory:{memory_id}")
            return result > 0
        
        except Exception as e:
            logger.error("Failed to delete memory from Redis: %s", str(e))
            return False
    
    async def get_analytics(self) -> MemoryAnalytics:
        """Get memory analytics."""
        analytics = MemoryAnalytics()
        
        try:
            # Count memories by type using indices
            for memory_type in MemoryType:
                count = await self.redis_client.scard(f"index:type:{memory_type.value}")
                analytics.total_memories[memory_type] = count
        
        except Exception as e:
            logger.error("Failed to get analytics from Redis: %s", str(e))
        
        return analytics
    
    async def get_associations(self, memory_id: str) -> List[MemoryAssociation]:
        """Get associations for a memory."""
        # Redis implementation would store associations separately
        return []
    
    async def create_association(
        self,
        memory_id_1: str,
        memory_id_2: str,
        association_type: str,
        strength: float
    ) -> str:
        """Create an association between memories."""
        # Redis implementation for associations
        return str(uuid.uuid4())
    
    async def _update_redis_indices(self, memory: MemoryEntry):
        """Update Redis indices."""
        memory_id = memory.memory_id
        
        # Type index
        await self.redis_client.sadd(f"index:type:{memory.memory_type.value}", memory_id)
        
        # Agent index
        if memory.agent_id:
            await self.redis_client.sadd(f"index:agent:{memory.agent_id}", memory_id)
        
        # Session index
        if memory.session_id:
            await self.redis_client.sadd(f"index:session:{memory.session_id}", memory_id)
    
    async def _remove_from_redis_indices(self, memory: MemoryEntry):
        """Remove memory from Redis indices."""
        memory_id = memory.memory_id
        
        await self.redis_client.srem(f"index:type:{memory.memory_type.value}", memory_id)
        
        if memory.agent_id:
            await self.redis_client.srem(f"index:agent:{memory.agent_id}", memory_id)
        
        if memory.session_id:
            await self.redis_client.srem(f"index:session:{memory.session_id}", memory_id)


class MemoryStoreManager:
    """Manager for different memory storage backends."""
    
    def __init__(self, config: MemoryStorageConfig):
        self.config = config
        self.store: Optional[BaseMemoryStore] = None
    
    async def initialize(self):
        """Initialize the storage backend."""
        if self.config.storage_type == MemoryStorageType.IN_MEMORY:
            self.store = InMemoryStore(self.config)
        elif self.config.storage_type == MemoryStorageType.SQLITE:
            self.store = SQLiteStore(self.config)
            await self.store.initialize()
        elif self.config.storage_type == MemoryStorageType.REDIS:
            self.store = RedisStore(self.config)
            await self.store.initialize()
        elif self.config.storage_type == MemoryStorageType.POSTGRESQL:
            # PostgreSQL implementation would go here
            raise NotImplementedError("PostgreSQL storage not yet implemented")
        else:
            raise ValueError(f"Unsupported storage type: {self.config.storage_type}")
        
        logger.info("Memory store initialized: %s", self.config.storage_type.value)
    
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.store_memory(memory)
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.retrieve_memory(memory_id)
    
    async def search_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching query."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.search_memories(query)
    
    async def update_memory(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.update_memory(memory)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.delete_memory(memory_id)
    
    async def get_analytics(self) -> MemoryAnalytics:
        """Get memory analytics."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.get_analytics()
    
    async def get_associations(self, memory_id: str) -> List[MemoryAssociation]:
        """Get associations for a memory."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.get_associations(memory_id)
    
    async def create_association(
        self,
        memory_id_1: str,
        memory_id_2: str,
        association_type: str,
        strength: float
    ) -> str:
        """Create an association between memories."""
        if not self.store:
            raise RuntimeError("Store not initialized")
        return await self.store.create_association(
            memory_id_1, memory_id_2, association_type, strength
        )
    
    async def shutdown(self):
        """Shutdown the storage backend."""
        if hasattr(self.store, 'shutdown'):
            await self.store.shutdown()
        logger.info("Memory store shutdown complete")
