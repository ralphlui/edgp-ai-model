"""
Memory Manager

Central manager for all memory operations including storage, retrieval,
consolidation, and analytics across different memory types.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import json

from .types import (
    MemoryEntry, MemoryQuery, MemorySearchResult, MemoryType,
    MemorySystemConfig, MemoryAnalytics, MemoryImportance,
    MemoryPersistenceLevel, EpisodicMemory, SemanticMemory,
    WorkingMemory, ProceduralMemory, MemoryPattern, MemoryAssociation,
    BaseMemoryStore
)
from .stores import MemoryStoreManager
from .consolidation import MemoryConsolidationManager
from .indexing import MemoryIndexManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central memory management system for agents.
    
    Manages different types of memory (episodic, semantic, working, procedural)
    with automatic consolidation, indexing, and retrieval optimization.
    """
    
    def __init__(self, config: MemorySystemConfig):
        self.config = config
        
        # Component managers
        self.store_manager = MemoryStoreManager(config.storage_config)
        self.consolidation_manager = MemoryConsolidationManager(config.consolidation_config)
        self.index_manager = MemoryIndexManager(config)
        
        # Memory caches by type
        self.memory_caches: Dict[MemoryType, Dict[str, MemoryEntry]] = {
            memory_type: {} for memory_type in MemoryType
        }
        
        # Active sessions and conversations
        self.active_sessions: Set[str] = set()
        self.session_memories: Dict[str, List[str]] = defaultdict(list)
        
        # Analytics tracking
        self.analytics = MemoryAnalytics()
        self._access_counts = defaultdict(int)
        self._retrieval_times = []
        
        # Background tasks
        self._consolidation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_stores': 0,
            'total_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_store_time': 0.0,
            'avg_retrieval_time': 0.0
        }
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory manager."""
        if self.initialized:
            return
        
        # Initialize component managers
        await self.store_manager.initialize()
        await self.consolidation_manager.initialize()
        await self.index_manager.initialize()
        
        # Start background tasks
        if self.config.consolidation_config.enable_consolidation:
            self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.enable_analytics:
            self._analytics_task = asyncio.create_task(self._analytics_loop())
        
        self.initialized = True
        logger.info("MemoryManager initialized with %d memory types", len(MemoryType))
    
    async def store_memory(self, memory: MemoryEntry) -> str:
        """Store a memory entry."""
        start_time = datetime.now()
        
        try:
            # Validate memory
            await self._validate_memory(memory)
            
            # Check memory limits
            await self._check_memory_limits(memory.memory_type)
            
            # Generate embedding if enabled
            if self.config.enable_embeddings and not memory.embedding:
                memory.embedding = await self._generate_embedding(memory)
            
            # Store in backend
            success = await self.store_manager.store_memory(memory)
            if not success:
                raise RuntimeError(f"Failed to store memory {memory.memory_id}")
            
            # Update cache
            self.memory_caches[memory.memory_type][memory.memory_id] = memory
            
            # Update index
            await self.index_manager.index_memory(memory)
            
            # Track session/conversation
            if memory.session_id:
                self.active_sessions.add(memory.session_id)
                self.session_memories[memory.session_id].append(memory.memory_id)
            
            # Update analytics
            self.analytics.total_memories[memory.memory_type] = (
                self.analytics.total_memories.get(memory.memory_type, 0) + 1
            )
            
            # Performance tracking
            store_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['total_stores'] += 1
            self._update_average_time('avg_store_time', store_time)
            
            logger.debug("Stored %s memory %s in %.3fs",
                        memory.memory_type.value, memory.memory_id, store_time)
            
            return memory.memory_id
        
        except Exception as e:
            logger.error("Failed to store memory: %s", str(e))
            raise
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        start_time = datetime.now()
        
        try:
            # Check caches first
            for memory_type, cache in self.memory_caches.items():
                if memory_id in cache:
                    memory = cache[memory_id]
                    await self._update_access_metadata(memory)
                    self.performance_metrics['cache_hits'] += 1
                    return memory
            
            # Retrieve from backend
            memory = await self.store_manager.retrieve_memory(memory_id)
            
            if memory:
                # Update cache
                self.memory_caches[memory.memory_type][memory_id] = memory
                
                # Update access metadata
                await self._update_access_metadata(memory)
                
                self.performance_metrics['cache_misses'] += 1
            
            # Performance tracking
            retrieval_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['total_retrievals'] += 1
            self._update_average_time('avg_retrieval_time', retrieval_time)
            self._retrieval_times.append(retrieval_time)
            
            return memory
        
        except Exception as e:
            logger.error("Failed to retrieve memory %s: %s", memory_id, str(e))
            return None
    
    async def search_memories(self, query: MemoryQuery) -> MemorySearchResult:
        """Search for memories matching the query."""
        start_time = datetime.now()
        
        try:
            # Use index for initial filtering if available
            candidate_ids = await self.index_manager.search_index(query)
            
            # If no index results, fall back to store search
            if not candidate_ids:
                return await self.store_manager.search_memories(query)
            
            # Retrieve candidate memories
            candidate_memories = []
            for memory_id in candidate_ids:
                memory = await self.retrieve_memory(memory_id)
                if memory:
                    candidate_memories.append(memory)
            
            # Apply additional filtering and scoring
            filtered_memories = await self._filter_and_score_memories(candidate_memories, query)
            
            # Sort by score and apply limit
            filtered_memories.sort(key=lambda x: x[1], reverse=True)
            limited_results = filtered_memories[:query.limit]
            
            memories = [memory for memory, _ in limited_results]
            scores = [score for _, score in limited_results]
            
            # Update access metadata for retrieved memories
            for memory in memories:
                await self._update_access_metadata(memory)
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return MemorySearchResult(
                memories=memories,
                scores=scores,
                total_found=len(filtered_memories),
                query_time=search_time,
                search_strategy="index_filtered"
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
            # Update metadata
            memory.metadata.last_modification = datetime.now()
            
            # Update in backend
            success = await self.store_manager.update_memory(memory)
            if not success:
                return False
            
            # Update cache
            self.memory_caches[memory.memory_type][memory.memory_id] = memory
            
            # Update index
            await self.index_manager.update_index(memory)
            
            logger.debug("Updated memory %s", memory.memory_id)
            return True
        
        except Exception as e:
            logger.error("Failed to update memory %s: %s", memory.memory_id, str(e))
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            # Get memory type for cache cleanup
            memory = await self.retrieve_memory(memory_id)
            if not memory:
                return False
            
            # Delete from backend
            success = await self.store_manager.delete_memory(memory_id)
            if not success:
                return False
            
            # Remove from cache
            if memory_id in self.memory_caches[memory.memory_type]:
                del self.memory_caches[memory.memory_type][memory_id]
            
            # Remove from index
            await self.index_manager.remove_from_index(memory_id)
            
            # Update analytics
            self.analytics.total_memories[memory.memory_type] = max(
                0, self.analytics.total_memories.get(memory.memory_type, 0) - 1
            )
            
            logger.debug("Deleted memory %s", memory_id)
            return True
        
        except Exception as e:
            logger.error("Failed to delete memory %s: %s", memory_id, str(e))
            return False
    
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
        memory = EpisodicMemory(
            content=content,
            agent_id=agent_id,
            session_id=session_id,
            conversation_id=conversation_id,
            participants=participants or [],
            actions=actions or [],
            outcomes=outcomes or [],
            **kwargs
        )
        return await self.store_memory(memory)
    
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
        memory = SemanticMemory(
            content={"concept": concept, "definition": definition},
            concept=concept,
            definition=definition,
            agent_id=agent_id,
            properties=properties or {},
            relations=relations or {},
            confidence=confidence,
            evidence=evidence or [],
            **kwargs
        )
        return await self.store_memory(memory)
    
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
        memory = WorkingMemory(
            content=content,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            current_focus=current_focus,
            persistence_level=MemoryPersistenceLevel.TEMPORARY,
            **kwargs
        )
        return await self.store_memory(memory)
    
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
        memory = ProceduralMemory(
            content={"skill": skill_name, "steps": steps},
            skill_name=skill_name,
            steps=steps,
            agent_id=agent_id,
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            **kwargs
        )
        return await self.store_memory(memory)
    
    async def get_session_memories(self, session_id: str) -> List[MemoryEntry]:
        """Get all memories for a session."""
        memory_ids = self.session_memories.get(session_id, [])
        memories = []
        
        for memory_id in memory_ids:
            memory = await self.retrieve_memory(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    async def clear_session_memories(self, session_id: str) -> int:
        """Clear all temporary memories for a session."""
        memories = await self.get_session_memories(session_id)
        deleted_count = 0
        
        for memory in memories:
            if memory.persistence_level == MemoryPersistenceLevel.TEMPORARY:
                if await self.delete_memory(memory.memory_id):
                    deleted_count += 1
        
        # Clean up session tracking
        if session_id in self.session_memories:
            del self.session_memories[session_id]
        self.active_sessions.discard(session_id)
        
        logger.info("Cleared %d temporary memories for session %s", deleted_count, session_id)
        return deleted_count
    
    async def consolidate_memories(self, memory_type: Optional[MemoryType] = None) -> Dict[str, int]:
        """Manually trigger memory consolidation."""
        return await self.consolidation_manager.consolidate_memories(
            memory_type=memory_type,
            memory_manager=self
        )
    
    async def get_memory_associations(self, memory_id: str) -> List[MemoryAssociation]:
        """Get associations for a memory."""
        return await self.store_manager.get_associations(memory_id)
    
    async def create_memory_association(
        self,
        memory_id_1: str,
        memory_id_2: str,
        association_type: str,
        strength: float
    ) -> str:
        """Create an association between memories."""
        return await self.store_manager.create_association(
            memory_id_1, memory_id_2, association_type, strength
        )
    
    async def find_memory_patterns(self, pattern_type: str) -> List[MemoryPattern]:
        """Find patterns in memory access or content."""
        return await self.index_manager.find_patterns(pattern_type)
    
    async def get_analytics(self) -> MemoryAnalytics:
        """Get comprehensive memory analytics."""
        # Update analytics from store
        store_analytics = await self.store_manager.get_analytics()
        
        # Combine with local analytics
        self.analytics.cache_hit_rate = (
            self.performance_metrics['cache_hits'] / 
            max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
        )
        
        self.analytics.average_retrieval_time = self.performance_metrics['avg_retrieval_time']
        
        return self.analytics
    
    async def cleanup_expired_memories(self) -> int:
        """Clean up expired memories based on retention policies."""
        deleted_count = 0
        current_time = datetime.now()
        
        for memory_type, retention_period in self.config.retention_policies.items():
            expiry_time = current_time - retention_period
            
            # Query for expired memories
            query = MemoryQuery(
                memory_types=[memory_type],
                time_range=(datetime.min, expiry_time),
                limit=1000
            )
            
            result = await self.search_memories(query)
            
            # Delete expired memories
            for memory in result.memories:
                # Skip permanent memories
                if memory.persistence_level == MemoryPersistenceLevel.PERMANENT:
                    continue
                
                # Skip high importance memories in long-term storage
                if (memory.memory_type == MemoryType.LONG_TERM and 
                    memory.metadata.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]):
                    continue
                
                if await self.delete_memory(memory.memory_id):
                    deleted_count += 1
        
        logger.info("Cleaned up %d expired memories", deleted_count)
        return deleted_count
    
    async def optimize_memory_usage(self):
        """Optimize memory usage and performance."""
        # Clear oversized caches
        for memory_type, cache in self.memory_caches.items():
            if len(cache) > self.config.storage_config.cache_size:
                # Keep most recently accessed
                sorted_memories = sorted(
                    cache.values(),
                    key=lambda m: m.metadata.last_accessed,
                    reverse=True
                )
                
                # Clear excess memories
                excess_count = len(cache) - self.config.storage_config.cache_size
                for memory in sorted_memories[self.config.storage_config.cache_size:]:
                    del cache[memory.memory_id]
                
                logger.debug("Cleared %d memories from %s cache", excess_count, memory_type.value)
        
        # Trigger consolidation
        await self.consolidate_memories()
        
        # Update analytics
        await self.get_analytics()
    
    async def export_memories(
        self,
        export_format: str = "json",
        memory_types: Optional[List[MemoryType]] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export memories for backup or transfer."""
        query = MemoryQuery(
            memory_types=memory_types or list(MemoryType),
            agent_id=agent_id,
            limit=10000
        )
        
        result = await self.search_memories(query)
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_memories": len(result.memories),
            "memory_types": [mt.value for mt in (memory_types or list(MemoryType))],
            "agent_id": agent_id,
            "memories": []
        }
        
        for memory in result.memories:
            if export_format == "json":
                export_data["memories"].append(memory.dict())
        
        return export_data
    
    async def import_memories(self, import_data: Dict[str, Any]) -> int:
        """Import memories from export data."""
        imported_count = 0
        
        for memory_data in import_data.get("memories", []):
            try:
                # Determine memory type and create appropriate object
                memory_type = MemoryType(memory_data["memory_type"])
                
                if memory_type == MemoryType.EPISODIC:
                    memory = EpisodicMemory(**memory_data)
                elif memory_type == MemoryType.SEMANTIC:
                    memory = SemanticMemory(**memory_data)
                elif memory_type == MemoryType.WORKING:
                    memory = WorkingMemory(**memory_data)
                elif memory_type == MemoryType.PROCEDURAL:
                    memory = ProceduralMemory(**memory_data)
                else:
                    memory = MemoryEntry(**memory_data)
                
                await self.store_memory(memory)
                imported_count += 1
            
            except Exception as e:
                logger.warning("Failed to import memory: %s", str(e))
                continue
        
        logger.info("Imported %d memories", imported_count)
        return imported_count
    
    async def shutdown(self):
        """Shutdown the memory manager."""
        # Cancel background tasks
        for task in [self._consolidation_task, self._cleanup_task, self._analytics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown component managers
        await self.store_manager.shutdown()
        await self.consolidation_manager.shutdown()
        await self.index_manager.shutdown()
        
        logger.info("MemoryManager shutdown complete")
    
    # Private methods
    async def _validate_memory(self, memory: MemoryEntry):
        """Validate memory entry."""
        if not memory.content:
            raise ValueError("Memory content cannot be empty")
        
        if memory.memory_type not in MemoryType:
            raise ValueError(f"Invalid memory type: {memory.memory_type}")
    
    async def _check_memory_limits(self, memory_type: MemoryType):
        """Check if memory type has reached its limit."""
        current_count = len(self.memory_caches[memory_type])
        limit = self.config.memory_limits.get(memory_type, 1000)
        
        if current_count >= limit:
            # Trigger cleanup for this memory type
            await self._cleanup_memory_type(memory_type)
    
    async def _cleanup_memory_type(self, memory_type: MemoryType):
        """Clean up oldest memories for a specific type."""
        cache = self.memory_caches[memory_type]
        limit = self.config.memory_limits.get(memory_type, 1000)
        
        if len(cache) >= limit:
            # Sort by last accessed time
            sorted_memories = sorted(
                cache.values(),
                key=lambda m: m.metadata.last_accessed
            )
            
            # Delete oldest memories (keep 80% of limit)
            keep_count = int(limit * 0.8)
            to_delete = sorted_memories[:len(sorted_memories) - keep_count]
            
            for memory in to_delete:
                await self.delete_memory(memory.memory_id)
    
    async def _generate_embedding(self, memory: MemoryEntry) -> Optional[List[float]]:
        """Generate embedding for memory content."""
        if not self.config.enable_embeddings:
            return None
        
        # This would integrate with the embedding system from RAG
        # For now, return None
        return None
    
    async def _update_access_metadata(self, memory: MemoryEntry):
        """Update memory access metadata."""
        memory.metadata.last_accessed = datetime.now()
        memory.metadata.access_count += 1
        
        # Update in backend asynchronously
        asyncio.create_task(self.store_manager.update_memory(memory))
    
    async def _filter_and_score_memories(
        self,
        memories: List[MemoryEntry],
        query: MemoryQuery
    ) -> List[tuple[MemoryEntry, float]]:
        """Filter and score memories based on query."""
        scored_memories = []
        
        for memory in memories:
            score = 0.0
            
            # Time-based scoring
            if query.time_range:
                start_time, end_time = query.time_range
                if start_time <= memory.metadata.created_at <= end_time:
                    score += 0.3
            
            # Importance scoring
            if query.importance_threshold:
                if memory.metadata.importance.value >= query.importance_threshold.value:
                    score += 0.2
            
            # Tag matching
            if query.tags:
                matching_tags = set(query.tags) & set(memory.metadata.tags)
                score += len(matching_tags) / len(query.tags) * 0.3
            
            # Content filtering
            if query.content_filter:
                # Simple content matching
                content_str = str(memory.content).lower()
                for key, value in query.content_filter.items():
                    if str(value).lower() in content_str:
                        score += 0.2
            
            # Base relevance score
            score += 0.1
            
            if score > 0:
                scored_memories.append((memory, score))
        
        return scored_memories
    
    def _update_average_time(self, metric_key: str, new_time: float):
        """Update running average for timing metrics."""
        current_avg = self.performance_metrics[metric_key]
        count_key = 'total_stores' if 'store' in metric_key else 'total_retrievals'
        count = self.performance_metrics[count_key]
        
        self.performance_metrics[metric_key] = (
            (current_avg * (count - 1) + new_time) / count
        )
    
    async def _consolidation_loop(self):
        """Background consolidation loop."""
        while True:
            try:
                await asyncio.sleep(
                    self.config.consolidation_config.consolidation_interval.total_seconds()
                )
                await self.consolidate_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in consolidation loop: %s", str(e))
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_expired_memories()
                await self.optimize_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop: %s", str(e))
    
    async def _analytics_loop(self):
        """Background analytics update loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                await self.get_analytics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in analytics loop: %s", str(e))
