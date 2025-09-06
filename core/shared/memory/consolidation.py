"""
Memory Consolidation Management

Handles memory consolidation strategies including compression,
clustering, and importance-based retention.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
import json

from .types import (
    MemoryEntry, MemoryType, MemoryImportance, MemoryConsolidationConfig,
    MemoryCompressionType, BaseMemoryConsolidator, EpisodicMemory,
    SemanticMemory, WorkingMemory, ProceduralMemory
)

logger = logging.getLogger(__name__)


class SummarizationConsolidator(BaseMemoryConsolidator):
    """Consolidates memories using summarization techniques."""
    
    async def consolidate_memories(
        self,
        memories: List[MemoryEntry],
        config: MemoryConsolidationConfig
    ) -> List[MemoryEntry]:
        """Consolidate memories by summarizing related content."""
        if len(memories) < 2:
            return memories
        
        # Group memories by similarity
        groups = await self._group_similar_memories(memories, config)
        
        consolidated_memories = []
        
        for group in groups:
            if len(group) >= config.cluster_size_threshold:
                # Create summarized memory
                summary_memory = await self._create_summary_memory(group, config)
                consolidated_memories.append(summary_memory)
                
                # Mark original memories for deletion (by setting very low importance)
                for memory in group:
                    memory.metadata.importance = MemoryImportance.MINIMAL
            else:
                # Keep individual memories
                consolidated_memories.extend(group)
        
        return consolidated_memories
    
    async def should_consolidate(
        self,
        memory: MemoryEntry,
        config: MemoryConsolidationConfig
    ) -> bool:
        """Determine if a memory should be consolidated."""
        # Check age
        age = datetime.now() - memory.metadata.created_at
        if age < config.consolidation_interval:
            return False
        
        # Check importance
        if memory.metadata.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
            return False
        
        # Check access frequency
        if memory.metadata.access_count > 10:  # Frequently accessed
            return False
        
        return True
    
    async def _group_similar_memories(
        self,
        memories: List[MemoryEntry],
        config: MemoryConsolidationConfig
    ) -> List[List[MemoryEntry]]:
        """Group memories by similarity."""
        groups = []
        unassigned = memories.copy()
        
        while unassigned:
            current_memory = unassigned.pop(0)
            current_group = [current_memory]
            
            # Find similar memories
            for memory in unassigned[:]:
                if await self._are_similar(current_memory, memory, config):
                    current_group.append(memory)
                    unassigned.remove(memory)
            
            groups.append(current_group)
        
        return groups
    
    async def _are_similar(
        self,
        memory1: MemoryEntry,
        memory2: MemoryEntry,
        config: MemoryConsolidationConfig
    ) -> bool:
        """Check if two memories are similar enough to consolidate."""
        # Same memory type
        if memory1.memory_type != memory2.memory_type:
            return False
        
        # Same agent
        if memory1.agent_id != memory2.agent_id:
            return False
        
        # Time proximity (within a day)
        time_diff = abs((memory1.metadata.created_at - memory2.metadata.created_at).total_seconds())
        if time_diff > 86400:  # 24 hours
            return False
        
        # Content similarity (simple text comparison)
        content1_str = str(memory1.content).lower()
        content2_str = str(memory2.content).lower()
        
        words1 = set(content1_str.split())
        words2 = set(content2_str.split())
        
        if not words1 or not words2:
            return False
        
        similarity = len(words1 & words2) / len(words1 | words2)
        return similarity >= config.similarity_threshold
    
    async def _create_summary_memory(
        self,
        memories: List[MemoryEntry],
        config: MemoryConsolidationConfig
    ) -> MemoryEntry:
        """Create a summarized memory from a group of memories."""
        # Use the first memory as template
        template = memories[0]
        
        # Combine content
        combined_content = {
            "summary_type": "consolidated",
            "original_count": len(memories),
            "original_memory_ids": [m.memory_id for m in memories],
            "time_range": {
                "start": min(m.metadata.created_at for m in memories).isoformat(),
                "end": max(m.metadata.created_at for m in memories).isoformat()
            },
            "content_summary": await self._summarize_content(memories)
        }
        
        # Create new memory
        if template.memory_type == MemoryType.EPISODIC:
            summary_memory = EpisodicMemory(
                content=combined_content,
                agent_id=template.agent_id,
                session_id=template.session_id,
                conversation_id=template.conversation_id
            )
        elif template.memory_type == MemoryType.SEMANTIC:
            summary_memory = SemanticMemory(
                content=combined_content,
                concept=f"consolidated_{template.concept}",
                definition="Consolidated semantic knowledge",
                agent_id=template.agent_id
            )
        else:
            summary_memory = MemoryEntry(
                content=combined_content,
                memory_type=template.memory_type,
                agent_id=template.agent_id,
                session_id=template.session_id,
                conversation_id=template.conversation_id
            )
        
        # Set metadata
        summary_memory.metadata.importance = MemoryImportance.MEDIUM
        summary_memory.metadata.tags = list(set().union(*(m.metadata.tags for m in memories)))
        summary_memory.metadata.related_memories = [m.memory_id for m in memories]
        
        return summary_memory
    
    async def _summarize_content(self, memories: List[MemoryEntry]) -> str:
        """Create a text summary of memory contents."""
        # Simple summarization - in practice, this could use LLMs
        contents = []
        
        for memory in memories:
            if isinstance(memory.content, dict):
                content_str = json.dumps(memory.content)
            else:
                content_str = str(memory.content)
            contents.append(content_str)
        
        # Extract key themes
        all_words = []
        for content in contents:
            all_words.extend(content.lower().split())
        
        word_counts = Counter(all_words)
        key_words = [word for word, count in word_counts.most_common(10) if len(word) > 3]
        
        return f"Consolidated memory covering themes: {', '.join(key_words)}. " \
               f"Based on {len(memories)} original memories. " \
               f"Key content: {' | '.join(contents[:3])}"


class ClusteringConsolidator(BaseMemoryConsolidator):
    """Consolidates memories using clustering algorithms."""
    
    async def consolidate_memories(
        self,
        memories: List[MemoryEntry],
        config: MemoryConsolidationConfig
    ) -> List[MemoryEntry]:
        """Consolidate memories using clustering."""
        if len(memories) < config.cluster_size_threshold:
            return memories
        
        # Create feature vectors for clustering
        feature_vectors = await self._create_feature_vectors(memories)
        
        # Perform clustering
        clusters = await self._cluster_memories(feature_vectors, config)
        
        consolidated_memories = []
        
        for cluster_indices in clusters:
            cluster_memories = [memories[i] for i in cluster_indices]
            
            if len(cluster_memories) >= config.cluster_size_threshold:
                # Create cluster representative
                representative = await self._create_cluster_representative(cluster_memories)
                consolidated_memories.append(representative)
            else:
                consolidated_memories.extend(cluster_memories)
        
        return consolidated_memories
    
    async def should_consolidate(
        self,
        memory: MemoryEntry,
        config: MemoryConsolidationConfig
    ) -> bool:
        """Determine if a memory should be consolidated."""
        # Basic criteria similar to summarization
        age = datetime.now() - memory.metadata.created_at
        return (age > config.consolidation_interval and 
                memory.metadata.importance not in [MemoryImportance.CRITICAL, MemoryImportance.HIGH])
    
    async def _create_feature_vectors(self, memories: List[MemoryEntry]) -> List[List[float]]:
        """Create feature vectors for clustering."""
        vectors = []
        
        for memory in memories:
            # Simple feature extraction
            features = []
            
            # Memory type (one-hot)
            type_features = [0.0] * len(MemoryType)
            type_features[list(MemoryType).index(memory.memory_type)] = 1.0
            features.extend(type_features)
            
            # Importance level
            importance_values = {
                MemoryImportance.MINIMAL: 0.1,
                MemoryImportance.LOW: 0.3,
                MemoryImportance.MEDIUM: 0.5,
                MemoryImportance.HIGH: 0.8,
                MemoryImportance.CRITICAL: 1.0
            }
            features.append(importance_values[memory.metadata.importance])
            
            # Access frequency (normalized)
            features.append(min(memory.metadata.access_count / 100.0, 1.0))
            
            # Age (normalized)
            age_days = (datetime.now() - memory.metadata.created_at).days
            features.append(min(age_days / 365.0, 1.0))
            
            # Content length (normalized)
            content_length = len(str(memory.content))
            features.append(min(content_length / 10000.0, 1.0))
            
            vectors.append(features)
        
        return vectors
    
    async def _cluster_memories(
        self,
        feature_vectors: List[List[float]],
        config: MemoryConsolidationConfig
    ) -> List[List[int]]:
        """Perform simple clustering."""
        # Simple distance-based clustering
        clusters = []
        unassigned = list(range(len(feature_vectors)))
        
        while unassigned:
            # Start new cluster with first unassigned point
            current_idx = unassigned.pop(0)
            current_cluster = [current_idx]
            current_vector = feature_vectors[current_idx]
            
            # Find similar points
            for idx in unassigned[:]:
                distance = self._calculate_distance(current_vector, feature_vectors[idx])
                if distance < (1.0 - config.similarity_threshold):
                    current_cluster.append(idx)
                    unassigned.remove(idx)
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _calculate_distance(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate Euclidean distance between feature vectors."""
        if len(vector1) != len(vector2):
            return float('inf')
        
        distance = sum((a - b) ** 2 for a, b in zip(vector1, vector2)) ** 0.5
        return distance / len(vector1) ** 0.5  # Normalize
    
    async def _create_cluster_representative(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """Create a representative memory for a cluster."""
        # Use the most important or most accessed memory as base
        representative = max(memories, key=lambda m: (
            {MemoryImportance.MINIMAL: 1, MemoryImportance.LOW: 2, 
             MemoryImportance.MEDIUM: 3, MemoryImportance.HIGH: 4, 
             MemoryImportance.CRITICAL: 5}[m.metadata.importance],
            m.metadata.access_count
        ))
        
        # Update metadata to indicate consolidation
        representative.metadata.related_memories = [m.memory_id for m in memories if m != representative]
        representative.metadata.tags.append("consolidated_cluster")
        
        return representative


class HierarchicalConsolidator(BaseMemoryConsolidator):
    """Consolidates memories using hierarchical structures."""
    
    async def consolidate_memories(
        self,
        memories: List[MemoryEntry],
        config: MemoryConsolidationConfig
    ) -> List[MemoryEntry]:
        """Consolidate memories into hierarchical structure."""
        # Build hierarchy based on content relationships
        hierarchy = await self._build_hierarchy(memories, config)
        
        consolidated_memories = []
        
        for level_memories in hierarchy.values():
            if len(level_memories) >= config.cluster_size_threshold:
                # Create parent memory for this level
                parent_memory = await self._create_parent_memory(level_memories)
                consolidated_memories.append(parent_memory)
                
                # Keep only important children
                for memory in level_memories:
                    if memory.metadata.importance in [MemoryImportance.HIGH, MemoryImportance.CRITICAL]:
                        memory.metadata.parent_memory = parent_memory.memory_id
                        consolidated_memories.append(memory)
            else:
                consolidated_memories.extend(level_memories)
        
        return consolidated_memories
    
    async def should_consolidate(
        self,
        memory: MemoryEntry,
        config: MemoryConsolidationConfig
    ) -> bool:
        """Determine if a memory should be consolidated."""
        return (memory.metadata.importance in [MemoryImportance.LOW, MemoryImportance.MINIMAL] and
                memory.metadata.access_count < 5)
    
    async def _build_hierarchy(
        self,
        memories: List[MemoryEntry],
        config: MemoryConsolidationConfig
    ) -> Dict[str, List[MemoryEntry]]:
        """Build hierarchical structure of memories."""
        hierarchy = defaultdict(list)
        
        # Group by topic/theme
        for memory in memories:
            # Simple topic extraction from content
            topic = await self._extract_topic(memory)
            hierarchy[topic].append(memory)
        
        return hierarchy
    
    async def _extract_topic(self, memory: MemoryEntry) -> str:
        """Extract main topic from memory."""
        # Simple topic extraction
        content_str = str(memory.content).lower()
        
        # Use memory type and first few words
        words = content_str.split()[:5]
        topic = f"{memory.memory_type.value}_{'-'.join(words)}"
        
        return topic
    
    async def _create_parent_memory(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """Create parent memory for a group."""
        template = memories[0]
        
        parent_content = {
            "type": "hierarchical_parent",
            "child_count": len(memories),
            "child_memory_ids": [m.memory_id for m in memories],
            "topic_summary": await self._summarize_topic(memories),
            "time_span": {
                "start": min(m.metadata.created_at for m in memories).isoformat(),
                "end": max(m.metadata.created_at for m in memories).isoformat()
            }
        }
        
        parent_memory = MemoryEntry(
            content=parent_content,
            memory_type=template.memory_type,
            agent_id=template.agent_id,
            session_id=template.session_id
        )
        
        parent_memory.metadata.importance = MemoryImportance.MEDIUM
        parent_memory.metadata.child_memories = [m.memory_id for m in memories]
        
        return parent_memory
    
    async def _summarize_topic(self, memories: List[MemoryEntry]) -> str:
        """Summarize the main topic of a group of memories."""
        # Extract common themes
        all_content = " ".join(str(m.content) for m in memories)
        words = all_content.lower().split()
        
        word_counts = Counter(words)
        common_words = [word for word, count in word_counts.most_common(5) if len(word) > 3]
        
        return f"Topic covering: {', '.join(common_words)}"


class MemoryConsolidationManager:
    """Manager for memory consolidation strategies."""
    
    def __init__(self, config: MemoryConsolidationConfig):
        self.config = config
        self.consolidators: Dict[MemoryCompressionType, BaseMemoryConsolidator] = {}
        
        # Statistics
        self.consolidation_stats = {
            'total_consolidations': 0,
            'memories_processed': 0,
            'memories_consolidated': 0,
            'last_consolidation': None
        }
    
    async def initialize(self):
        """Initialize consolidation strategies."""
        self.consolidators[MemoryCompressionType.SUMMARIZATION] = SummarizationConsolidator()
        self.consolidators[MemoryCompressionType.CLUSTERING] = ClusteringConsolidator()
        self.consolidators[MemoryCompressionType.HIERARCHICAL] = HierarchicalConsolidator()
        
        logger.info("Memory consolidation manager initialized with %d strategies", 
                   len(self.consolidators))
    
    async def consolidate_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        memory_manager=None
    ) -> Dict[str, int]:
        """Perform memory consolidation."""
        if not self.config.enable_consolidation:
            return {}
        
        start_time = datetime.now()
        results = {}
        
        try:
            # Get consolidator
            consolidator = self.consolidators.get(
                self.config.compression_type,
                self.consolidators[MemoryCompressionType.SUMMARIZATION]
            )
            
            # Process each memory type
            memory_types = [memory_type] if memory_type else list(MemoryType)
            
            for mt in memory_types:
                # Get candidates for consolidation
                candidates = await self._get_consolidation_candidates(mt, memory_manager)
                
                if len(candidates) < self.config.cluster_size_threshold:
                    results[mt.value] = 0
                    continue
                
                # Perform consolidation
                consolidated = await consolidator.consolidate_memories(candidates, self.config)
                
                # Update memory store
                if memory_manager:
                    consolidated_count = await self._update_memory_store(
                        candidates, consolidated, memory_manager
                    )
                    results[mt.value] = consolidated_count
                else:
                    results[mt.value] = len(candidates) - len(consolidated)
            
            # Update statistics
            total_processed = sum(len(candidates) for candidates in results.values())
            total_consolidated = sum(results.values())
            
            self.consolidation_stats['total_consolidations'] += 1
            self.consolidation_stats['memories_processed'] += total_processed
            self.consolidation_stats['memories_consolidated'] += total_consolidated
            self.consolidation_stats['last_consolidation'] = start_time.isoformat()
            
            consolidation_time = (datetime.now() - start_time).total_seconds()
            logger.info("Consolidated %d memories in %.2fs", total_consolidated, consolidation_time)
            
            return results
        
        except Exception as e:
            logger.error("Error during memory consolidation: %s", str(e))
            return {}
    
    async def _get_consolidation_candidates(
        self,
        memory_type: MemoryType,
        memory_manager
    ) -> List[MemoryEntry]:
        """Get memories that are candidates for consolidation."""
        if not memory_manager:
            return []
        
        # Query for old, low-importance memories
        from .types import MemoryQuery  # Import here to avoid circular imports
        
        cutoff_time = datetime.now() - self.config.consolidation_interval
        
        query = MemoryQuery(
            memory_types=[memory_type],
            time_range=(datetime.min, cutoff_time),
            importance_threshold=MemoryImportance.MEDIUM,
            limit=1000
        )
        
        result = await memory_manager.search_memories(query)
        
        # Filter by consolidation criteria
        candidates = []
        consolidator = self.consolidators.get(
            self.config.compression_type,
            self.consolidators[MemoryCompressionType.SUMMARIZATION]
        )
        
        for memory in result.memories:
            if await consolidator.should_consolidate(memory, self.config):
                candidates.append(memory)
        
        return candidates
    
    async def _update_memory_store(
        self,
        original_memories: List[MemoryEntry],
        consolidated_memories: List[MemoryEntry],
        memory_manager
    ) -> int:
        """Update memory store with consolidated memories."""
        consolidated_count = 0
        
        # Store new consolidated memories
        for memory in consolidated_memories:
            # Check if this is a new memory (consolidation result)
            if memory.memory_id not in [m.memory_id for m in original_memories]:
                await memory_manager.store_memory(memory)
                consolidated_count += 1
        
        # Remove or update original memories that were consolidated
        original_ids = {m.memory_id for m in original_memories}
        remaining_ids = {m.memory_id for m in consolidated_memories}
        
        for memory_id in original_ids - remaining_ids:
            await memory_manager.delete_memory(memory_id)
        
        return consolidated_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return self.consolidation_stats.copy()
    
    async def shutdown(self):
        """Shutdown consolidation manager."""
        logger.info("Memory consolidation manager shutdown complete")
