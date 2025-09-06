"""
Memory Indexing Management

Provides indexing strategies for fast memory retrieval including
temporal, semantic, and behavioral indices.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
import json
import hashlib

from .types import (
    MemoryEntry, MemoryQuery, MemoryType, MemorySystemConfig,
    MemoryPattern, MemoryImportance, BaseMemoryIndex
)

logger = logging.getLogger(__name__)


class TemporalIndex(BaseMemoryIndex):
    """Index memories by time for temporal queries."""
    
    def __init__(self):
        # Time-based indices
        self.daily_index: Dict[str, Set[str]] = defaultdict(set)  # YYYY-MM-DD -> memory_ids
        self.hourly_index: Dict[str, Set[str]] = defaultdict(set)  # YYYY-MM-DD-HH -> memory_ids
        self.memory_timestamps: Dict[str, datetime] = {}
    
    async def index_memory(self, memory: MemoryEntry):
        """Index a memory by its temporal properties."""
        memory_id = memory.memory_id
        timestamp = memory.metadata.created_at
        
        # Store timestamp
        self.memory_timestamps[memory_id] = timestamp
        
        # Daily index
        daily_key = timestamp.strftime("%Y-%m-%d")
        self.daily_index[daily_key].add(memory_id)
        
        # Hourly index
        hourly_key = timestamp.strftime("%Y-%m-%d-%H")
        self.hourly_index[hourly_key].add(memory_id)
    
    async def search_index(self, query: MemoryQuery) -> List[str]:
        """Search temporal index for matching memory IDs."""
        matching_ids = set()
        
        # Time range query
        if query.time_range:
            start_time, end_time = query.time_range
            
            # Find all days in range
            current_date = start_time.date()
            end_date = end_time.date()
            
            while current_date <= end_date:
                daily_key = current_date.strftime("%Y-%m-%d")
                
                if daily_key in self.daily_index:
                    # Check each memory in this day
                    for memory_id in self.daily_index[daily_key]:
                        memory_time = self.memory_timestamps.get(memory_id)
                        if memory_time and start_time <= memory_time <= end_time:
                            matching_ids.add(memory_id)
                
                current_date += timedelta(days=1)
        
        # Max age query
        elif query.max_age:
            cutoff_time = datetime.now() - query.max_age
            
            for memory_id, timestamp in self.memory_timestamps.items():
                if timestamp >= cutoff_time:
                    matching_ids.add(memory_id)
        
        return list(matching_ids)
    
    async def update_index(self, memory: MemoryEntry):
        """Update index for modified memory."""
        # Remove old entries
        await self.remove_from_index(memory.memory_id)
        
        # Re-index
        await self.index_memory(memory)
    
    async def remove_from_index(self, memory_id: str):
        """Remove memory from temporal index."""
        timestamp = self.memory_timestamps.get(memory_id)
        if not timestamp:
            return
        
        # Remove from daily index
        daily_key = timestamp.strftime("%Y-%m-%d")
        self.daily_index[daily_key].discard(memory_id)
        
        # Remove from hourly index
        hourly_key = timestamp.strftime("%Y-%m-%d-%H")
        self.hourly_index[hourly_key].discard(memory_id)
        
        # Remove timestamp
        del self.memory_timestamps[memory_id]
    
    def get_temporal_patterns(self) -> List[MemoryPattern]:
        """Identify temporal patterns in memory creation."""
        patterns = []
        
        # Daily patterns
        daily_counts = {day: len(memory_ids) for day, memory_ids in self.daily_index.items()}
        if daily_counts:
            avg_daily = sum(daily_counts.values()) / len(daily_counts)
            
            # Find high activity days
            for day, count in daily_counts.items():
                if count > avg_daily * 2:
                    pattern = MemoryPattern(
                        pattern_type="temporal_spike",
                        description=f"High memory activity on {day}: {count} memories",
                        frequency=count,
                        confidence=min(count / (avg_daily * 3), 1.0),
                        metadata={"date": day, "count": count, "avg_daily": avg_daily}
                    )
                    patterns.append(pattern)
        
        return patterns


class SemanticIndex(BaseMemoryIndex):
    """Index memories by semantic content."""
    
    def __init__(self):
        # Semantic indices
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> memory_ids
        self.concept_index: Dict[str, Set[str]] = defaultdict(set)  # concept -> memory_ids
        self.memory_keywords: Dict[str, Set[str]] = defaultdict(set)  # memory_id -> keywords
        self.memory_concepts: Dict[str, Set[str]] = defaultdict(set)  # memory_id -> concepts
    
    async def index_memory(self, memory: MemoryEntry):
        """Index a memory by its semantic content."""
        memory_id = memory.memory_id
        
        # Extract keywords
        keywords = await self._extract_keywords(memory)
        concepts = await self._extract_concepts(memory)
        
        # Store memory mappings
        self.memory_keywords[memory_id] = keywords
        self.memory_concepts[memory_id] = concepts
        
        # Update indices
        for keyword in keywords:
            self.keyword_index[keyword].add(memory_id)
        
        for concept in concepts:
            self.concept_index[concept].add(memory_id)
    
    async def search_index(self, query: MemoryQuery) -> List[str]:
        """Search semantic index for matching memory IDs."""
        matching_ids = set()
        
        if query.query_text:
            # Extract keywords from query
            query_keywords = await self._extract_keywords_from_text(query.query_text)
            
            # Find memories with matching keywords
            for keyword in query_keywords:
                if keyword in self.keyword_index:
                    matching_ids.update(self.keyword_index[keyword])
        
        # Tag-based search
        if query.tags:
            for tag in query.tags:
                if tag in self.keyword_index:
                    matching_ids.update(self.keyword_index[tag])
        
        return list(matching_ids)
    
    async def update_index(self, memory: MemoryEntry):
        """Update index for modified memory."""
        await self.remove_from_index(memory.memory_id)
        await self.index_memory(memory)
    
    async def remove_from_index(self, memory_id: str):
        """Remove memory from semantic index."""
        # Remove from keyword index
        keywords = self.memory_keywords.get(memory_id, set())
        for keyword in keywords:
            self.keyword_index[keyword].discard(memory_id)
        
        # Remove from concept index
        concepts = self.memory_concepts.get(memory_id, set())
        for concept in concepts:
            self.concept_index[concept].discard(memory_id)
        
        # Clean up memory mappings
        self.memory_keywords.pop(memory_id, None)
        self.memory_concepts.pop(memory_id, None)
    
    async def _extract_keywords(self, memory: MemoryEntry) -> Set[str]:
        """Extract keywords from memory content."""
        keywords = set()
        
        # Extract from content
        content_text = str(memory.content).lower()
        keywords.update(await self._extract_keywords_from_text(content_text))
        
        # Add tags
        keywords.update(memory.metadata.tags)
        
        # Add memory type
        keywords.add(memory.memory_type.value)
        
        # Add importance level
        keywords.add(memory.metadata.importance.value)
        
        return keywords
    
    async def _extract_concepts(self, memory: MemoryEntry) -> Set[str]:
        """Extract concepts from memory content."""
        concepts = set()
        
        # For semantic memories, use the concept field
        if hasattr(memory, 'concept'):
            concepts.add(memory.concept)
        
        # For procedural memories, use skill name
        if hasattr(memory, 'skill_name'):
            concepts.add(memory.skill_name)
        
        # Extract concepts from content
        content_text = str(memory.content).lower()
        
        # Simple concept extraction (could be enhanced with NLP)
        if 'learning' in content_text or 'knowledge' in content_text:
            concepts.add('learning')
        
        if 'problem' in content_text or 'issue' in content_text:
            concepts.add('problem_solving')
        
        if 'decision' in content_text or 'choice' in content_text:
            concepts.add('decision_making')
        
        return concepts
    
    async def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter out common words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        keywords = set()
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            
            # Filter criteria
            if (len(word) >= 3 and 
                word not in stop_words and
                not word.isdigit()):
                keywords.add(word)
        
        return keywords


class BehavioralIndex(BaseMemoryIndex):
    """Index memories by access patterns and behavior."""
    
    def __init__(self):
        self.access_frequency: Dict[str, int] = defaultdict(int)
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.memory_sessions: Dict[str, Set[str]] = defaultdict(set)  # memory_id -> session_ids
        self.session_memories: Dict[str, Set[str]] = defaultdict(set)  # session_id -> memory_ids
    
    async def index_memory(self, memory: MemoryEntry):
        """Index a memory by its behavioral properties."""
        memory_id = memory.memory_id
        
        # Track access frequency
        self.access_frequency[memory_id] = memory.metadata.access_count
        
        # Track session associations
        if memory.session_id:
            self.memory_sessions[memory_id].add(memory.session_id)
            self.session_memories[memory.session_id].add(memory_id)
    
    async def search_index(self, query: MemoryQuery) -> List[str]:
        """Search behavioral index for matching memory IDs."""
        matching_ids = set()
        
        # Session-based search
        if query.session_id:
            matching_ids.update(self.session_memories.get(query.session_id, set()))
        
        return list(matching_ids)
    
    async def update_index(self, memory: MemoryEntry):
        """Update index for modified memory."""
        memory_id = memory.memory_id
        
        # Update access frequency
        self.access_frequency[memory_id] = memory.metadata.access_count
        
        # Track access time
        self.access_patterns[memory_id].append(memory.metadata.last_accessed)
        
        # Keep only recent access times
        recent_cutoff = datetime.now() - timedelta(days=30)
        self.access_patterns[memory_id] = [
            access_time for access_time in self.access_patterns[memory_id]
            if access_time > recent_cutoff
        ]
    
    async def remove_from_index(self, memory_id: str):
        """Remove memory from behavioral index."""
        # Clean up access data
        self.access_frequency.pop(memory_id, None)
        self.access_patterns.pop(memory_id, None)
        
        # Clean up session associations
        sessions = self.memory_sessions.pop(memory_id, set())
        for session_id in sessions:
            self.session_memories[session_id].discard(memory_id)
    
    def get_access_patterns(self) -> List[MemoryPattern]:
        """Identify access patterns in memory usage."""
        patterns = []
        
        # Frequently accessed memories
        if self.access_frequency:
            avg_access = sum(self.access_frequency.values()) / len(self.access_frequency)
            
            for memory_id, access_count in self.access_frequency.items():
                if access_count > avg_access * 3:
                    pattern = MemoryPattern(
                        pattern_type="high_access_frequency",
                        description=f"Memory {memory_id} accessed {access_count} times",
                        frequency=access_count,
                        confidence=min(access_count / (avg_access * 5), 1.0),
                        associated_memories=[memory_id],
                        metadata={"memory_id": memory_id, "access_count": access_count}
                    )
                    patterns.append(pattern)
        
        # Session clustering patterns
        session_sizes = {sid: len(mids) for sid, mids in self.session_memories.items()}
        if session_sizes:
            avg_session_size = sum(session_sizes.values()) / len(session_sizes)
            
            for session_id, size in session_sizes.items():
                if size > avg_session_size * 2:
                    pattern = MemoryPattern(
                        pattern_type="large_session",
                        description=f"Session {session_id} has {size} memories",
                        frequency=size,
                        confidence=min(size / (avg_session_size * 3), 1.0),
                        metadata={"session_id": session_id, "memory_count": size}
                    )
                    patterns.append(pattern)
        
        return patterns


class CompositeIndex(BaseMemoryIndex):
    """Composite index combining multiple indexing strategies."""
    
    def __init__(self):
        self.temporal_index = TemporalIndex()
        self.semantic_index = SemanticIndex()
        self.behavioral_index = BehavioralIndex()
    
    async def index_memory(self, memory: MemoryEntry):
        """Index memory using all strategies."""
        await asyncio.gather(
            self.temporal_index.index_memory(memory),
            self.semantic_index.index_memory(memory),
            self.behavioral_index.index_memory(memory)
        )
    
    async def search_index(self, query: MemoryQuery) -> List[str]:
        """Search all indices and combine results."""
        # Get results from all indices
        temporal_results, semantic_results, behavioral_results = await asyncio.gather(
            self.temporal_index.search_index(query),
            self.semantic_index.search_index(query),
            self.behavioral_index.search_index(query)
        )
        
        # Combine results with scoring
        memory_scores = defaultdict(float)
        
        # Temporal matches
        for memory_id in temporal_results:
            memory_scores[memory_id] += 0.3
        
        # Semantic matches
        for memory_id in semantic_results:
            memory_scores[memory_id] += 0.5
        
        # Behavioral matches
        for memory_id in behavioral_results:
            memory_scores[memory_id] += 0.2
        
        # Sort by score and return
        sorted_results = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return [memory_id for memory_id, _ in sorted_results]
    
    async def update_index(self, memory: MemoryEntry):
        """Update all indices."""
        await asyncio.gather(
            self.temporal_index.update_index(memory),
            self.semantic_index.update_index(memory),
            self.behavioral_index.update_index(memory)
        )
    
    async def remove_from_index(self, memory_id: str):
        """Remove from all indices."""
        await asyncio.gather(
            self.temporal_index.remove_from_index(memory_id),
            self.semantic_index.remove_from_index(memory_id),
            self.behavioral_index.remove_from_index(memory_id)
        )


class MemoryIndexManager:
    """Manager for memory indexing strategies."""
    
    def __init__(self, config: MemorySystemConfig):
        self.config = config
        self.index = CompositeIndex()
        
        # Pattern tracking
        self.discovered_patterns: List[MemoryPattern] = []
        self.pattern_update_interval = timedelta(hours=1)
        self.last_pattern_update = datetime.now()
    
    async def initialize(self):
        """Initialize indexing manager."""
        logger.info("Memory index manager initialized")
    
    async def index_memory(self, memory: MemoryEntry):
        """Index a memory entry."""
        await self.index.index_memory(memory)
    
    async def search_index(self, query: MemoryQuery) -> List[str]:
        """Search index for matching memory IDs."""
        return await self.index.search_index(query)
    
    async def update_index(self, memory: MemoryEntry):
        """Update index for modified memory."""
        await self.index.update_index(memory)
    
    async def remove_from_index(self, memory_id: str):
        """Remove memory from index."""
        await self.index.remove_from_index(memory_id)
    
    async def find_patterns(self, pattern_type: Optional[str] = None) -> List[MemoryPattern]:
        """Find patterns in memory data."""
        # Update patterns if needed
        if datetime.now() - self.last_pattern_update > self.pattern_update_interval:
            await self._update_patterns()
        
        # Filter by pattern type if specified
        if pattern_type:
            return [p for p in self.discovered_patterns if p.pattern_type == pattern_type]
        
        return self.discovered_patterns.copy()
    
    async def _update_patterns(self):
        """Update discovered patterns."""
        self.discovered_patterns.clear()
        
        # Get patterns from all indices
        temporal_patterns = self.index.temporal_index.get_temporal_patterns()
        behavioral_patterns = self.index.behavioral_index.get_access_patterns()
        
        self.discovered_patterns.extend(temporal_patterns)
        self.discovered_patterns.extend(behavioral_patterns)
        
        # Sort patterns by confidence
        self.discovered_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        self.last_pattern_update = datetime.now()
        
        logger.debug("Updated patterns: found %d patterns", len(self.discovered_patterns))
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        return {
            "temporal_index": {
                "daily_buckets": len(self.index.temporal_index.daily_index),
                "hourly_buckets": len(self.index.temporal_index.hourly_index),
                "indexed_memories": len(self.index.temporal_index.memory_timestamps)
            },
            "semantic_index": {
                "keywords": len(self.index.semantic_index.keyword_index),
                "concepts": len(self.index.semantic_index.concept_index),
                "indexed_memories": len(self.index.semantic_index.memory_keywords)
            },
            "behavioral_index": {
                "tracked_sessions": len(self.index.behavioral_index.session_memories),
                "access_patterns": len(self.index.behavioral_index.access_patterns),
                "frequency_tracked": len(self.index.behavioral_index.access_frequency)
            },
            "patterns": {
                "discovered_patterns": len(self.discovered_patterns),
                "last_update": self.last_pattern_update.isoformat()
            }
        }
    
    async def optimize_indices(self):
        """Optimize index performance."""
        # Clean up empty buckets in temporal index
        empty_daily = [k for k, v in self.index.temporal_index.daily_index.items() if not v]
        for key in empty_daily:
            del self.index.temporal_index.daily_index[key]
        
        empty_hourly = [k for k, v in self.index.temporal_index.hourly_index.items() if not v]
        for key in empty_hourly:
            del self.index.temporal_index.hourly_index[key]
        
        # Clean up empty keyword indices
        empty_keywords = [k for k, v in self.index.semantic_index.keyword_index.items() if not v]
        for key in empty_keywords:
            del self.index.semantic_index.keyword_index[key]
        
        logger.debug("Optimized indices: removed %d empty buckets", 
                    len(empty_daily) + len(empty_hourly) + len(empty_keywords))
    
    async def shutdown(self):
        """Shutdown index manager."""
        logger.info("Memory index manager shutdown complete")
