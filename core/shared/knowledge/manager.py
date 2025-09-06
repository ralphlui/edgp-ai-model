"""
Knowledge Base Manager

Central manager for knowledge base operations including entity management,
relationship tracking, inference, and knowledge graph construction.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
import json

from .types import (
    Entity, Relationship, KnowledgeFact, KnowledgeGraph, OntologyClass,
    KnowledgeQuery, KnowledgeSearchResult, EntityType, RelationshipType,
    KnowledgeBaseConfig, KnowledgeBaseAnalytics, KnowledgeSource,
    BaseKnowledgeStore, BaseKnowledgeExtractor, BaseInferenceEngine
)
from .stores import KnowledgeStoreManager
from .extraction import KnowledgeExtractionManager
from .inference import InferenceEngineManager

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    Central knowledge base management system.
    
    Manages entities, relationships, facts, and provides intelligent
    knowledge extraction, inference, and graph operations.
    """
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
        
        # Component managers
        self.store_manager = KnowledgeStoreManager(config)
        self.extraction_manager = KnowledgeExtractionManager(config)
        self.inference_manager = InferenceEngineManager(config)
        
        # Local cache
        self.entity_cache: Dict[str, Entity] = {}
        self.relationship_cache: Dict[str, Relationship] = {}
        self.fact_cache: Dict[str, KnowledgeFact] = {}
        
        # Graph structure
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        
        # Analytics tracking
        self.analytics = KnowledgeBaseAnalytics()
        self.query_count = 0
        self.total_query_time = 0.0
        
        # Background tasks
        self._inference_task: Optional[asyncio.Task] = None
        self._validation_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize the knowledge base manager."""
        if self.initialized:
            return
        
        # Initialize component managers
        await self.store_manager.initialize()
        await self.extraction_manager.initialize()
        await self.inference_manager.initialize()
        
        # Load or create knowledge graph
        self.knowledge_graph = await self._load_or_create_graph()
        
        # Start background tasks
        if self.config.enable_inference:
            self._inference_task = asyncio.create_task(self._inference_loop())
        
        if self.config.enable_validation:
            self._validation_task = asyncio.create_task(self._validation_loop())
        
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        
        self.initialized = True
        logger.info("KnowledgeBaseManager initialized")
    
    # Entity Management
    async def create_entity(
        self,
        name: str,
        entity_type: EntityType,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Create a new entity."""
        entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
            **kwargs
        )
        
        # Generate embedding if enabled
        if self.config.enable_embeddings:
            entity.embedding = await self._generate_entity_embedding(entity)
        
        # Store entity
        success = await self.store_manager.store_entity(entity)
        if not success:
            raise RuntimeError(f"Failed to store entity {entity.entity_id}")
        
        # Update cache and graph
        self.entity_cache[entity.entity_id] = entity
        if self.knowledge_graph:
            self.knowledge_graph.entities[entity.entity_id] = entity
            self.knowledge_graph.entity_count += 1
            self.knowledge_graph.updated_at = datetime.now()
        
        # Update analytics
        self.analytics.total_entities += 1
        self.analytics.entities_added_today += 1
        
        logger.debug("Created entity %s: %s", entity.entity_id, entity.name)
        return entity.entity_id
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        # Check cache first
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]
        
        # Retrieve from store
        entity = await self.store_manager.get_entity(entity_id)
        if entity:
            self.entity_cache[entity_id] = entity
        
        return entity
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        entity.updated_at = datetime.now()
        
        # Update in store
        success = await self.store_manager.update_entity(entity)
        if not success:
            return False
        
        # Update cache and graph
        self.entity_cache[entity.entity_id] = entity
        if self.knowledge_graph and entity.entity_id in self.knowledge_graph.entities:
            self.knowledge_graph.entities[entity.entity_id] = entity
            self.knowledge_graph.updated_at = datetime.now()
        
        return True
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        # Get related relationships
        related_relationships = await self._get_entity_relationships(entity_id)
        
        # Delete relationships first
        for rel in related_relationships:
            await self.delete_relationship(rel.relationship_id)
        
        # Delete entity
        success = await self.store_manager.delete_entity(entity_id)
        if not success:
            return False
        
        # Update cache and graph
        self.entity_cache.pop(entity_id, None)
        if self.knowledge_graph:
            self.knowledge_graph.entities.pop(entity_id, None)
            self.knowledge_graph.entity_count = max(0, self.knowledge_graph.entity_count - 1)
            self.knowledge_graph.updated_at = datetime.now()
        
        # Update analytics
        self.analytics.total_entities = max(0, self.analytics.total_entities - 1)
        
        return True
    
    # Relationship Management
    async def create_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None,
        strength: float = 0.5,
        confidence: float = 0.8,
        **kwargs
    ) -> str:
        """Create a new relationship."""
        # Verify entities exist
        source_entity = await self.get_entity(source_entity_id)
        target_entity = await self.get_entity(target_entity_id)
        
        if not source_entity or not target_entity:
            raise ValueError("Source or target entity not found")
        
        relationship = Relationship(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            properties=properties or {},
            strength=strength,
            confidence=confidence,
            **kwargs
        )
        
        # Store relationship
        success = await self.store_manager.store_relationship(relationship)
        if not success:
            raise RuntimeError(f"Failed to store relationship {relationship.relationship_id}")
        
        # Update cache and graph
        self.relationship_cache[relationship.relationship_id] = relationship
        if self.knowledge_graph:
            self.knowledge_graph.relationships[relationship.relationship_id] = relationship
            self.knowledge_graph.relationship_count += 1
            self.knowledge_graph.updated_at = datetime.now()
        
        # Update analytics
        self.analytics.total_relationships += 1
        self.analytics.relationships_added_today += 1
        
        logger.debug("Created relationship %s: %s -> %s (%s)",
                    relationship.relationship_id, source_entity_id, 
                    target_entity_id, relationship_type.value)
        
        return relationship.relationship_id
    
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        # Check cache first
        if relationship_id in self.relationship_cache:
            return self.relationship_cache[relationship_id]
        
        # Retrieve from store
        relationship = await self.store_manager.get_relationship(relationship_id)
        if relationship:
            self.relationship_cache[relationship_id] = relationship
        
        return relationship
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        success = await self.store_manager.delete_relationship(relationship_id)
        if not success:
            return False
        
        # Update cache and graph
        self.relationship_cache.pop(relationship_id, None)
        if self.knowledge_graph:
            self.knowledge_graph.relationships.pop(relationship_id, None)
            self.knowledge_graph.relationship_count = max(0, self.knowledge_graph.relationship_count - 1)
            self.knowledge_graph.updated_at = datetime.now()
        
        # Update analytics
        self.analytics.total_relationships = max(0, self.analytics.total_relationships - 1)
        
        return True
    
    # Knowledge Search and Query
    async def search_knowledge(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """Search the knowledge base."""
        start_time = datetime.now()
        
        try:
            # Use store manager for search
            result = await self.store_manager.search_knowledge(query)
            
            # Enhanced semantic search if query text provided
            if query.query_text and self.config.enable_embeddings:
                semantic_results = await self._semantic_search(query)
                result = await self._merge_search_results(result, semantic_results)
            
            # Apply additional filtering
            result = await self._filter_search_results(result, query)
            
            # Update analytics
            search_time = (datetime.now() - start_time).total_seconds()
            self.query_count += 1
            self.total_query_time += search_time
            
            result.query_time = search_time
            
            return result
        
        except Exception as e:
            logger.error("Failed to search knowledge: %s", str(e))
            return KnowledgeSearchResult(
                entities=[], relationships=[], facts=[],
                total_found=0, query_time=0.0, search_strategy="error"
            )
    
    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"  # "incoming", "outgoing", "both"
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        return await self._get_entity_relationships(entity_id, relationship_types, direction)
    
    async def get_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_depth: int = 1
    ) -> List[Entity]:
        """Get entities related to the given entity."""
        related_entities = []
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get relationships
            relationships = await self._get_entity_relationships(
                current_id, relationship_types
            )
            
            for rel in relationships:
                # Determine the related entity
                related_id = (rel.target_entity_id if rel.source_entity_id == current_id 
                            else rel.source_entity_id)
                
                if related_id not in visited:
                    related_entity = await self.get_entity(related_id)
                    if related_entity:
                        related_entities.append(related_entity)
                    
                    if depth < max_depth:
                        queue.append((related_id, depth + 1))
        
        return related_entities
    
    # Knowledge Extraction
    async def extract_knowledge_from_text(
        self,
        text: str,
        source: KnowledgeSource = KnowledgeSource.DOCUMENT_EXTRACTION
    ) -> Dict[str, List[str]]:
        """Extract knowledge from text."""
        if not self.config.enable_auto_extraction:
            return {"entities": [], "relationships": [], "facts": []}
        
        # Extract entities
        entities = await self.extraction_manager.extract_entities(text)
        entity_ids = []
        
        for entity in entities:
            entity.source = source
            entity_id = await self.create_entity(
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                properties=entity.properties
            )
            entity_ids.append(entity_id)
        
        # Extract relationships
        relationships = await self.extraction_manager.extract_relationships(text, entities)
        relationship_ids = []
        
        for rel in relationships:
            rel.source = source
            # Find corresponding entity IDs
            source_id = next((eid for eid, e in zip(entity_ids, entities) 
                            if e.name == rel.source_entity_id), None)
            target_id = next((eid for eid, e in zip(entity_ids, entities) 
                            if e.name == rel.target_entity_id), None)
            
            if source_id and target_id:
                rel_id = await self.create_relationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel.relationship_type,
                    properties=rel.properties,
                    strength=rel.strength,
                    confidence=rel.confidence
                )
                relationship_ids.append(rel_id)
        
        return {
            "entities": entity_ids,
            "relationships": relationship_ids,
            "facts": []  # Fact extraction could be added here
        }
    
    # Inference and Reasoning
    async def infer_new_knowledge(self) -> Dict[str, int]:
        """Infer new knowledge from existing data."""
        if not self.config.enable_inference:
            return {"relationships": 0, "facts": 0}
        
        # Get current entities and relationships
        entities = list(self.entity_cache.values())
        relationships = list(self.relationship_cache.values())
        
        # Infer new relationships
        new_relationships = await self.inference_manager.infer_relationships(entities)
        
        inferred_count = 0
        for rel in new_relationships:
            # Check if relationship already exists
            existing = await self._find_existing_relationship(
                rel.source_entity_id, rel.target_entity_id, rel.relationship_type
            )
            
            if not existing:
                rel.source = KnowledgeSource.INFERENCE
                await self.create_relationship(
                    source_entity_id=rel.source_entity_id,
                    target_entity_id=rel.target_entity_id,
                    relationship_type=rel.relationship_type,
                    properties=rel.properties,
                    strength=rel.strength,
                    confidence=rel.confidence
                )
                inferred_count += 1
        
        logger.info("Inferred %d new relationships", inferred_count)
        return {"relationships": inferred_count, "facts": 0}
    
    async def validate_knowledge_consistency(self) -> List[str]:
        """Validate knowledge base consistency."""
        if not self.knowledge_graph:
            return []
        
        return await self.inference_manager.validate_consistency(self.knowledge_graph)
    
    # Graph Operations
    async def export_knowledge_graph(self, format: str = "json") -> Any:
        """Export the knowledge graph."""
        if not self.knowledge_graph:
            return None
        
        if format == "json":
            return self.knowledge_graph.dict()
        elif format == "rdf":
            # RDF export could be implemented
            pass
        elif format == "cypher":
            # Cypher export for Neo4j could be implemented
            pass
        
        return None
    
    async def import_knowledge_graph(self, data: Any, format: str = "json") -> int:
        """Import knowledge graph data."""
        imported_count = 0
        
        if format == "json":
            if isinstance(data, dict):
                # Import entities
                for entity_data in data.get("entities", {}).values():
                    try:
                        entity = Entity(**entity_data)
                        await self.create_entity(
                            name=entity.name,
                            entity_type=entity.entity_type,
                            description=entity.description,
                            properties=entity.properties
                        )
                        imported_count += 1
                    except Exception as e:
                        logger.warning("Failed to import entity: %s", str(e))
                
                # Import relationships
                for rel_data in data.get("relationships", {}).values():
                    try:
                        rel = Relationship(**rel_data)
                        await self.create_relationship(
                            source_entity_id=rel.source_entity_id,
                            target_entity_id=rel.target_entity_id,
                            relationship_type=rel.relationship_type,
                            properties=rel.properties,
                            strength=rel.strength,
                            confidence=rel.confidence
                        )
                        imported_count += 1
                    except Exception as e:
                        logger.warning("Failed to import relationship: %s", str(e))
        
        return imported_count
    
    # Analytics and Monitoring
    async def get_analytics(self) -> KnowledgeBaseAnalytics:
        """Get knowledge base analytics."""
        # Update counts from actual data
        self.analytics.total_entities = len(self.entity_cache)
        self.analytics.total_relationships = len(self.relationship_cache)
        
        # Calculate averages
        if self.query_count > 0:
            self.analytics.average_query_time = self.total_query_time / self.query_count
        
        # Calculate confidence average
        if self.entity_cache:
            total_confidence = sum(e.confidence for e in self.entity_cache.values())
            self.analytics.average_confidence = total_confidence / len(self.entity_cache)
        
        # Calculate verified percentage
        if self.entity_cache:
            verified_count = sum(1 for e in self.entity_cache.values() if e.verified)
            self.analytics.verified_percentage = verified_count / len(self.entity_cache) * 100
        
        return self.analytics
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get detailed graph statistics."""
        if not self.knowledge_graph:
            return {}
        
        # Entity type distribution
        entity_types = defaultdict(int)
        for entity in self.entity_cache.values():
            entity_types[entity.entity_type.value] += 1
        
        # Relationship type distribution
        relationship_types = defaultdict(int)
        for rel in self.relationship_cache.values():
            relationship_types[rel.relationship_type.value] += 1
        
        # Connectivity metrics
        entity_connectivity = defaultdict(int)
        for rel in self.relationship_cache.values():
            entity_connectivity[rel.source_entity_id] += 1
            entity_connectivity[rel.target_entity_id] += 1
        
        avg_connectivity = (sum(entity_connectivity.values()) / len(entity_connectivity) 
                          if entity_connectivity else 0)
        
        return {
            "entity_type_distribution": dict(entity_types),
            "relationship_type_distribution": dict(relationship_types),
            "average_entity_connectivity": avg_connectivity,
            "most_connected_entities": sorted(
                entity_connectivity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    async def shutdown(self):
        """Shutdown the knowledge base manager."""
        # Cancel background tasks
        for task in [self._inference_task, self._validation_task, self._analytics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown component managers
        await self.store_manager.shutdown()
        await self.extraction_manager.shutdown()
        await self.inference_manager.shutdown()
        
        logger.info("KnowledgeBaseManager shutdown complete")
    
    # Private methods
    async def _load_or_create_graph(self) -> KnowledgeGraph:
        """Load existing graph or create new one."""
        # Try to load existing graph
        existing_graph = await self.store_manager.get_knowledge_graph()
        
        if existing_graph:
            # Load entities and relationships into cache
            for entity in existing_graph.entities.values():
                self.entity_cache[entity.entity_id] = entity
            
            for rel in existing_graph.relationships.values():
                self.relationship_cache[rel.relationship_id] = rel
            
            return existing_graph
        
        # Create new graph
        return KnowledgeGraph(
            name="Default Knowledge Graph",
            description="Main knowledge graph for the system"
        )
    
    async def _get_entity_relationships(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        relationships = []
        
        for rel in self.relationship_cache.values():
            # Check if entity is involved
            involved = False
            if direction == "outgoing" and rel.source_entity_id == entity_id:
                involved = True
            elif direction == "incoming" and rel.target_entity_id == entity_id:
                involved = True
            elif direction == "both" and (rel.source_entity_id == entity_id or 
                                        rel.target_entity_id == entity_id):
                involved = True
            
            # Check relationship type filter
            if involved and (not relationship_types or rel.relationship_type in relationship_types):
                relationships.append(rel)
        
        return relationships
    
    async def _generate_entity_embedding(self, entity: Entity) -> Optional[List[float]]:
        """Generate embedding for entity."""
        if not self.config.enable_embeddings:
            return None
        
        # This would integrate with embedding models
        # For now, return None
        return None
    
    async def _semantic_search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """Perform semantic search using embeddings."""
        # Placeholder for semantic search implementation
        return KnowledgeSearchResult(
            entities=[], relationships=[], facts=[],
            total_found=0, query_time=0.0, search_strategy="semantic"
        )
    
    async def _merge_search_results(
        self,
        result1: KnowledgeSearchResult,
        result2: KnowledgeSearchResult
    ) -> KnowledgeSearchResult:
        """Merge two search results."""
        # Combine and deduplicate results
        all_entities = {e.entity_id: e for e in result1.entities + result2.entities}
        all_relationships = {r.relationship_id: r for r in result1.relationships + result2.relationships}
        all_facts = {f.fact_id: f for f in result1.facts + result2.facts}
        
        return KnowledgeSearchResult(
            entities=list(all_entities.values()),
            relationships=list(all_relationships.values()),
            facts=list(all_facts.values()),
            total_found=len(all_entities) + len(all_relationships) + len(all_facts),
            query_time=max(result1.query_time, result2.query_time),
            search_strategy="merged"
        )
    
    async def _filter_search_results(
        self,
        result: KnowledgeSearchResult,
        query: KnowledgeQuery
    ) -> KnowledgeSearchResult:
        """Apply additional filtering to search results."""
        # Filter entities
        filtered_entities = []
        for entity in result.entities:
            if (not query.entity_types or entity.entity_type in query.entity_types) and \
               entity.confidence >= query.min_confidence and \
               entity.quality_score >= query.min_quality_score and \
               (not query.verified_only or entity.verified):
                filtered_entities.append(entity)
        
        # Filter relationships
        filtered_relationships = []
        for rel in result.relationships:
            if (not query.relationship_types or rel.relationship_type in query.relationship_types) and \
               rel.confidence >= query.min_confidence and \
               rel.strength >= query.relationship_strength_min and \
               (not query.verified_only or rel.verified):
                filtered_relationships.append(rel)
        
        # Apply limit
        filtered_entities = filtered_entities[:query.limit]
        filtered_relationships = filtered_relationships[:query.limit]
        
        return KnowledgeSearchResult(
            entities=filtered_entities,
            relationships=filtered_relationships,
            facts=result.facts[:query.limit],
            total_found=len(filtered_entities) + len(filtered_relationships) + len(result.facts),
            query_time=result.query_time,
            search_strategy=result.search_strategy + "_filtered"
        )
    
    async def _find_existing_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType
    ) -> Optional[Relationship]:
        """Find existing relationship between entities."""
        for rel in self.relationship_cache.values():
            if (rel.source_entity_id == source_id and 
                rel.target_entity_id == target_id and 
                rel.relationship_type == rel_type):
                return rel
        return None
    
    async def _inference_loop(self):
        """Background inference loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.infer_new_knowledge()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in inference loop: %s", str(e))
    
    async def _validation_loop(self):
        """Background validation loop."""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                inconsistencies = await self.validate_knowledge_consistency()
                if inconsistencies:
                    logger.warning("Found %d knowledge inconsistencies", len(inconsistencies))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in validation loop: %s", str(e))
    
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
