"""
Knowledge Inference and Reasoning System

Provides inference capabilities for deriving new knowledge from existing
data using various reasoning techniques including rule-based, graph-based,
and machine learning approaches.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, deque
import json

from .types import (
    Entity, Relationship, KnowledgeFact, KnowledgeGraph, EntityType, RelationshipType,
    FactType, KnowledgeSource, KnowledgeBaseConfig, BaseInferenceEngine
)

logger = logging.getLogger(__name__)


class RuleBasedInferenceEngine(BaseInferenceEngine):
    """Rule-based inference engine using predefined rules."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        
        # Inference rules for relationships
        self.relationship_rules = [
            # Transitivity rules
            {
                "name": "works_at_transitivity",
                "pattern": [(RelationshipType.WORKS_AT, RelationshipType.PART_OF)],
                "inferred": RelationshipType.WORKS_AT,
                "confidence": 0.8
            },
            {
                "name": "location_transitivity",
                "pattern": [(RelationshipType.LOCATED_IN, RelationshipType.LOCATED_IN)],
                "inferred": RelationshipType.LOCATED_IN,
                "confidence": 0.9
            },
            
            # Symmetry rules
            {
                "name": "collaboration_symmetry",
                "pattern": [(RelationshipType.COLLABORATES_WITH,)],
                "inferred": RelationshipType.COLLABORATES_WITH,
                "symmetric": True,
                "confidence": 0.9
            },
            {
                "name": "competition_symmetry",
                "pattern": [(RelationshipType.COMPETES_WITH,)],
                "inferred": RelationshipType.COMPETES_WITH,
                "symmetric": True,
                "confidence": 0.9
            },
            
            # Inverse rules
            {
                "name": "owns_inverse",
                "pattern": [(RelationshipType.OWNS,)],
                "inferred": RelationshipType.OWNED_BY,
                "inverse": True,
                "confidence": 0.95
            },
            {
                "name": "leads_implies_works_at",
                "pattern": [(RelationshipType.LEADS,)],
                "inferred": RelationshipType.WORKS_AT,
                "same_direction": True,
                "confidence": 0.85
            }
        ]
        
        # Entity type inference rules
        self.entity_type_rules = [
            {
                "name": "ceo_is_person",
                "condition": lambda entity: "CEO" in entity.name or "president" in entity.name.lower(),
                "inferred_type": EntityType.PERSON,
                "confidence": 0.9
            },
            {
                "name": "university_is_organization",
                "condition": lambda entity: "university" in entity.name.lower() or "college" in entity.name.lower(),
                "inferred_type": EntityType.ORGANIZATION,
                "confidence": 0.95
            },
            {
                "name": "conference_is_event",
                "condition": lambda entity: any(word in entity.name.lower() for word in ["conference", "summit", "meeting", "symposium"]),
                "inferred_type": EntityType.EVENT,
                "confidence": 0.9
            }
        ]
    
    async def infer_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Infer new relationships using rules."""
        inferred_relationships = []
        entity_map = {e.entity_id: e for e in entities}
        
        # Build relationship graph
        relationship_graph = self._build_relationship_graph(entities)
        
        # Apply relationship rules
        for rule in self.relationship_rules:
            rule_relationships = await self._apply_relationship_rule(
                rule, relationship_graph, entity_map
            )
            inferred_relationships.extend(rule_relationships)
        
        return inferred_relationships
    
    async def infer_entity_types(self, entities: List[Entity]) -> List[Entity]:
        """Infer entity types using rules."""
        updated_entities = []
        
        for entity in entities:
            for rule in self.entity_type_rules:
                if rule["condition"](entity):
                    if entity.entity_type != rule["inferred_type"]:
                        # Create updated entity with new type
                        updated_entity = Entity(
                            entity_id=entity.entity_id,
                            name=entity.name,
                            entity_type=rule["inferred_type"],
                            description=f"Type inferred by rule: {rule['name']}",
                            properties=entity.properties.copy(),
                            confidence=rule["confidence"],
                            quality_score=entity.quality_score,
                            verified=False,
                            source=KnowledgeSource.INFERENCE,
                            tags=entity.tags + [f"inferred_by:{rule['name']}"]
                        )
                        updated_entities.append(updated_entity)
                        break
        
        return updated_entities
    
    async def validate_consistency(self, graph: KnowledgeGraph) -> List[str]:
        """Validate knowledge consistency using rules."""
        inconsistencies = []
        
        # Check for conflicting relationships
        for entity_id, entity in graph.entities.items():
            relationships = [r for r in graph.relationships.values() 
                           if r.source_entity_id == entity_id or r.target_entity_id == entity_id]
            
            # Check for logical conflicts
            conflicts = self._check_relationship_conflicts(entity_id, relationships)
            inconsistencies.extend(conflicts)
        
        # Check entity type consistency
        type_conflicts = self._check_entity_type_consistency(graph.entities, graph.relationships)
        inconsistencies.extend(type_conflicts)
        
        return inconsistencies
    
    def _build_relationship_graph(self, entities: List[Entity]) -> Dict[str, List[Relationship]]:
        """Build a graph of relationships for inference."""
        # This would be populated with actual relationships
        # For now, return empty graph as placeholder
        return defaultdict(list)
    
    async def _apply_relationship_rule(
        self,
        rule: Dict[str, Any],
        relationship_graph: Dict[str, List[Relationship]],
        entity_map: Dict[str, Entity]
    ) -> List[Relationship]:
        """Apply a single relationship rule."""
        inferred = []
        
        if rule.get("symmetric"):
            # Apply symmetry rule
            inferred.extend(self._apply_symmetry_rule(rule, relationship_graph))
        
        elif rule.get("inverse"):
            # Apply inverse rule
            inferred.extend(self._apply_inverse_rule(rule, relationship_graph))
        
        elif rule.get("same_direction"):
            # Apply same direction rule (A->B implies A->B with different type)
            inferred.extend(self._apply_same_direction_rule(rule, relationship_graph))
        
        else:
            # Apply transitivity rule
            inferred.extend(self._apply_transitivity_rule(rule, relationship_graph))
        
        return inferred
    
    def _apply_symmetry_rule(
        self,
        rule: Dict[str, Any],
        relationship_graph: Dict[str, List[Relationship]]
    ) -> List[Relationship]:
        """Apply symmetry rule: if A->B then B->A."""
        inferred = []
        target_type = rule["pattern"][0][0]
        
        for relationships in relationship_graph.values():
            for rel in relationships:
                if rel.relationship_type == target_type:
                    # Check if reverse relationship exists
                    reverse_exists = any(
                        r.source_entity_id == rel.target_entity_id and
                        r.target_entity_id == rel.source_entity_id and
                        r.relationship_type == rule["inferred"]
                        for r_list in relationship_graph.values()
                        for r in r_list
                    )
                    
                    if not reverse_exists:
                        reverse_rel = Relationship(
                            source_entity_id=rel.target_entity_id,
                            target_entity_id=rel.source_entity_id,
                            relationship_type=rule["inferred"],
                            confidence=rule["confidence"],
                            strength=rel.strength * 0.9,  # Slightly lower strength
                            source=KnowledgeSource.INFERENCE,
                            properties={
                                "inference_rule": rule["name"],
                                "original_relationship": rel.relationship_id
                            }
                        )
                        inferred.append(reverse_rel)
        
        return inferred
    
    def _apply_inverse_rule(
        self,
        rule: Dict[str, Any],
        relationship_graph: Dict[str, List[Relationship]]
    ) -> List[Relationship]:
        """Apply inverse rule: if A owns B then B is owned by A."""
        inferred = []
        target_type = rule["pattern"][0][0]
        
        for relationships in relationship_graph.values():
            for rel in relationships:
                if rel.relationship_type == target_type:
                    inverse_rel = Relationship(
                        source_entity_id=rel.target_entity_id,
                        target_entity_id=rel.source_entity_id,
                        relationship_type=rule["inferred"],
                        confidence=rule["confidence"],
                        strength=rel.strength,
                        source=KnowledgeSource.INFERENCE,
                        properties={
                            "inference_rule": rule["name"],
                            "original_relationship": rel.relationship_id
                        }
                    )
                    inferred.append(inverse_rel)
        
        return inferred
    
    def _apply_same_direction_rule(
        self,
        rule: Dict[str, Any],
        relationship_graph: Dict[str, List[Relationship]]
    ) -> List[Relationship]:
        """Apply same direction rule: if A leads B then A works at B."""
        inferred = []
        target_type = rule["pattern"][0][0]
        
        for relationships in relationship_graph.values():
            for rel in relationships:
                if rel.relationship_type == target_type:
                    same_dir_rel = Relationship(
                        source_entity_id=rel.source_entity_id,
                        target_entity_id=rel.target_entity_id,
                        relationship_type=rule["inferred"],
                        confidence=rule["confidence"],
                        strength=rel.strength * 0.8,  # Lower strength for inferred
                        source=KnowledgeSource.INFERENCE,
                        properties={
                            "inference_rule": rule["name"],
                            "original_relationship": rel.relationship_id
                        }
                    )
                    inferred.append(same_dir_rel)
        
        return inferred
    
    def _apply_transitivity_rule(
        self,
        rule: Dict[str, Any],
        relationship_graph: Dict[str, List[Relationship]]
    ) -> List[Relationship]:
        """Apply transitivity rule: if A->B and B->C then A->C."""
        inferred = []
        pattern = rule["pattern"]
        
        if len(pattern) == 1 and len(pattern[0]) == 2:
            # Two-step transitivity: A-[type1]->B-[type2]->C implies A-[inferred]->C
            type1, type2 = pattern[0]
            
            # Find all A-[type1]->B relationships
            for relationships in relationship_graph.values():
                for rel1 in relationships:
                    if rel1.relationship_type == type1:
                        # Find B-[type2]->C relationships
                        for rel2_list in relationship_graph.values():
                            for rel2 in rel2_list:
                                if (rel2.relationship_type == type2 and
                                    rel2.source_entity_id == rel1.target_entity_id):
                                    
                                    # Create A-[inferred]->C relationship
                                    transitive_rel = Relationship(
                                        source_entity_id=rel1.source_entity_id,
                                        target_entity_id=rel2.target_entity_id,
                                        relationship_type=rule["inferred"],
                                        confidence=rule["confidence"],
                                        strength=min(rel1.strength, rel2.strength) * 0.9,
                                        source=KnowledgeSource.INFERENCE,
                                        properties={
                                            "inference_rule": rule["name"],
                                            "path": [rel1.relationship_id, rel2.relationship_id]
                                        }
                                    )
                                    inferred.append(transitive_rel)
        
        return inferred
    
    def _check_relationship_conflicts(
        self,
        entity_id: str,
        relationships: List[Relationship]
    ) -> List[str]:
        """Check for conflicting relationships."""
        conflicts = []
        
        # Define conflicting relationship types
        conflict_pairs = [
            (RelationshipType.COLLABORATES_WITH, RelationshipType.COMPETES_WITH),
            (RelationshipType.OWNS, RelationshipType.OWNED_BY),  # If both directions exist
        ]
        
        # Group relationships by target
        by_target = defaultdict(list)
        for rel in relationships:
            if rel.source_entity_id == entity_id:
                by_target[rel.target_entity_id].append(rel)
            elif rel.target_entity_id == entity_id:
                by_target[rel.source_entity_id].append(rel)
        
        # Check for conflicts
        for target_entity, target_rels in by_target.items():
            rel_types = {rel.relationship_type for rel in target_rels}
            
            for type1, type2 in conflict_pairs:
                if type1 in rel_types and type2 in rel_types:
                    conflicts.append(
                        f"Conflicting relationships between {entity_id} and {target_entity}: "
                        f"{type1.value} vs {type2.value}"
                    )
        
        return conflicts
    
    def _check_entity_type_consistency(
        self,
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship]
    ) -> List[str]:
        """Check entity type consistency with relationships."""
        inconsistencies = []
        
        # Define type constraints for relationships
        type_constraints = {
            RelationshipType.WORKS_AT: {
                "source": [EntityType.PERSON],
                "target": [EntityType.ORGANIZATION]
            },
            RelationshipType.LEADS: {
                "source": [EntityType.PERSON],
                "target": [EntityType.ORGANIZATION, EntityType.CONCEPT]
            },
            RelationshipType.LOCATED_IN: {
                "source": [EntityType.ORGANIZATION, EntityType.PERSON, EntityType.EVENT],
                "target": [EntityType.LOCATION]
            }
        }
        
        for rel in relationships.values():
            if rel.relationship_type in type_constraints:
                constraints = type_constraints[rel.relationship_type]
                
                source_entity = entities.get(rel.source_entity_id)
                target_entity = entities.get(rel.target_entity_id)
                
                if source_entity and target_entity:
                    # Check source type
                    if (constraints.get("source") and 
                        source_entity.entity_type not in constraints["source"]):
                        inconsistencies.append(
                            f"Type inconsistency: {source_entity.name} ({source_entity.entity_type.value}) "
                            f"cannot be source of {rel.relationship_type.value}"
                        )
                    
                    # Check target type
                    if (constraints.get("target") and 
                        target_entity.entity_type not in constraints["target"]):
                        inconsistencies.append(
                            f"Type inconsistency: {target_entity.name} ({target_entity.entity_type.value}) "
                            f"cannot be target of {rel.relationship_type.value}"
                        )
        
        return inconsistencies


class GraphBasedInferenceEngine(BaseInferenceEngine):
    """Graph-based inference using network analysis."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        self.min_path_confidence = 0.6
        self.max_inference_depth = 3
    
    async def infer_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Infer relationships using graph analysis."""
        inferred = []
        
        # Build entity similarity graph
        similarity_graph = await self._build_similarity_graph(entities)
        
        # Find potential relationships based on similarity
        for entity1_id, similarities in similarity_graph.items():
            for entity2_id, similarity in similarities.items():
                if similarity > 0.8:  # High similarity threshold
                    # Infer general relationship
                    inferred_rel = Relationship(
                        source_entity_id=entity1_id,
                        target_entity_id=entity2_id,
                        relationship_type=RelationshipType.RELATED_TO,
                        confidence=similarity,
                        strength=similarity * 0.8,
                        source=KnowledgeSource.INFERENCE,
                        properties={
                            "inference_method": "similarity",
                            "similarity_score": similarity
                        }
                    )
                    inferred.append(inferred_rel)
        
        return inferred
    
    async def infer_entity_types(self, entities: List[Entity]) -> List[Entity]:
        """Infer entity types using graph clustering."""
        # Placeholder for graph-based type inference
        return []
    
    async def validate_consistency(self, graph: KnowledgeGraph) -> List[str]:
        """Validate using graph metrics."""
        inconsistencies = []
        
        # Check for disconnected components that should be connected
        components = self._find_connected_components(graph)
        if len(components) > 1:
            inconsistencies.append(f"Graph has {len(components)} disconnected components")
        
        # Check for suspicious patterns
        suspicious_patterns = self._find_suspicious_patterns(graph)
        inconsistencies.extend(suspicious_patterns)
        
        return inconsistencies
    
    async def _build_similarity_graph(self, entities: List[Entity]) -> Dict[str, Dict[str, float]]:
        """Build similarity graph between entities."""
        similarity_graph = defaultdict(dict)
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                similarity = self._calculate_entity_similarity(entity1, entity2)
                if similarity > 0.5:  # Only store significant similarities
                    similarity_graph[entity1.entity_id][entity2.entity_id] = similarity
                    similarity_graph[entity2.entity_id][entity1.entity_id] = similarity
        
        return similarity_graph
    
    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between two entities."""
        similarity = 0.0
        
        # Type similarity
        if entity1.entity_type == entity2.entity_type:
            similarity += 0.3
        
        # Name similarity (simple overlap)
        name1_words = set(entity1.name.lower().split())
        name2_words = set(entity2.name.lower().split())
        if name1_words and name2_words:
            name_overlap = len(name1_words & name2_words) / len(name1_words | name2_words)
            similarity += name_overlap * 0.4
        
        # Tag similarity
        tags1 = set(entity1.tags)
        tags2 = set(entity2.tags)
        if tags1 and tags2:
            tag_overlap = len(tags1 & tags2) / len(tags1 | tags2)
            similarity += tag_overlap * 0.3
        
        return min(similarity, 1.0)
    
    def _find_connected_components(self, graph: KnowledgeGraph) -> List[Set[str]]:
        """Find connected components in the graph."""
        visited = set()
        components = []
        
        def dfs(entity_id: str, component: Set[str]):
            if entity_id in visited:
                return
            
            visited.add(entity_id)
            component.add(entity_id)
            
            # Find connected entities through relationships
            for rel in graph.relationships.values():
                if rel.source_entity_id == entity_id and rel.target_entity_id not in visited:
                    dfs(rel.target_entity_id, component)
                elif rel.target_entity_id == entity_id and rel.source_entity_id not in visited:
                    dfs(rel.source_entity_id, component)
        
        for entity_id in graph.entities:
            if entity_id not in visited:
                component = set()
                dfs(entity_id, component)
                if component:
                    components.append(component)
        
        return components
    
    def _find_suspicious_patterns(self, graph: KnowledgeGraph) -> List[str]:
        """Find suspicious patterns in the graph."""
        suspicious = []
        
        # Find entities with too many relationships (potential hubs)
        relationship_counts = defaultdict(int)
        for rel in graph.relationships.values():
            relationship_counts[rel.source_entity_id] += 1
            relationship_counts[rel.target_entity_id] += 1
        
        for entity_id, count in relationship_counts.items():
            if count > 20:  # Arbitrary threshold
                entity = graph.entities.get(entity_id)
                if entity:
                    suspicious.append(
                        f"Entity '{entity.name}' has unusually high relationship count: {count}"
                    )
        
        return suspicious


class MLBasedInferenceEngine(BaseInferenceEngine):
    """Machine learning-based inference engine."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        self.models_loaded = False
    
    async def infer_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Infer relationships using ML models."""
        if not self.models_loaded:
            logger.warning("ML models not loaded, skipping ML inference")
            return []
        
        # Placeholder for ML-based relationship inference
        # This would use trained models to predict relationships
        return []
    
    async def infer_entity_types(self, entities: List[Entity]) -> List[Entity]:
        """Infer entity types using ML classification."""
        if not self.models_loaded:
            logger.warning("ML models not loaded, skipping ML inference")
            return []
        
        # Placeholder for ML-based type inference
        # This would use trained classifiers to predict entity types
        return []
    
    async def validate_consistency(self, graph: KnowledgeGraph) -> List[str]:
        """Validate using ML anomaly detection."""
        if not self.models_loaded:
            return []
        
        # Placeholder for ML-based consistency checking
        # This would use anomaly detection models
        return []


class InferenceEngineManager:
    """Manager for inference operations."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
        
        # Initialize inference engines
        self.engines = []
        
        if config.enable_rule_inference:
            self.engines.append(RuleBasedInferenceEngine(config))
        
        if config.enable_graph_inference:
            self.engines.append(GraphBasedInferenceEngine(config))
        
        if config.enable_ml_inference:
            self.engines.append(MLBasedInferenceEngine(config))
        
        # Default to rule-based if no engines specified
        if not self.engines:
            self.engines.append(RuleBasedInferenceEngine(config))
    
    async def initialize(self):
        """Initialize the inference manager."""
        logger.info("InferenceEngineManager initialized with %d engines", len(self.engines))
    
    async def infer_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Infer relationships using all available engines."""
        all_inferred = []
        
        for engine in self.engines:
            try:
                inferred = await engine.infer_relationships(entities)
                all_inferred.extend(inferred)
            except Exception as e:
                logger.error("Error in relationship inference: %s", str(e))
        
        # Deduplicate and merge results
        return self._deduplicate_relationships(all_inferred)
    
    async def infer_entity_types(self, entities: List[Entity]) -> List[Entity]:
        """Infer entity types using all available engines."""
        all_inferred = []
        
        for engine in self.engines:
            try:
                inferred = await engine.infer_entity_types(entities)
                all_inferred.extend(inferred)
            except Exception as e:
                logger.error("Error in entity type inference: %s", str(e))
        
        return all_inferred
    
    async def validate_consistency(self, graph: KnowledgeGraph) -> List[str]:
        """Validate consistency using all available engines."""
        all_inconsistencies = []
        
        for engine in self.engines:
            try:
                inconsistencies = await engine.validate_consistency(graph)
                all_inconsistencies.extend(inconsistencies)
            except Exception as e:
                logger.error("Error in consistency validation: %s", str(e))
        
        # Deduplicate inconsistencies
        return list(set(all_inconsistencies))
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships."""
        seen = set()
        deduplicated = []
        
        for rel in relationships:
            # Create a key based on source, target, and type
            key = (rel.source_entity_id, rel.target_entity_id, rel.relationship_type)
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
            else:
                # If duplicate, keep the one with higher confidence
                for i, existing in enumerate(deduplicated):
                    existing_key = (existing.source_entity_id, existing.target_entity_id, existing.relationship_type)
                    if existing_key == key and rel.confidence > existing.confidence:
                        deduplicated[i] = rel
                        break
        
        return deduplicated
    
    async def shutdown(self):
        """Shutdown the inference manager."""
        logger.info("InferenceEngineManager shutdown complete")
