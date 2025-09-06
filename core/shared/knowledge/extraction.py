"""
Knowledge Extraction System

Automated extraction of entities, relationships, and facts from text
using various extraction techniques including NLP, ML, and rule-based approaches.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
import json

from .types import (
    Entity, Relationship, KnowledgeFact, EntityType, RelationshipType, FactType,
    KnowledgeSource, KnowledgeBaseConfig, BaseKnowledgeExtractor
)

logger = logging.getLogger(__name__)


class RegexKnowledgeExtractor(BaseKnowledgeExtractor):
    """Rule-based knowledge extraction using regex patterns."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        
        # Entity patterns
        self.entity_patterns = {
            EntityType.PERSON: [
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last name
                r'\b(Mr\.|Mrs\.|Ms\.|Dr\.) ([A-Z][a-z]+)\b',  # Title + name
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|mentioned|stated)\b'  # Speaker
            ],
            EntityType.ORGANIZATION: [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|Corp\.|LLC|Ltd\.)\b',
                r'\b(University of [A-Z][a-z]+)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Company|Corporation)\b'
            ],
            EntityType.LOCATION: [
                r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\b([A-Z][a-z]+),\s*([A-Z][A-Z])\b'  # City, State
            ],
            EntityType.CONCEPT: [
                r'\b(artificial intelligence|machine learning|data science|blockchain)\b',
                r'\b([a-z]+(?:\s+[a-z]+)*)\s+(?:algorithm|model|framework|methodology)\b'
            ],
            EntityType.EVENT: [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:conference|summit|meeting|event)\b',
                r'\bon\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b'  # Date patterns
            ]
        }
        
        # Relationship patterns
        self.relationship_patterns = [
            (r'\b(\w+(?:\s+\w+)*)\s+works\s+at\s+(\w+(?:\s+\w+)*)\b', RelationshipType.WORKS_AT),
            (r'\b(\w+(?:\s+\w+)*)\s+is\s+(?:the\s+)?(?:CEO|president|director)\s+of\s+(\w+(?:\s+\w+)*)\b', RelationshipType.LEADS),
            (r'\b(\w+(?:\s+\w+)*)\s+owns\s+(\w+(?:\s+\w+)*)\b', RelationshipType.OWNS),
            (r'\b(\w+(?:\s+\w+)*)\s+is\s+located\s+in\s+(\w+(?:\s+\w+)*)\b', RelationshipType.LOCATED_IN),
            (r'\b(\w+(?:\s+\w+)*)\s+collaborates\s+with\s+(\w+(?:\s+\w+)*)\b', RelationshipType.COLLABORATES_WITH),
            (r'\b(\w+(?:\s+\w+)*)\s+competes\s+with\s+(\w+(?:\s+\w+)*)\b', RelationshipType.COMPETES_WITH),
            (r'\b(\w+(?:\s+\w+)*)\s+depends\s+on\s+(\w+(?:\s+\w+)*)\b', RelationshipType.DEPENDS_ON),
            (r'\b(\w+(?:\s+\w+)*)\s+is\s+part\s+of\s+(\w+(?:\s+\w+)*)\b', RelationshipType.PART_OF),
            (r'\b(\w+(?:\s+\w+)*)\s+creates\s+(\w+(?:\s+\w+)*)\b', RelationshipType.CREATES),
            (r'\b(\w+(?:\s+\w+)*)\s+uses\s+(\w+(?:\s+\w+)*)\b', RelationshipType.USES)
        ]
        
        # Fact patterns
        self.fact_patterns = [
            r'\b(\w+(?:\s+\w+)*)\s+is\s+(\w+(?:\s+\w+)*)\b',  # X is Y
            r'\b(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)\b',  # X has Y
            r'\b(\w+(?:\s+\w+)*)\s+was\s+founded\s+in\s+(\d{4})\b',  # Founded date
            r'\b(\w+(?:\s+\w+)*)\s+costs\s+\$?([\d,]+)\b'  # Cost/price
        ]
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using regex patterns."""
        entities = []
        entity_names = set()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_name = match.group(1).strip()
                    
                    # Skip if already found or too short
                    if entity_name.lower() in entity_names or len(entity_name) < 2:
                        continue
                    
                    entity_names.add(entity_name.lower())
                    
                    entity = Entity(
                        name=entity_name,
                        entity_type=entity_type,
                        description=f"Extracted {entity_type.value}",
                        confidence=0.7,  # Lower confidence for regex extraction
                        quality_score=0.6,
                        source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                        properties={
                            "extraction_method": "regex",
                            "pattern_matched": pattern,
                            "context": self._get_context(text, match.start(), match.end())
                        }
                    )
                    entities.append(entity)
        
        return entities
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from text using regex patterns."""
        relationships = []
        entity_name_map = {entity.name.lower(): entity.name for entity in entities}
        
        for pattern, rel_type in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                source_name = match.group(1).strip()
                target_name = match.group(2).strip()
                
                # Find corresponding entities
                source_entity_name = entity_name_map.get(source_name.lower())
                target_entity_name = entity_name_map.get(target_name.lower())
                
                if source_entity_name and target_entity_name:
                    relationship = Relationship(
                        source_entity_id=source_entity_name,  # Will be replaced with actual ID
                        target_entity_id=target_entity_name,  # Will be replaced with actual ID
                        relationship_type=rel_type,
                        confidence=0.7,
                        strength=0.8,
                        source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                        properties={
                            "extraction_method": "regex",
                            "pattern_matched": pattern,
                            "context": self._get_context(text, match.start(), match.end())
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def extract_facts(self, text: str, entities: List[Entity]) -> List[KnowledgeFact]:
        """Extract facts from text using regex patterns."""
        facts = []
        entity_name_map = {entity.name.lower(): entity.name for entity in entities}
        
        for pattern in self.fact_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                subject = match.group(1).strip()
                object_value = match.group(2).strip()
                
                # Find corresponding entity
                subject_entity_name = entity_name_map.get(subject.lower())
                
                if subject_entity_name:
                    # Determine fact type and predicate
                    if "is" in pattern:
                        predicate = "is"
                        fact_type = FactType.CLASSIFICATION
                    elif "has" in pattern:
                        predicate = "has"
                        fact_type = FactType.PROPERTY
                    elif "founded" in pattern:
                        predicate = "founded_in"
                        fact_type = FactType.TEMPORAL
                    elif "costs" in pattern:
                        predicate = "costs"
                        fact_type = FactType.QUANTITATIVE
                    else:
                        predicate = "relates_to"
                        fact_type = FactType.GENERAL
                    
                    fact = KnowledgeFact(
                        statement=f"{subject} {predicate} {object_value}",
                        subject_entity_id=subject_entity_name,  # Will be replaced with actual ID
                        predicate=predicate,
                        object_value=object_value,
                        fact_type=fact_type,
                        confidence=0.7,
                        source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                        properties={
                            "extraction_method": "regex",
                            "pattern_matched": pattern,
                            "context": self._get_context(text, match.start(), match.end())
                        }
                    )
                    facts.append(fact)
        
        return facts
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around matched text."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()


class NLPKnowledgeExtractor(BaseKnowledgeExtractor):
    """NLP-based knowledge extraction using spaCy."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        self.nlp = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy NLP model."""
        try:
            import spacy
            # Try to load English model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Using regex extractor instead.")
                self.nlp = None
        except ImportError:
            logger.warning("spaCy not installed. Using regex extractor instead.")
            self.nlp = None
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using NLP named entity recognition."""
        if not self.nlp:
            # Fallback to regex
            regex_extractor = RegexKnowledgeExtractor(self.config)
            return await regex_extractor.extract_entities(text)
        
        entities = []
        doc = self.nlp(text)
        
        # Map spaCy entity types to our entity types
        entity_type_map = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,  # Geopolitical entity
            "LOC": EntityType.LOCATION,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.CONCEPT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LANGUAGE": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT
        }
        
        entity_names = set()
        
        for ent in doc.ents:
            entity_type = entity_type_map.get(ent.label_, EntityType.CONCEPT)
            entity_name = ent.text.strip()
            
            # Skip if already found or too short
            if entity_name.lower() in entity_names or len(entity_name) < 2:
                continue
            
            entity_names.add(entity_name.lower())
            
            entity = Entity(
                name=entity_name,
                entity_type=entity_type,
                description=f"NLP extracted {entity_type.value} ({ent.label_})",
                confidence=0.8,  # Higher confidence for NLP extraction
                quality_score=0.7,
                source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                properties={
                    "extraction_method": "nlp",
                    "spacy_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "context": self._get_sentence_context(doc, ent)
                }
            )
            entities.append(entity)
        
        return entities
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using dependency parsing."""
        if not self.nlp:
            # Fallback to regex
            regex_extractor = RegexKnowledgeExtractor(self.config)
            return await regex_extractor.extract_relationships(text, entities)
        
        relationships = []
        doc = self.nlp(text)
        entity_name_map = {entity.name.lower(): entity.name for entity in entities}
        
        # Extract relationships based on dependency patterns
        for sent in doc.sents:
            for token in sent:
                # Look for verb relationships
                if token.pos_ == "VERB":
                    subject = self._find_subject(token)
                    obj = self._find_object(token)
                    
                    if subject and obj:
                        subject_name = subject.text.strip()
                        object_name = obj.text.strip()
                        
                        # Check if both are entities
                        source_entity = entity_name_map.get(subject_name.lower())
                        target_entity = entity_name_map.get(object_name.lower())
                        
                        if source_entity and target_entity:
                            rel_type = self._verb_to_relationship_type(token.lemma_)
                            
                            relationship = Relationship(
                                source_entity_id=source_entity,
                                target_entity_id=target_entity,
                                relationship_type=rel_type,
                                confidence=0.8,
                                strength=0.7,
                                source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                                properties={
                                    "extraction_method": "nlp",
                                    "verb": token.lemma_,
                                    "sentence": sent.text.strip(),
                                    "dependency_pattern": f"{subject.dep_}-{token.dep_}-{obj.dep_}"
                                }
                            )
                            relationships.append(relationship)
        
        return relationships
    
    async def extract_facts(self, text: str, entities: List[Entity]) -> List[KnowledgeFact]:
        """Extract facts using NLP patterns."""
        if not self.nlp:
            # Fallback to regex
            regex_extractor = RegexKnowledgeExtractor(self.config)
            return await regex_extractor.extract_facts(text, entities)
        
        facts = []
        doc = self.nlp(text)
        entity_name_map = {entity.name.lower(): entity.name for entity in entities}
        
        # Extract facts from copula constructions (X is Y)
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ == "be" and token.pos_ == "AUX":
                    subject = self._find_subject(token)
                    predicate = self._find_predicate_complement(token)
                    
                    if subject and predicate:
                        subject_name = subject.text.strip()
                        predicate_text = predicate.text.strip()
                        
                        # Check if subject is an entity
                        subject_entity = entity_name_map.get(subject_name.lower())
                        
                        if subject_entity:
                            fact = KnowledgeFact(
                                statement=f"{subject_name} is {predicate_text}",
                                subject_entity_id=subject_entity,
                                predicate="is",
                                object_value=predicate_text,
                                fact_type=FactType.CLASSIFICATION,
                                confidence=0.8,
                                source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                                properties={
                                    "extraction_method": "nlp",
                                    "sentence": sent.text.strip(),
                                    "dependency_pattern": f"{subject.dep_}-{token.dep_}-{predicate.dep_}"
                                }
                            )
                            facts.append(fact)
        
        return facts
    
    def _find_subject(self, verb_token):
        """Find the subject of a verb token."""
        for child in verb_token.children:
            if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                return child
        return None
    
    def _find_object(self, verb_token):
        """Find the object of a verb token."""
        for child in verb_token.children:
            if child.dep_ in ("dobj", "pobj", "iobj"):
                return child
        return None
    
    def _find_predicate_complement(self, verb_token):
        """Find predicate complement (for copula constructions)."""
        for child in verb_token.children:
            if child.dep_ in ("attr", "acomp", "xcomp"):
                return child
        return None
    
    def _verb_to_relationship_type(self, verb: str) -> RelationshipType:
        """Map verb lemma to relationship type."""
        verb_mapping = {
            "work": RelationshipType.WORKS_AT,
            "lead": RelationshipType.LEADS,
            "own": RelationshipType.OWNS,
            "locate": RelationshipType.LOCATED_IN,
            "collaborate": RelationshipType.COLLABORATES_WITH,
            "compete": RelationshipType.COMPETES_WITH,
            "depend": RelationshipType.DEPENDS_ON,
            "create": RelationshipType.CREATES,
            "use": RelationshipType.USES,
            "contain": RelationshipType.PART_OF,
            "include": RelationshipType.PART_OF
        }
        
        return verb_mapping.get(verb, RelationshipType.RELATED_TO)
    
    def _get_sentence_context(self, doc, entity):
        """Get sentence context for an entity."""
        for sent in doc.sents:
            if entity.start >= sent.start and entity.end <= sent.end:
                return sent.text.strip()
        return ""


class LLMKnowledgeExtractor(BaseKnowledgeExtractor):
    """LLM-based knowledge extraction using language models."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        self.model_name = config.llm_model if hasattr(config, 'llm_model') else "gpt-3.5-turbo"
        self.max_chunk_size = 2000  # Limit text size for LLM processing
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using LLM prompting."""
        if not self._has_llm_capabilities():
            # Fallback to NLP extractor
            nlp_extractor = NLPKnowledgeExtractor(self.config)
            return await nlp_extractor.extract_entities(text)
        
        entities = []
        chunks = self._chunk_text(text)
        
        for chunk in chunks:
            prompt = self._create_entity_extraction_prompt(chunk)
            response = await self._call_llm(prompt)
            
            if response:
                chunk_entities = self._parse_entity_response(response, chunk)
                entities.extend(chunk_entities)
        
        return entities
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using LLM prompting."""
        if not self._has_llm_capabilities():
            # Fallback to NLP extractor
            nlp_extractor = NLPKnowledgeExtractor(self.config)
            return await nlp_extractor.extract_relationships(text, entities)
        
        relationships = []
        entity_names = [entity.name for entity in entities]
        chunks = self._chunk_text(text)
        
        for chunk in chunks:
            prompt = self._create_relationship_extraction_prompt(chunk, entity_names)
            response = await self._call_llm(prompt)
            
            if response:
                chunk_relationships = self._parse_relationship_response(response, entities)
                relationships.extend(chunk_relationships)
        
        return relationships
    
    async def extract_facts(self, text: str, entities: List[Entity]) -> List[KnowledgeFact]:
        """Extract facts using LLM prompting."""
        if not self._has_llm_capabilities():
            # Fallback to NLP extractor
            nlp_extractor = NLPKnowledgeExtractor(self.config)
            return await nlp_extractor.extract_facts(text, entities)
        
        facts = []
        entity_names = [entity.name for entity in entities]
        chunks = self._chunk_text(text)
        
        for chunk in chunks:
            prompt = self._create_fact_extraction_prompt(chunk, entity_names)
            response = await self._call_llm(prompt)
            
            if response:
                chunk_facts = self._parse_fact_response(response, entities)
                facts.extend(chunk_facts)
        
        return facts
    
    def _has_llm_capabilities(self) -> bool:
        """Check if LLM capabilities are available."""
        # This would check for API keys, model availability, etc.
        return False  # Placeholder - would implement actual check
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for LLM processing."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    # Single word too long, add anyway
                    chunks.append(word)
                    current_length = 0
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _create_entity_extraction_prompt(self, text: str) -> str:
        """Create prompt for entity extraction."""
        return f"""
        Extract entities from the following text. Identify:
        - PERSON: Names of people
        - ORGANIZATION: Companies, institutions, groups
        - LOCATION: Places, cities, countries
        - CONCEPT: Ideas, technologies, methodologies
        - EVENT: Meetings, conferences, incidents
        
        Text: {text}
        
        Return entities in JSON format:
        [
            {{"name": "entity_name", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT", "confidence": 0.9}}
        ]
        """
    
    def _create_relationship_extraction_prompt(self, text: str, entity_names: List[str]) -> str:
        """Create prompt for relationship extraction."""
        entities_str = ", ".join(entity_names)
        return f"""
        Extract relationships between entities from the following text.
        
        Known entities: {entities_str}
        
        Text: {text}
        
        Return relationships in JSON format:
        [
            {{"source": "entity1", "target": "entity2", "type": "WORKS_AT|LEADS|OWNS|RELATED_TO", "confidence": 0.9}}
        ]
        """
    
    def _create_fact_extraction_prompt(self, text: str, entity_names: List[str]) -> str:
        """Create prompt for fact extraction."""
        entities_str = ", ".join(entity_names)
        return f"""
        Extract factual statements about entities from the following text.
        
        Known entities: {entities_str}
        
        Text: {text}
        
        Return facts in JSON format:
        [
            {{"entity": "entity_name", "predicate": "is|has|founded_in", "object": "value", "confidence": 0.9}}
        ]
        """
    
    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API (placeholder implementation)."""
        # This would implement actual LLM API calls
        return None
    
    def _parse_entity_response(self, response: str, text: str) -> List[Entity]:
        """Parse entity extraction response."""
        entities = []
        try:
            data = json.loads(response)
            for item in data:
                entity_type = EntityType(item["type"])
                entity = Entity(
                    name=item["name"],
                    entity_type=entity_type,
                    description=f"LLM extracted {entity_type.value}",
                    confidence=item.get("confidence", 0.9),
                    quality_score=0.8,
                    source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                    properties={
                        "extraction_method": "llm",
                        "model": self.model_name,
                        "context": text[:200] + "..." if len(text) > 200 else text
                    }
                )
                entities.append(entity)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse entity response: %s", str(e))
        
        return entities
    
    def _parse_relationship_response(self, response: str, entities: List[Entity]) -> List[Relationship]:
        """Parse relationship extraction response."""
        relationships = []
        entity_name_map = {entity.name.lower(): entity.name for entity in entities}
        
        try:
            data = json.loads(response)
            for item in data:
                source_name = entity_name_map.get(item["source"].lower())
                target_name = entity_name_map.get(item["target"].lower())
                
                if source_name and target_name:
                    rel_type = RelationshipType(item["type"])
                    relationship = Relationship(
                        source_entity_id=source_name,
                        target_entity_id=target_name,
                        relationship_type=rel_type,
                        confidence=item.get("confidence", 0.9),
                        strength=0.8,
                        source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                        properties={
                            "extraction_method": "llm",
                            "model": self.model_name
                        }
                    )
                    relationships.append(relationship)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse relationship response: %s", str(e))
        
        return relationships
    
    def _parse_fact_response(self, response: str, entities: List[Entity]) -> List[KnowledgeFact]:
        """Parse fact extraction response."""
        facts = []
        entity_name_map = {entity.name.lower(): entity.name for entity in entities}
        
        try:
            data = json.loads(response)
            for item in data:
                entity_name = entity_name_map.get(item["entity"].lower())
                
                if entity_name:
                    predicate = item["predicate"]
                    object_value = item["object"]
                    
                    # Determine fact type
                    if predicate == "is":
                        fact_type = FactType.CLASSIFICATION
                    elif predicate == "has":
                        fact_type = FactType.PROPERTY
                    elif "founded" in predicate:
                        fact_type = FactType.TEMPORAL
                    else:
                        fact_type = FactType.GENERAL
                    
                    fact = KnowledgeFact(
                        statement=f"{item['entity']} {predicate} {object_value}",
                        subject_entity_id=entity_name,
                        predicate=predicate,
                        object_value=object_value,
                        fact_type=fact_type,
                        confidence=item.get("confidence", 0.9),
                        source=KnowledgeSource.AUTOMATIC_EXTRACTION,
                        properties={
                            "extraction_method": "llm",
                            "model": self.model_name
                        }
                    )
                    facts.append(fact)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse fact response: %s", str(e))
        
        return facts


class KnowledgeExtractionManager:
    """Manager for knowledge extraction operations."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
        
        # Initialize extractors based on configuration
        self.extractors = []
        
        if config.enable_llm_extraction:
            self.extractors.append(LLMKnowledgeExtractor(config))
        
        if config.enable_nlp_extraction:
            self.extractors.append(NLPKnowledgeExtractor(config))
        
        # Always include regex as fallback
        self.extractors.append(RegexKnowledgeExtractor(config))
        
        # Use the first available extractor as primary
        self.primary_extractor = self.extractors[0] if self.extractors else RegexKnowledgeExtractor(config)
    
    async def initialize(self):
        """Initialize the extraction manager."""
        logger.info("KnowledgeExtractionManager initialized with %d extractors", len(self.extractors))
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        return await self.primary_extractor.extract_entities(text)
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from text."""
        return await self.primary_extractor.extract_relationships(text, entities)
    
    async def extract_facts(self, text: str, entities: List[Entity]) -> List[KnowledgeFact]:
        """Extract facts from text."""
        return await self.primary_extractor.extract_facts(text, entities)
    
    async def extract_all(self, text: str) -> Dict[str, List]:
        """Extract all knowledge from text."""
        # First extract entities
        entities = await self.extract_entities(text)
        
        # Then extract relationships and facts
        relationships = await self.extract_relationships(text, entities)
        facts = await self.extract_facts(text, entities)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "facts": facts
        }
    
    async def shutdown(self):
        """Shutdown the extraction manager."""
        logger.info("KnowledgeExtractionManager shutdown complete")
