"""
Knowledge Storage Interfaces and Implementations

Provides storage backends for knowledge base data including entities,
relationships, facts, and knowledge graphs.
"""

import asyncio
import json
import sqlite3
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import aiosqlite

from .types import (
    Entity, Relationship, KnowledgeFact, KnowledgeGraph, OntologyClass,
    KnowledgeQuery, KnowledgeSearchResult, EntityType, RelationshipType,
    KnowledgeBaseConfig, BaseKnowledgeStore
)

logger = logging.getLogger(__name__)


class SQLiteKnowledgeStore(BaseKnowledgeStore):
    """SQLite-based knowledge storage implementation."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        self.db_path = Path(config.storage_path) / "knowledge.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection_pool: Dict[str, aiosqlite.Connection] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the SQLite database."""
        if self._initialized:
            return
        
        # Create database and tables
        async with aiosqlite.connect(str(self.db_path)) as db:
            await self._create_tables(db)
            await db.commit()
        
        self._initialized = True
        logger.info("SQLiteKnowledgeStore initialized at %s", self.db_path)
    
    async def _create_tables(self, db: aiosqlite.Connection):
        """Create database tables."""
        # Entities table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                embedding BLOB,
                confidence REAL DEFAULT 0.8,
                quality_score REAL DEFAULT 0.5,
                verified BOOLEAN DEFAULT FALSE,
                source TEXT,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT,
                version INTEGER DEFAULT 1
            )
        """)
        
        # Relationships table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                properties TEXT,
                strength REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.8,
                verified BOOLEAN DEFAULT FALSE,
                source TEXT,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT,
                version INTEGER DEFAULT 1,
                FOREIGN KEY (source_entity_id) REFERENCES entities (entity_id),
                FOREIGN KEY (target_entity_id) REFERENCES entities (entity_id)
            )
        """)
        
        # Facts table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                subject_entity_id TEXT,
                predicate TEXT,
                object_value TEXT,
                fact_type TEXT NOT NULL,
                properties TEXT,
                confidence REAL DEFAULT 0.8,
                verified BOOLEAN DEFAULT FALSE,
                source TEXT,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT,
                version INTEGER DEFAULT 1,
                FOREIGN KEY (subject_entity_id) REFERENCES entities (entity_id)
            )
        """)
        
        # Knowledge graphs table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graphs (
                graph_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                metadata TEXT,
                entity_count INTEGER DEFAULT 0,
                relationship_count INTEGER DEFAULT 0,
                fact_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Ontology classes table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ontology_classes (
                class_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                parent_class_id TEXT,
                properties TEXT,
                constraints TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (parent_class_id) REFERENCES ontology_classes (class_id)
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships (source_entity_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships (target_entity_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships (relationship_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject_entity_id)")
    
    async def store_entity(self, entity: Entity) -> bool:
        """Store an entity."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO entities (
                        entity_id, name, entity_type, description, properties,
                        embedding, confidence, quality_score, verified, source,
                        tags, created_at, updated_at, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.entity_id,
                    entity.name,
                    entity.entity_type.value,
                    entity.description,
                    json.dumps(entity.properties),
                    json.dumps(entity.embedding) if entity.embedding else None,
                    entity.confidence,
                    entity.quality_score,
                    entity.verified,
                    entity.source.value if entity.source else None,
                    json.dumps(entity.tags),
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    entity.version
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store entity %s: %s", entity.entity_id, str(e))
            return False
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM entities WHERE entity_id = ?",
                    (entity_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return self._row_to_entity(row)
                return None
        except Exception as e:
            logger.error("Failed to get entity %s: %s", entity_id, str(e))
            return None
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        return await self.store_entity(entity)
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to delete entity %s: %s", entity_id, str(e))
            return False
    
    async def store_relationship(self, relationship: Relationship) -> bool:
        """Store a relationship."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO relationships (
                        relationship_id, source_entity_id, target_entity_id,
                        relationship_type, properties, strength, confidence,
                        verified, source, tags, created_at, updated_at, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relationship.relationship_id,
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    relationship.relationship_type.value,
                    json.dumps(relationship.properties),
                    relationship.strength,
                    relationship.confidence,
                    relationship.verified,
                    relationship.source.value if relationship.source else None,
                    json.dumps(relationship.tags),
                    relationship.created_at.isoformat(),
                    relationship.updated_at.isoformat(),
                    relationship.version
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store relationship %s: %s", relationship.relationship_id, str(e))
            return False
    
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve a relationship by ID."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM relationships WHERE relationship_id = ?",
                    (relationship_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return self._row_to_relationship(row)
                return None
        except Exception as e:
            logger.error("Failed to get relationship %s: %s", relationship_id, str(e))
            return None
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("DELETE FROM relationships WHERE relationship_id = ?", (relationship_id,))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to delete relationship %s: %s", relationship_id, str(e))
            return False
    
    async def store_fact(self, fact: KnowledgeFact) -> bool:
        """Store a knowledge fact."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO facts (
                        fact_id, statement, subject_entity_id, predicate,
                        object_value, fact_type, properties, confidence,
                        verified, source, tags, created_at, updated_at, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fact.fact_id,
                    fact.statement,
                    fact.subject_entity_id,
                    fact.predicate,
                    fact.object_value,
                    fact.fact_type.value,
                    json.dumps(fact.properties),
                    fact.confidence,
                    fact.verified,
                    fact.source.value if fact.source else None,
                    json.dumps(fact.tags),
                    fact.created_at.isoformat(),
                    fact.updated_at.isoformat(),
                    fact.version
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store fact %s: %s", fact.fact_id, str(e))
            return False
    
    async def get_fact(self, fact_id: str) -> Optional[KnowledgeFact]:
        """Retrieve a fact by ID."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM facts WHERE fact_id = ?",
                    (fact_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return self._row_to_fact(row)
                return None
        except Exception as e:
            logger.error("Failed to get fact %s: %s", fact_id, str(e))
            return None
    
    async def search_knowledge(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """Search knowledge base."""
        start_time = datetime.now()
        
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                
                entities = []
                relationships = []
                facts = []
                
                # Search entities
                entity_sql = self._build_entity_search_sql(query)
                if entity_sql:
                    cursor = await db.execute(entity_sql[0], entity_sql[1])
                    rows = await cursor.fetchall()
                    entities = [self._row_to_entity(row) for row in rows]
                
                # Search relationships
                rel_sql = self._build_relationship_search_sql(query)
                if rel_sql:
                    cursor = await db.execute(rel_sql[0], rel_sql[1])
                    rows = await cursor.fetchall()
                    relationships = [self._row_to_relationship(row) for row in rows]
                
                # Search facts
                fact_sql = self._build_fact_search_sql(query)
                if fact_sql:
                    cursor = await db.execute(fact_sql[0], fact_sql[1])
                    rows = await cursor.fetchall()
                    facts = [self._row_to_fact(row) for row in rows]
                
                query_time = (datetime.now() - start_time).total_seconds()
                
                return KnowledgeSearchResult(
                    entities=entities[:query.limit],
                    relationships=relationships[:query.limit],
                    facts=facts[:query.limit],
                    total_found=len(entities) + len(relationships) + len(facts),
                    query_time=query_time,
                    search_strategy="sql"
                )
        
        except Exception as e:
            logger.error("Failed to search knowledge: %s", str(e))
            return KnowledgeSearchResult(
                entities=[], relationships=[], facts=[],
                total_found=0, query_time=0.0, search_strategy="error"
            )
    
    async def store_knowledge_graph(self, graph: KnowledgeGraph) -> bool:
        """Store knowledge graph metadata."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO knowledge_graphs (
                        graph_id, name, description, metadata, entity_count,
                        relationship_count, fact_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    graph.graph_id,
                    graph.name,
                    graph.description,
                    json.dumps(graph.metadata),
                    graph.entity_count,
                    graph.relationship_count,
                    graph.fact_count,
                    graph.created_at.isoformat(),
                    graph.updated_at.isoformat()
                ))
                await db.commit()
                return True
        except Exception as e:
            logger.error("Failed to store knowledge graph %s: %s", graph.graph_id, str(e))
            return False
    
    async def get_knowledge_graph(self, graph_id: Optional[str] = None) -> Optional[KnowledgeGraph]:
        """Retrieve knowledge graph."""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                
                if graph_id:
                    cursor = await db.execute(
                        "SELECT * FROM knowledge_graphs WHERE graph_id = ?",
                        (graph_id,)
                    )
                else:
                    # Get the first/default graph
                    cursor = await db.execute("SELECT * FROM knowledge_graphs LIMIT 1")
                
                row = await cursor.fetchone()
                if not row:
                    return None
                
                # Load entities
                entities = {}
                cursor = await db.execute("SELECT * FROM entities")
                entity_rows = await cursor.fetchall()
                for entity_row in entity_rows:
                    entity = self._row_to_entity(entity_row)
                    entities[entity.entity_id] = entity
                
                # Load relationships
                relationships = {}
                cursor = await db.execute("SELECT * FROM relationships")
                rel_rows = await cursor.fetchall()
                for rel_row in rel_rows:
                    rel = self._row_to_relationship(rel_row)
                    relationships[rel.relationship_id] = rel
                
                # Load facts
                facts = {}
                cursor = await db.execute("SELECT * FROM facts")
                fact_rows = await cursor.fetchall()
                for fact_row in fact_rows:
                    fact = self._row_to_fact(fact_row)
                    facts[fact.fact_id] = fact
                
                return KnowledgeGraph(
                    graph_id=row['graph_id'],
                    name=row['name'],
                    description=row['description'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    entities=entities,
                    relationships=relationships,
                    facts=facts,
                    entity_count=row['entity_count'],
                    relationship_count=row['relationship_count'],
                    fact_count=row['fact_count'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                
        except Exception as e:
            logger.error("Failed to get knowledge graph: %s", str(e))
            return None
    
    async def shutdown(self):
        """Shutdown the store."""
        # Close any open connections
        for conn in self._connection_pool.values():
            try:
                await conn.close()
            except:
                pass
        self._connection_pool.clear()
        logger.info("SQLiteKnowledgeStore shutdown complete")
    
    def _row_to_entity(self, row) -> Entity:
        """Convert database row to Entity object."""
        return Entity(
            entity_id=row['entity_id'],
            name=row['name'],
            entity_type=EntityType(row['entity_type']),
            description=row['description'],
            properties=json.loads(row['properties']) if row['properties'] else {},
            embedding=json.loads(row['embedding']) if row['embedding'] else None,
            confidence=row['confidence'],
            quality_score=row['quality_score'],
            verified=bool(row['verified']),
            source=row['source'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            version=row['version']
        )
    
    def _row_to_relationship(self, row) -> Relationship:
        """Convert database row to Relationship object."""
        return Relationship(
            relationship_id=row['relationship_id'],
            source_entity_id=row['source_entity_id'],
            target_entity_id=row['target_entity_id'],
            relationship_type=RelationshipType(row['relationship_type']),
            properties=json.loads(row['properties']) if row['properties'] else {},
            strength=row['strength'],
            confidence=row['confidence'],
            verified=bool(row['verified']),
            source=row['source'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            version=row['version']
        )
    
    def _row_to_fact(self, row) -> KnowledgeFact:
        """Convert database row to KnowledgeFact object."""
        from .types import FactType
        return KnowledgeFact(
            fact_id=row['fact_id'],
            statement=row['statement'],
            subject_entity_id=row['subject_entity_id'],
            predicate=row['predicate'],
            object_value=row['object_value'],
            fact_type=FactType(row['fact_type']),
            properties=json.loads(row['properties']) if row['properties'] else {},
            confidence=row['confidence'],
            verified=bool(row['verified']),
            source=row['source'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            version=row['version']
        )
    
    def _build_entity_search_sql(self, query: KnowledgeQuery) -> Optional[tuple]:
        """Build SQL query for entity search."""
        conditions = []
        params = []
        
        # Text search
        if query.query_text:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{query.query_text}%", f"%{query.query_text}%"])
        
        # Entity type filter
        if query.entity_types:
            type_placeholders = ",".join("?" * len(query.entity_types))
            conditions.append(f"entity_type IN ({type_placeholders})")
            params.extend([et.value for et in query.entity_types])
        
        # Confidence filter
        if query.min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(query.min_confidence)
        
        # Quality score filter
        if query.min_quality_score > 0:
            conditions.append("quality_score >= ?")
            params.append(query.min_quality_score)
        
        # Verified filter
        if query.verified_only:
            conditions.append("verified = 1")
        
        if not conditions:
            return None
        
        sql = f"SELECT * FROM entities WHERE {' AND '.join(conditions)} LIMIT ?"
        params.append(query.limit)
        
        return (sql, params)
    
    def _build_relationship_search_sql(self, query: KnowledgeQuery) -> Optional[tuple]:
        """Build SQL query for relationship search."""
        conditions = []
        params = []
        
        # Entity filters
        if query.entity_ids:
            entity_placeholders = ",".join("?" * len(query.entity_ids))
            conditions.append(f"(source_entity_id IN ({entity_placeholders}) OR target_entity_id IN ({entity_placeholders}))")
            params.extend(query.entity_ids)
            params.extend(query.entity_ids)
        
        # Relationship type filter
        if query.relationship_types:
            type_placeholders = ",".join("?" * len(query.relationship_types))
            conditions.append(f"relationship_type IN ({type_placeholders})")
            params.extend([rt.value for rt in query.relationship_types])
        
        # Confidence filter
        if query.min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(query.min_confidence)
        
        # Strength filter
        if query.relationship_strength_min > 0:
            conditions.append("strength >= ?")
            params.append(query.relationship_strength_min)
        
        # Verified filter
        if query.verified_only:
            conditions.append("verified = 1")
        
        if not conditions:
            return None
        
        sql = f"SELECT * FROM relationships WHERE {' AND '.join(conditions)} LIMIT ?"
        params.append(query.limit)
        
        return (sql, params)
    
    def _build_fact_search_sql(self, query: KnowledgeQuery) -> Optional[tuple]:
        """Build SQL query for fact search."""
        conditions = []
        params = []
        
        # Text search
        if query.query_text:
            conditions.append("statement LIKE ?")
            params.append(f"%{query.query_text}%")
        
        # Entity filter
        if query.entity_ids:
            entity_placeholders = ",".join("?" * len(query.entity_ids))
            conditions.append(f"subject_entity_id IN ({entity_placeholders})")
            params.extend(query.entity_ids)
        
        # Confidence filter
        if query.min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(query.min_confidence)
        
        # Verified filter
        if query.verified_only:
            conditions.append("verified = 1")
        
        if not conditions:
            return None
        
        sql = f"SELECT * FROM facts WHERE {' AND '.join(conditions)} LIMIT ?"
        params.append(query.limit)
        
        return (sql, params)


class InMemoryKnowledgeStore(BaseKnowledgeStore):
    """In-memory knowledge storage for testing and development."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        super().__init__(config)
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.facts: Dict[str, KnowledgeFact] = {}
        self.graphs: Dict[str, KnowledgeGraph] = {}
    
    async def initialize(self):
        """Initialize the in-memory store."""
        logger.info("InMemoryKnowledgeStore initialized")
    
    async def store_entity(self, entity: Entity) -> bool:
        """Store an entity."""
        self.entities[entity.entity_id] = entity
        return True
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self.entities.get(entity_id)
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        if entity.entity_id in self.entities:
            self.entities[entity.entity_id] = entity
            return True
        return False
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        return self.entities.pop(entity_id, None) is not None
    
    async def store_relationship(self, relationship: Relationship) -> bool:
        """Store a relationship."""
        self.relationships[relationship.relationship_id] = relationship
        return True
    
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve a relationship by ID."""
        return self.relationships.get(relationship_id)
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        return self.relationships.pop(relationship_id, None) is not None
    
    async def store_fact(self, fact: KnowledgeFact) -> bool:
        """Store a knowledge fact."""
        self.facts[fact.fact_id] = fact
        return True
    
    async def get_fact(self, fact_id: str) -> Optional[KnowledgeFact]:
        """Retrieve a fact by ID."""
        return self.facts.get(fact_id)
    
    async def search_knowledge(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """Search knowledge base."""
        start_time = datetime.now()
        
        # Simple in-memory search implementation
        entities = []
        relationships = []
        facts = []
        
        # Search entities
        for entity in self.entities.values():
            if self._matches_entity_query(entity, query):
                entities.append(entity)
        
        # Search relationships
        for rel in self.relationships.values():
            if self._matches_relationship_query(rel, query):
                relationships.append(rel)
        
        # Search facts
        for fact in self.facts.values():
            if self._matches_fact_query(fact, query):
                facts.append(fact)
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return KnowledgeSearchResult(
            entities=entities[:query.limit],
            relationships=relationships[:query.limit],
            facts=facts[:query.limit],
            total_found=len(entities) + len(relationships) + len(facts),
            query_time=query_time,
            search_strategy="in_memory"
        )
    
    async def store_knowledge_graph(self, graph: KnowledgeGraph) -> bool:
        """Store knowledge graph metadata."""
        self.graphs[graph.graph_id] = graph
        return True
    
    async def get_knowledge_graph(self, graph_id: Optional[str] = None) -> Optional[KnowledgeGraph]:
        """Retrieve knowledge graph."""
        if graph_id:
            return self.graphs.get(graph_id)
        elif self.graphs:
            return next(iter(self.graphs.values()))
        return None
    
    async def shutdown(self):
        """Shutdown the store."""
        self.entities.clear()
        self.relationships.clear()
        self.facts.clear()
        self.graphs.clear()
        logger.info("InMemoryKnowledgeStore shutdown complete")
    
    def _matches_entity_query(self, entity: Entity, query: KnowledgeQuery) -> bool:
        """Check if entity matches query."""
        # Text search
        if query.query_text:
            text = query.query_text.lower()
            if (text not in entity.name.lower() and 
                (not entity.description or text not in entity.description.lower())):
                return False
        
        # Type filter
        if query.entity_types and entity.entity_type not in query.entity_types:
            return False
        
        # Confidence filter
        if entity.confidence < query.min_confidence:
            return False
        
        # Quality filter
        if entity.quality_score < query.min_quality_score:
            return False
        
        # Verified filter
        if query.verified_only and not entity.verified:
            return False
        
        return True
    
    def _matches_relationship_query(self, rel: Relationship, query: KnowledgeQuery) -> bool:
        """Check if relationship matches query."""
        # Entity filter
        if query.entity_ids:
            if (rel.source_entity_id not in query.entity_ids and 
                rel.target_entity_id not in query.entity_ids):
                return False
        
        # Type filter
        if query.relationship_types and rel.relationship_type not in query.relationship_types:
            return False
        
        # Confidence filter
        if rel.confidence < query.min_confidence:
            return False
        
        # Strength filter
        if rel.strength < query.relationship_strength_min:
            return False
        
        # Verified filter
        if query.verified_only and not rel.verified:
            return False
        
        return True
    
    def _matches_fact_query(self, fact: KnowledgeFact, query: KnowledgeQuery) -> bool:
        """Check if fact matches query."""
        # Text search
        if query.query_text:
            text = query.query_text.lower()
            if text not in fact.statement.lower():
                return False
        
        # Entity filter
        if query.entity_ids and fact.subject_entity_id not in query.entity_ids:
            return False
        
        # Confidence filter
        if fact.confidence < query.min_confidence:
            return False
        
        # Verified filter
        if query.verified_only and not fact.verified:
            return False
        
        return True


class KnowledgeStoreManager:
    """Manager for knowledge storage operations."""
    
    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
        
        # Initialize appropriate store
        if config.storage_backend == "sqlite":
            self.store = SQLiteKnowledgeStore(config)
        elif config.storage_backend == "memory":
            self.store = InMemoryKnowledgeStore(config)
        else:
            raise ValueError(f"Unsupported storage backend: {config.storage_backend}")
    
    async def initialize(self):
        """Initialize the store manager."""
        await self.store.initialize()
    
    async def store_entity(self, entity: Entity) -> bool:
        """Store an entity."""
        return await self.store.store_entity(entity)
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity."""
        return await self.store.get_entity(entity_id)
    
    async def update_entity(self, entity: Entity) -> bool:
        """Update an entity."""
        return await self.store.update_entity(entity)
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        return await self.store.delete_entity(entity_id)
    
    async def store_relationship(self, relationship: Relationship) -> bool:
        """Store a relationship."""
        return await self.store.store_relationship(relationship)
    
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship."""
        return await self.store.get_relationship(relationship_id)
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        return await self.store.delete_relationship(relationship_id)
    
    async def store_fact(self, fact: KnowledgeFact) -> bool:
        """Store a fact."""
        return await self.store.store_fact(fact)
    
    async def get_fact(self, fact_id: str) -> Optional[KnowledgeFact]:
        """Get a fact."""
        return await self.store.get_fact(fact_id)
    
    async def search_knowledge(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        """Search knowledge."""
        return await self.store.search_knowledge(query)
    
    async def store_knowledge_graph(self, graph: KnowledgeGraph) -> bool:
        """Store knowledge graph."""
        return await self.store.store_knowledge_graph(graph)
    
    async def get_knowledge_graph(self, graph_id: Optional[str] = None) -> Optional[KnowledgeGraph]:
        """Get knowledge graph."""
        return await self.store.get_knowledge_graph(graph_id)
    
    async def shutdown(self):
        """Shutdown the store manager."""
        await self.store.shutdown()
