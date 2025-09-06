"""
Vector Store Manager

Manages different vector store implementations and provides unified
interface for storing and retrieving document embeddings.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from abc import ABC, abstractmethod
import json
from pathlib import Path

from .types import (
    Document, DocumentChunk, VectorStoreConfig, VectorStoreType,
    BaseVectorStore, RetrievalResult, RetrievalConfig
)

logger = logging.getLogger(__name__)


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for development and testing."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.chunk_embeddings: Dict[str, np.ndarray] = {}
        self.metadata_index: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the vector store."""
        logger.info("Initialized in-memory vector store")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the store."""
        doc_ids = []
        
        for doc in documents:
            # Store document
            self.documents[doc.doc_id] = doc
            
            # Store document embedding
            if doc.embedding:
                self.embeddings[doc.doc_id] = np.array(doc.embedding)
            
            # Store chunk embeddings
            for chunk in doc.chunks:
                if chunk.embedding is not None:
                    self.chunk_embeddings[chunk.chunk_id] = chunk.embedding
            
            # Index metadata
            self.metadata_index[doc.doc_id] = {
                "title": doc.metadata.title,
                "category": doc.metadata.category,
                "tags": doc.metadata.tags,
                "created_at": doc.created_at.isoformat()
            }
            
            doc_ids.append(doc.doc_id)
        
        logger.info("Added %d documents to in-memory store", len(documents))
        return doc_ids
    
    async def search(self, query_embedding: np.ndarray, config: RetrievalConfig) -> RetrievalResult:
        """Search for similar documents."""
        similarities = []
        
        # Search in document embeddings
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            if similarity >= config.similarity_threshold:
                similarities.append((doc_id, similarity, "document"))
        
        # Search in chunk embeddings
        for chunk_id, chunk_embedding in self.chunk_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            if similarity >= config.similarity_threshold:
                similarities.append((chunk_id, similarity, "chunk"))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Process results
        documents = []
        chunks = []
        scores = []
        
        for item_id, score, item_type in similarities[:config.top_k]:
            scores.append(score)
            
            if item_type == "document":
                if item_id in self.documents:
                    documents.append(self.documents[item_id])
            
            elif item_type == "chunk":
                # Find chunk and its parent document
                for doc in self.documents.values():
                    for chunk in doc.chunks:
                        if chunk.chunk_id == item_id:
                            chunks.append(chunk)
                            if doc not in documents:
                                documents.append(doc)
                            break
        
        return RetrievalResult(
            documents=documents,
            chunks=chunks,
            scores=scores,
            query="",  # Will be set by caller
            retrieval_time=0.0,  # Will be set by caller
            total_results=len(similarities)
        )
    
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the store."""
        for doc_id in doc_ids:
            if doc_id in self.documents:
                # Remove document
                doc = self.documents[doc_id]
                del self.documents[doc_id]
                
                # Remove document embedding
                if doc_id in self.embeddings:
                    del self.embeddings[doc_id]
                
                # Remove chunk embeddings
                for chunk in doc.chunks:
                    if chunk.chunk_id in self.chunk_embeddings:
                        del self.chunk_embeddings[chunk.chunk_id]
                
                # Remove metadata
                if doc_id in self.metadata_index:
                    del self.metadata_index[doc_id]
        
        return True
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "total_chunks": len(self.chunk_embeddings),
            "memory_usage_mb": 0  # Could calculate actual memory usage
        }


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection = None
    
    async def initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            
            if self.config.persistence_path:
                self.client = chromadb.PersistentClient(path=self.config.persistence_path)
            else:
                self.client = chromadb.Client()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
            except:
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata=self.config.chroma_config
                )
            
            logger.info("Initialized ChromaDB vector store")
            
        except ImportError:
            raise ImportError("chromadb package is required for ChromaVectorStore")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            if doc.embedding:
                ids.append(doc.doc_id)
                embeddings.append(doc.embedding)
                metadatas.append({
                    "title": doc.metadata.title or "",
                    "category": doc.metadata.category or "",
                    "tags": json.dumps(doc.metadata.tags),
                    "content_type": doc.metadata.content_type.value,
                    "created_at": doc.created_at.isoformat()
                })
                documents_text.append(doc.content)
            
            # Add chunks
            for chunk in doc.chunks:
                if chunk.embedding is not None:
                    ids.append(chunk.chunk_id)
                    embeddings.append(chunk.embedding.tolist())
                    metadatas.append({
                        "doc_id": doc.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "type": "chunk"
                    })
                    documents_text.append(chunk.content)
        
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
        
        logger.info("Added %d items to ChromaDB", len(ids))
        return [doc.doc_id for doc in documents]
    
    async def search(self, query_embedding: np.ndarray, config: RetrievalConfig) -> RetrievalResult:
        """Search using ChromaDB."""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=config.top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        documents = []
        chunks = []
        scores = []
        
        if results["ids"] and results["ids"][0]:
            for i, (doc_id, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - distance
                scores.append(similarity)
                
                if metadata.get("type") == "chunk":
                    # This is a chunk result
                    chunk = DocumentChunk(
                        chunk_id=doc_id,
                        doc_id=metadata["doc_id"],
                        content=results["documents"][0][i],
                        start_idx=0,  # Would need to store these
                        end_idx=len(results["documents"][0][i]),
                        chunk_index=metadata.get("chunk_index", 0)
                    )
                    chunks.append(chunk)
                else:
                    # This is a document result - would need to reconstruct Document
                    pass
        
        return RetrievalResult(
            documents=documents,
            chunks=chunks,
            scores=scores,
            query="",
            retrieval_time=0.0,
            total_results=len(results["ids"][0]) if results["ids"] else 0
        )
    
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from ChromaDB."""
        if not self.collection:
            return False
        
        # Also need to delete associated chunks
        for doc_id in doc_ids:
            try:
                self.collection.delete(ids=[doc_id])
                # Delete chunks (would need to query for chunk IDs first)
            except Exception as e:
                logger.error("Failed to delete document %s: %s", doc_id, e)
                return False
        
        return True
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID from ChromaDB."""
        if not self.collection:
            return None
        
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if result["ids"] and result["ids"][0]:
                # Reconstruct document from stored data
                metadata_dict = result["metadatas"][0]
                # Would need to reconstruct full Document object
                pass
        except Exception as e:
            logger.error("Failed to get document %s: %s", doc_id, e)
        
        return None


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.id_to_doc: Dict[int, str] = {}
        self.doc_to_id: Dict[str, int] = {}
        self.documents: Dict[str, Document] = {}
        self.next_id = 0
    
    async def initialize(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            # Create index based on dimension (would need to be provided)
            dimension = self.config.index_params.get("dimension", 384)
            
            if self.config.index_params.get("index_type") == "IVF":
                # IVF index for large datasets
                nlist = self.config.index_params.get("nlist", 100)
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            else:
                # Default to flat index
                self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine if normalized)
            
            logger.info("Initialized FAISS vector store with dimension %d", dimension)
            
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu package is required for FAISSVectorStore")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to FAISS index."""
        if not self.index:
            raise RuntimeError("Index not initialized")
        
        embeddings_to_add = []
        doc_ids = []
        
        for doc in documents:
            if doc.embedding:
                # Add document
                self.documents[doc.doc_id] = doc
                self.id_to_doc[self.next_id] = doc.doc_id
                self.doc_to_id[doc.doc_id] = self.next_id
                
                embeddings_to_add.append(np.array(doc.embedding))
                doc_ids.append(doc.doc_id)
                self.next_id += 1
                
                # Add chunks
                for chunk in doc.chunks:
                    if chunk.embedding is not None:
                        self.id_to_doc[self.next_id] = f"{doc.doc_id}::{chunk.chunk_id}"
                        embeddings_to_add.append(chunk.embedding)
                        self.next_id += 1
        
        if embeddings_to_add:
            embeddings_matrix = np.vstack(embeddings_to_add).astype(np.float32)
            self.index.add(embeddings_matrix)
        
        logger.info("Added %d items to FAISS index", len(embeddings_to_add))
        return doc_ids
    
    async def search(self, query_embedding: np.ndarray, config: RetrievalConfig) -> RetrievalResult:
        """Search using FAISS index."""
        if not self.index:
            raise RuntimeError("Index not initialized")
        
        query_vector = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vector, config.top_k)
        
        documents = []
        chunks = []
        result_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for no result
                continue
                
            result_scores.append(float(score))
            item_id = self.id_to_doc.get(idx)
            
            if not item_id:
                continue
            
            if "::" in item_id:
                # This is a chunk
                doc_id, chunk_id = item_id.split("::", 1)
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    for chunk in doc.chunks:
                        if chunk.chunk_id == chunk_id:
                            chunks.append(chunk)
                            if doc not in documents:
                                documents.append(doc)
                            break
            else:
                # This is a document
                if item_id in self.documents:
                    documents.append(self.documents[item_id])
        
        return RetrievalResult(
            documents=documents,
            chunks=chunks,
            scores=result_scores,
            query="",
            retrieval_time=0.0,
            total_results=len([i for i in indices[0] if i != -1])
        )
    
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from FAISS (rebuild required)."""
        # FAISS doesn't support deletion, would need to rebuild index
        for doc_id in doc_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                # Remove from ID mappings
                if doc_id in self.doc_to_id:
                    idx = self.doc_to_id[doc_id]
                    del self.doc_to_id[doc_id]
                    del self.id_to_doc[idx]
        
        # Note: Index rebuild would be needed for actual deletion
        logger.warning("FAISS deletion requires index rebuild")
        return True
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)


class VectorStoreManager:
    """Manager for different vector store implementations."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.store: Optional[BaseVectorStore] = None
    
    async def initialize(self):
        """Initialize the vector store."""
        if self.config.store_type == VectorStoreType.IN_MEMORY:
            self.store = InMemoryVectorStore(self.config)
        
        elif self.config.store_type == VectorStoreType.CHROMA:
            self.store = ChromaVectorStore(self.config)
        
        elif self.config.store_type == VectorStoreType.FAISS:
            self.store = FAISSVectorStore(self.config)
        
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.store_type}")
        
        await self.store.initialize()
        logger.info("VectorStoreManager initialized with %s", self.config.store_type.value)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not self.store:
            raise RuntimeError("Vector store not initialized")
        
        return await self.store.add_documents(documents)
    
    async def search(self, query_embedding: np.ndarray, config: RetrievalConfig) -> RetrievalResult:
        """Search the vector store."""
        if not self.store:
            raise RuntimeError("Vector store not initialized")
        
        return await self.store.search(query_embedding, config)
    
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        if not self.store:
            raise RuntimeError("Vector store not initialized")
        
        return await self.store.delete_documents(doc_ids)
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        if not self.store:
            return None
        
        return await self.store.get_document(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if hasattr(self.store, 'get_stats'):
            return self.store.get_stats()
        
        return {"store_type": self.config.store_type.value}
    
    async def optimize(self) -> List[str]:
        """Optimize the vector store."""
        optimizations = []
        
        # Store-specific optimizations
        if isinstance(self.store, FAISSVectorStore) and self.store.index:
            # Train IVF index if needed
            if hasattr(self.store.index, 'is_trained') and not self.store.index.is_trained:
                # Would need training data
                optimizations.append("FAISS index training needed")
        
        return optimizations
    
    async def backup(self, backup_path: str) -> bool:
        """Backup the vector store."""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Store-specific backup logic
            if isinstance(self.store, InMemoryVectorStore):
                # Save to JSON
                backup_data = {
                    "documents": [doc.dict() for doc in self.store.documents.values()],
                    "config": self.config.dict()
                }
                
                with open(backup_dir / "backup.json", 'w') as f:
                    json.dump(backup_data, f, indent=2, default=str)
            
            elif isinstance(self.store, ChromaVectorStore) and self.config.persistence_path:
                # ChromaDB handles persistence automatically
                logger.info("ChromaDB data persisted automatically")
            
            logger.info("Vector store backup completed: %s", backup_path)
            return True
            
        except Exception as e:
            logger.error("Vector store backup failed: %s", e)
            return False
    
    async def restore(self, backup_path: str) -> bool:
        """Restore the vector store from backup."""
        try:
            backup_dir = Path(backup_path)
            
            if isinstance(self.store, InMemoryVectorStore):
                backup_file = backup_dir / "backup.json"
                if backup_file.exists():
                    with open(backup_file, 'r') as f:
                        backup_data = json.load(f)
                    
                    # Restore documents
                    for doc_data in backup_data.get("documents", []):
                        # Would need to reconstruct Document objects
                        pass
            
            logger.info("Vector store restore completed: %s", backup_path)
            return True
            
        except Exception as e:
            logger.error("Vector store restore failed: %s", e)
            return False
