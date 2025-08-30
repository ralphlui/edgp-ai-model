"""
RAG (Retrieval-Augmented Generation) system for enterprise knowledge management.
Supports multiple vector stores and embedding models for context-aware responses.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging
import os

from sentence_transformers import SentenceTransformer
import numpy as np

from .config import settings

logger = logging.getLogger(__name__)


class Document:
    """Represents a document in the RAG system."""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or f"doc_{hash(content)}"
        self.created_at = datetime.utcnow()
        self.embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "has_embedding": self.embedding is not None
        }


class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store implementation for development and testing."""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model = embedding_model
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the in-memory store."""
        doc_ids = []
        
        for doc in documents:
            # Generate embedding
            embedding = self.embedding_model.encode([doc.content])[0]
            doc.embedding = embedding
            
            # Store document and embedding
            self.documents[doc.doc_id] = doc
            self.embeddings[doc.doc_id] = embedding
            doc_ids.append(doc.doc_id)
        
        logger.info(f"Added {len(documents)} documents to in-memory vector store")
        return doc_ids
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search using cosine similarity."""
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            if similarity >= threshold:
                similarities.append((doc_id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for doc_id, similarity in similarities[:k]:
            document = self.documents[doc_id]
            results.append((document, similarity))
        
        return results
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.embeddings[doc_id]
            return True
        return False
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, collection_name: str = "edgp_documents"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Create client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(allow_reset=True)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "EDGP document collection"}
            )
            
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
            
        except ImportError:
            logger.error("ChromaDB not installed. Please install chromadb package.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB."""
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")
        
        # Prepare data for ChromaDB
        doc_ids = [doc.doc_id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=doc_ids
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return doc_ids
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search using ChromaDB."""
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Convert results to Document objects
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, (content, metadata, doc_id, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["ids"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity (ChromaDB returns distances)
                similarity = 1.0 - distance
                
                if similarity >= threshold:
                    doc = Document(
                        content=content,
                        metadata=metadata,
                        doc_id=doc_id
                    )
                    documents.append((doc, similarity))
        
        return documents
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from ChromaDB."""
        if not self.collection:
            return False
        
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID from ChromaDB."""
        if not self.collection:
            return None
        
        try:
            results = self.collection.get(ids=[doc_id])
            if results["documents"] and results["documents"][0]:
                return Document(
                    content=results["documents"][0],
                    metadata=results["metadatas"][0],
                    doc_id=doc_id
                )
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
        
        return None


class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system for enterprise knowledge management.
    Provides context-aware responses by retrieving relevant documents.
    """
    
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_model_name: str = None
    ):
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = self._create_default_vector_store()
        
        # Document processing settings
        self.chunk_size = settings.rag_chunk_size
        self.chunk_overlap = settings.rag_chunk_overlap
        self.top_k = settings.rag_top_k
        self.similarity_threshold = settings.rag_similarity_threshold
        
        logger.info(f"Initialized RAG system with {type(self.vector_store).__name__}")
    
    def _create_default_vector_store(self) -> BaseVectorStore:
        """Create default vector store based on configuration."""
        if settings.vector_store_type.lower() == "chroma":
            try:
                return ChromaVectorStore()
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB, falling back to in-memory: {e}")
                return InMemoryVectorStore(self.embedding_model)
        else:
            return InMemoryVectorStore(self.embedding_model)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            
            if chunk.strip():
                chunks.append(chunk)
            
            # Break if no overlap needed for the last chunk
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        chunk_document: bool = True
    ) -> List[str]:
        """Add a document to the RAG system."""
        documents = []
        
        if chunk_document and len(content.split()) > self.chunk_size:
            # Split into chunks
            chunks = self.chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id or 'doc'}_{i}" if doc_id else None
                chunk_metadata = {**(metadata or {}), "chunk_index": i, "total_chunks": len(chunks)}
                
                doc = Document(
                    content=chunk,
                    metadata=chunk_metadata,
                    doc_id=chunk_id
                )
                documents.append(doc)
        else:
            # Add as single document
            doc = Document(
                content=content,
                metadata=metadata,
                doc_id=doc_id
            )
            documents.append(doc)
        
        # Add to vector store
        doc_ids = await self.vector_store.add_documents(documents)
        
        logger.info(f"Added document with {len(documents)} chunks to RAG system")
        return doc_ids
    
    async def add_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        chunk_documents: bool = True
    ) -> List[str]:
        """Add multiple documents in batch."""
        all_doc_ids = []
        
        for doc_data in documents:
            doc_ids = await self.add_document(
                content=doc_data["content"],
                metadata=doc_data.get("metadata"),
                doc_id=doc_data.get("doc_id"),
                chunk_document=chunk_documents
            )
            all_doc_ids.extend(doc_ids)
        
        return all_doc_ids
    
    async def retrieve_context(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for a query."""
        k = k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        results = await self.vector_store.similarity_search(
            query=query,
            k=k,
            threshold=threshold
        )
        
        logger.info(f"Retrieved {len(results)} relevant documents for query")
        return results
    
    async def generate_context_prompt(
        self,
        query: str,
        max_context_length: int = 4000
    ) -> Tuple[str, List[Document]]:
        """Generate a context-enhanced prompt for LLM."""
        # Retrieve relevant documents
        relevant_docs = await self.retrieve_context(query)
        
        if not relevant_docs:
            return query, []
        
        # Build context string
        context_parts = []
        used_docs = []
        current_length = 0
        
        for doc, similarity in relevant_docs:
            doc_text = f"Document {doc.doc_id} (similarity: {similarity:.3f}):\n{doc.content}\n"
            
            if current_length + len(doc_text) > max_context_length:
                break
            
            context_parts.append(doc_text)
            used_docs.append(doc)
            current_length += len(doc_text)
        
        if context_parts:
            context = "\n---\n".join(context_parts)
            enhanced_prompt = f"""
Context Information:
{context}

---

Based on the above context, please answer the following question:
{query}

If the context doesn't contain relevant information, please indicate that in your response.
"""
        else:
            enhanced_prompt = query
        
        return enhanced_prompt, used_docs
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the RAG system."""
        return await self.vector_store.delete_document(doc_id)
    
    async def search_documents(
        self,
        query: str,
        k: int = 10,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Search documents and return formatted results."""
        results = await self.retrieve_context(query, k=k)
        
        formatted_results = []
        for doc, similarity in results:
            result = {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "similarity": similarity
            }
            
            if include_metadata:
                result["metadata"] = doc.metadata
            
            formatted_results.append(result)
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "embedding_model": self.embedding_model_name,
            "vector_store_type": type(self.vector_store).__name__,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }


# Global RAG system instance
rag_system = RAGSystem()
