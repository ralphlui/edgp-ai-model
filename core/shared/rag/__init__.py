"""
RAG (Retrieval-Augmented Generation) Management System

This module provides a comprehensive RAG system for managing documents,
embeddings, vector stores, and retrieval strategies.

Key Components:
- Document processing and chunking
- Multiple embedding model support
- Vector store management
- Advanced retrieval strategies
- Performance optimization and caching

Example Usage:
    from core.shared.rag import RAGSystem, RAGSystemConfig
    
    # Initialize RAG system
    config = RAGSystemConfig(
        embedding_model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        vector_store_type=VectorStoreType.CHROMA
    )
    
    rag_system = RAGSystem(config)
    await rag_system.initialize()
    
    # Add documents
    documents = [
        Document(
            doc_id="doc1",
            content="Your document content here",
            doc_type=DocumentType.TEXT
        )
    ]
    
    await rag_system.add_documents(documents)
    
    # Query documents
    from core.shared.rag.types import RAGQuery, RetrievalStrategy
    
    query = RAGQuery(
        query_text="What is the main topic?",
        retrieval_config=RetrievalConfig(
            strategy=RetrievalStrategy.MMR,
            top_k=5
        )
    )
    
    result = await rag_system.query(query)
    print(f"Found {len(result.documents)} relevant documents")
"""

# Type definitions
from .types import (
    # Enums
    VectorStoreType,
    EmbeddingModelType,
    DocumentType,
    RetrievalStrategy,
    ChunkingStrategy,
    
    # Data Models
    Document,
    DocumentChunk,
    DocumentMetadata,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RAGSystemConfig,
    
    # Query and Result Types
    RAGQuery,
    RetrievalResult,
    
    # Base Classes
    BaseEmbeddingModel,
    BaseVectorStore,
    BaseDocumentProcessor,
    BaseRetriever
)

# Core components
from .manager import RAGManager
from .processors import (
    DocumentProcessorRegistry,
    TextProcessor,
    MarkdownProcessor,
    JSONProcessor,
    HTMLProcessor
)
from .embeddings import (
    EmbeddingModelManager,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    HuggingFaceEmbedding
)
from .stores import (
    VectorStoreManager,
    InMemoryVectorStore,
    ChromaVectorStore,
    FAISSVectorStore
)
from .retrievers import (
    RetrieverManager,
    SimilarityRetriever,
    MMRRetriever,
    HybridRetriever,
    ContextualRetriever
)

# Main RAG System
class RAGSystem:
    """
    Complete RAG system combining all components.
    
    This is the main interface for using the RAG system. It manages
    all components including document processing, embeddings, vector
    stores, and retrieval.
    """
    
    def __init__(self, config: RAGSystemConfig):
        """Initialize RAG system with configuration."""
        self.config = config
        
        # Initialize managers
        self.rag_manager = RAGManager(config)
        self.embedding_manager = None
        self.vector_store_manager = None
        self.retriever_manager = None
        
        # State
        self.initialized = False
    
    async def initialize(self):
        """Initialize all RAG system components."""
        if self.initialized:
            return
        
        # Initialize RAG manager
        await self.rag_manager.initialize()
        
        # Get component managers
        self.embedding_manager = self.rag_manager.embedding_manager
        self.vector_store_manager = self.rag_manager.vector_store_manager
        
        # Initialize retriever manager
        self.retriever_manager = RetrieverManager(
            self.embedding_manager,
            self.vector_store_manager,
            self.config.retrieval_config
        )
        await self.retriever_manager.initialize()
        
        self.initialized = True
    
    async def add_documents(self, documents: list[Document]) -> dict:
        """Add documents to the RAG system."""
        if not self.initialized:
            await self.initialize()
        
        return await self.rag_manager.add_documents(documents)
    
    async def query(self, query: RAGQuery) -> RetrievalResult:
        """Query the RAG system for relevant documents."""
        if not self.initialized:
            await self.initialize()
        
        return await self.retriever_manager.retrieve(query)
    
    async def multi_strategy_query(
        self,
        query: RAGQuery,
        strategies: list[RetrievalStrategy],
        combine_method: str = "rank_fusion"
    ) -> RetrievalResult:
        """Query using multiple retrieval strategies."""
        if not self.initialized:
            await self.initialize()
        
        return await self.retriever_manager.multi_strategy_retrieve(
            query, strategies, combine_method
        )
    
    async def get_document(self, doc_id: str) -> Document:
        """Get a specific document by ID."""
        if not self.initialized:
            await self.initialize()
        
        return await self.rag_manager.get_document(doc_id)
    
    async def delete_documents(self, doc_ids: list[str]) -> dict:
        """Delete documents from the RAG system."""
        if not self.initialized:
            await self.initialize()
        
        return await self.rag_manager.delete_documents(doc_ids)
    
    async def update_document(self, document: Document) -> dict:
        """Update an existing document."""
        if not self.initialized:
            await self.initialize()
        
        return await self.rag_manager.update_document(document)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        stats = {}
        
        if self.rag_manager:
            stats['rag_manager'] = self.rag_manager.get_stats()
        
        if self.retriever_manager:
            stats['retriever'] = self.retriever_manager.get_stats()
        
        return stats
    
    async def optimize_system(self):
        """Optimize system performance."""
        if not self.initialized:
            return
        
        await self.rag_manager.optimize_system()


# Convenience functions for quick setup
def create_simple_rag_system(
    embedding_model: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS,
    vector_store: VectorStoreType = VectorStoreType.IN_MEMORY
) -> RAGSystem:
    """Create a simple RAG system with default configuration."""
    config = RAGSystemConfig(
        embedding_model_type=embedding_model,
        vector_store_type=vector_store
    )
    return RAGSystem(config)


def create_production_rag_system(
    vector_store_config: VectorStoreConfig,
    embedding_config: EmbeddingConfig
) -> RAGSystem:
    """Create a production-ready RAG system."""
    config = RAGSystemConfig(
        embedding_model_type=embedding_config.model_type,
        vector_store_type=vector_store_config.store_type,
        embedding_config=embedding_config,
        vector_store_config=vector_store_config,
        enable_caching=True,
        enable_async_processing=True,
        enable_performance_monitoring=True
    )
    return RAGSystem(config)


# Export all components
__all__ = [
    # Main system
    'RAGSystem',
    'create_simple_rag_system',
    'create_production_rag_system',
    
    # Types
    'VectorStoreType',
    'EmbeddingModelType',
    'DocumentType',
    'RetrievalStrategy',
    'ChunkingStrategy',
    'Document',
    'DocumentChunk',
    'DocumentMetadata',
    'EmbeddingConfig',
    'VectorStoreConfig',
    'RetrievalConfig',
    'RAGSystemConfig',
    'RAGQuery',
    'RetrievalResult',
    'BaseEmbeddingModel',
    'BaseVectorStore',
    'BaseDocumentProcessor',
    'BaseRetriever',
    
    # Managers
    'RAGManager',
    'EmbeddingModelManager',
    'VectorStoreManager',
    'RetrieverManager',
    
    # Processors
    'DocumentProcessorRegistry',
    'TextProcessor',
    'MarkdownProcessor',
    'JSONProcessor',
    'HTMLProcessor',
    
    # Embeddings
    'SentenceTransformerEmbedding',
    'OpenAIEmbedding',
    'HuggingFaceEmbedding',
    
    # Vector Stores
    'InMemoryVectorStore',
    'ChromaVectorStore',
    'FAISSVectorStore',
    
    # Retrievers
    'SimilarityRetriever',
    'MMRRetriever',
    'HybridRetriever',
    'ContextualRetriever'
]
