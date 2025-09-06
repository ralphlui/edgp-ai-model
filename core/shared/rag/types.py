"""
RAG (Retrieval-Augmented Generation) Type Definitions

Comprehensive type system for RAG management including:
- Document types and metadata
- Vector store configurations
- Embedding models and strategies
- Retrieval configurations and results
- Performance metrics and optimization
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    IN_MEMORY = "in_memory"
    CHROMA = "chroma"
    PINECONE = "pinecone" 
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    FAISS = "faiss"
    ELASTICSEARCH = "elasticsearch"


class EmbeddingModelType(str, Enum):
    """Supported embedding model types."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"


class DocumentType(str, Enum):
    """Document content types."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    CODE = "code"


class ChunkingStrategy(str, Enum):
    """Text chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"
    TOKEN_BASED = "token_based"


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for RAG."""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximal Marginal Relevance
    HYBRID = "hybrid"  # Vector + keyword search
    RERANK = "rerank"  # With reranking model
    CONTEXTUAL = "contextual"  # Context-aware retrieval


class IndexingStatus(str, Enum):
    """Document indexing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    UPDATING = "updating"


@dataclass
class DocumentMetadata:
    """Rich metadata for documents."""
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    language: str = "en"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    content_type: DocumentType = DocumentType.TEXT
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)


class Document(BaseModel):
    """Document model for RAG system."""
    doc_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    chunks: List['DocumentChunk'] = Field(default_factory=list)
    indexing_status: IndexingStatus = Field(default=IndexingStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Document content cannot be empty')
        return v


@dataclass
class DocumentChunk:
    """Chunk of a document for fine-grained retrieval."""
    chunk_id: str
    doc_id: str
    content: str
    start_idx: int
    end_idx: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    created_at: datetime = field(default_factory=datetime.now)


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = Field(default=1000, ge=100, le=8192)
    chunk_overlap: int = Field(default=200, ge=0)
    separators: List[str] = Field(default=["\n\n", "\n", " ", ""])
    max_chunks_per_doc: Optional[int] = Field(None, ge=1)
    preserve_structure: bool = True
    
    @model_validator(mode='after')
    def overlap_less_than_size(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('Chunk overlap must be less than chunk size')
        return self


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model_type: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: Optional[int] = None
    max_input_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""
    store_type: VectorStoreType = VectorStoreType.IN_MEMORY
    connection_string: Optional[str] = None
    collection_name: str = "rag_documents"
    distance_metric: str = "cosine"
    index_params: Dict[str, Any] = Field(default_factory=dict)
    search_params: Dict[str, Any] = Field(default_factory=dict)
    persistence_path: Optional[str] = None
    
    # Specific configurations for different stores
    chroma_config: Dict[str, Any] = Field(default_factory=dict)
    pinecone_config: Dict[str, Any] = Field(default_factory=dict)
    weaviate_config: Dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    """Configuration for retrieval operations."""
    strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    
    # MMR specific
    mmr_diversity_bias: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Hybrid search specific
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Reranking specific
    rerank_model: Optional[str] = None
    rerank_top_k: Optional[int] = None
    
    # Context-aware specific
    context_window: int = Field(default=3, ge=1)
    context_overlap: bool = True
    
    @field_validator('keyword_weight', 'semantic_weight')
    @classmethod
    def weights_sum_to_one(cls, v, info):
        if info.field_name == 'semantic_weight' and 'keyword_weight' in info.data:
            if abs(v + info.data['keyword_weight'] - 1.0) > 1e-6:
                raise ValueError('Keyword and semantic weights must sum to 1.0')
        return v


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    documents: List[Document]
    chunks: List[DocumentChunk]
    scores: List[float]
    query: str
    retrieval_time: float
    total_results: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "documents": [doc.dict() for doc in self.documents],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index
                }
                for chunk in self.chunks
            ],
            "scores": self.scores,
            "query": self.query,
            "retrieval_time": self.retrieval_time,
            "total_results": self.total_results,
            "metadata": self.metadata
        }


class RAGMetrics(BaseModel):
    """Metrics for RAG system performance."""
    total_documents: int = 0
    total_chunks: int = 0
    index_size_mb: float = 0.0
    avg_retrieval_time: float = 0.0
    cache_hit_rate: float = 0.0
    query_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)
    
    # Quality metrics
    relevance_scores: List[float] = Field(default_factory=list)
    avg_relevance: float = 0.0
    precision_at_k: Dict[int, float] = Field(default_factory=dict)
    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    
    # Performance metrics
    indexing_time_per_doc: float = 0.0
    memory_usage_mb: float = 0.0
    storage_usage_mb: float = 0.0


class RAGQuery(BaseModel):
    """Query for RAG system."""
    query_text: str = Field(..., description="Query text")
    query_id: Optional[str] = None
    retrieval_config: Optional[RetrievalConfig] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('query_text')
    @classmethod
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty')
        return v


class DocumentSource(BaseModel):
    """Source configuration for document ingestion."""
    source_id: str
    source_type: str  # file, url, database, api, etc.
    source_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    auto_update: bool = False
    update_frequency: Optional[str] = None  # cron expression
    last_updated: Optional[datetime] = None
    active: bool = True


class IndexingJob(BaseModel):
    """Background indexing job."""
    job_id: str
    job_type: str  # index, update, delete, rebuild
    status: IndexingStatus
    source: Optional[DocumentSource] = None
    documents: List[str] = Field(default_factory=list)  # document IDs
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    total_items: int = 0
    processed_items: int = 0


class RAGSystemConfig(BaseModel):
    """Complete configuration for RAG system."""
    system_id: str = "rag_system"
    embedding_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
    
    # System settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000
    enable_async: bool = True
    max_concurrent_operations: int = 10
    
    # Monitoring and logging
    enable_metrics: bool = True
    metrics_retention_days: int = 30
    log_level: str = "INFO"
    
    # Security
    enable_auth: bool = False
    api_keys: List[str] = Field(default_factory=list)
    rate_limit_per_minute: int = 100


# Abstract base classes for extensibility
class BaseDocumentProcessor(ABC):
    """Base class for document processors."""
    
    @abstractmethod
    def process(self, content: str, metadata: DocumentMetadata) -> Document:
        """Process raw content into a document."""
        pass
    
    @abstractmethod
    def supports(self, content_type: DocumentType) -> bool:
        """Check if processor supports content type."""
        pass


class BaseEmbeddingModel(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class BaseVectorStore(ABC):
    """Base class for vector stores."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, config: RetrievalConfig) -> RetrievalResult:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the store."""
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        pass


class BaseRetriever(ABC):
    """Base class for retrievers."""
    
    @abstractmethod
    async def retrieve(self, query: RAGQuery) -> RetrievalResult:
        """Retrieve relevant documents for query."""
        pass
    
    @abstractmethod
    def configure(self, config: RetrievalConfig):
        """Configure retriever parameters."""
        pass
