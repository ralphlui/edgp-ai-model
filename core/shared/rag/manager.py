"""
RAG Manager

Centralized management system for Retrieval-Augmented Generation (RAG) operations.
Provides enterprise-grade RAG capabilities with document management, vector indexing,
retrieval strategies, and performance optimization.
"""

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict

from .types import (
    Document, DocumentChunk, DocumentMetadata, DocumentType,
    RAGSystemConfig, RAGQuery, RetrievalResult, RAGMetrics,
    IndexingJob, IndexingStatus, DocumentSource,
    EmbeddingConfig, VectorStoreConfig, ChunkingConfig, RetrievalConfig,
    BaseDocumentProcessor, BaseEmbeddingModel, BaseVectorStore, BaseRetriever
)
from .processors import DocumentProcessorRegistry
from .embeddings import EmbeddingModelManager
from .stores import VectorStoreManager
from .retrievers import RetrieverManager

logger = logging.getLogger(__name__)


class RAGManager:
    """
    Centralized RAG management system providing:
    - Document ingestion and processing
    - Vector indexing and storage management
    - Multi-strategy retrieval
    - Performance monitoring and optimization
    - Caching and async operations
    - Background indexing jobs
    """
    
    def __init__(self, config: RAGSystemConfig = None):
        self.config = config or RAGSystemConfig()
        
        # Core components
        self.document_processor_registry = DocumentProcessorRegistry()
        self.embedding_manager = EmbeddingModelManager(self.config.embedding_config)
        self.vector_store_manager = VectorStoreManager(self.config.vector_store_config)
        self.retriever_manager = RetrieverManager(
            self.embedding_manager,
            self.vector_store_manager,
            self.config.retrieval_config
        )
        
        # Document management
        self.documents: Dict[str, Document] = {}
        self.document_sources: Dict[str, DocumentSource] = {}
        
        # Job management
        self.indexing_jobs: Dict[str, IndexingJob] = {}
        self.job_queue = asyncio.Queue() if self.config.enable_async else None
        
        # Caching
        self._query_cache: Dict[str, RetrievalResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Metrics
        self.metrics = RAGMetrics()
        self._query_times: List[float] = []
        
        # Background workers
        self._workers_started = False
        
        logger.info("RAGManager initialized with config: %s", self.config.system_id)
    
    async def initialize(self):
        """Initialize all components and start background workers."""
        await self.embedding_manager.initialize()
        await self.vector_store_manager.initialize()
        await self.retriever_manager.initialize()
        
        if self.config.enable_async and not self._workers_started:
            await self._start_background_workers()
            self._workers_started = True
        
        logger.info("RAGManager fully initialized")
    
    async def add_document(
        self,
        content: str,
        metadata: DocumentMetadata = None,
        doc_id: Optional[str] = None,
        process_async: bool = True
    ) -> str:
        """
        Add a single document to the RAG system.
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID (generated if not provided)
            process_async: Whether to process asynchronously
            
        Returns:
            Document ID
        """
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty")
        
        # Generate document ID if not provided
        if not doc_id:
            doc_id = self._generate_doc_id(content)
        
        # Create document
        document = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata or DocumentMetadata()
        )
        
        # Store document
        self.documents[doc_id] = document
        
        if process_async and self.config.enable_async:
            # Queue for background processing
            job = IndexingJob(
                job_id=f"index_{doc_id}",
                job_type="index",
                status=IndexingStatus.PENDING,
                documents=[doc_id],
                total_items=1
            )
            
            self.indexing_jobs[job.job_id] = job
            await self.job_queue.put(job)
            
            logger.info("Queued document %s for async indexing", doc_id)
        else:
            # Process immediately
            await self._process_document(document)
            logger.info("Processed document %s immediately", doc_id)
        
        return doc_id
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        process_async: bool = True
    ) -> List[str]:
        """
        Add multiple documents to the RAG system.
        
        Args:
            documents: List of document dictionaries with 'content' and optional 'metadata', 'doc_id'
            process_async: Whether to process asynchronously
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        for doc_data in documents:
            content = doc_data.get('content')
            if not content:
                continue
            
            metadata_data = doc_data.get('metadata', {})
            metadata = DocumentMetadata(**metadata_data) if metadata_data else DocumentMetadata()
            
            doc_id = await self.add_document(
                content=content,
                metadata=metadata,
                doc_id=doc_data.get('doc_id'),
                process_async=process_async
            )
            doc_ids.append(doc_id)
        
        logger.info("Added %d documents to RAG system", len(doc_ids))
        return doc_ids
    
    async def add_document_source(
        self,
        source: DocumentSource,
        initial_load: bool = True
    ) -> str:
        """
        Add a document source for automatic ingestion.
        
        Args:
            source: Document source configuration
            initial_load: Whether to perform initial load
            
        Returns:
            Source ID
        """
        self.document_sources[source.source_id] = source
        
        if initial_load:
            job = IndexingJob(
                job_id=f"source_load_{source.source_id}",
                job_type="index",
                status=IndexingStatus.PENDING,
                source=source
            )
            
            self.indexing_jobs[job.job_id] = job
            
            if self.config.enable_async:
                await self.job_queue.put(job)
            else:
                await self._process_source(source)
        
        logger.info("Added document source: %s", source.source_id)
        return source.source_id
    
    async def query(
        self,
        query: Union[str, RAGQuery],
        config: Optional[RetrievalConfig] = None
    ) -> RetrievalResult:
        """
        Query the RAG system for relevant documents.
        
        Args:
            query: Query string or RAGQuery object
            config: Optional retrieval configuration override
            
        Returns:
            RetrievalResult with relevant documents and metadata
        """
        start_time = datetime.now()
        
        # Convert string to RAGQuery if needed
        if isinstance(query, str):
            rag_query = RAGQuery(query_text=query, retrieval_config=config)
        else:
            rag_query = query
            if config:
                rag_query.retrieval_config = config
        
        # Check cache
        if self.config.enable_caching:
            cache_key = self._get_cache_key(rag_query)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug("Returning cached result for query: %s", rag_query.query_text[:50])
                return cached_result
        
        # Perform retrieval
        result = await self.retriever_manager.retrieve(rag_query)
        
        # Calculate timing
        retrieval_time = (datetime.now() - start_time).total_seconds()
        result.retrieval_time = retrieval_time
        
        # Cache result
        if self.config.enable_caching:
            self._cache_result(cache_key, result)
        
        # Update metrics
        self._update_query_metrics(retrieval_time, result)
        
        logger.info("Query completed in %.3fs: '%s'", retrieval_time, rag_query.query_text[:50])
        return result
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the RAG system.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if doc_id not in self.documents:
            return False
        
        # Remove from vector store
        await self.vector_store_manager.delete_documents([doc_id])
        
        # Remove from local storage
        del self.documents[doc_id]
        
        # Clear related cache entries
        self._invalidate_cache()
        
        # Update metrics
        self.metrics.total_documents -= 1
        
        logger.info("Deleted document: %s", doc_id)
        return True
    
    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            content: New content (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if updated, False if not found
        """
        if doc_id not in self.documents:
            return False
        
        document = self.documents[doc_id]
        updated = False
        
        if content is not None:
            document.content = content
            updated = True
        
        if metadata is not None:
            document.metadata = metadata
            updated = True
        
        if updated:
            document.updated_at = datetime.now()
            document.indexing_status = IndexingStatus.PENDING
            
            # Reprocess document
            await self._process_document(document)
            
            # Clear cache
            self._invalidate_cache()
            
            logger.info("Updated document: %s", doc_id)
        
        return updated
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    async def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Document]:
        """
        List documents with optional filtering.
        
        Args:
            filters: Optional filters (e.g., {"category": "technical", "tags": ["ai"]})
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of documents
        """
        documents = list(self.documents.values())
        
        # Apply filters
        if filters:
            filtered_docs = []
            for doc in documents:
                if self._matches_filters(doc, filters):
                    filtered_docs.append(doc)
            documents = filtered_docs
        
        # Sort by creation date (newest first)
        documents.sort(key=lambda d: d.created_at, reverse=True)
        
        # Apply pagination
        if limit is not None:
            documents = documents[offset:offset + limit]
        else:
            documents = documents[offset:]
        
        return documents
    
    async def get_job_status(self, job_id: str) -> Optional[IndexingJob]:
        """Get status of an indexing job."""
        return self.indexing_jobs.get(job_id)
    
    async def list_jobs(
        self,
        status: Optional[IndexingStatus] = None
    ) -> List[IndexingJob]:
        """List indexing jobs with optional status filter."""
        jobs = list(self.indexing_jobs.values())
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation date (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs
    
    async def rebuild_index(self, doc_ids: Optional[List[str]] = None) -> str:
        """
        Rebuild the vector index for all or specific documents.
        
        Args:
            doc_ids: Optional list of document IDs to rebuild (all if None)
            
        Returns:
            Job ID for the rebuild operation
        """
        documents_to_rebuild = doc_ids or list(self.documents.keys())
        
        job = IndexingJob(
            job_id=f"rebuild_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            job_type="rebuild",
            status=IndexingStatus.PENDING,
            documents=documents_to_rebuild,
            total_items=len(documents_to_rebuild)
        )
        
        self.indexing_jobs[job.job_id] = job
        
        if self.config.enable_async:
            await self.job_queue.put(job)
        else:
            await self._process_rebuild_job(job)
        
        logger.info("Started index rebuild job: %s", job.job_id)
        return job.job_id
    
    def get_metrics(self) -> RAGMetrics:
        """Get current system metrics."""
        # Update real-time metrics
        self.metrics.total_documents = len(self.documents)
        self.metrics.query_count = len(self._query_times)
        
        if self._query_times:
            self.metrics.avg_retrieval_time = sum(self._query_times) / len(self._query_times)
        
        # Calculate cache hit rate
        total_queries = len(self._query_times)
        if total_queries > 0:
            cache_hits = total_queries - len([t for t in self._query_times if t > 0])
            self.metrics.cache_hit_rate = cache_hits / total_queries
        
        self.metrics.last_updated = datetime.now()
        
        return self.metrics
    
    async def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cleared RAG query cache")
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze and optimize system performance.
        
        Returns:
            Dictionary with optimization results and recommendations
        """
        optimization_result = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "recommendations": []
        }
        
        # Clean up old cache entries
        if self.config.enable_caching:
            old_cache_size = len(self._query_cache)
            self._cleanup_cache()
            new_cache_size = len(self._query_cache)
            
            if old_cache_size > new_cache_size:
                optimization_result["actions_taken"].append(
                    f"Cleaned up {old_cache_size - new_cache_size} expired cache entries"
                )
        
        # Analyze query patterns
        if len(self._query_times) > 100:
            avg_time = sum(self._query_times) / len(self._query_times)
            if avg_time > 1.0:  # More than 1 second average
                optimization_result["recommendations"].append(
                    "Consider optimizing vector store configuration for faster queries"
                )
        
        # Check document distribution
        if len(self.documents) > 10000:
            optimization_result["recommendations"].append(
                "Consider implementing document archiving for better performance"
            )
        
        # Vector store optimization
        vector_optimization = await self.vector_store_manager.optimize()
        if vector_optimization:
            optimization_result["actions_taken"].extend(vector_optimization)
        
        logger.info("Performance optimization completed: %d actions, %d recommendations",
                   len(optimization_result["actions_taken"]),
                   len(optimization_result["recommendations"]))
        
        return optimization_result
    
    async def export_data(self, export_path: str) -> str:
        """
        Export RAG system data to a file.
        
        Args:
            export_path: Path for export file
            
        Returns:
            Path to exported file
        """
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.dict(),
            "documents": [doc.dict() for doc in self.documents.values()],
            "sources": [source.dict() for source in self.document_sources.values()],
            "metrics": self.metrics.dict()
        }
        
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("Exported RAG data to: %s", export_path)
        return str(export_file)
    
    async def import_data(self, import_path: str, overwrite: bool = False) -> Dict[str, int]:
        """
        Import RAG system data from a file.
        
        Args:
            import_path: Path to import file
            overwrite: Whether to overwrite existing documents
            
        Returns:
            Dictionary with import statistics
        """
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        stats = {
            "documents_imported": 0,
            "documents_skipped": 0,
            "sources_imported": 0,
            "errors": 0
        }
        
        # Import documents
        for doc_data in import_data.get("documents", []):
            try:
                doc_id = doc_data["doc_id"]
                
                if doc_id in self.documents and not overwrite:
                    stats["documents_skipped"] += 1
                    continue
                
                # Recreate document
                metadata = DocumentMetadata(**doc_data["metadata"])
                document = Document(
                    doc_id=doc_id,
                    content=doc_data["content"],
                    metadata=metadata
                )
                
                self.documents[doc_id] = document
                await self._process_document(document)
                
                stats["documents_imported"] += 1
                
            except Exception as e:
                logger.error("Failed to import document: %s", e)
                stats["errors"] += 1
        
        # Import sources
        for source_data in import_data.get("sources", []):
            try:
                source = DocumentSource(**source_data)
                self.document_sources[source.source_id] = source
                stats["sources_imported"] += 1
                
            except Exception as e:
                logger.error("Failed to import source: %s", e)
                stats["errors"] += 1
        
        # Clear cache after import
        await self.clear_cache()
        
        logger.info("Import completed: %s", stats)
        return stats
    
    # Private methods
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate a unique document ID from content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"doc_{timestamp}_{content_hash[:8]}"
    
    async def _process_document(self, document: Document):
        """Process a document for indexing."""
        try:
            document.indexing_status = IndexingStatus.PROCESSING
            
            # Process document through processors
            processed_doc = await self.document_processor_registry.process(document)
            
            # Generate embeddings and chunks
            processed_doc = await self.embedding_manager.process_document(processed_doc)
            
            # Add to vector store
            await self.vector_store_manager.add_documents([processed_doc])
            
            # Update status
            processed_doc.indexing_status = IndexingStatus.INDEXED
            self.documents[processed_doc.doc_id] = processed_doc
            
            # Update metrics
            self.metrics.total_documents = len(self.documents)
            
        except Exception as e:
            document.indexing_status = IndexingStatus.FAILED
            logger.error("Failed to process document %s: %s", document.doc_id, e)
            raise
    
    async def _process_source(self, source: DocumentSource):
        """Process documents from a source."""
        # This would be implemented based on source type
        # For now, placeholder implementation
        logger.info("Processing source: %s", source.source_id)
    
    async def _process_rebuild_job(self, job: IndexingJob):
        """Process a rebuild job."""
        try:
            job.status = IndexingStatus.PROCESSING
            job.started_at = datetime.now()
            
            for i, doc_id in enumerate(job.documents):
                if doc_id in self.documents:
                    await self._process_document(self.documents[doc_id])
                    job.processed_items = i + 1
                    job.progress = job.processed_items / job.total_items
            
            job.status = IndexingStatus.INDEXED
            job.completed_at = datetime.now()
            
        except Exception as e:
            job.status = IndexingStatus.FAILED
            job.error_message = str(e)
            logger.error("Rebuild job failed: %s", e)
    
    def _get_cache_key(self, query: RAGQuery) -> str:
        """Generate cache key for query."""
        query_parts = [
            query.query_text,
            str(query.retrieval_config.dict() if query.retrieval_config else ""),
            str(sorted(query.filters.items()) if query.filters else "")
        ]
        return hashlib.md5("|".join(query_parts).encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[RetrievalResult]:
        """Get result from cache if not expired."""
        if cache_key not in self._query_cache:
            return None
        
        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time:
            return None
        
        if datetime.now() - cache_time > timedelta(seconds=self.config.cache_ttl):
            # Expired
            del self._query_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        return self._query_cache[cache_key]
    
    def _cache_result(self, cache_key: str, result: RetrievalResult):
        """Cache query result."""
        if len(self._query_cache) >= self.config.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps.keys(),
                           key=lambda k: self._cache_timestamps[k])
            del self._query_cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        self._query_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > timedelta(seconds=self.config.cache_ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._query_cache[key]
            del self._cache_timestamps[key]
    
    def _invalidate_cache(self):
        """Invalidate entire cache."""
        self._query_cache.clear()
        self._cache_timestamps.clear()
    
    def _update_query_metrics(self, retrieval_time: float, result: RetrievalResult):
        """Update query performance metrics."""
        self._query_times.append(retrieval_time)
        
        # Keep only recent query times (last 1000)
        if len(self._query_times) > 1000:
            self._query_times = self._query_times[-1000:]
        
        # Update relevance metrics if scores available
        if result.scores:
            self.metrics.relevance_scores.extend(result.scores)
            if len(self.metrics.relevance_scores) > 1000:
                self.metrics.relevance_scores = self.metrics.relevance_scores[-1000:]
            
            self.metrics.avg_relevance = sum(self.metrics.relevance_scores) / len(self.metrics.relevance_scores)
    
    def _matches_filters(self, document: Document, filters: Dict[str, Any]) -> bool:
        """Check if document matches filters."""
        for key, value in filters.items():
            if key == "category":
                if document.metadata.category != value:
                    return False
            elif key == "tags":
                if isinstance(value, list):
                    if not all(tag in document.metadata.tags for tag in value):
                        return False
                elif value not in document.metadata.tags:
                    return False
            elif key == "content_type":
                if document.metadata.content_type != value:
                    return False
            elif key == "author":
                if document.metadata.author != value:
                    return False
            # Add more filter conditions as needed
        
        return True
    
    async def _start_background_workers(self):
        """Start background workers for async processing."""
        if not self.config.enable_async:
            return
        
        # Start job processing workers
        for i in range(self.config.max_concurrent_operations):
            asyncio.create_task(self._job_worker(f"worker_{i}"))
        
        logger.info("Started %d background workers", self.config.max_concurrent_operations)
    
    async def _job_worker(self, worker_name: str):
        """Background worker for processing jobs."""
        while True:
            try:
                job = await self.job_queue.get()
                logger.debug("Worker %s processing job: %s", worker_name, job.job_id)
                
                if job.job_type == "index":
                    if job.source:
                        await self._process_source(job.source)
                    else:
                        for doc_id in job.documents:
                            if doc_id in self.documents:
                                await self._process_document(self.documents[doc_id])
                
                elif job.job_type == "rebuild":
                    await self._process_rebuild_job(job)
                
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error("Worker %s error: %s", worker_name, e)
