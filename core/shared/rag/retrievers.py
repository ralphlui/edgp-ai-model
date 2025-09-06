"""
Retrieval Manager

Manages different retrieval strategies for RAG systems including
similarity search, MMR, hybrid search, and reranking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime

from .types import (
    RAGQuery, RetrievalResult, RetrievalConfig, RetrievalStrategy,
    Document, DocumentChunk, BaseRetriever
)
from .embeddings import EmbeddingModelManager
from .stores import VectorStoreManager

logger = logging.getLogger(__name__)


class SimilarityRetriever(BaseRetriever):
    """Basic similarity-based retriever."""
    
    def __init__(self, embedding_manager: EmbeddingModelManager, vector_store_manager: VectorStoreManager):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.config = RetrievalConfig()
    
    async def retrieve(self, query: RAGQuery) -> RetrievalResult:
        """Retrieve documents using similarity search."""
        start_time = datetime.now()
        
        # Use query config or default
        config = query.retrieval_config or self.config
        
        # Generate query embedding
        query_embedding = await self.embedding_manager.embed_text(query.query_text)
        
        # Search vector store
        result = await self.vector_store_manager.search(query_embedding, config)
        
        # Set query and timing
        result.query = query.query_text
        result.retrieval_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def configure(self, config: RetrievalConfig):
        """Configure retriever parameters."""
        self.config = config


class MMRRetriever(BaseRetriever):
    """Maximal Marginal Relevance retriever for diverse results."""
    
    def __init__(self, embedding_manager: EmbeddingModelManager, vector_store_manager: VectorStoreManager):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.config = RetrievalConfig()
    
    async def retrieve(self, query: RAGQuery) -> RetrievalResult:
        """Retrieve documents using MMR for diversity."""
        start_time = datetime.now()
        
        # Use query config or default
        config = query.retrieval_config or self.config
        
        # Generate query embedding
        query_embedding = await self.embedding_manager.embed_text(query.query_text)
        
        # Get initial candidates (more than needed)
        candidate_config = RetrievalConfig(
            strategy=RetrievalStrategy.SIMILARITY,
            top_k=config.top_k * 3,  # Get more candidates
            similarity_threshold=config.similarity_threshold
        )
        
        candidates = await self.vector_store_manager.search(query_embedding, candidate_config)
        
        # Apply MMR selection
        selected_docs, selected_chunks, selected_scores = self._mmr_selection(
            query_embedding=query_embedding,
            candidates=candidates,
            top_k=config.top_k,
            diversity_bias=config.mmr_diversity_bias
        )
        
        result = RetrievalResult(
            documents=selected_docs,
            chunks=selected_chunks,
            scores=selected_scores,
            query=query.query_text,
            retrieval_time=(datetime.now() - start_time).total_seconds(),
            total_results=len(candidates.documents)
        )
        
        return result
    
    def configure(self, config: RetrievalConfig):
        """Configure retriever parameters."""
        self.config = config
    
    def _mmr_selection(
        self,
        query_embedding: np.ndarray,
        candidates: RetrievalResult,
        top_k: int,
        diversity_bias: float
    ) -> Tuple[List[Document], List[DocumentChunk], List[float]]:
        """Select documents using MMR algorithm."""
        if not candidates.documents:
            return [], [], []
        
        # Get embeddings for all candidates
        candidate_embeddings = []
        for doc in candidates.documents:
            if doc.embedding:
                candidate_embeddings.append(np.array(doc.embedding))
            else:
                # Fallback to zero vector
                candidate_embeddings.append(np.zeros_like(query_embedding))
        
        if not candidate_embeddings:
            return [], [], []
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(candidates.documents)))
        
        for _ in range(min(top_k, len(candidates.documents))):
            if not remaining_indices:
                break
            
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Relevance score
                relevance = np.dot(query_embedding, candidate_embeddings[idx]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embeddings[idx])
                )
                
                # Diversity score (maximum similarity to already selected)
                diversity = 0.0
                if selected_indices:
                    similarities = []
                    for selected_idx in selected_indices:
                        sim = np.dot(candidate_embeddings[idx], candidate_embeddings[selected_idx]) / (
                            np.linalg.norm(candidate_embeddings[idx]) * np.linalg.norm(candidate_embeddings[selected_idx])
                        )
                        similarities.append(sim)
                    diversity = max(similarities)
                
                # MMR score
                mmr_score = (1 - diversity_bias) * relevance - diversity_bias * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        # Return selected documents
        selected_docs = [candidates.documents[i] for i in selected_indices]
        selected_chunks = [candidates.chunks[i] for i in selected_indices if i < len(candidates.chunks)]
        selected_scores = [candidates.scores[i] for i in selected_indices if i < len(candidates.scores)]
        
        return selected_docs, selected_chunks, selected_scores


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(self, embedding_manager: EmbeddingModelManager, vector_store_manager: VectorStoreManager):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.config = RetrievalConfig()
        
        # Keyword search index (simple implementation)
        self.keyword_index: Dict[str, List[str]] = {}  # word -> list of doc_ids
        self.documents: Dict[str, Document] = {}
    
    async def retrieve(self, query: RAGQuery) -> RetrievalResult:
        """Retrieve documents using hybrid search."""
        start_time = datetime.now()
        
        # Use query config or default
        config = query.retrieval_config or self.config
        
        # Semantic search
        semantic_results = await self._semantic_search(query.query_text, config)
        
        # Keyword search
        keyword_results = await self._keyword_search(query.query_text, config)
        
        # Combine results
        combined_results = self._combine_results(
            semantic_results, keyword_results, config
        )
        
        combined_results.query = query.query_text
        combined_results.retrieval_time = (datetime.now() - start_time).total_seconds()
        
        return combined_results
    
    def configure(self, config: RetrievalConfig):
        """Configure retriever parameters."""
        self.config = config
    
    async def _semantic_search(self, query: str, config: RetrievalConfig) -> RetrievalResult:
        """Perform semantic search."""
        query_embedding = await self.embedding_manager.embed_text(query)
        return await self.vector_store_manager.search(query_embedding, config)
    
    async def _keyword_search(self, query: str, config: RetrievalConfig) -> RetrievalResult:
        """Perform keyword search."""
        query_words = set(query.lower().split())
        
        # Score documents by keyword overlap
        doc_scores = {}
        
        for word in query_words:
            if word in self.keyword_index:
                for doc_id in self.keyword_index[word]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        
        # Normalize scores and get top documents
        if doc_scores:
            max_score = max(doc_scores.values())
            for doc_id in doc_scores:
                doc_scores[doc_id] /= max_score
        
        # Sort by score and get top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:config.top_k]
        
        # Create result
        documents = []
        scores = []
        
        for doc_id, score in top_docs:
            if doc_id in self.documents:
                documents.append(self.documents[doc_id])
                scores.append(score)
        
        return RetrievalResult(
            documents=documents,
            chunks=[],
            scores=scores,
            query=query,
            retrieval_time=0.0,
            total_results=len(doc_scores)
        )
    
    def _combine_results(
        self,
        semantic_results: RetrievalResult,
        keyword_results: RetrievalResult,
        config: RetrievalConfig
    ) -> RetrievalResult:
        """Combine semantic and keyword search results."""
        # Create combined scoring
        doc_scores = {}
        
        # Add semantic scores
        for doc, score in zip(semantic_results.documents, semantic_results.scores):
            doc_scores[doc.doc_id] = {
                'semantic': score * config.semantic_weight,
                'keyword': 0.0,
                'document': doc
            }
        
        # Add keyword scores
        for doc, score in zip(keyword_results.documents, keyword_results.scores):
            if doc.doc_id in doc_scores:
                doc_scores[doc.doc_id]['keyword'] = score * config.keyword_weight
            else:
                doc_scores[doc.doc_id] = {
                    'semantic': 0.0,
                    'keyword': score * config.keyword_weight,
                    'document': doc
                }
        
        # Calculate final scores
        final_results = []
        for doc_id, scores in doc_scores.items():
            final_score = scores['semantic'] + scores['keyword']
            final_results.append((scores['document'], final_score))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top k
        top_results = final_results[:config.top_k]
        
        documents = [doc for doc, _ in top_results]
        scores = [score for _, score in top_results]
        
        return RetrievalResult(
            documents=documents,
            chunks=semantic_results.chunks,  # Use semantic chunks
            scores=scores,
            query="",
            retrieval_time=0.0,
            total_results=len(final_results)
        )
    
    async def index_documents(self, documents: List[Document]):
        """Index documents for keyword search."""
        for doc in documents:
            self.documents[doc.doc_id] = doc
            
            # Simple keyword indexing
            words = set(doc.content.lower().split())
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                if doc.doc_id not in self.keyword_index[word]:
                    self.keyword_index[word].append(doc.doc_id)


class ContextualRetriever(BaseRetriever):
    """Context-aware retriever that considers conversation history."""
    
    def __init__(self, embedding_manager: EmbeddingModelManager, vector_store_manager: VectorStoreManager):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.config = RetrievalConfig()
        self.conversation_history: List[str] = []
    
    async def retrieve(self, query: RAGQuery) -> RetrievalResult:
        """Retrieve documents with context awareness."""
        start_time = datetime.now()
        
        # Use query config or default
        config = query.retrieval_config or self.config
        
        # Build context-aware query
        contextual_query = self._build_contextual_query(query.query_text, query.context)
        
        # Generate embedding for contextual query
        query_embedding = await self.embedding_manager.embed_text(contextual_query)
        
        # Search with context
        result = await self.vector_store_manager.search(query_embedding, config)
        
        # Re-rank based on context relevance
        if result.documents and query.context:
            result = await self._rerank_by_context(result, query.context, config)
        
        result.query = query.query_text
        result.retrieval_time = (datetime.now() - start_time).total_seconds()
        
        # Update conversation history
        self.conversation_history.append(query.query_text)
        if len(self.conversation_history) > config.context_window:
            self.conversation_history = self.conversation_history[-config.context_window:]
        
        return result
    
    def configure(self, config: RetrievalConfig):
        """Configure retriever parameters."""
        self.config = config
    
    def _build_contextual_query(self, query: str, context: Optional[str] = None) -> str:
        """Build query with context information."""
        context_parts = []
        
        # Add conversation history
        if self.conversation_history:
            recent_history = self.conversation_history[-self.config.context_window:]
            context_parts.append("Previous queries: " + " | ".join(recent_history))
        
        # Add provided context
        if context:
            context_parts.append("Context: " + context)
        
        # Add current query
        context_parts.append("Current query: " + query)
        
        return " ".join(context_parts)
    
    async def _rerank_by_context(
        self,
        result: RetrievalResult,
        context: str,
        config: RetrievalConfig
    ) -> RetrievalResult:
        """Re-rank results based on context relevance."""
        if not result.documents:
            return result
        
        # Generate context embedding
        context_embedding = await self.embedding_manager.embed_text(context)
        
        # Calculate context relevance scores
        context_scores = []
        for doc in result.documents:
            if doc.embedding:
                doc_embedding = np.array(doc.embedding)
                context_relevance = np.dot(context_embedding, doc_embedding) / (
                    np.linalg.norm(context_embedding) * np.linalg.norm(doc_embedding)
                )
                context_scores.append(context_relevance)
            else:
                context_scores.append(0.0)
        
        # Combine original scores with context scores
        combined_scores = []
        for i, (original_score, context_score) in enumerate(zip(result.scores, context_scores)):
            # Weight: 70% original relevance, 30% context relevance
            combined_score = 0.7 * original_score + 0.3 * context_score
            combined_scores.append((i, combined_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Reorder results
        reordered_docs = [result.documents[i] for i, _ in combined_scores]
        reordered_chunks = [result.chunks[i] for i, _ in combined_scores if i < len(result.chunks)]
        reordered_scores = [score for _, score in combined_scores]
        
        return RetrievalResult(
            documents=reordered_docs,
            chunks=reordered_chunks,
            scores=reordered_scores,
            query=result.query,
            retrieval_time=result.retrieval_time,
            total_results=result.total_results
        )


class RetrieverManager:
    """Manager for different retrieval strategies."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingModelManager,
        vector_store_manager: VectorStoreManager,
        default_config: RetrievalConfig
    ):
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.default_config = default_config
        
        # Initialize retrievers
        self.retrievers: Dict[RetrievalStrategy, BaseRetriever] = {}
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'strategy_usage': {}
        }
    
    async def initialize(self):
        """Initialize all retrievers."""
        self.retrievers[RetrievalStrategy.SIMILARITY] = SimilarityRetriever(
            self.embedding_manager, self.vector_store_manager
        )
        
        self.retrievers[RetrievalStrategy.MMR] = MMRRetriever(
            self.embedding_manager, self.vector_store_manager
        )
        
        self.retrievers[RetrievalStrategy.HYBRID] = HybridRetriever(
            self.embedding_manager, self.vector_store_manager
        )
        
        self.retrievers[RetrievalStrategy.CONTEXTUAL] = ContextualRetriever(
            self.embedding_manager, self.vector_store_manager
        )
        
        # Configure all retrievers with default config
        for retriever in self.retrievers.values():
            retriever.configure(self.default_config)
        
        logger.info("RetrieverManager initialized with %d strategies", len(self.retrievers))
    
    async def retrieve(self, query: RAGQuery) -> RetrievalResult:
        """Retrieve documents using specified strategy."""
        # Determine strategy
        strategy = (
            query.retrieval_config.strategy if query.retrieval_config 
            else self.default_config.strategy
        )
        
        # Get retriever
        retriever = self.retrievers.get(strategy)
        if not retriever:
            # Fallback to similarity search
            retriever = self.retrievers[RetrievalStrategy.SIMILARITY]
            strategy = RetrievalStrategy.SIMILARITY
        
        # Perform retrieval
        start_time = datetime.now()
        result = await retriever.retrieve(query)
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        # Update statistics
        self._update_stats(strategy, retrieval_time)
        
        logger.debug("Retrieved %d documents using %s strategy in %.3fs",
                    len(result.documents), strategy.value, retrieval_time)
        
        return result
    
    async def multi_strategy_retrieve(
        self,
        query: RAGQuery,
        strategies: List[RetrievalStrategy],
        combine_method: str = "rank_fusion"
    ) -> RetrievalResult:
        """Retrieve using multiple strategies and combine results."""
        if not strategies:
            return await self.retrieve(query)
        
        start_time = datetime.now()
        
        # Get results from all strategies
        strategy_results = []
        for strategy in strategies:
            # Create query copy with specific strategy
            strategy_query = RAGQuery(
                query_text=query.query_text,
                retrieval_config=RetrievalConfig(
                    strategy=strategy,
                    top_k=query.retrieval_config.top_k if query.retrieval_config else self.default_config.top_k
                ),
                filters=query.filters,
                context=query.context
            )
            
            retriever = self.retrievers.get(strategy)
            if retriever:
                result = await retriever.retrieve(strategy_query)
                strategy_results.append((strategy, result))
        
        # Combine results
        if combine_method == "rank_fusion":
            combined_result = self._rank_fusion_combine(strategy_results, query)
        elif combine_method == "score_averaging":
            combined_result = self._score_averaging_combine(strategy_results, query)
        else:
            # Default to first result
            combined_result = strategy_results[0][1] if strategy_results else RetrievalResult(
                documents=[], chunks=[], scores=[], query=query.query_text,
                retrieval_time=0.0, total_results=0
            )
        
        combined_result.retrieval_time = (datetime.now() - start_time).total_seconds()
        
        return combined_result
    
    def _update_stats(self, strategy: RetrievalStrategy, retrieval_time: float):
        """Update retrieval statistics."""
        self.retrieval_stats['total_queries'] += 1
        
        # Update average retrieval time
        total_time = (
            self.retrieval_stats['avg_retrieval_time'] * (self.retrieval_stats['total_queries'] - 1) +
            retrieval_time
        )
        self.retrieval_stats['avg_retrieval_time'] = total_time / self.retrieval_stats['total_queries']
        
        # Update strategy usage
        strategy_name = strategy.value
        if strategy_name not in self.retrieval_stats['strategy_usage']:
            self.retrieval_stats['strategy_usage'][strategy_name] = 0
        self.retrieval_stats['strategy_usage'][strategy_name] += 1
    
    def _rank_fusion_combine(
        self,
        strategy_results: List[Tuple[RetrievalStrategy, RetrievalResult]],
        query: RAGQuery
    ) -> RetrievalResult:
        """Combine results using rank fusion."""
        doc_ranks = {}
        
        # Calculate ranks for each strategy
        for strategy, result in strategy_results:
            for i, doc in enumerate(result.documents):
                if doc.doc_id not in doc_ranks:
                    doc_ranks[doc.doc_id] = {'document': doc, 'ranks': []}
                doc_ranks[doc.doc_id]['ranks'].append(i + 1)
        
        # Calculate reciprocal rank fusion scores
        doc_scores = []
        for doc_id, data in doc_ranks.items():
            rrf_score = sum(1 / (rank + 60) for rank in data['ranks'])  # k=60 is common
            doc_scores.append((data['document'], rrf_score))
        
        # Sort by RRF score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract results
        top_k = query.retrieval_config.top_k if query.retrieval_config else self.default_config.top_k
        top_results = doc_scores[:top_k]
        
        documents = [doc for doc, _ in top_results]
        scores = [score for _, score in top_results]
        
        return RetrievalResult(
            documents=documents,
            chunks=[],  # Would need to collect chunks
            scores=scores,
            query=query.query_text,
            retrieval_time=0.0,
            total_results=len(doc_scores)
        )
    
    def _score_averaging_combine(
        self,
        strategy_results: List[Tuple[RetrievalStrategy, RetrievalResult]],
        query: RAGQuery
    ) -> RetrievalResult:
        """Combine results using score averaging."""
        doc_scores = {}
        
        # Collect scores from all strategies
        for strategy, result in strategy_results:
            for doc, score in zip(result.documents, result.scores):
                if doc.doc_id not in doc_scores:
                    doc_scores[doc.doc_id] = {'document': doc, 'scores': []}
                doc_scores[doc.doc_id]['scores'].append(score)
        
        # Calculate average scores
        avg_scores = []
        for doc_id, data in doc_scores.items():
            avg_score = sum(data['scores']) / len(data['scores'])
            avg_scores.append((data['document'], avg_score))
        
        # Sort by average score
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract results
        top_k = query.retrieval_config.top_k if query.retrieval_config else self.default_config.top_k
        top_results = avg_scores[:top_k]
        
        documents = [doc for doc, _ in top_results]
        scores = [score for _, score in top_results]
        
        return RetrievalResult(
            documents=documents,
            chunks=[],
            scores=scores,
            query=query.query_text,
            retrieval_time=0.0,
            total_results=len(avg_scores)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return self.retrieval_stats.copy()
    
    def configure_strategy(self, strategy: RetrievalStrategy, config: RetrievalConfig):
        """Configure a specific retrieval strategy."""
        if strategy in self.retrievers:
            self.retrievers[strategy].configure(config)
            logger.info("Configured %s strategy", strategy.value)
