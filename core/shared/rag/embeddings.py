"""
Embedding Model Manager

Manages different embedding models and provides unified interface
for generating embeddings for documents and queries.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod

from .types import (
    Document, DocumentChunk, EmbeddingConfig, EmbeddingModelType,
    BaseEmbeddingModel
)

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Sentence Transformers embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self.kwargs = kwargs
    
    async def initialize(self):
        """Initialize the model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, **self.kwargs)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info("Initialized SentenceTransformer model: %s (dim=%d)", 
                       self.model_name, self.dimension)
        except ImportError:
            raise ImportError("sentence-transformers package is required for SentenceTransformerEmbedding")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI embedding model."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.dimension = 1536  # Default for Ada-002
        self.client = None
        self.kwargs = kwargs
    
    async def initialize(self):
        """Initialize the OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info("Initialized OpenAI embedding model: %s", self.model_name)
        except ImportError:
            raise ImportError("openai package is required for OpenAIEmbedding")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class HuggingFaceEmbedding(BaseEmbeddingModel):
    """HuggingFace embedding model."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.dimension = None
        self.kwargs = kwargs
    
    async def initialize(self):
        """Initialize the HuggingFace model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Get dimension from model config
            self.dimension = self.model.config.hidden_size
            
            logger.info("Initialized HuggingFace model: %s (dim=%d)", 
                       self.model_name, self.dimension)
        except ImportError:
            raise ImportError("transformers and torch packages are required for HuggingFaceEmbedding")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        import torch
        
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class EmbeddingModelManager:
    """Manager for different embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model: Optional[BaseEmbeddingModel] = None
        self.batch_size = config.batch_size
        self.max_input_length = config.max_input_length
        
        # Performance tracking
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0
    
    async def initialize(self):
        """Initialize the embedding model."""
        if self.config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformerEmbedding(
                model_name=self.config.model_name,
                **self.config.model_kwargs
            )
        
        elif self.config.model_type == EmbeddingModelType.OPENAI:
            self.model = OpenAIEmbedding(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                **self.config.model_kwargs
            )
        
        elif self.config.model_type == EmbeddingModelType.HUGGINGFACE:
            self.model = HuggingFaceEmbedding(
                model_name=self.config.model_name,
                **self.config.model_kwargs
            )
        
        else:
            raise ValueError(f"Unsupported embedding model type: {self.config.model_type}")
        
        await self.model.initialize()
        
        # Update dimension in config if not set
        if not self.config.dimensions:
            self.config.dimensions = self.model.get_dimension()
        
        logger.info("EmbeddingModelManager initialized with %s model", 
                   self.config.model_type.value)
    
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        if not texts:
            return np.array([])
        
        import time
        start_time = time.time()
        
        # Truncate texts if needed
        truncated_texts = [
            text[:self.max_input_length] if len(text) > self.max_input_length else text
            for text in texts
        ]
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(truncated_texts), self.batch_size):
            batch = truncated_texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch)
            
            if len(batch_embeddings.shape) == 1:
                # Single embedding
                batch_embeddings = batch_embeddings.reshape(1, -1)
            
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Normalize if requested
        if self.config.normalize_embeddings and len(embeddings) > 0:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Update metrics
        processing_time = time.time() - start_time
        self.total_embeddings_generated += len(texts)
        self.total_processing_time += processing_time
        
        logger.debug("Generated %d embeddings in %.3fs", len(texts), processing_time)
        
        return embeddings
    
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            NumPy array with single embedding
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    async def process_document(self, document: Document) -> Document:
        """
        Process a document to generate embeddings for content and chunks.
        
        Args:
            document: Document to process
            
        Returns:
            Document with embeddings added
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Generate embedding for full document content
        doc_embedding = await self.embed_text(document.content)
        document.embedding = doc_embedding.tolist()
        
        # Generate embeddings for chunks if they exist
        if document.chunks:
            chunk_texts = [chunk.content for chunk in document.chunks]
            chunk_embeddings = await self.embed_texts(chunk_texts)
            
            for chunk, embedding in zip(document.chunks, chunk_embeddings):
                chunk.embedding = embedding
        
        logger.debug("Generated embeddings for document %s and %d chunks", 
                    document.doc_id, len(document.chunks))
        
        return document
    
    async def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process multiple documents to generate embeddings.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of documents with embeddings added
        """
        processed_docs = []
        
        for document in documents:
            processed_doc = await self.process_document(document)
            processed_docs.append(processed_doc)
        
        logger.info("Processed embeddings for %d documents", len(documents))
        return processed_docs
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.model:
            return self.model.get_dimension()
        return self.config.dimensions or 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get embedding generation metrics."""
        avg_time_per_embedding = (
            self.total_processing_time / self.total_embeddings_generated
            if self.total_embeddings_generated > 0 else 0
        )
        
        return {
            "model_type": self.config.model_type.value,
            "model_name": self.config.model_name,
            "dimension": self.get_dimension(),
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_processing_time": self.total_processing_time,
            "avg_time_per_embedding": avg_time_per_embedding,
            "batch_size": self.batch_size,
            "max_input_length": self.max_input_length
        }
    
    async def similarity_search(
        self, 
        query_text: str, 
        document_embeddings: List[np.ndarray],
        k: int = 5
    ) -> List[tuple]:
        """
        Perform similarity search against document embeddings.
        
        Args:
            query_text: Query text
            document_embeddings: List of document embeddings
            k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not document_embeddings:
            return []
        
        # Generate query embedding
        query_embedding = await self.embed_text(query_text)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(document_embeddings):
            if len(doc_embedding) > 0:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((i, float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    async def batch_similarity_search(
        self,
        query_texts: List[str],
        document_embeddings: List[np.ndarray],
        k: int = 5
    ) -> List[List[tuple]]:
        """
        Perform batch similarity search.
        
        Args:
            query_texts: List of query texts
            document_embeddings: List of document embeddings
            k: Number of results per query
            
        Returns:
            List of similarity results for each query
        """
        results = []
        
        for query_text in query_texts:
            query_results = await self.similarity_search(query_text, document_embeddings, k)
            results.append(query_results)
        
        return results
    
    def update_config(self, new_config: EmbeddingConfig):
        """Update configuration (requires reinitialization)."""
        self.config = new_config
        self.batch_size = new_config.batch_size
        self.max_input_length = new_config.max_input_length
        
        # Model needs to be reinitialized
        self.model = None
        
        logger.info("Updated embedding configuration")
    
    async def warm_up(self, sample_texts: List[str] = None):
        """
        Warm up the model with sample texts.
        
        Args:
            sample_texts: Optional sample texts for warm-up
        """
        if not sample_texts:
            sample_texts = [
                "This is a sample text for model warm-up.",
                "Another example to initialize the embedding model.",
                "Warming up the neural network for better performance."
            ]
        
        logger.info("Warming up embedding model...")
        await self.embed_texts(sample_texts)
        logger.info("Model warm-up completed")
