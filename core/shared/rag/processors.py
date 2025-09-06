"""
Document Processors

Registry and implementations for processing different document types
before indexing in the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import hashlib

from .types import (
    Document, DocumentMetadata, DocumentType, DocumentChunk,
    BaseDocumentProcessor, ChunkingConfig, ChunkingStrategy
)

logger = logging.getLogger(__name__)


class DocumentProcessorRegistry:
    """Registry for document processors."""
    
    def __init__(self):
        self.processors: Dict[DocumentType, BaseDocumentProcessor] = {}
        self._register_default_processors()
    
    def register_processor(self, content_type: DocumentType, processor: BaseDocumentProcessor):
        """Register a processor for a content type."""
        self.processors[content_type] = processor
        logger.info("Registered processor for %s", content_type.value)
    
    def get_processor(self, content_type: DocumentType) -> Optional[BaseDocumentProcessor]:
        """Get processor for content type."""
        return self.processors.get(content_type)
    
    async def process(self, document: Document) -> Document:
        """Process document using appropriate processor."""
        processor = self.get_processor(document.metadata.content_type)
        if processor:
            return processor.process(document.content, document.metadata)
        
        # Fallback to text processor
        text_processor = self.processors.get(DocumentType.TEXT)
        if text_processor:
            return text_processor.process(document.content, document.metadata)
        
        # Return original document if no processor available
        return document
    
    def _register_default_processors(self):
        """Register default processors."""
        self.register_processor(DocumentType.TEXT, TextProcessor())
        self.register_processor(DocumentType.MARKDOWN, MarkdownProcessor())
        self.register_processor(DocumentType.JSON, JSONProcessor())
        self.register_processor(DocumentType.HTML, HTMLProcessor())


class TextProcessor(BaseDocumentProcessor):
    """Basic text document processor."""
    
    def __init__(self, chunking_config: ChunkingConfig = None):
        self.chunking_config = chunking_config or ChunkingConfig()
    
    def process(self, content: str, metadata: DocumentMetadata) -> Document:
        """Process text content."""
        # Clean content
        cleaned_content = self._clean_text(content)
        
        # Create document
        doc_id = self._generate_doc_id(cleaned_content, metadata)
        document = Document(
            doc_id=doc_id,
            content=cleaned_content,
            metadata=metadata
        )
        
        # Create chunks
        document.chunks = self._create_chunks(cleaned_content, doc_id)
        
        return document
    
    def supports(self, content_type: DocumentType) -> bool:
        """Check if processor supports content type."""
        return content_type == DocumentType.TEXT
    
    def _clean_text(self, content: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = ' '.join(line.split())  # Normalize whitespace
            if cleaned_line.strip():  # Skip empty lines
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _create_chunks(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Create document chunks based on chunking strategy."""
        if self.chunking_config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(content, doc_id)
        elif self.chunking_config.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(content, doc_id)
        elif self.chunking_config.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(content, doc_id)
        elif self.chunking_config.strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(content, doc_id)
        else:
            # Default to fixed size
            return self._chunk_fixed_size(content, doc_id)
    
    def _chunk_fixed_size(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Create fixed-size chunks."""
        chunks = []
        chunk_size = self.chunking_config.chunk_size
        overlap = self.chunking_config.chunk_overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at word boundary
            if end < len(content):
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    doc_id=doc_id,
                    content=chunk_content,
                    start_idx=start,
                    end_idx=end,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def _chunk_by_sentence(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Create chunks by sentence."""
        # Simple sentence splitting (could be improved with NLP libraries)
        sentences = []
        current_sentence = ""
        
        for char in content:
            current_sentence += char
            if char in '.!?' and current_sentence.strip():
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        chunks = []
        current_chunk = ""
        start_idx = 0
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunking_config.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    end_idx = start_idx + len(current_chunk)
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_index}",
                        doc_id=doc_id,
                        content=current_chunk.strip(),
                        start_idx=start_idx,
                        end_idx=end_idx,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_idx = end_idx
                
                current_chunk = sentence + " "
        
        # Add final chunk
        if current_chunk.strip():
            end_idx = start_idx + len(current_chunk)
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                content=current_chunk.strip(),
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraph(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Create chunks by paragraph."""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        start_idx = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunking_config.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    end_idx = start_idx + len(current_chunk)
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_index}",
                        doc_id=doc_id,
                        content=current_chunk.strip(),
                        start_idx=start_idx,
                        end_idx=end_idx,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_idx = end_idx
                
                current_chunk = paragraph + "\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            end_idx = start_idx + len(current_chunk)
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                content=current_chunk.strip(),
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_recursive(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Create chunks using recursive splitting."""
        separators = self.chunking_config.separators
        
        def split_text(text: str, seps: List[str]) -> List[str]:
            if not seps or len(text) <= self.chunking_config.chunk_size:
                return [text]
            
            separator = seps[0]
            parts = text.split(separator)
            
            result = []
            current = ""
            
            for part in parts:
                if len(current) + len(part) + len(separator) <= self.chunking_config.chunk_size:
                    current += part + separator
                else:
                    if current:
                        result.extend(split_text(current.rstrip(separator), seps[1:]))
                        current = ""
                    
                    if len(part) > self.chunking_config.chunk_size:
                        result.extend(split_text(part, seps[1:]))
                    else:
                        current = part + separator
            
            if current:
                result.extend(split_text(current.rstrip(separator), seps[1:]))
            
            return result
        
        text_chunks = split_text(content, separators)
        
        chunks = []
        start_idx = 0
        
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                end_idx = start_idx + len(chunk_text)
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{i}",
                    doc_id=doc_id,
                    content=chunk_text.strip(),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    chunk_index=i
                )
                chunks.append(chunk)
                start_idx = end_idx
        
        return chunks
    
    def _generate_doc_id(self, content: str, metadata: DocumentMetadata) -> str:
        """Generate document ID."""
        if metadata.title:
            base = metadata.title.lower().replace(' ', '_')
        else:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
            base = f"doc_{content_hash}"
        
        return base


class MarkdownProcessor(TextProcessor):
    """Markdown document processor."""
    
    def process(self, content: str, metadata: DocumentMetadata) -> Document:
        """Process markdown content."""
        # Extract title from markdown if not provided
        if not metadata.title:
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    metadata.title = line[2:].strip()
                    break
        
        # Remove markdown formatting for better text processing
        cleaned_content = self._clean_markdown(content)
        
        # Use parent text processing
        return super().process(cleaned_content, metadata)
    
    def supports(self, content_type: DocumentType) -> bool:
        return content_type == DocumentType.MARKDOWN
    
    def _clean_markdown(self, content: str) -> str:
        """Clean markdown formatting."""
        # Simple markdown cleaning (could be improved with markdown library)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove markdown headers
            if line.startswith('#'):
                line = line.lstrip('#').strip()
            
            # Remove markdown formatting
            line = line.replace('**', '').replace('*', '').replace('`', '')
            
            if line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class JSONProcessor(BaseDocumentProcessor):
    """JSON document processor."""
    
    def process(self, content: str, metadata: DocumentMetadata) -> Document:
        """Process JSON content."""
        import json
        
        try:
            data = json.loads(content)
            # Convert JSON to readable text
            text_content = self._json_to_text(data)
        except json.JSONDecodeError:
            # Fallback to original content
            text_content = content
        
        doc_id = self._generate_doc_id(content, metadata)
        
        return Document(
            doc_id=doc_id,
            content=text_content,
            metadata=metadata
        )
    
    def supports(self, content_type: DocumentType) -> bool:
        return content_type == DocumentType.JSON
    
    def _json_to_text(self, data: Any, prefix: str = "") -> str:
        """Convert JSON data to readable text."""
        text_parts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text_parts.append(f"{prefix}{key}:")
                    text_parts.append(self._json_to_text(value, prefix + "  "))
                else:
                    text_parts.append(f"{prefix}{key}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                text_parts.append(f"{prefix}Item {i + 1}:")
                text_parts.append(self._json_to_text(item, prefix + "  "))
        
        else:
            text_parts.append(f"{prefix}{data}")
        
        return '\n'.join(text_parts)
    
    def _generate_doc_id(self, content: str, metadata: DocumentMetadata) -> str:
        """Generate document ID."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"json_{content_hash}"


class HTMLProcessor(BaseDocumentProcessor):
    """HTML document processor."""
    
    def process(self, content: str, metadata: DocumentMetadata) -> Document:
        """Process HTML content."""
        # Extract text from HTML
        text_content = self._extract_text_from_html(content)
        
        doc_id = self._generate_doc_id(content, metadata)
        
        return Document(
            doc_id=doc_id,
            content=text_content,
            metadata=metadata
        )
    
    def supports(self, content_type: DocumentType) -> bool:
        return content_type == DocumentType.HTML
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content."""
        # Simple HTML tag removal (could be improved with BeautifulSoup)
        import re
        
        # Remove script and style content
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _generate_doc_id(self, content: str, metadata: DocumentMetadata) -> str:
        """Generate document ID."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"html_{content_hash}"
