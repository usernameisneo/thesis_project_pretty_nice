"""
Text processing and chunking functionality.
"""
import re
import uuid
from typing import List, Optional
from core.types import TextChunk, DocumentMetadata
from core.exceptions import DocumentProcessingError


def clean_text(text: str) -> str:
    """Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers (simple heuristic)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip very short lines that might be page numbers
        if len(line) < 3:
            continue
        # Skip lines that are just numbers
        if line.isdigit():
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def chunk_text_by_paragraph(text: str, document_id: str, 
                          max_chunk_size: int = 512, 
                          overlap: int = 50) -> List[TextChunk]:
    """Split text into chunks by paragraphs with size limits.
    
    Args:
        text: Text to chunk
        document_id: Unique identifier for the document
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    if not text.strip():
        return []
    
    # Clean the text first
    text = clean_text(text)
    
    # Split by paragraphs (double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    current_start = 0
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size, finalize current chunk
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            # Create chunk
            chunk_id = str(uuid.uuid4())
            chunk = TextChunk(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                document_id=document_id,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            )
            chunks.append(chunk)
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + paragraph
                current_start = current_start + len(current_chunk) - len(overlap_text) - 2
            else:
                current_chunk = paragraph
                current_start = current_start + len(current_chunk)
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk if there's content
    if current_chunk.strip():
        chunk_id = str(uuid.uuid4())
        chunk = TextChunk(
            text=current_chunk.strip(),
            chunk_id=chunk_id,
            document_id=document_id,
            start_pos=current_start,
            end_pos=current_start + len(current_chunk)
        )
        chunks.append(chunk)
    
    return chunks


def chunk_text_by_sentences(text: str, document_id: str,
                          max_chunk_size: int = 512,
                          overlap: int = 50) -> List[TextChunk]:
    """Split text into chunks by sentences with size limits.
    
    Args:
        text: Text to chunk
        document_id: Unique identifier for the document
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    if not text.strip():
        return []
    
    # Clean the text first
    text = clean_text(text)
    
    # Simple sentence splitting (can be improved with spaCy if needed)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    current_start = 0
    
    for sentence in sentences:
        # If adding this sentence would exceed max size, finalize current chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            # Create chunk
            chunk_id = str(uuid.uuid4())
            chunk = TextChunk(
                text=current_chunk.strip(),
                chunk_id=chunk_id,
                document_id=document_id,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            )
            chunks.append(chunk)
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) - 1
            else:
                current_chunk = sentence
                current_start = current_start + len(current_chunk)
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add final chunk if there's content
    if current_chunk.strip():
        chunk_id = str(uuid.uuid4())
        chunk = TextChunk(
            text=current_chunk.strip(),
            chunk_id=chunk_id,
            document_id=document_id,
            start_pos=current_start,
            end_pos=current_start + len(current_chunk)
        )
        chunks.append(chunk)
    
    return chunks


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction - can be improved with NLP libraries
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
        'did', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 'have',
        'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
        'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many',
        'over', 'such', 'take', 'than', 'them', 'well', 'were'
    }
    
    # Filter and count words
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


class TextProcessor:
    """
    Text processing class for document chunking and analysis.

    This class provides methods for processing text documents into
    searchable chunks with various chunking strategies.
    """

    def __init__(self,
                 default_chunk_size: int = 512,
                 default_overlap: int = 50,
                 chunking_method: str = "paragraph"):
        """
        Initialize text processor.

        Args:
            default_chunk_size: Default maximum chunk size
            default_overlap: Default overlap between chunks
            chunking_method: Default chunking method ("paragraph" or "sentence")
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.chunking_method = chunking_method

    def process_text(self,
                    text: str,
                    document_id: str,
                    chunk_size: Optional[int] = None,
                    overlap: Optional[int] = None,
                    method: Optional[str] = None) -> List[TextChunk]:
        """
        Process text into chunks using specified method.

        Args:
            text: Text to process
            document_id: Document identifier
            chunk_size: Chunk size (uses default if None)
            overlap: Overlap size (uses default if None)
            method: Chunking method (uses default if None)

        Returns:
            List of TextChunk objects
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        method = method or self.chunking_method

        if method == "paragraph":
            return chunk_text_by_paragraph(text, document_id, chunk_size, overlap)
        elif method == "sentence":
            return chunk_text_by_sentences(text, document_id, chunk_size, overlap)
        else:
            raise DocumentProcessingError(f"Unknown chunking method: {method}")

    def clean_text(self, text: str) -> str:
        """Clean text content."""
        return clean_text(text)

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        return extract_keywords(text, max_keywords)

