"""
Document parsing functionality for various file formats.

This module provides comprehensive document parsing capabilities for the thesis
assistant application. It supports multiple file formats including PDF, text,
and markdown files, with robust error handling and metadata extraction.

Key Features:
    - Multi-format document parsing (PDF, TXT, MD)
    - Automatic text encoding detection and fallback
    - Metadata extraction (title, DOI, year, abstract)
    - File integrity verification through hashing
    - Comprehensive error handling with detailed messages

Supported Formats:
    - PDF: Using pdfplumber for accurate text extraction
    - TXT: Plain text files with encoding detection
    - MD: Markdown files treated as plain text

Author: AI-Powered Thesis Assistant Team
Version: 2.0
License: MIT
"""

import os
import hashlib
import re
from pathlib import Path
from typing import Tuple
import logging

# Third-party imports
import pdfplumber

# Local imports
from core.types import DocumentMetadata
from core.exceptions import DocumentProcessingError

# Configure module logger
logger = logging.getLogger(__name__)

# Constants for document processing
CHUNK_SIZE = 4096  # Size for file reading chunks
MAX_TITLE_LENGTH = 200  # Maximum length for extracted titles
MAX_ABSTRACT_LENGTH = 500  # Maximum length for extracted abstracts
SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']  # Encoding fallback order


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file for integrity verification.

    Computes a cryptographic hash of the file content to detect changes
    and ensure file integrity. Uses chunked reading to handle large files
    efficiently without loading them entirely into memory.

    Args:
        file_path: Absolute or relative path to the file

    Returns:
        Hexadecimal string representation of the SHA-256 hash

    Raises:
        DocumentProcessingError: If file cannot be read or accessed

    Example:
        >>> hash_value = calculate_file_hash("document.pdf")
        >>> print(f"File hash: {hash_value}")
    """
    logger.debug(f"Calculating hash for file: {file_path}")

    hash_sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hash_sha256.update(chunk)

        hash_value = hash_sha256.hexdigest()
        logger.debug(f"File hash calculated: {hash_value[:16]}...")
        return hash_value

    except IOError as e:
        error_msg = f"Could not read file {file_path}: {e}"
        logger.error(error_msg)
        raise DocumentProcessingError(error_msg) from e


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from PDF files using pdfplumber.

    Processes PDF files page by page to extract text content while
    preserving document structure. Handles various PDF formats including
    scanned documents (though OCR is not performed).

    Args:
        file_path: Path to the PDF file to process

    Returns:
        Extracted text content with pages separated by newlines

    Raises:
        DocumentProcessingError: If PDF processing fails or no text is found

    Note:
        This function does not perform OCR on scanned PDFs. For scanned
        documents, consider using an OCR library like pytesseract.

    Example:
        >>> text = extract_text_from_pdf("research_paper.pdf")
        >>> print(f"Extracted {len(text)} characters")
    """
    logger.info(f"Extracting text from PDF: {file_path}")

    try:
        text_content = []
        page_count = 0

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.debug(f"PDF has {total_pages} pages")

            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text from current page
                    text = page.extract_text()
                    if text and text.strip():  # Only add non-empty pages
                        text_content.append(text.strip())
                        page_count += 1

                except Exception as page_error:
                    # Log page-specific errors but continue processing
                    logger.warning(f"Failed to extract text from page {page_num}: {page_error}")
                    continue

        # Validate that we extracted some content
        if not text_content:
            raise DocumentProcessingError(
                f"No text could be extracted from {file_path}. "
                "The PDF may be scanned or corrupted."
            )

        extracted_text = "\n\n".join(text_content)
        logger.info(f"Successfully extracted text from {page_count}/{total_pages} pages")
        return extracted_text

    except DocumentProcessingError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        error_msg = f"Failed to process PDF {file_path}: {e}"
        logger.error(error_msg)
        raise DocumentProcessingError(error_msg) from e


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text content from plain text and markdown files.

    Attempts to read text files using multiple encoding strategies to handle
    various character sets and file origins. Provides robust fallback mechanisms
    for files with different encodings.

    Args:
        file_path: Path to the text or markdown file

    Returns:
        Complete file content as a string

    Raises:
        DocumentProcessingError: If file cannot be read with any supported encoding

    Note:
        Supports UTF-8, Latin-1, CP1252, and ISO-8859-1 encodings with
        automatic fallback. This covers most common text file encodings.

    Example:
        >>> content = extract_text_from_txt("notes.md")
        >>> print(f"File contains {len(content)} characters")
    """
    logger.info(f"Extracting text from file: {file_path}")

    # Try multiple encodings in order of preference
    for encoding in SUPPORTED_ENCODINGS:
        try:
            logger.debug(f"Attempting to read with {encoding} encoding")
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Validate that we got meaningful content
            if not content.strip():
                logger.warning(f"File {file_path} appears to be empty")
            else:
                logger.info(f"Successfully read file with {encoding} encoding")

            return content

        except UnicodeDecodeError as e:
            logger.debug(f"Failed to read with {encoding}: {e}")
            continue  # Try next encoding

        except Exception as e:
            # Non-encoding related error, don't try other encodings
            error_msg = f"Failed to read text file {file_path}: {e}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    # If we get here, all encodings failed
    error_msg = (
        f"Failed to read text file {file_path} with any supported encoding "
        f"({', '.join(SUPPORTED_ENCODINGS)}). File may be corrupted or in an unsupported format."
    )
    logger.error(error_msg)
    raise DocumentProcessingError(error_msg)


def extract_metadata_from_text(text: str, file_path: str) -> DocumentMetadata:
    """
    Extract comprehensive metadata from document text content.

    Analyzes the document text to automatically extract key metadata fields
    including title, DOI, publication year, and abstract. Uses pattern matching
    and heuristics to identify relevant information.

    Args:
        text: Complete document text content
        file_path: Path to the source file for hash calculation

    Returns:
        DocumentMetadata object populated with extracted information

    Note:
        Extraction is best-effort and may not find all metadata in every document.
        The quality of extraction depends on document formatting and structure.

    Example:
        >>> with open("paper.txt") as f:
        ...     text = f.read()
        >>> metadata = extract_metadata_from_text(text, "paper.txt")
        >>> print(f"Title: {metadata.title}")
    """
    logger.debug(f"Extracting metadata from text ({len(text)} characters)")

    # Initialize metadata object with basic file information
    metadata = DocumentMetadata()
    metadata.file_path = file_path
    metadata.file_hash = calculate_file_hash(file_path)

    # Extract title from the first meaningful line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        # Use the first non-trivial line as title, with length limit
        potential_title = lines[0][:MAX_TITLE_LENGTH]

        # Clean up common title artifacts
        potential_title = re.sub(r'^(title|abstract|introduction):\s*', '', potential_title, flags=re.IGNORECASE)
        metadata.title = potential_title.strip()
        logger.debug(f"Extracted title: {metadata.title[:50]}...")

    # Extract DOI using standard DOI pattern
    doi_pattern = r'10\.\d{4,}/[^\s\]\)>]+'
    doi_match = re.search(doi_pattern, text, re.IGNORECASE)
    if doi_match:
        metadata.doi = doi_match.group().strip()
        logger.debug(f"Found DOI: {metadata.doi}")

    # Extract publication year (4-digit years between 1900-2030)
    current_year = 2024  # Update this periodically
    year_pattern = r'\b(19[0-9]{2}|20[0-3][0-9])\b'
    year_matches = re.findall(year_pattern, text)

    if year_matches:
        # Convert to integers and find the most reasonable year
        years = [int(year) for year in year_matches]
        # Prefer years closer to current time for academic papers
        reasonable_years = [y for y in years if 1950 <= y <= current_year + 1]

        if reasonable_years:
            metadata.year = max(reasonable_years)  # Most recent reasonable year
            logger.debug(f"Extracted year: {metadata.year}")

    # Extract abstract using multiple patterns
    abstract_patterns = [
        r'(?i)abstract[:\s]*\n?\s*([^.]*(?:\.[^.]*){0,8}\.)',  # Standard abstract
        r'(?i)summary[:\s]*\n?\s*([^.]*(?:\.[^.]*){0,5}\.)',   # Summary section
        r'(?i)overview[:\s]*\n?\s*([^.]*(?:\.[^.]*){0,5}\.)'   # Overview section
    ]

    for pattern in abstract_patterns:
        abstract_match = re.search(pattern, text)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            # Clean up the abstract text
            abstract_text = re.sub(r'\s+', ' ', abstract_text)  # Normalize whitespace
            metadata.abstract = abstract_text[:MAX_ABSTRACT_LENGTH]
            logger.debug(f"Extracted abstract: {metadata.abstract[:100]}...")
            break

    # Extract potential keywords (words that appear frequently and are capitalized)
    keyword_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    potential_keywords = re.findall(keyword_pattern, text)

    # Filter and deduplicate keywords
    if potential_keywords:
        # Count frequency and keep most common ones
        from collections import Counter
        keyword_counts = Counter(potential_keywords)
        # Keep keywords that appear at least twice and are not too common
        filtered_keywords = [
            word for word, count in keyword_counts.items()
            if 2 <= count <= 10 and len(word) > 3
        ]
        metadata.keywords = filtered_keywords[:10]  # Limit to top 10
        logger.debug(f"Extracted {len(metadata.keywords)} keywords")

    logger.info(f"Metadata extraction complete for {file_path}")
    return metadata


def parse_document(file_path: str) -> Tuple[str, DocumentMetadata]:
    """
    Parse a document and extract both text content and metadata.

    This is the main entry point for document processing. It automatically
    detects the file format and applies the appropriate extraction method,
    then extracts metadata from the resulting text.

    Args:
        file_path: Path to the document file to process

    Returns:
        Tuple containing:
            - str: Extracted text content from the document
            - DocumentMetadata: Metadata object with extracted information

    Raises:
        DocumentProcessingError: If file doesn't exist, format is unsupported,
                               or processing fails for any reason

    Supported Formats:
        - .pdf: PDF documents (using pdfplumber)
        - .txt: Plain text files
        - .md: Markdown files (treated as plain text)

    Example:
        >>> text, metadata = parse_document("research_paper.pdf")
        >>> print(f"Title: {metadata.title}")
        >>> print(f"Content length: {len(text)} characters")
    """
    logger.info(f"Starting document parsing: {file_path}")

    # Validate file existence
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise DocumentProcessingError(error_msg)

    # Get file information
    file_path_obj = Path(file_path)
    file_ext = file_path_obj.suffix.lower()
    file_size = file_path_obj.stat().st_size

    logger.debug(f"Processing file: {file_path_obj.name} ({file_size:,} bytes, {file_ext} format)")

    # Validate file size (prevent processing extremely large files)
    max_file_size = 100 * 1024 * 1024  # 100 MB limit
    if file_size > max_file_size:
        error_msg = f"File too large: {file_size:,} bytes (max: {max_file_size:,} bytes)"
        logger.error(error_msg)
        raise DocumentProcessingError(error_msg)

    # Extract text based on file format
    try:
        if file_ext == '.pdf':
            logger.debug("Processing as PDF document")
            text = extract_text_from_pdf(file_path)
        elif file_ext in ['.txt', '.md']:
            logger.debug(f"Processing as text document ({file_ext})")
            text = extract_text_from_txt(file_path)
        else:
            supported_formats = ['.pdf', '.txt', '.md']
            error_msg = (
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)

        # Validate extracted text
        if not text or not text.strip():
            error_msg = f"No text content extracted from {file_path}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)

        logger.info(f"Successfully extracted {len(text):,} characters from {file_path}")

        # Extract metadata from the text
        metadata = extract_metadata_from_text(text, file_path)

        # Add file size to metadata
        metadata.chunk_count = 0  # Will be set later during chunking

        logger.info(f"Document parsing completed successfully: {file_path}")
        return text, metadata

    except DocumentProcessingError:
        # Re-raise our custom exceptions without modification
        raise
    except Exception as e:
        # Wrap unexpected exceptions
        error_msg = f"Unexpected error processing document {file_path}: {e}"
        logger.error(error_msg, exc_info=True)
        raise DocumentProcessingError(error_msg) from e


# Additional utility functions for document validation and analysis

def validate_document_content(text: str, min_length: int = 100) -> bool:
    """
    Validate that document content meets minimum quality requirements.

    Args:
        text: Document text content to validate
        min_length: Minimum required text length

    Returns:
        True if content is valid, False otherwise
    """
    if not text or len(text.strip()) < min_length:
        return False

    # Check for reasonable text-to-whitespace ratio
    non_whitespace = len(re.sub(r'\s', '', text))
    if non_whitespace / len(text) < 0.1:  # Less than 10% actual content
        return False

    return True


def get_supported_formats() -> list[str]:
    """
    Get list of supported document formats.

    Returns:
        List of supported file extensions (including the dot)
    """
    return ['.pdf', '.txt', '.md']

