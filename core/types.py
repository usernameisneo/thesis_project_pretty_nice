"""
Custom data types for the thesis assistant.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class ChapterType(Enum):
    """Standard thesis chapter types."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    CUSTOM = "custom"


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    doi: str = ""
    year: Optional[int] = None
    journal: str = ""
    keywords: List[str] = field(default_factory=list)
    citation_count: int = 0
    file_path: str = ""
    file_hash: str = ""
    processed_date: datetime = field(default_factory=datetime.now)
    chunk_count: int = 0


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    chunk_id: str
    document_id: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result with relevance score."""
    chunk: TextChunk
    score: float
    rank: int
    search_type: str  # "vector", "keyword", or "hybrid"


@dataclass
class WritingSession:
    """Track writing productivity."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    words_written: int = 0
    chapter_id: Optional[str] = None
    notes: str = ""


@dataclass
class WritingPrompt:
    """AI prompt template for writing assistance."""
    prompt_id: str
    name: str
    description: str
    template: str
    category: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CitationEntry:
    """Citation entry in APA 7 format."""
    citation_id: str
    authors: List[str]
    title: str
    year: int
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    access_date: Optional[datetime] = None
    citation_type: str = "journal"  # journal, book, website, etc.
    
    def to_apa7(self) -> str:
        """Format citation in APA 7 style."""
        if not self.authors:
            author_str = "Unknown Author"
        elif len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) == 2:
            author_str = f"{self.authors[0]} & {self.authors[1]}"
        else:
            author_str = f"{self.authors[0]} et al."
        
        citation = f"{author_str} ({self.year}). {self.title}."
        
        if self.journal:
            citation += f" {self.journal}"
            if self.volume:
                citation += f", {self.volume}"
                if self.issue:
                    citation += f"({self.issue})"
            if self.pages:
                citation += f", {self.pages}"
        
        if self.doi:
            citation += f" https://doi.org/{self.doi}"
        elif self.url:
            citation += f" {self.url}"
        
        return citation

