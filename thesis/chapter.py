"""
Thesis Chapter Management for the AI-Powered Thesis Assistant.

This module implements comprehensive chapter organization and tracking
with version control, progress monitoring, and AI-powered assistance.

Features:
    - Chapter structure management
    - Section and subsection organization
    - Progress tracking and analytics
    - Version control integration
    - AI-powered writing assistance
    - Citation tracking per chapter

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ChapterStatus(Enum):
    """Chapter status enumeration."""
    PLANNED = "planned"
    OUTLINE = "outline"
    DRAFT = "draft"
    REVIEW = "review"
    REVISION = "revision"
    COMPLETED = "completed"


class SectionType(Enum):
    """Section type enumeration."""
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    CUSTOM = "custom"


@dataclass
class ChapterSection:
    """Chapter section structure."""
    title: str
    content: str = ""
    section_type: SectionType = SectionType.CUSTOM
    word_count: int = 0
    page_count: int = 0
    citations_count: int = 0
    subsections: List['ChapterSection'] = field(default_factory=list)
    notes: str = ""
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class ChapterMetadata:
    """Chapter metadata structure."""
    title: str
    number: int
    description: str
    status: ChapterStatus
    target_word_count: int
    target_page_count: int
    deadline: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)


@dataclass
class ChapterStatistics:
    """Chapter statistics tracking."""
    word_count: int = 0
    page_count: int = 0
    paragraph_count: int = 0
    sentence_count: int = 0
    citations_count: int = 0
    figures_count: int = 0
    tables_count: int = 0
    writing_time: float = 0.0  # hours
    revision_count: int = 0
    completion_percentage: float = 0.0


class ThesisChapter:
    """
    Comprehensive thesis chapter management system.
    
    This class provides complete chapter lifecycle management including
    content organization, progress tracking, and AI-powered assistance.
    """
    
    def __init__(self, chapter_path: str, project_path: Optional[str] = None):
        """
        Initialize a thesis chapter.
        
        Args:
            chapter_path: Path to the chapter file
            project_path: Path to the parent project
        """
        self.chapter_path = Path(chapter_path)
        self.project_path = Path(project_path) if project_path else None
        self.metadata: Optional[ChapterMetadata] = None
        self.statistics = ChapterStatistics()
        self.sections: List[ChapterSection] = []
        
        # Chapter files
        self.content_file = self.chapter_path
        self.metadata_file = self.chapter_path.with_suffix('.meta.json')
        self.stats_file = self.chapter_path.with_suffix('.stats.json')
        
        # Load existing chapter or initialize new one
        if self.metadata_file.exists():
            self._load_chapter()
        
        logger.info(f"Thesis chapter initialized: {self.chapter_path}")
    
    def create_chapter(self, metadata: ChapterMetadata) -> None:
        """
        Create a new thesis chapter.
        
        Args:
            metadata: Chapter metadata
        """
        try:
            # Ensure directory exists
            self.chapter_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set metadata
            self.metadata = metadata
            
            # Create initial content
            self._create_initial_content()
            
            # Save chapter configuration
            self._save_chapter()
            
            # Initialize statistics
            self.statistics = ChapterStatistics()
            self._save_statistics()
            
            logger.info(f"New thesis chapter created: {metadata.title}")
            
        except Exception as e:
            logger.error(f"Failed to create chapter: {e}")
            raise Exception(f"Chapter creation failed: {e}")
    
    def _create_initial_content(self) -> None:
        """Create initial chapter content."""
        if not self.metadata:
            return
        
        initial_content = f"""# Chapter {self.metadata.number}: {self.metadata.title}

## Overview
{self.metadata.description}

## Introduction
[Write your introduction here]

## Main Content
[Write your main content here]

## Conclusion
[Write your conclusion here]

---
Created: {self.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}
Target Word Count: {self.metadata.target_word_count}
Target Page Count: {self.metadata.target_page_count}
"""
        
        self.chapter_path.write_text(initial_content, encoding='utf-8')
    
    def _load_chapter(self) -> None:
        """Load existing chapter configuration."""
        try:
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert datetime strings
                for date_field in ['deadline', 'created_at', 'last_modified']:
                    if date_field in data and data[date_field]:
                        data[date_field] = datetime.fromisoformat(data[date_field])
                
                # Convert status string to enum
                if 'status' in data:
                    data['status'] = ChapterStatus(data['status'])
                
                self.metadata = ChapterMetadata(**data)
            
            # Load statistics
            if self.stats_file.exists():
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                
                self.statistics = ChapterStatistics(**stats_data)
            
            # Load sections (if any)
            self._load_sections()
            
            logger.info("Chapter configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load chapter: {e}")
            raise Exception(f"Chapter loading failed: {e}")
    
    def _load_sections(self) -> None:
        """Load chapter sections from content."""
        if not self.chapter_path.exists():
            return
        
        try:
            content = self.chapter_path.read_text(encoding='utf-8')
            
            # Parse sections from markdown headers
            import re
            
            # Find all headers (## Section Title)
            header_pattern = r'^##\s+(.+)$'
            headers = re.findall(header_pattern, content, re.MULTILINE)
            
            # Create sections
            self.sections = []
            for header in headers:
                section = ChapterSection(
                    title=header.strip(),
                    section_type=self._detect_section_type(header),
                    last_modified=datetime.now()
                )
                self.sections.append(section)
            
        except Exception as e:
            logger.error(f"Failed to load sections: {e}")
    
    def _detect_section_type(self, title: str) -> SectionType:
        """Detect section type from title."""
        title_lower = title.lower()
        
        if 'introduction' in title_lower:
            return SectionType.INTRODUCTION
        elif 'literature' in title_lower or 'review' in title_lower:
            return SectionType.LITERATURE_REVIEW
        elif 'method' in title_lower:
            return SectionType.METHODOLOGY
        elif 'result' in title_lower:
            return SectionType.RESULTS
        elif 'discussion' in title_lower:
            return SectionType.DISCUSSION
        elif 'conclusion' in title_lower:
            return SectionType.CONCLUSION
        elif 'appendix' in title_lower:
            return SectionType.APPENDIX
        else:
            return SectionType.CUSTOM
    
    def _save_chapter(self) -> None:
        """Save chapter configuration."""
        try:
            if not self.metadata:
                return
            
            data = {
                'title': self.metadata.title,
                'number': self.metadata.number,
                'description': self.metadata.description,
                'status': self.metadata.status.value,
                'target_word_count': self.metadata.target_word_count,
                'target_page_count': self.metadata.target_page_count,
                'deadline': self.metadata.deadline.isoformat() if self.metadata.deadline else None,
                'tags': self.metadata.tags,
                'keywords': self.metadata.keywords,
                'created_at': self.metadata.created_at.isoformat(),
                'last_modified': datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Chapter configuration saved")
            
        except Exception as e:
            logger.error(f"Failed to save chapter: {e}")
    
    def _save_statistics(self) -> None:
        """Save chapter statistics."""
        try:
            data = {
                'word_count': self.statistics.word_count,
                'page_count': self.statistics.page_count,
                'paragraph_count': self.statistics.paragraph_count,
                'sentence_count': self.statistics.sentence_count,
                'citations_count': self.statistics.citations_count,
                'figures_count': self.statistics.figures_count,
                'tables_count': self.statistics.tables_count,
                'writing_time': self.statistics.writing_time,
                'revision_count': self.statistics.revision_count,
                'completion_percentage': self.statistics.completion_percentage
            }
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Chapter statistics saved")
            
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
    
    def update_content(self, content: str) -> None:
        """
        Update chapter content.
        
        Args:
            content: New chapter content
        """
        try:
            # Save content
            self.chapter_path.write_text(content, encoding='utf-8')
            
            # Update statistics
            self._update_statistics_from_content(content)
            
            # Update metadata
            if self.metadata:
                self.metadata.last_modified = datetime.now()
                self._save_chapter()
            
            logger.info(f"Chapter content updated: {self.chapter_path}")
            
        except Exception as e:
            logger.error(f"Failed to update content: {e}")
            raise Exception(f"Content update failed: {e}")
    
    def _update_statistics_from_content(self, content: str) -> None:
        """Update statistics based on content analysis."""
        # Count words
        words = content.split()
        self.statistics.word_count = len(words)
        
        # Count paragraphs (double newlines)
        paragraphs = content.split('\n\n')
        self.statistics.paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Count sentences (rough estimate)
        import re
        sentences = re.split(r'[.!?]+', content)
        self.statistics.sentence_count = len([s for s in sentences if s.strip()])
        
        # Estimate page count (250 words per page)
        self.statistics.page_count = max(1, self.statistics.word_count // 250)
        
        # Count citations (rough estimate)
        citation_patterns = [r'\([^)]*\d{4}[^)]*\)', r'\[[^\]]*\d{4}[^\]]*\]']
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content))
        self.statistics.citations_count = citation_count
        
        # Count figures and tables
        self.statistics.figures_count = len(re.findall(r'!\[.*?\]\(.*?\)', content))
        self.statistics.tables_count = len(re.findall(r'\|.*?\|', content))
        
        # Calculate completion percentage
        if self.metadata and self.metadata.target_word_count > 0:
            self.statistics.completion_percentage = min(100.0, 
                (self.statistics.word_count / self.metadata.target_word_count) * 100)
        
        self._save_statistics()
    
    def add_section(self, section: ChapterSection) -> None:
        """
        Add a new section to the chapter.
        
        Args:
            section: Section to add
        """
        self.sections.append(section)
        logger.info(f"Section added: {section.title}")
    
    def get_chapter_info(self) -> Dict[str, Any]:
        """
        Get comprehensive chapter information.
        
        Returns:
            Chapter information dictionary
        """
        if not self.metadata:
            return {}
        
        return {
            'metadata': {
                'title': self.metadata.title,
                'number': self.metadata.number,
                'description': self.metadata.description,
                'status': self.metadata.status.value,
                'target_word_count': self.metadata.target_word_count,
                'target_page_count': self.metadata.target_page_count,
                'deadline': self.metadata.deadline.isoformat() if self.metadata.deadline else None,
                'tags': self.metadata.tags,
                'keywords': self.metadata.keywords,
                'created_at': self.metadata.created_at.isoformat(),
                'last_modified': self.metadata.last_modified.isoformat()
            },
            'statistics': {
                'word_count': self.statistics.word_count,
                'page_count': self.statistics.page_count,
                'paragraph_count': self.statistics.paragraph_count,
                'sentence_count': self.statistics.sentence_count,
                'citations_count': self.statistics.citations_count,
                'figures_count': self.statistics.figures_count,
                'tables_count': self.statistics.tables_count,
                'writing_time': self.statistics.writing_time,
                'revision_count': self.statistics.revision_count,
                'completion_percentage': self.statistics.completion_percentage
            },
            'sections': [
                {
                    'title': section.title,
                    'type': section.section_type.value,
                    'word_count': section.word_count,
                    'citations_count': section.citations_count
                }
                for section in self.sections
            ],
            'paths': {
                'content_file': str(self.content_file),
                'metadata_file': str(self.metadata_file),
                'stats_file': str(self.stats_file)
            }
        }
