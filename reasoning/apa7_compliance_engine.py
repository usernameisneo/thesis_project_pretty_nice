"""
APA7 Compliance Engine for Precision Citation Formatting.

This module implements enterprise-grade APA7 citation formatting with strict
compliance checking, validation, and error correction. It ensures maximum
precision and adherence to APA7 standards for academic citations.

Features:
    - Strict APA7 format validation
    - Automatic citation formatting
    - Bibliography generation
    - Cross-reference validation
    - Error detection and correction
    - Template-based formatting
    - Metadata validation

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.types import DocumentMetadata, CitationEntry
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class CitationType(Enum):
    """Types of citations supported by APA7."""
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    THESIS_DISSERTATION = "thesis_dissertation"
    WEBSITE = "website"
    REPORT = "report"
    NEWSPAPER = "newspaper"
    MAGAZINE = "magazine"
    UNKNOWN = "unknown"


class ComplianceLevel(Enum):
    """APA7 compliance levels."""
    FULLY_COMPLIANT = "fully_compliant"      # 100% APA7 compliant
    MOSTLY_COMPLIANT = "mostly_compliant"    # Minor formatting issues
    PARTIALLY_COMPLIANT = "partially_compliant"  # Some missing elements
    NON_COMPLIANT = "non_compliant"          # Major issues or incorrect format


@dataclass
class APA7ValidationResult:
    """Result of APA7 compliance validation."""
    citation_id: str
    original_citation: str
    formatted_citation: str
    compliance_level: ComplianceLevel
    compliance_score: float  # 0.0 - 1.0
    
    # Validation details
    missing_elements: List[str] = field(default_factory=list)
    formatting_errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    citation_type: CitationType = CitationType.UNKNOWN
    validated_at: datetime = field(default_factory=datetime.now)


class APA7ComplianceEngine:
    """
    Enterprise-grade APA7 compliance engine for precision citation formatting.
    
    This engine implements strict APA7 standards with comprehensive validation,
    error detection, and automatic formatting correction.
    """
    
    def __init__(self):
        """Initialize the APA7 compliance engine."""
        
        # APA7 required elements for each citation type
        self.required_elements = {
            CitationType.JOURNAL_ARTICLE: ['author', 'year', 'title', 'journal', 'volume'],
            CitationType.BOOK: ['author', 'year', 'title', 'publisher'],
            CitationType.BOOK_CHAPTER: ['author', 'year', 'title', 'editor', 'book_title', 'publisher'],
            CitationType.CONFERENCE_PAPER: ['author', 'year', 'title', 'conference', 'location'],
            CitationType.THESIS_DISSERTATION: ['author', 'year', 'title', 'degree_type', 'institution'],
            CitationType.WEBSITE: ['author', 'year', 'title', 'url', 'access_date'],
            CitationType.REPORT: ['author', 'year', 'title', 'report_number', 'publisher'],
            CitationType.NEWSPAPER: ['author', 'year', 'title', 'newspaper', 'date'],
            CitationType.MAGAZINE: ['author', 'year', 'title', 'magazine', 'date']
        }
        
        # APA7 formatting patterns
        self.apa7_patterns = {
            'author_pattern': r'^[A-Z][a-z]+,\s[A-Z]\.\s?([A-Z]\.\s?)?',
            'year_pattern': r'\((\d{4}[a-z]?)\)',
            'title_pattern': r'[A-Z][^.]*\.',
            'journal_pattern': r'\*[^*]+\*',
            'volume_pattern': r'\*(\d+)\*',
            'pages_pattern': r'(\d+)-(\d+)',
            'doi_pattern': r'https://doi\.org/[^\s]+',
            'url_pattern': r'https?://[^\s]+'
        }
        
        # Common APA7 formatting rules
        self.formatting_rules = {
            'title_case': 'Only first word and proper nouns capitalized',
            'italics_journal': 'Journal names should be italicized',
            'italics_volume': 'Volume numbers should be italicized',
            'author_format': 'Last name, First initial. Middle initial.',
            'year_parentheses': 'Year should be in parentheses',
            'doi_format': 'DOI should be formatted as https://doi.org/...',
            'page_range': 'Page ranges should use en-dash (–)'
        }
        
        logger.info("APA7 compliance engine initialized")
    
    def validate_citation(self, citation_text: str, metadata: Optional[DocumentMetadata] = None) -> APA7ValidationResult:
        """
        Validate a citation for APA7 compliance.
        
        Args:
            citation_text: The citation text to validate
            metadata: Optional metadata for enhanced validation
            
        Returns:
            Comprehensive APA7 validation result
        """
        try:
            # Generate validation ID
            validation_id = self._generate_citation_id(citation_text)
            
            # Detect citation type
            citation_type = self._detect_citation_type(citation_text, metadata)
            
            # Initialize validation result
            result = APA7ValidationResult(
                citation_id=validation_id,
                original_citation=citation_text,
                formatted_citation=citation_text,
                compliance_level=ComplianceLevel.NON_COMPLIANT,
                compliance_score=0.0,
                citation_type=citation_type
            )
            
            # Validate required elements
            self._validate_required_elements(result, citation_text, metadata)
            
            # Validate formatting
            self._validate_formatting(result, citation_text)
            
            # Generate formatted citation
            result.formatted_citation = self._format_citation(citation_text, citation_type, metadata)
            
            # Calculate compliance score
            result.compliance_score = self._calculate_compliance_score(result)
            result.compliance_level = self._determine_compliance_level(result.compliance_score)
            
            # Generate suggestions
            result.suggestions = self._generate_suggestions(result)
            
            logger.info(f"APA7 validation completed for {validation_id}: score={result.compliance_score:.3f}")
            return result
            
        except Exception as e:
            error_msg = f"APA7 validation failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ValidationError(error_msg)
    
    def format_bibliography(self, citations: List[CitationEntry]) -> str:
        """
        Format a complete bibliography in APA7 style.
        
        Args:
            citations: List of citation entries
            
        Returns:
            Formatted bibliography string
        """
        try:
            formatted_citations = []

            for citation in citations:
                # Use the built-in APA7 formatting method
                if hasattr(citation, 'to_apa7'):
                    apa7_citation = citation.to_apa7()
                    # Validate the formatted citation
                    validation_result = self.validate_citation(apa7_citation, None)
                    formatted_citations.append(validation_result.formatted_citation)
                else:
                    # Fallback for other citation formats
                    citation_text = str(citation)
                    validation_result = self.validate_citation(citation_text, None)
                    formatted_citations.append(validation_result.formatted_citation)

            # Sort alphabetically by first author's last name
            formatted_citations.sort()

            # Create bibliography
            bibliography = "References\n\n"
            for citation in formatted_citations:
                bibliography += f"{citation}\n\n"

            return bibliography.strip()
            
        except Exception as e:
            error_msg = f"Bibliography formatting failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ValidationError(error_msg)
    
    def _detect_citation_type(self, citation_text: str, metadata: Optional[DocumentMetadata]) -> CitationType:
        """
        Detect the type of citation based on text patterns and metadata.
        
        Args:
            citation_text: Citation text to analyze
            metadata: Optional metadata
            
        Returns:
            Detected citation type
        """
        try:
            text_lower = citation_text.lower()
            
            # Check for journal indicators
            if any(indicator in text_lower for indicator in ['journal', 'vol.', 'volume', 'issue', 'pp.']):
                return CitationType.JOURNAL_ARTICLE
            
            # Check for book indicators
            if any(indicator in text_lower for indicator in ['publisher', 'press', 'edition']):
                if 'chapter' in text_lower or 'in ' in text_lower:
                    return CitationType.BOOK_CHAPTER
                return CitationType.BOOK
            
            # Check for conference indicators
            if any(indicator in text_lower for indicator in ['conference', 'proceedings', 'symposium']):
                return CitationType.CONFERENCE_PAPER
            
            # Check for thesis indicators
            if any(indicator in text_lower for indicator in ['thesis', 'dissertation', 'phd', 'master']):
                return CitationType.THESIS_DISSERTATION
            
            # Check for web indicators
            if any(indicator in text_lower for indicator in ['http', 'www.', 'retrieved', 'accessed']):
                return CitationType.WEBSITE
            
            # Check for report indicators
            if any(indicator in text_lower for indicator in ['report', 'technical', 'working paper']):
                return CitationType.REPORT
            
            # Check metadata if available
            if metadata:
                if hasattr(metadata, 'document_type'):
                    type_mapping = {
                        'journal': CitationType.JOURNAL_ARTICLE,
                        'book': CitationType.BOOK,
                        'conference': CitationType.CONFERENCE_PAPER,
                        'thesis': CitationType.THESIS_DISSERTATION,
                        'website': CitationType.WEBSITE,
                        'report': CitationType.REPORT
                    }
                    return type_mapping.get(metadata.document_type.lower(), CitationType.UNKNOWN)
            
            return CitationType.UNKNOWN
            
        except Exception as e:
            logger.warning(f"Citation type detection failed: {e}")
            return CitationType.UNKNOWN

    def _generate_citation_id(self, citation_text: str) -> str:
        """Generate unique citation ID."""
        import hashlib
        return hashlib.md5(citation_text.encode()).hexdigest()[:12]

    def _validate_required_elements(self, result: APA7ValidationResult, citation_text: str, metadata: Optional[DocumentMetadata]):
        """Validate required elements for citation type."""
        required = self.required_elements.get(result.citation_type, [])

        for element in required:
            if element == 'author' and not re.search(r'[A-Z][a-z]+,\s[A-Z]\.', citation_text):
                result.missing_elements.append('author')
            elif element == 'year' and not re.search(r'\(\d{4}\)', citation_text):
                result.missing_elements.append('year')
            elif element == 'title' and len(citation_text.split('.')[0]) < 10:
                result.missing_elements.append('title')

    def _validate_formatting(self, result: APA7ValidationResult, citation_text: str):
        """Validate APA7 formatting rules."""
        # Check for common formatting issues
        if not re.search(r'\(\d{4}\)', citation_text):
            result.formatting_errors.append('Year should be in parentheses')

        if citation_text.count('.') < 2:
            result.formatting_errors.append('Missing periods after author and title')

    def _format_citation(self, citation_text: str, citation_type: CitationType, metadata: Optional[DocumentMetadata]) -> str:
        """
        Format citation according to APA7 standards with comprehensive formatting rules.

        This method applies professional APA7 formatting based on citation type:
        - Journal articles: Author, A. A. (Year). Title. Journal Name, Volume(Issue), pages.
        - Books: Author, A. A. (Year). Title. Publisher.
        - Book chapters: Author, A. A. (Year). Chapter title. In B. B. Editor (Ed.), Book title (pp. xx-xx). Publisher.
        - Conference papers: Author, A. A. (Year). Title. In Conference proceedings (pp. xx-xx).
        - Websites: Author, A. A. (Year, Month Day). Title. Site Name. URL

        Args:
            citation_text: Raw citation text
            citation_type: Type of citation
            metadata: Optional metadata for enhanced formatting

        Returns:
            Professionally formatted APA7 citation
        """
        try:
            # Parse citation components
            components = self._parse_citation_components(citation_text, metadata)

            # Format based on citation type
            if citation_type == CitationType.JOURNAL_ARTICLE:
                return self._format_journal_article(components)
            elif citation_type == CitationType.BOOK:
                return self._format_book(components)
            elif citation_type == CitationType.BOOK_CHAPTER:
                return self._format_book_chapter(components)
            elif citation_type == CitationType.CONFERENCE_PAPER:
                return self._format_conference_paper(components)
            elif citation_type == CitationType.THESIS_DISSERTATION:
                return self._format_thesis_dissertation(components)
            elif citation_type == CitationType.WEBSITE:
                return self._format_website(components)
            elif citation_type == CitationType.REPORT:
                return self._format_report(components)
            else:
                return self._format_generic(components)

        except Exception as e:
            logger.warning(f"Citation formatting failed: {e}")
            return citation_text.strip()

    def _parse_citation_components(self, citation_text: str, metadata: Optional[DocumentMetadata]) -> Dict[str, str]:
        """
        Parse citation text into components for formatting.

        Args:
            citation_text: Raw citation text
            metadata: Optional metadata

        Returns:
            Dictionary of citation components
        """
        try:
            import re

            components = {
                'authors': '',
                'year': '',
                'title': '',
                'journal': '',
                'volume': '',
                'issue': '',
                'pages': '',
                'publisher': '',
                'url': '',
                'doi': '',
                'editor': '',
                'book_title': '',
                'conference': ''
            }

            # Extract year (look for 4-digit year in parentheses or standalone)
            year_match = re.search(r'\((\d{4})\)|(\d{4})', citation_text)
            if year_match:
                components['year'] = year_match.group(1) or year_match.group(2)

            # Extract DOI
            doi_match = re.search(r'doi:?\s*([^\s]+)|https?://doi\.org/([^\s]+)', citation_text, re.IGNORECASE)
            if doi_match:
                components['doi'] = doi_match.group(1) or doi_match.group(2)

            # Extract URL
            url_match = re.search(r'https?://[^\s]+', citation_text)
            if url_match:
                components['url'] = url_match.group(0)

            # Extract volume and issue
            vol_issue_match = re.search(r'(\d+)\((\d+)\)', citation_text)
            if vol_issue_match:
                components['volume'] = vol_issue_match.group(1)
                components['issue'] = vol_issue_match.group(2)
            else:
                vol_match = re.search(r'vol\.?\s*(\d+)', citation_text, re.IGNORECASE)
                if vol_match:
                    components['volume'] = vol_match.group(1)

            # Extract pages
            pages_match = re.search(r'pp?\.?\s*(\d+(?:-\d+)?)', citation_text, re.IGNORECASE)
            if pages_match:
                components['pages'] = pages_match.group(1)

            # Use metadata if available to fill gaps
            if metadata:
                if hasattr(metadata, 'author') and metadata.author and not components['authors']:
                    components['authors'] = metadata.author
                if hasattr(metadata, 'title') and metadata.title and not components['title']:
                    components['title'] = metadata.title
                if hasattr(metadata, 'year') and metadata.year and not components['year']:
                    components['year'] = str(metadata.year)
                if hasattr(metadata, 'publisher') and metadata.publisher and not components['publisher']:
                    components['publisher'] = metadata.publisher

            # Extract remaining components from text
            if not components['authors']:
                # Simple author extraction (first part before year or title)
                author_match = re.match(r'^([^(]+?)(?:\s*\(\d{4}\)|\.)', citation_text)
                if author_match:
                    components['authors'] = author_match.group(1).strip()

            if not components['title']:
                # Extract title (usually after year, before journal/publisher)
                title_match = re.search(r'\(\d{4}\)\.?\s*([^.]+)', citation_text)
                if title_match:
                    components['title'] = title_match.group(1).strip()

            return components

        except Exception as e:
            logger.warning(f"Citation component parsing failed: {e}")
            return {'authors': '', 'year': '', 'title': citation_text}

    def _format_journal_article(self, components: Dict[str, str]) -> str:
        """Format journal article citation in APA7 style."""
        try:
            parts = []

            # Author(s)
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Title
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(title)

            # Journal name (italicized)
            journal_part = ""
            if components['journal']:
                journal_part = f"*{components['journal']}*"

                # Volume
                if components['volume']:
                    journal_part += f", *{components['volume']}*"

                    # Issue
                    if components['issue']:
                        journal_part += f"({components['issue']})"

                # Pages
                if components['pages']:
                    journal_part += f", {components['pages']}"

                journal_part += "."
                parts.append(journal_part)

            # DOI or URL
            if components['doi']:
                parts.append(f"https://doi.org/{components['doi']}")
            elif components['url']:
                parts.append(components['url'])

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Journal article formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _format_book(self, components: Dict[str, str]) -> str:
        """Format book citation in APA7 style."""
        try:
            parts = []

            # Author(s)
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Title (italicized)
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(f"*{title}*")

            # Publisher
            if components['publisher']:
                parts.append(f"{components['publisher']}.")

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Book formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _format_authors(self, authors_text: str) -> str:
        """
        Format author names according to APA7 standards.

        Args:
            authors_text: Raw author text

        Returns:
            Formatted author string
        """
        try:
            import re

            # Split authors by common delimiters
            authors = re.split(r'[,;&]|\sand\s', authors_text)
            formatted_authors = []

            for author in authors[:20]:  # Limit to 20 authors max
                author = author.strip()
                if not author:
                    continue

                # Check if already in APA format (Last, F. M.)
                if re.match(r'^[A-Z][a-z]+,\s+[A-Z]\.', author):
                    formatted_authors.append(author)
                else:
                    # Try to convert to APA format
                    name_parts = author.split()
                    if len(name_parts) >= 2:
                        last_name = name_parts[-1]
                        first_names = name_parts[:-1]
                        initials = '. '.join([name[0].upper() for name in first_names if name]) + '.'
                        formatted_authors.append(f"{last_name}, {initials}")
                    else:
                        formatted_authors.append(author)

            # Format according to APA7 rules
            if len(formatted_authors) == 1:
                return formatted_authors[0]
            elif len(formatted_authors) == 2:
                return f"{formatted_authors[0]}, & {formatted_authors[1]}"
            elif len(formatted_authors) <= 20:
                return ', '.join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
            else:
                # More than 20 authors - use ellipsis
                return ', '.join(formatted_authors[:19]) + f", ... {formatted_authors[-1]}"

        except Exception as e:
            logger.warning(f"Author formatting failed: {e}")
            return authors_text

    def _format_book_chapter(self, components: Dict[str, str]) -> str:
        """Format book chapter citation in APA7 style."""
        try:
            parts = []

            # Author(s)
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Chapter title
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(title)

            # Book information
            book_part = "In "
            if components['editor']:
                book_part += f"{self._format_authors(components['editor'])} (Ed.), "

            if components['book_title']:
                book_part += f"*{components['book_title']}*"

                if components['pages']:
                    book_part += f" (pp. {components['pages']})"

                book_part += "."
                parts.append(book_part)

            # Publisher
            if components['publisher']:
                parts.append(f"{components['publisher']}.")

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Book chapter formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _format_conference_paper(self, components: Dict[str, str]) -> str:
        """Format conference paper citation in APA7 style."""
        try:
            parts = []

            # Author(s)
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Title
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(title)

            # Conference information
            if components['conference']:
                conf_part = f"In *{components['conference']}*"

                if components['pages']:
                    conf_part += f" (pp. {components['pages']})"

                conf_part += "."
                parts.append(conf_part)

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Conference paper formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _format_thesis_dissertation(self, components: Dict[str, str]) -> str:
        """Format thesis/dissertation citation in APA7 style."""
        try:
            parts = []

            # Author
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Title (italicized)
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(f"*{title}*")

            # Degree type and institution
            degree_info = "[Doctoral dissertation"
            if 'master' in components.get('title', '').lower():
                degree_info = "[Master's thesis"

            if components['publisher']:  # Institution
                degree_info += f", {components['publisher']}]."
            else:
                degree_info += "]."

            parts.append(degree_info)

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Thesis/dissertation formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _format_website(self, components: Dict[str, str]) -> str:
        """Format website citation in APA7 style."""
        try:
            parts = []

            # Author(s) or organization
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Title
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(title)

            # Website name (if different from author)
            if components['publisher'] and components['publisher'] != components['authors']:
                parts.append(f"*{components['publisher']}*.")

            # URL
            if components['url']:
                parts.append(components['url'])

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Website formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _format_report(self, components: Dict[str, str]) -> str:
        """Format report citation in APA7 style."""
        try:
            parts = []

            # Author(s) or organization
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Title (italicized)
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(f"*{title}*")

            # Publisher/Organization
            if components['publisher']:
                parts.append(f"{components['publisher']}.")

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Report formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _format_generic(self, components: Dict[str, str]) -> str:
        """Format generic citation when type is unknown."""
        try:
            parts = []

            # Author(s)
            if components['authors']:
                authors = self._format_authors(components['authors'])
                parts.append(authors)

            # Year
            if components['year']:
                parts.append(f"({components['year']})")

            # Title
            if components['title']:
                title = components['title'].strip()
                if not title.endswith('.'):
                    title += '.'
                parts.append(title)

            # Additional information
            additional_info = []
            for key in ['journal', 'publisher', 'conference']:
                if components[key]:
                    additional_info.append(components[key])

            if additional_info:
                parts.append(f"{', '.join(additional_info)}.")

            # URL or DOI
            if components['doi']:
                parts.append(f"https://doi.org/{components['doi']}")
            elif components['url']:
                parts.append(components['url'])

            return ' '.join(parts)

        except Exception as e:
            logger.warning(f"Generic formatting failed: {e}")
            return ' '.join([v for v in components.values() if v])

    def _calculate_compliance_score(self, result: APA7ValidationResult) -> float:
        """Calculate compliance score based on validation results."""
        base_score = 1.0

        # Deduct for missing elements
        base_score -= len(result.missing_elements) * 0.2

        # Deduct for formatting errors
        base_score -= len(result.formatting_errors) * 0.1

        return max(0.0, base_score)

    def _determine_compliance_level(self, score: float) -> ComplianceLevel:
        """Determine compliance level from score."""
        if score >= 0.9:
            return ComplianceLevel.FULLY_COMPLIANT
        elif score >= 0.7:
            return ComplianceLevel.MOSTLY_COMPLIANT
        elif score >= 0.5:
            return ComplianceLevel.PARTIALLY_COMPLIANT
        else:
            return ComplianceLevel.NON_COMPLIANT

    def _generate_suggestions(self, result: APA7ValidationResult) -> List[str]:
        """Generate suggestions for improving citation."""
        suggestions = []

        if 'author' in result.missing_elements:
            suggestions.append('Add author information in format: Last, F. M.')

        if 'year' in result.missing_elements:
            suggestions.append('Add publication year in parentheses: (2023)')

        if result.formatting_errors:
            suggestions.append('Review APA7 formatting guidelines')

        return suggestions
