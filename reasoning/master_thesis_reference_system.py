"""
Master Thesis Reference System - Enterprise-Grade Academic Citation Engine.

This is the main orchestration system that integrates all components to provide
a complete, production-ready solution for master thesis citation analysis and
generation with maximum precision and reliability.

Features:
    - Complete thesis analysis and indexing
    - Advanced claim detection with AI validation
    - Precision citation matching with anti-hallucination measures
    - APA7 compliance validation and formatting
    - Human-in-the-loop validation queues
    - Comprehensive reporting and analytics
    - Batch processing for large document collections
    - Real-time progress tracking

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from core.types import DocumentMetadata, TextChunk, SearchResult, CitationEntry
from core.exceptions import ProcessingError, ValidationError
from processing.document_parser import parse_document
from processing.text_processor import TextProcessor
from indexing.hybrid_search import HybridSearchEngine
from analysis.master_thesis_claim_detector import MasterThesisClaimDetector, DetectedClaim
from reasoning.advanced_citation_validator import AdvancedCitationValidator, ValidationResult
from reasoning.apa7_compliance_engine import APA7ComplianceEngine, APA7ValidationResult
from api.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class ProcessingProgress:
    """Progress tracking for thesis processing."""
    total_documents: int = 0
    processed_documents: int = 0
    total_claims: int = 0
    validated_claims: int = 0
    generated_citations: int = 0
    current_stage: str = "Initializing"
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100


@dataclass
class ThesisAnalysisResult:
    """Complete analysis result for a master thesis."""
    thesis_metadata: DocumentMetadata
    detected_claims: List[DetectedClaim]
    citation_validations: List[ValidationResult]
    apa7_validations: List[APA7ValidationResult]
    generated_bibliography: str
    
    # Analytics
    total_claims: int = 0
    high_confidence_claims: int = 0
    citations_needed: int = 0
    citations_generated: int = 0
    compliance_score: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    processed_at: datetime = field(default_factory=datetime.now)


class MasterThesisReferenceSystem:
    """
    Enterprise-grade master thesis reference system.
    
    This system orchestrates all components to provide a complete solution
    for academic citation analysis and generation with maximum precision.
    """
    
    def __init__(self, 
                 openrouter_client: OpenRouterClient,
                 index_directory: str = "thesis_index",
                 enable_human_review: bool = True,
                 min_confidence_threshold: float = 0.7):
        """
        Initialize the master thesis reference system.
        
        Args:
            openrouter_client: Client for AI model access
            index_directory: Directory for storing indices
            enable_human_review: Whether to enable human review queues
            min_confidence_threshold: Minimum confidence for auto-approval
        """
        self.openrouter_client = openrouter_client
        self.index_directory = Path(index_directory)
        self.enable_human_review = enable_human_review
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.search_engine = HybridSearchEngine(str(self.index_directory))
        self.claim_detector = MasterThesisClaimDetector(openrouter_client)
        self.citation_validator = AdvancedCitationValidator(
            openrouter_client, 
            None,  # Will be initialized with semantic matcher
            min_confidence_threshold,
            enable_human_review
        )
        self.apa7_engine = APA7ComplianceEngine()
        
        # Processing state
        self.progress = ProcessingProgress()
        self.indexed_documents: Dict[str, DocumentMetadata] = {}
        
        # Create index directory
        self.index_directory.mkdir(exist_ok=True)
        
        logger.info("Master thesis reference system initialized")
    
    async def process_complete_thesis_project(self, 
                                            thesis_file: str,
                                            source_directory: str,
                                            output_directory: str = "thesis_output") -> ThesisAnalysisResult:
        """
        Process a complete thesis project with source materials.
        
        Args:
            thesis_file: Path to the master thesis document
            source_directory: Directory containing source materials
            output_directory: Directory for output files
            
        Returns:
            Complete analysis result
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting complete thesis project processing")
            
            # Stage 1: Index all source materials
            self.progress.current_stage = "Indexing source materials"
            await self._index_source_materials(source_directory)
            
            # Stage 2: Process thesis document
            self.progress.current_stage = "Processing thesis document"
            thesis_text, thesis_metadata = await self._process_thesis_document(thesis_file)
            
            # Stage 3: Detect claims requiring citations
            self.progress.current_stage = "Detecting claims"
            detected_claims = await self.claim_detector.detect_claims(thesis_text, thesis_metadata, True)
            self.progress.total_claims = len(detected_claims)
            
            # Stage 4: Validate citations for each claim
            self.progress.current_stage = "Validating citations"
            citation_validations = await self._validate_all_claims(detected_claims, thesis_metadata)
            
            # Stage 5: Generate APA7 compliant citations
            self.progress.current_stage = "Generating APA7 citations"
            apa7_validations = await self._generate_apa7_citations(citation_validations)
            
            # Stage 6: Create bibliography
            self.progress.current_stage = "Creating bibliography"
            bibliography = await self._create_bibliography(apa7_validations)
            
            # Stage 7: Generate analysis report
            self.progress.current_stage = "Generating report"
            analysis_result = self._create_analysis_result(
                thesis_metadata, detected_claims, citation_validations, 
                apa7_validations, bibliography, start_time
            )
            
            # Stage 8: Save outputs
            await self._save_outputs(analysis_result, output_directory)
            
            self.progress.current_stage = "Completed"
            logger.info(f"Thesis project processing completed in {analysis_result.processing_time:.2f} seconds")
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"Thesis project processing failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)
    
    async def _index_source_materials(self, source_directory: str) -> None:
        """
        Index all source materials in the directory.
        
        Args:
            source_directory: Directory containing source files
        """
        try:
            source_path = Path(source_directory)
            if not source_path.exists():
                raise ProcessingError(f"Source directory not found: {source_directory}")
            
            # Find all supported files
            supported_extensions = ['.pdf', '.txt', '.md', '.doc', '.docx']
            source_files = []
            
            for ext in supported_extensions:
                source_files.extend(source_path.glob(f"**/*{ext}"))
            
            self.progress.total_documents = len(source_files)
            logger.info(f"Found {len(source_files)} source documents to index")
            
            # Process each file
            for file_path in source_files:
                try:
                    # Parse document
                    text, metadata = parse_document(str(file_path))
                    
                    # Process text into chunks
                    chunks = self.text_processor.process_text(text, str(file_path))
                    
                    # Add to search index
                    self.search_engine.add_documents(chunks)
                    
                    # Store metadata
                    self.indexed_documents[str(file_path)] = metadata
                    
                    self.progress.processed_documents += 1
                    logger.debug(f"Indexed: {file_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")
                    continue
            
            logger.info(f"Indexing completed: {self.progress.processed_documents}/{self.progress.total_documents} documents")
            
        except Exception as e:
            error_msg = f"Source material indexing failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)
    
    async def _process_thesis_document(self, thesis_file: str) -> Tuple[str, DocumentMetadata]:
        """
        Process the main thesis document.
        
        Args:
            thesis_file: Path to thesis file
            
        Returns:
            Tuple of (thesis_text, thesis_metadata)
        """
        try:
            logger.info(f"Processing thesis document: {thesis_file}")
            
            # Parse thesis document
            text, metadata = parse_document(thesis_file)
            
            # Validate thesis content
            if len(text) < 10000:  # Minimum length check
                logger.warning("Thesis document seems unusually short")
            
            # Extract additional metadata
            metadata.document_type = "master_thesis"
            metadata.processing_date = datetime.now()
            
            logger.info(f"Thesis processed: {len(text)} characters, {len(text.split())} words")
            return text, metadata
            
        except Exception as e:
            error_msg = f"Thesis document processing failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)
    
    async def _validate_all_claims(self, 
                                 detected_claims: List[DetectedClaim],
                                 thesis_metadata: DocumentMetadata) -> List[ValidationResult]:
        """
        Validate citations for all detected claims.
        
        Args:
            detected_claims: List of detected claims
            thesis_metadata: Thesis metadata
            
        Returns:
            List of validation results
        """
        validation_results = []
        
        try:
            for claim in detected_claims:
                # Search for relevant sources
                search_results = self.search_engine.search(
                    claim.text, 
                    k=10,  # Get top 10 candidates
                    threshold=0.3
                )
                
                # Validate each potential source
                for search_result in search_results[:5]:  # Validate top 5
                    try:
                        validation = await self.citation_validator.validate_citation(
                            claim.text,
                            search_result.chunk,
                            thesis_metadata
                        )
                        
                        # Add claim context to validation
                        validation.claim_context = {
                            'claim_id': claim.claim_id,
                            'claim_type': claim.claim_type.value,
                            'paragraph_number': claim.paragraph_number,
                            'search_score': search_result.score
                        }
                        
                        validation_results.append(validation)
                        
                    except Exception as e:
                        logger.warning(f"Validation failed for claim {claim.claim_id}: {e}")
                        continue
                
                self.progress.validated_claims += 1
            
            logger.info(f"Citation validation completed: {len(validation_results)} validations")
            return validation_results
            
        except Exception as e:
            error_msg = f"Claim validation failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ValidationError(error_msg)
