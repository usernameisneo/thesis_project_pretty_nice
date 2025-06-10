#!/usr/bin/env python3
"""
Complete Thesis Analysis System - Production-Grade Multi-API Academic Citation Engine.

This script provides the complete, enterprise-grade thesis analysis system with
full integration of Semantic Scholar, Perplexity, and OpenRouter APIs for
maximum precision academic citation generation and validation.

Usage:
    python scripts/complete_thesis_analysis.py --thesis thesis.pdf --sources ./sources --output ./results

Features:
    - Complete multi-API integration (Semantic Scholar + Perplexity + OpenRouter)
    - Advanced claim detection with AI validation
    - Real-time academic paper discovery and verification
    - Precision citation matching with anti-hallucination measures
    - APA7 compliance validation and formatting
    - Comprehensive analytics and reporting
    - Human-in-the-loop validation queues
    - Progress tracking and performance monitoring

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.exceptions import ProcessingError, ValidationError, APIError
from processing.document_parser import parse_document
from processing.text_processor import TextProcessor
from indexing.hybrid_search import HybridSearchEngine
from analysis.master_thesis_claim_detector import MasterThesisClaimDetector
from reasoning.enhanced_citation_engine import EnhancedCitationEngine
from api.openrouter_client import OpenRouterClient

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_thesis_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class CompleteThesisAnalysisSystem:
    """
    Complete production-grade thesis analysis system.
    
    This system orchestrates all components to provide the most comprehensive
    academic citation analysis and generation available.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the complete thesis analysis system.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = Config(config_path)
        
        # API keys
        self.semantic_scholar_api_key = self._get_api_key('SEMANTIC_SCHOLAR_API_KEY', required=False)
        self.perplexity_api_key = self._get_api_key('PERPLEXITY_API_KEY', required=True)
        self.openrouter_api_key = self._get_api_key('OPENROUTER_API_KEY', required=True)
        
        # System components
        self.text_processor = None
        self.search_engine = None
        self.claim_detector = None
        self.citation_engine = None
        
        logger.info("Complete thesis analysis system initialized")
    
    def _get_api_key(self, env_var: str, required: bool = True) -> str:
        """
        Get API key from environment or config.
        
        Args:
            env_var: Environment variable name
            required: Whether the key is required
            
        Returns:
            API key string
        """
        key = os.getenv(env_var) or self.config.get(env_var.lower())
        
        if required and not key:
            raise ProcessingError(
                f"{env_var} is required. Please set it as an environment variable "
                f"or add it to your configuration file."
            )
        
        return key
    
    async def initialize_system(self) -> None:
        """Initialize all system components."""
        try:
            logger.info("Initializing complete thesis analysis system...")
            
            # Initialize text processor
            self.text_processor = TextProcessor()
            
            # Initialize search engine
            index_directory = self.config.get('index_directory', 'complete_thesis_index')
            self.search_engine = HybridSearchEngine(index_directory)
            
            # Initialize OpenRouter client for claim detection
            openrouter_client = OpenRouterClient(self.openrouter_api_key)
            
            # Test OpenRouter connection
            await self._test_openrouter_connection(openrouter_client)
            
            # Initialize claim detector
            self.claim_detector = MasterThesisClaimDetector(openrouter_client)
            
            # Initialize enhanced citation engine
            self.citation_engine = EnhancedCitationEngine(
                semantic_scholar_api_key=self.semantic_scholar_api_key,
                perplexity_api_key=self.perplexity_api_key,
                openrouter_api_key=self.openrouter_api_key,
                cache_directory=self.config.get('cache_directory', 'complete_citation_cache'),
                min_confidence_threshold=self.config.get('min_confidence_threshold', 0.75),
                enable_human_review=self.config.get('enable_human_review', True),
                max_concurrent_requests=self.config.get('max_concurrent_requests', 5)
            )
            
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            error_msg = f"System initialization failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)
    
    async def run_complete_analysis(self,
                                  thesis_file: str,
                                  sources_directory: str,
                                  output_directory: str = "complete_thesis_results") -> None:
        """
        Run complete thesis analysis with all APIs and validation.
        
        Args:
            thesis_file: Path to thesis document
            sources_directory: Directory containing source materials
            output_directory: Directory for output files
        """
        try:
            start_time = datetime.now()
            logger.info("="*80)
            logger.info("STARTING COMPLETE THESIS ANALYSIS")
            logger.info("="*80)
            logger.info(f"Thesis file: {thesis_file}")
            logger.info(f"Sources directory: {sources_directory}")
            logger.info(f"Output directory: {output_directory}")
            
            # Validate inputs
            await self._validate_inputs(thesis_file, sources_directory)
            
            # Create output directory
            output_path = Path(output_directory)
            output_path.mkdir(exist_ok=True)
            
            # Stage 1: Index source materials
            logger.info("\n" + "="*50)
            logger.info("STAGE 1: INDEXING SOURCE MATERIALS")
            logger.info("="*50)
            await self._index_source_materials(sources_directory)
            
            # Stage 2: Process thesis document
            logger.info("\n" + "="*50)
            logger.info("STAGE 2: PROCESSING THESIS DOCUMENT")
            logger.info("="*50)
            thesis_text, thesis_metadata = await self._process_thesis_document(thesis_file)
            
            # Stage 3: Detect claims requiring citations
            logger.info("\n" + "="*50)
            logger.info("STAGE 3: DETECTING CLAIMS REQUIRING CITATIONS")
            logger.info("="*50)
            detected_claims = await self.claim_detector.detect_claims(
                thesis_text, 
                thesis_metadata, 
                use_ai_validation=True
            )
            logger.info(f"Detected {len(detected_claims)} claims requiring citations")
            
            # Stage 4: Search local index for each claim
            logger.info("\n" + "="*50)
            logger.info("STAGE 4: SEARCHING LOCAL INDEX FOR CLAIMS")
            logger.info("="*50)
            local_search_results = await self._search_local_index_for_claims(detected_claims)
            
            # Stage 5: Complete citation analysis with all APIs
            logger.info("\n" + "="*50)
            logger.info("STAGE 5: COMPREHENSIVE CITATION ANALYSIS")
            logger.info("="*50)
            analysis_result = await self.citation_engine.process_complete_thesis(
                thesis_text=thesis_text,
                thesis_metadata=thesis_metadata,
                detected_claims=detected_claims,
                local_search_results=local_search_results
            )
            
            # Stage 6: Generate comprehensive outputs
            logger.info("\n" + "="*50)
            logger.info("STAGE 6: GENERATING COMPREHENSIVE OUTPUTS")
            logger.info("="*50)
            await self._save_comprehensive_outputs(analysis_result, output_path)
            
            # Stage 7: Generate final report
            total_time = (datetime.now() - start_time).total_seconds()
            await self._generate_final_report(analysis_result, total_time, output_path)
            
            logger.info("\n" + "="*80)
            logger.info("COMPLETE THESIS ANALYSIS FINISHED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Results saved to: {output_directory}")
            
            # Print executive summary
            self._print_executive_summary(analysis_result, total_time)
            
        except Exception as e:
            error_msg = f"Complete thesis analysis failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)
    
    async def _test_openrouter_connection(self, client: OpenRouterClient) -> None:
        """Test OpenRouter API connection."""
        try:
            logger.info("Testing OpenRouter API connection...")
            
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Test connection"}],
                model="openai/gpt-3.5-turbo",
                max_tokens=10
            )
            
            if response and 'choices' in response:
                logger.info("OpenRouter API connection test successful")
            else:
                raise APIError("Invalid OpenRouter API response")
                
        except Exception as e:
            error_msg = f"OpenRouter API connection test failed: {e}"
            logger.error(error_msg)
            raise APIError(error_msg)
    
    async def _validate_inputs(self, thesis_file: str, sources_directory: str) -> None:
        """Validate input files and directories."""
        # Check thesis file
        thesis_path = Path(thesis_file)
        if not thesis_path.exists():
            raise ProcessingError(f"Thesis file not found: {thesis_file}")
        
        if not thesis_path.is_file():
            raise ProcessingError(f"Thesis path is not a file: {thesis_file}")
        
        # Check file extension
        supported_extensions = ['.pdf', '.txt', '.md', '.doc', '.docx']
        if thesis_path.suffix.lower() not in supported_extensions:
            raise ProcessingError(
                f"Unsupported thesis file format: {thesis_path.suffix}. "
                f"Supported formats: {', '.join(supported_extensions)}"
            )
        
        # Check sources directory
        sources_path = Path(sources_directory)
        if not sources_path.exists():
            raise ProcessingError(f"Sources directory not found: {sources_directory}")
        
        if not sources_path.is_dir():
            raise ProcessingError(f"Sources path is not a directory: {sources_directory}")
        
        # Check for source files
        source_files = []
        for ext in supported_extensions:
            source_files.extend(sources_path.glob(f"**/*{ext}"))
        
        if not source_files:
            raise ProcessingError(
                f"No supported source files found in: {sources_directory}. "
                f"Supported formats: {', '.join(supported_extensions)}"
            )
        
        logger.info(f"Input validation passed: {len(source_files)} source files found")
    
    async def _index_source_materials(self, sources_directory: str) -> None:
        """Index all source materials."""
        try:
            sources_path = Path(sources_directory)
            supported_extensions = ['.pdf', '.txt', '.md', '.doc', '.docx']
            
            # Find all source files
            source_files = []
            for ext in supported_extensions:
                source_files.extend(sources_path.glob(f"**/*{ext}"))
            
            logger.info(f"Indexing {len(source_files)} source documents...")
            
            processed_count = 0
            for file_path in source_files:
                try:
                    # Parse document
                    text, metadata = parse_document(str(file_path))
                    
                    # Process into chunks
                    chunks = self.text_processor.process_text(text, str(file_path))
                    
                    # Add to search index
                    self.search_engine.add_documents(chunks)
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Indexed {processed_count}/{len(source_files)} documents")
                    
                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")
                    continue
            
            logger.info(f"Source indexing completed: {processed_count}/{len(source_files)} documents indexed")
            
        except Exception as e:
            error_msg = f"Source material indexing failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)
    
    async def _process_thesis_document(self, thesis_file: str) -> tuple:
        """Process the main thesis document."""
        try:
            logger.info(f"Processing thesis document: {thesis_file}")
            
            # Parse thesis document
            text, metadata = parse_document(thesis_file)
            
            # Validate thesis content
            if len(text) < 10000:
                logger.warning("Thesis document seems unusually short")
            
            # Enhance metadata
            metadata.document_type = "master_thesis"
            metadata.processing_date = datetime.now()
            
            logger.info(f"Thesis processed: {len(text)} characters, {len(text.split())} words")
            return text, metadata
            
        except Exception as e:
            error_msg = f"Thesis document processing failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)

    async def _search_local_index_for_claims(self, detected_claims) -> dict:
        """Search local index for each detected claim."""
        try:
            logger.info(f"Searching local index for {len(detected_claims)} claims...")

            local_search_results = {}

            for i, claim in enumerate(detected_claims):
                try:
                    # Search for relevant sources
                    search_results = self.search_engine.search(
                        claim.text,
                        k=10,
                        threshold=0.3
                    )

                    local_search_results[claim.claim_id] = search_results

                    if (i + 1) % 20 == 0:
                        logger.info(f"Searched {i + 1}/{len(detected_claims)} claims")

                except Exception as e:
                    logger.warning(f"Local search failed for claim {claim.claim_id}: {e}")
                    local_search_results[claim.claim_id] = []

            total_results = sum(len(results) for results in local_search_results.values())
            logger.info(f"Local search completed: {total_results} total search results")

            return local_search_results

        except Exception as e:
            error_msg = f"Local index search failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg)

    async def _save_comprehensive_outputs(self, analysis_result, output_path: Path) -> None:
        """Save all comprehensive outputs."""
        try:
            logger.info("Saving comprehensive analysis outputs...")

            # Save formatted bibliography
            bibliography_file = output_path / "bibliography.txt"
            with open(bibliography_file, 'w', encoding='utf-8') as f:
                f.write(analysis_result.formatted_bibliography)

            # Save citation report
            report_file = output_path / "citation_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(analysis_result.citation_report)

            # Save detailed analysis data
            analysis_data = {
                "thesis_metadata": {
                    "title": getattr(analysis_result.thesis_metadata, 'title', 'Unknown'),
                    "author": getattr(analysis_result.thesis_metadata, 'author', 'Unknown'),
                    "year": getattr(analysis_result.thesis_metadata, 'year', 'Unknown'),
                    "document_type": getattr(analysis_result.thesis_metadata, 'document_type', 'Unknown')
                },
                "analysis_summary": {
                    "total_claims": analysis_result.total_claims,
                    "claims_with_citations": analysis_result.claims_with_citations,
                    "high_confidence_citations": analysis_result.high_confidence_citations,
                    "peer_reviewed_citations": analysis_result.peer_reviewed_citations,
                    "requires_human_review": analysis_result.requires_human_review,
                    "overall_citation_quality": analysis_result.overall_citation_quality,
                    "apa7_compliance_score": analysis_result.apa7_compliance_score,
                    "source_credibility_score": analysis_result.source_credibility_score,
                    "processing_time": analysis_result.processing_time,
                    "api_calls_made": analysis_result.api_calls_made
                },
                "detected_claims": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "claim_type": claim.claim_type.value,
                        "citation_need": claim.citation_need.value,
                        "confidence_score": claim.confidence_score,
                        "paragraph_number": claim.paragraph_number,
                        "keywords": claim.keywords,
                        "suggested_search_terms": claim.suggested_search_terms
                    }
                    for claim in analysis_result.detected_claims
                ],
                "citation_matches": [
                    {
                        "claim_id": match.claim_id,
                        "claim_text": match.claim_text[:200] + "..." if len(match.claim_text) > 200 else match.claim_text,
                        "discovery_method": match.discovery_method,
                        "semantic_similarity": match.semantic_similarity,
                        "factual_verification": match.factual_verification,
                        "temporal_validity": match.temporal_validity,
                        "source_credibility": match.source_credibility,
                        "overall_confidence": match.overall_confidence,
                        "apa7_citation": match.apa7_citation,
                        "requires_human_review": match.requires_human_review,
                        "validation_trace": match.validation_trace,
                        "peer_reviewed": match.peer_reviewed,
                        "citation_count": match.citation_count,
                        "publication_year": match.publication_year
                    }
                    for match in analysis_result.citation_matches
                ]
            }

            analysis_file = output_path / "complete_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)

            # Save human review queue if needed
            review_items = [m for m in analysis_result.citation_matches if m.requires_human_review]
            if review_items:
                review_file = output_path / "human_review_queue.json"
                review_data = {
                    "review_items": [
                        {
                            "claim_id": item.claim_id,
                            "claim_text": item.claim_text,
                            "confidence": item.overall_confidence,
                            "apa7_citation": item.apa7_citation,
                            "validation_trace": item.validation_trace,
                            "discovery_method": item.discovery_method
                        }
                        for item in review_items
                    ],
                    "total_items": len(review_items),
                    "generated_at": datetime.now().isoformat()
                }

                with open(review_file, 'w', encoding='utf-8') as f:
                    json.dump(review_data, f, indent=2, ensure_ascii=False)

                logger.info(f"Human review queue saved: {len(review_items)} items require review")

            logger.info(f"All outputs saved to: {output_path}")

        except Exception as e:
            logger.warning(f"Output saving failed: {e}")

    async def _generate_final_report(self, analysis_result, total_time: float, output_path: Path) -> None:
        """Generate final comprehensive report."""
        try:
            report = f"""
COMPLETE THESIS ANALYSIS - FINAL REPORT
=======================================

Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total processing time: {total_time:.2f} seconds

THESIS INFORMATION
-----------------
Title: {getattr(analysis_result.thesis_metadata, 'title', 'Unknown')}
Author: {getattr(analysis_result.thesis_metadata, 'author', 'Unknown')}
Year: {getattr(analysis_result.thesis_metadata, 'year', 'Unknown')}

ANALYSIS RESULTS
---------------
Total Claims Detected: {analysis_result.total_claims}
Claims with Valid Citations: {analysis_result.claims_with_citations}
High Confidence Citations: {analysis_result.high_confidence_citations}
Peer-Reviewed Citations: {analysis_result.peer_reviewed_citations}
Items Requiring Human Review: {analysis_result.requires_human_review}

QUALITY METRICS
--------------
Overall Citation Quality: {analysis_result.overall_citation_quality:.3f} / 1.000
APA7 Compliance Score: {analysis_result.apa7_compliance_score:.3f} / 1.000
Source Credibility Score: {analysis_result.source_credibility_score:.3f} / 1.000

API USAGE STATISTICS
-------------------
Semantic Scholar API Calls: {analysis_result.api_calls_made.get('semantic_scholar', 0)}
Perplexity API Calls: {analysis_result.api_calls_made.get('perplexity', 0)}
OpenRouter API Calls: {analysis_result.api_calls_made.get('openrouter', 0)}

PERFORMANCE METRICS
------------------
Claims Processing Rate: {analysis_result.total_claims / total_time:.2f} claims/second
Average Time per Claim: {total_time / max(analysis_result.total_claims, 1):.2f} seconds

RECOMMENDATIONS
--------------
"""

            # Add recommendations based on results
            if analysis_result.requires_human_review > 0:
                report += f"- Review {analysis_result.requires_human_review} low-confidence citations in human_review_queue.json\n"

            if analysis_result.apa7_compliance_score < 0.8:
                report += "- Consider manual review of APA7 formatting for improved compliance\n"

            if analysis_result.peer_reviewed_citations < analysis_result.claims_with_citations * 0.5:
                report += "- Consider seeking more peer-reviewed sources for stronger academic support\n"

            if analysis_result.overall_citation_quality < 0.7:
                report += "- Overall citation quality could be improved with additional source validation\n"

            report += f"""
FILES GENERATED
--------------
- bibliography.txt: Formatted APA7 bibliography
- citation_report.txt: Detailed citation analysis report
- complete_analysis.json: Complete analysis data in JSON format
- human_review_queue.json: Items requiring human review (if any)
- final_report.txt: This comprehensive report

SYSTEM INFORMATION
-----------------
Analysis Engine: Complete Thesis Analysis System v2.0
APIs Used: Semantic Scholar, Perplexity, OpenRouter
Anti-Hallucination: Multi-layer validation enabled
Human Review: {'Enabled' if analysis_result.requires_human_review > 0 else 'Not required'}

Analysis completed successfully.
"""

            report_file = output_path / "final_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info("Final comprehensive report generated")

        except Exception as e:
            logger.warning(f"Final report generation failed: {e}")

    def _print_executive_summary(self, analysis_result, total_time: float) -> None:
        """Print executive summary to console."""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY - COMPLETE THESIS ANALYSIS")
        print("="*80)
        print(f"Processing Time: {total_time:.2f} seconds")
        print(f"Total Claims Detected: {analysis_result.total_claims}")
        print(f"Valid Citations Generated: {analysis_result.claims_with_citations}")
        print(f"High Confidence Citations: {analysis_result.high_confidence_citations}")
        print(f"Peer-Reviewed Citations: {analysis_result.peer_reviewed_citations}")
        print(f"Human Review Required: {analysis_result.requires_human_review}")
        print(f"Overall Citation Quality: {analysis_result.overall_citation_quality:.3f}")
        print(f"APA7 Compliance Score: {analysis_result.apa7_compliance_score:.3f}")
        print(f"Source Credibility Score: {analysis_result.source_credibility_score:.3f}")
        print("="*80)
        print("Analysis completed successfully!")
        print("="*80)


async def main():
    """Main entry point for complete thesis analysis."""
    parser = argparse.ArgumentParser(
        description="Complete Thesis Analysis System - Production-Grade Multi-API Academic Citation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Complete analysis with all APIs
    python scripts/complete_thesis_analysis.py --thesis thesis.pdf --sources ./sources

    # With custom output directory
    python scripts/complete_thesis_analysis.py --thesis thesis.pdf --sources ./sources --output ./results

    # With custom configuration
    python scripts/complete_thesis_analysis.py --thesis thesis.pdf --sources ./sources --config config.json

Required Environment Variables:
    PERPLEXITY_API_KEY: Your Perplexity API key (required)
    OPENROUTER_API_KEY: Your OpenRouter API key (required)
    SEMANTIC_SCHOLAR_API_KEY: Your Semantic Scholar API key (optional, improves rate limits)
        """
    )

    parser.add_argument(
        '--thesis',
        required=True,
        help='Path to the master thesis document (PDF, TXT, MD, DOC, DOCX)'
    )

    parser.add_argument(
        '--sources',
        required=True,
        help='Directory containing source materials for citation'
    )

    parser.add_argument(
        '--output',
        default='complete_thesis_results',
        help='Output directory for results (default: complete_thesis_results)'
    )

    parser.add_argument(
        '--config',
        help='Path to configuration file (optional)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize system
        system = CompleteThesisAnalysisSystem(args.config)
        await system.initialize_system()

        # Run complete analysis
        await system.run_complete_analysis(
            thesis_file=args.thesis,
            sources_directory=args.sources,
            output_directory=args.output
        )

        print("\nComplete thesis analysis finished successfully!")

    except Exception as e:
        logger.error(f"Complete analysis failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
