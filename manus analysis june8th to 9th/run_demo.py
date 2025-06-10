#!/usr/bin/env python3
"""
Demo Script for Complete Thesis Analysis System.

This script demonstrates the system capabilities without requiring API keys.
It shows the local processing, indexing, and search functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.document_parser import parse_document
from processing.text_processor import TextProcessor
from indexing.hybrid_search import HybridSearchEngine
from analysis.master_thesis_claim_detector import MasterThesisClaimDetector
from reasoning.apa7_compliance_engine import APA7ComplianceEngine

def run_demo():
    """Run the complete thesis analysis demo."""
    print("="*80)
    print("COMPLETE THESIS ANALYSIS SYSTEM - DEMO")
    print("="*80)
    print()

    # Stage 1: Document Processing
    print("STAGE 1: DOCUMENT PROCESSING")
    print("-" * 40)

    try:
        # Process thesis document
        thesis_path = "demo/sample_thesis.txt"
        print(f"Processing thesis: {thesis_path}")
        thesis_text, thesis_metadata = parse_document(thesis_path)
        print(f"✓ Thesis processed: {len(thesis_text)} characters")

        # Process source documents
        sources_dir = Path("demo/sources")
        source_files = list(sources_dir.glob("*.txt"))
        print(f"✓ Found {len(source_files)} source documents")

        all_sources = []
        for source_file in source_files:
            text, metadata = parse_document(str(source_file))
            all_sources.append((text, metadata, source_file.name))
            print(f"  - {source_file.name}: {len(text)} characters")

    except Exception as e:
        print(f"✗ Document processing failed: {e}")
        return

    print()

    # Stage 2: Text Processing and Chunking
    print("STAGE 2: TEXT PROCESSING AND CHUNKING")
    print("-" * 40)

    try:
        text_processor = TextProcessor(
            default_chunk_size=256,
            default_overlap=25,
            chunking_method="paragraph"
        )

        # Process thesis into chunks
        thesis_chunks = text_processor.process_text(thesis_text, "thesis")
        print(f"✓ Thesis chunked into {len(thesis_chunks)} segments")

        # Process source documents
        all_chunks = []
        for text, metadata, filename in all_sources:
            chunks = text_processor.process_text(text, filename)
            all_chunks.extend(chunks)
            print(f"✓ {filename}: {len(chunks)} chunks")

        print(f"✓ Total source chunks: {len(all_chunks)}")

    except Exception as e:
        print(f"✗ Text processing failed: {e}")
        return

    print()

    # Stage 3: Indexing and Search
    print("STAGE 3: INDEXING AND SEARCH")
    print("-" * 40)

    try:
        # Initialize search engine
        search_engine = HybridSearchEngine("demo/demo_index")
        print("✓ Search engine initialized")

        # Add documents to index
        search_engine.add_documents(all_chunks)
        print(f"✓ Indexed {len(all_chunks)} document chunks")

        # Test search functionality
        test_queries = [
            "machine learning accuracy",
            "deep learning medical images",
            "transformer architectures"
        ]

        for query in test_queries:
            results = search_engine.search(query, k=3, threshold=0.1)
            print(f"✓ Search '{query}': {len(results)} results")
            if results:
                best_result = results[0]
                print(f"  Best match (score: {best_result.score:.3f}): {best_result.chunk.text[:100]}...")

    except Exception as e:
        print(f"✗ Indexing and search failed: {e}")
        return

    print()

    # Stage 4: Claim Detection (Local Only)
    print("STAGE 4: CLAIM DETECTION (LOCAL SIMULATION)")
    print("-" * 40)

    try:
        # Simulate claim detection without API calls
        print("✓ Simulating claim detection...")

        # Simple pattern-based claim detection for demo
        import re

        claim_patterns = [
            r'according to.*studies',
            r'research shows',
            r'studies.*demonstrate',
            r'achieve.*accuracy',
            r'results.*show',
            r'\d+%.*accuracy'
        ]

        detected_claims = []
        sentences = thesis_text.split('.')

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    detected_claims.append({
                        'id': f"claim_{len(detected_claims)+1}",
                        'text': sentence,
                        'pattern': pattern,
                        'sentence_number': i
                    })
                    break

        print(f"✓ Detected {len(detected_claims)} potential claims requiring citations")

        for claim in detected_claims[:5]:  # Show first 5
            print(f"  - Claim {claim['id']}: {claim['text'][:80]}...")

    except Exception as e:
        print(f"✗ Claim detection failed: {e}")
        return

    print()

    # Stage 5: APA7 Compliance Testing
    print("STAGE 5: APA7 COMPLIANCE TESTING")
    print("-" * 40)

    try:
        apa7_engine = APA7ComplianceEngine()
        print("✓ APA7 compliance engine initialized")

        # Test citations
        test_citations = [
            "Smith, J., Johnson, M., & Williams, K. (2023). Deep Learning in Medical Image Analysis. Medical AI Review, 45(2), 123-145.",
            "Brown, A. (2023). Machine Learning Applications in Healthcare. Healthcare Technology Review.",
            "Garcia, L., Thompson, P., & Anderson, C. (2023). Transformer Architectures in Medical NLP. Computational Medicine, 12(3), 78-92. https://doi.org/10.1000/compmed.2023.078"
        ]

        for i, citation in enumerate(test_citations, 1):
            result = apa7_engine.validate_citation(citation)
            print(f"✓ Citation {i} validation:")
            print(f"  Compliance Level: {result.compliance_level.value}")
            print(f"  Compliance Score: {result.compliance_score:.3f}")
            if result.formatting_errors:
                print(f"  Errors: {', '.join(result.formatting_errors[:2])}")

    except Exception as e:
        print(f"✗ APA7 compliance testing failed: {e}")
        return

    print()

    # Stage 6: Demo Summary
    print("STAGE 6: DEMO SUMMARY")
    print("-" * 40)

    print("✓ Document processing: SUCCESSFUL")
    print("✓ Text chunking and indexing: SUCCESSFUL")
    print("✓ Hybrid search functionality: SUCCESSFUL")
    print("✓ Claim detection simulation: SUCCESSFUL")
    print("✓ APA7 compliance validation: SUCCESSFUL")
    print()
    print("DEMO COMPLETED SUCCESSFULLY!")
    print()
    print("To run the full system with API integration:")
    print("1. Set environment variables:")
    print("   export OPENROUTER_API_KEY='your_key'")
    print("   export PERPLEXITY_API_KEY='your_key'")
    print("   export SEMANTIC_SCHOLAR_API_KEY='your_key'  # optional")
    print()
    print("2. Run the complete analysis:")
    print("   python scripts/complete_thesis_analysis.py \\")
    print("     --thesis demo/sample_thesis.txt \\")
    print("     --sources demo/sources/ \\")
    print("     --output demo/results/ \\")
    print("     --config demo/demo_config.json")
    print()
    print("="*80)

if __name__ == "__main__":
    run_demo()
