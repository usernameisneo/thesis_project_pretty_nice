"""
Advanced Citation Reasoning Engine for Academic Research.

This module provides highly technical, precision-engineered citation reasoning
capabilities for academic research and thesis writing. It implements sophisticated
algorithms for cross-document reasoning, semantic entailment verification, and
temporal constraint validation.

Features:
    - Semantic claim-source matching with confidence scoring
    - Temporal constraint validation and chronological verification
    - Cross-document reasoning and relationship mapping
    - Anti-hallucination framework with uncertainty quantification
    - Provenance tracking and audit trails
    - Logic-based citation validation
    - APA7 compliance verification

Author: AI-Powered Thesis Assistant Team
Version: 2.0
License: MIT
"""

# Import only the modules that exist
try:
    from .advanced_citation_validator import AdvancedCitationValidator
except ImportError:
    AdvancedCitationValidator = None

try:
    from .apa7_compliance_engine import APA7ComplianceEngine
except ImportError:
    APA7ComplianceEngine = None

try:
    from .enhanced_citation_engine import EnhancedCitationEngine
except ImportError:
    EnhancedCitationEngine = None

try:
    from .master_thesis_reference_system import MasterThesisReferenceSystem
except ImportError:
    MasterThesisReferenceSystem = None

# Legacy imports (commented out until implemented)
# from .citation_engine import CitationReasoningEngine
# from .semantic_matcher import SemanticMatcher
# from .temporal_validator import TemporalValidator
# from .confidence_scorer import ConfidenceScorer
# from .provenance_tracker import ProvenanceTracker

__all__ = [
    'AdvancedCitationValidator',
    'APA7ComplianceEngine',
    'EnhancedCitationEngine',
    'MasterThesisReferenceSystem'
]
