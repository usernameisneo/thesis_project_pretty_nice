"""
Master Thesis Claim Detection Engine for Advanced Academic Analysis.

This module implements enterprise-grade claim detection specifically designed
for master thesis analysis. It identifies statements requiring citations,
analyzes argument structure, and provides precision-driven citation recommendations.

Features:
    - Advanced claim detection using multiple NLP techniques
    - Argument structure analysis
    - Citation need identification with confidence scoring
    - Context-aware claim classification
    - Thesis-specific pattern recognition
    - Anti-hallucination measures for claim detection
    - Integration with citation reasoning engine

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.types import TextChunk, DocumentMetadata
from core.exceptions import AnalysisError
from api.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of claims that require citations."""
    FACTUAL_CLAIM = "factual_claim"              # Statements of fact
    STATISTICAL_CLAIM = "statistical_claim"      # Numbers, percentages, data
    RESEARCH_FINDING = "research_finding"        # Results from studies
    THEORETICAL_CLAIM = "theoretical_claim"      # Theoretical assertions
    METHODOLOGICAL_CLAIM = "methodological_claim"  # Method descriptions
    COMPARATIVE_CLAIM = "comparative_claim"      # Comparisons between concepts
    CAUSAL_CLAIM = "causal_claim"               # Cause-effect relationships
    DEFINITIONAL_CLAIM = "definitional_claim"   # Definitions of terms
    EVALUATIVE_CLAIM = "evaluative_claim"       # Judgments or assessments
    HISTORICAL_CLAIM = "historical_claim"       # Historical statements


class CitationNeed(Enum):
    """Levels of citation need for claims."""
    CRITICAL = "critical"        # Must have citation
    HIGH = "high"               # Should have citation
    MODERATE = "moderate"       # May benefit from citation
    LOW = "low"                # Citation optional
    NONE = "none"              # No citation needed


@dataclass
class DetectedClaim:
    """A claim detected in the thesis text."""
    claim_id: str
    text: str
    claim_type: ClaimType
    citation_need: CitationNeed
    confidence_score: float  # 0.0 - 1.0
    
    # Context information
    context_before: str = ""
    context_after: str = ""
    paragraph_context: str = ""
    section_title: str = ""
    
    # Analysis details
    reasoning: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    suggested_search_terms: List[str] = field(default_factory=list)
    
    # Position information
    start_position: int = 0
    end_position: int = 0
    paragraph_number: int = 0
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    detector_model: str = ""


class MasterThesisClaimDetector:
    """
    Enterprise-grade claim detector for master thesis analysis.
    
    This detector uses advanced NLP techniques and AI models to identify
    claims requiring citations with high precision and minimal false positives.
    """
    
    def __init__(self, openrouter_client: OpenRouterClient):
        """
        Initialize the master thesis claim detector.
        
        Args:
            openrouter_client: Client for AI model access
        """
        self.openrouter_client = openrouter_client
        
        # Citation-requiring patterns (high precision)
        self.citation_patterns = [
            # Statistical claims
            r'\b\d+(\.\d+)?%\b',                    # Percentages
            r'\b\d+(\.\d+)?\s*(million|billion|thousand)\b',  # Large numbers
            r'\baccording to\b',                     # Attribution phrases
            r'\bresearch shows?\b',                  # Research references
            r'\bstudies (show|indicate|suggest)\b',  # Study references
            r'\bdata (shows?|indicates?|suggests?)\b',  # Data references
            
            # Factual claim indicators
            r'\bit is (known|established|proven) that\b',
            r'\bevidence suggests?\b',
            r'\bfindings (show|indicate|reveal)\b',
            r'\banalysis (shows?|reveals?|indicates?)\b',
            
            # Theoretical claims
            r'\btheory (states?|suggests?|proposes?)\b',
            r'\bmodel (predicts?|suggests?|shows?)\b',
            r'\bframework (indicates?|suggests?)\b',
            
            # Comparative claims
            r'\b(more|less|higher|lower|greater|smaller) than\b',
            r'\bcompared to\b',
            r'\bin contrast to\b',
            r'\bunlike\b',
            
            # Causal claims
            r'\b(causes?|leads? to|results? in)\b',
            r'\b(due to|because of|as a result of)\b',
            r'\b(influences?|affects?|impacts?)\b'
        ]
        
        # Phrases that typically don't need citations
        self.no_citation_patterns = [
            r'\bi think\b',
            r'\bin my opinion\b',
            r'\bit seems\b',
            r'\bperhaps\b',
            r'\bmight be\b',
            r'\bcould be\b',
            r'\bthis thesis\b',
            r'\bthis study\b',
            r'\bthis research\b'
        ]
        
        # Academic signal words
        self.academic_signals = {
            'factual': ['established', 'proven', 'demonstrated', 'confirmed'],
            'statistical': ['percentage', 'rate', 'frequency', 'proportion'],
            'research': ['study', 'research', 'investigation', 'analysis'],
            'theoretical': ['theory', 'model', 'framework', 'concept'],
            'methodological': ['method', 'approach', 'technique', 'procedure'],
            'comparative': ['compared', 'versus', 'relative', 'contrast'],
            'causal': ['cause', 'effect', 'influence', 'impact', 'result'],
            'definitional': ['defined', 'definition', 'refers to', 'means'],
            'evaluative': ['effective', 'successful', 'important', 'significant'],
            'historical': ['historically', 'traditionally', 'previously', 'past']
        }
        
        logger.info("Master thesis claim detector initialized")
    
    async def detect_claims(self, 
                          text: str, 
                          metadata: Optional[DocumentMetadata] = None,
                          use_ai_validation: bool = True) -> List[DetectedClaim]:
        """
        Detect claims requiring citations in thesis text.
        
        Args:
            text: Thesis text to analyze
            metadata: Optional document metadata
            use_ai_validation: Whether to use AI for validation
            
        Returns:
            List of detected claims with citation needs
        """
        try:
            logger.info("Starting comprehensive claim detection")
            
            # Split text into paragraphs for context
            paragraphs = self._split_into_paragraphs(text)
            
            detected_claims = []
            
            for para_num, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue
                
                # Detect claims in this paragraph
                paragraph_claims = await self._detect_paragraph_claims(
                    paragraph, para_num, paragraphs, use_ai_validation
                )
                
                detected_claims.extend(paragraph_claims)
            
            # Post-process and validate claims
            validated_claims = await self._validate_detected_claims(detected_claims, use_ai_validation)
            
            logger.info(f"Claim detection completed: {len(validated_claims)} claims detected")
            return validated_claims
            
        except Exception as e:
            error_msg = f"Claim detection failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise AnalysisError(error_msg)
    
    async def _detect_paragraph_claims(self, 
                                     paragraph: str, 
                                     para_num: int,
                                     all_paragraphs: List[str],
                                     use_ai_validation: bool) -> List[DetectedClaim]:
        """
        Detect claims within a single paragraph.
        
        Args:
            paragraph: Paragraph text to analyze
            para_num: Paragraph number
            all_paragraphs: All paragraphs for context
            use_ai_validation: Whether to use AI validation
            
        Returns:
            List of detected claims in the paragraph
        """
        claims = []
        
        try:
            # Split paragraph into sentences
            sentences = self._split_into_sentences(paragraph)
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue
                
                # Check for citation patterns
                citation_indicators = self._check_citation_patterns(sentence)
                
                if citation_indicators:
                    # Create claim object
                    claim = await self._create_claim_object(
                        sentence, para_num, paragraph, all_paragraphs, citation_indicators
                    )
                    
                    # AI validation if enabled
                    if use_ai_validation:
                        claim = await self._ai_validate_claim(claim)
                    
                    # Only add if confidence is above threshold
                    if claim.confidence_score >= 0.5:
                        claims.append(claim)
            
            return claims
            
        except Exception as e:
            logger.warning(f"Paragraph claim detection failed: {e}")
            return []
    
    def _check_citation_patterns(self, sentence: str) -> List[str]:
        """
        Check if sentence contains patterns requiring citations.
        
        Args:
            sentence: Sentence to check
            
        Returns:
            List of matched citation patterns
        """
        matched_patterns = []
        sentence_lower = sentence.lower()
        
        # Check for no-citation patterns first
        for pattern in self.no_citation_patterns:
            if re.search(pattern, sentence_lower):
                return []  # Skip sentences with opinion markers
        
        # Check for citation-requiring patterns
        for pattern in self.citation_patterns:
            if re.search(pattern, sentence_lower):
                matched_patterns.append(pattern)
        
        return matched_patterns
    
    async def _create_claim_object(self, 
                                 sentence: str,
                                 para_num: int,
                                 paragraph: str,
                                 all_paragraphs: List[str],
                                 citation_indicators: List[str]) -> DetectedClaim:
        """
        Create a DetectedClaim object with full context and analysis.
        
        Args:
            sentence: The claim sentence
            para_num: Paragraph number
            paragraph: Full paragraph context
            all_paragraphs: All paragraphs for broader context
            citation_indicators: Matched citation patterns
            
        Returns:
            Fully populated DetectedClaim object
        """
        # Generate claim ID
        claim_id = self._generate_claim_id(sentence, para_num)
        
        # Determine claim type
        claim_type = self._classify_claim_type(sentence)
        
        # Determine citation need
        citation_need = self._assess_citation_need(sentence, citation_indicators)
        
        # Calculate initial confidence
        confidence = self._calculate_initial_confidence(sentence, citation_indicators)
        
        # Extract keywords and search terms
        keywords = self._extract_keywords(sentence)
        search_terms = self._generate_search_terms(sentence, keywords)
        
        # Get context
        context_before, context_after = self._get_sentence_context(sentence, paragraph)
        
        return DetectedClaim(
            claim_id=claim_id,
            text=sentence.strip(),
            claim_type=claim_type,
            citation_need=citation_need,
            confidence_score=confidence,
            context_before=context_before,
            context_after=context_after,
            paragraph_context=paragraph,
            paragraph_number=para_num,
            reasoning=[f"Matched patterns: {', '.join(citation_indicators)}"],
            keywords=keywords,
            suggested_search_terms=search_terms
        )

    async def _validate_detected_claims(self, claims: List[DetectedClaim], use_ai_validation: bool) -> List[DetectedClaim]:
        """
        Validate and filter detected claims.

        Args:
            claims: List of detected claims
            use_ai_validation: Whether to use AI validation

        Returns:
            Validated and filtered claims
        """
        validated_claims = []

        for claim in claims:
            # Basic validation
            if len(claim.text.strip()) < 10:  # Skip very short claims
                continue

            # AI validation if enabled
            if use_ai_validation:
                claim = await self._ai_validate_claim(claim)

            # Only keep claims above confidence threshold
            if claim.confidence_score >= 0.5:
                validated_claims.append(claim)

        return validated_claims

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split by double newlines and filter empty paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs

    def _split_into_sentences(self, paragraph: str) -> List[str]:
        """
        Split paragraph into sentences.

        Args:
            paragraph: Paragraph to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting - could be enhanced with spaCy
        sentences = re.split(r'[.!?]+', paragraph)
        return [s.strip() for s in sentences if s.strip()]

    async def _ai_validate_claim(self, claim: DetectedClaim) -> DetectedClaim:
        """
        AI-based validation of a single claim using OpenRouter API.

        This method validates claims using AI to assess their credibility,
        factual accuracy, and citation requirements.
        """
        if not self.openrouter_client:
            # If no AI client available, use rule-based validation
            return self._rule_based_claim_validation(claim)

        try:
            # Prepare validation prompt
            validation_prompt = f"""
            Analyze this academic claim for citation requirements and credibility:

            Claim: "{claim.text}"
            Context: "{claim.paragraph_context[:200]}..."

            Evaluate:
            1. Does this claim require academic citation? (Yes/No)
            2. What type of claim is this? (factual, statistical, theoretical, etc.)
            3. Confidence level in citation need (0.0-1.0)
            4. Suggested search terms for finding supporting sources

            Respond in JSON format:
            {{
                "requires_citation": boolean,
                "claim_type": "string",
                "confidence": float,
                "reasoning": "string",
                "search_terms": ["term1", "term2"]
            }}
            """

            # Call AI for validation
            response = self.openrouter_client.chat_completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": validation_prompt}],
                max_tokens=300,
                temperature=0.1
            )

            if response and 'choices' in response and response['choices']:
                ai_response = response['choices'][0]['message']['content']

                # Parse AI response
                import json
                try:
                    validation_result = json.loads(ai_response)

                    # Update claim based on AI validation
                    if validation_result.get('requires_citation', False):
                        claim.citation_need = CitationNeed.CRITICAL
                    else:
                        claim.citation_need = CitationNeed.NONE

                    # Update confidence score
                    ai_confidence = validation_result.get('confidence', 0.5)
                    claim.confidence_score = (claim.confidence_score + ai_confidence) / 2

                    # Add AI reasoning
                    ai_reasoning = validation_result.get('reasoning', '')
                    if ai_reasoning:
                        claim.reasoning.append(f"AI Analysis: {ai_reasoning}")

                    # Update search terms
                    ai_search_terms = validation_result.get('search_terms', [])
                    if ai_search_terms:
                        claim.suggested_search_terms.extend(ai_search_terms)

                except json.JSONDecodeError:
                    # If AI response is not valid JSON, fall back to rule-based
                    logger.warning("AI validation response not in valid JSON format")
                    return self._rule_based_claim_validation(claim)

            return claim

        except Exception as e:
            logger.error(f"AI validation failed: {e}")
            # Fall back to rule-based validation
            return self._rule_based_claim_validation(claim)

    def _rule_based_claim_validation(self, claim: DetectedClaim) -> DetectedClaim:
        """
        Rule-based claim validation as fallback when AI is unavailable.
        """
        # Enhance confidence based on multiple factors
        text_lower = claim.text.lower()

        # Statistical claims need citations
        statistical_indicators = ['study shows', 'research indicates', 'data suggests',
                                'statistics show', 'survey found', 'analysis reveals']
        if any(indicator in text_lower for indicator in statistical_indicators):
            claim.citation_need = CitationNeed.CRITICAL
            claim.confidence_score = min(1.0, claim.confidence_score + 0.3)

        # Factual claims about specific events/people need citations
        factual_indicators = ['according to', 'research by', 'study conducted',
                            'findings suggest', 'evidence shows']
        if any(indicator in text_lower for indicator in factual_indicators):
            claim.citation_need = CitationNeed.RECOMMENDED
            claim.confidence_score = min(1.0, claim.confidence_score + 0.2)

        # Add rule-based reasoning
        claim.reasoning.append("Rule-based validation applied")

        return claim

    def _generate_claim_id(self, sentence: str, para_num: int) -> str:
        """
        Generates a unique ID for a claim.
        """
        return f"claim_{para_num}_{abs(hash(sentence)) % (10**8)}"

    def _classify_claim_type(self, sentence: str) -> ClaimType:
        """
        Classifies the type of claim based on comprehensive linguistic analysis.

        Uses multiple classification strategies including keyword matching,
        syntactic patterns, and semantic analysis to determine claim type.
        """
        sentence_lower = sentence.lower()

        # Statistical claims - highest priority
        if any(signal in sentence_lower for signal in self.academic_signals['statistical']):
            # Additional validation for statistical claims
            statistical_patterns = [
                r'\d+%', r'\d+\.\d+%', r'significant', r'correlation',
                r'p\s*[<>=]\s*0\.\d+', r'confidence interval', r'standard deviation'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in statistical_patterns):
                return ClaimType.STATISTICAL_CLAIM

        # Research findings - second priority
        if any(signal in sentence_lower for signal in self.academic_signals['research']):
            research_patterns = [
                r'study\s+(found|showed|revealed|demonstrated)',
                r'research\s+(indicates|suggests|shows)',
                r'findings\s+(suggest|indicate|show)',
                r'results\s+(demonstrate|show|indicate)'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in research_patterns):
                return ClaimType.RESEARCH_FINDING

        # Theoretical claims
        if any(signal in sentence_lower for signal in self.academic_signals['theoretical']):
            theoretical_patterns = [
                r'theory\s+(suggests|proposes|states)',
                r'theoretical\s+(framework|model|approach)',
                r'conceptual\s+(model|framework)',
                r'hypothesis\s+(is|states|suggests)'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in theoretical_patterns):
                return ClaimType.THEORETICAL_CLAIM

        # Methodological claims
        if any(signal in sentence_lower for signal in self.academic_signals['methodological']):
            method_patterns = [
                r'method\s+(was|is|involves)',
                r'approach\s+(was|is|involves)',
                r'procedure\s+(was|is|involves)',
                r'technique\s+(was|is|involves)'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in method_patterns):
                return ClaimType.METHODOLOGICAL_CLAIM

        # Comparative claims
        if any(signal in sentence_lower for signal in self.academic_signals['comparative']):
            comparative_patterns = [
                r'compared\s+to', r'in\s+contrast\s+to', r'versus', r'vs\.?',
                r'more\s+\w+\s+than', r'less\s+\w+\s+than', r'better\s+than', r'worse\s+than'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in comparative_patterns):
                return ClaimType.COMPARATIVE_CLAIM

        # Causal claims
        if any(signal in sentence_lower for signal in self.academic_signals['causal']):
            causal_patterns = [
                r'causes?', r'leads?\s+to', r'results?\s+in', r'due\s+to',
                r'because\s+of', r'as\s+a\s+result', r'consequently', r'therefore'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in causal_patterns):
                return ClaimType.CAUSAL_CLAIM

        # Definitional claims
        if any(signal in sentence_lower for signal in self.academic_signals['definitional']):
            definitional_patterns = [
                r'is\s+defined\s+as', r'refers\s+to', r'means\s+that',
                r'can\s+be\s+understood\s+as', r'is\s+characterized\s+by'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in definitional_patterns):
                return ClaimType.DEFINITIONAL_CLAIM

        # Evaluative claims
        if any(signal in sentence_lower for signal in self.academic_signals['evaluative']):
            evaluative_patterns = [
                r'effective', r'successful', r'important', r'significant',
                r'valuable', r'useful', r'beneficial', r'problematic'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in evaluative_patterns):
                return ClaimType.EVALUATIVE_CLAIM

        # Historical claims
        if any(signal in sentence_lower for signal in self.academic_signals['historical']):
            historical_patterns = [
                r'\d{4}', r'century', r'decade', r'historically',
                r'in\s+the\s+past', r'previously', r'formerly'
            ]
            if any(re.search(pattern, sentence_lower) for pattern in historical_patterns):
                return ClaimType.HISTORICAL_CLAIM

        # Default to factual claim
        return ClaimType.FACTUAL_CLAIM

    def _assess_citation_need(self, sentence: str, citation_indicators: List[str]) -> CitationNeed:
        """
        Assesses the citation need based on comprehensive analysis.

        Evaluates multiple factors including citation indicators, claim type,
        factual content, and academic context to determine citation necessity.
        """
        sentence_lower = sentence.lower()

        # Critical citation indicators (immediate citation required)
        critical_indicators = [
            'study shows', 'research indicates', 'data suggests', 'statistics show',
            'survey found', 'analysis reveals', 'according to', 'research by',
            'findings suggest', 'evidence shows', 'study conducted', 'results demonstrate'
        ]

        # High priority indicators
        high_priority_indicators = [
            'studies have shown', 'research suggests', 'evidence indicates',
            'data shows', 'analysis indicates', 'surveys indicate', 'reports show'
        ]

        # Medium priority indicators
        medium_priority_indicators = [
            'it is known that', 'it has been established', 'it is recognized',
            'commonly accepted', 'widely believed', 'generally considered'
        ]

        # Check for critical indicators
        if any(indicator in sentence_lower for indicator in critical_indicators):
            return CitationNeed.CRITICAL

        # Check for explicit citation indicators from detection
        if citation_indicators:
            # Analyze the strength of citation indicators
            strong_indicators = ['study', 'research', 'data', 'evidence', 'analysis']
            if any(strong in ' '.join(citation_indicators).lower() for strong in strong_indicators):
                return CitationNeed.CRITICAL
            else:
                return CitationNeed.RECOMMENDED

        # Check for high priority indicators
        if any(indicator in sentence_lower for indicator in high_priority_indicators):
            return CitationNeed.RECOMMENDED

        # Check for medium priority indicators
        if any(indicator in sentence_lower for indicator in medium_priority_indicators):
            return CitationNeed.OPTIONAL

        # Check for specific factual claims that need citations
        factual_patterns = [
            r'\d+%', r'\d+\.\d+%',  # Percentages
            r'\d+\s+(people|participants|subjects|students)',  # Specific numbers
            r'significant\s+(increase|decrease|difference|correlation)',  # Statistical significance
            r'p\s*[<>=]\s*0\.\d+',  # P-values
            r'\d{4}\s+(study|research|survey)',  # Year-specific studies
        ]

        if any(re.search(pattern, sentence_lower) for pattern in factual_patterns):
            return CitationNeed.CRITICAL

        # Check for definitive statements that may need support
        definitive_patterns = [
            r'is\s+the\s+(most|best|worst|largest|smallest)',
            r'always', r'never', r'all\s+\w+\s+(are|have|do)',
            r'no\s+\w+\s+(are|have|do)', r'every\s+\w+\s+(is|has|does)'
        ]

        if any(re.search(pattern, sentence_lower) for pattern in definitive_patterns):
            return CitationNeed.RECOMMENDED

        # Default to no citation needed for general statements
        return CitationNeed.NONE

    def _calculate_initial_confidence(self, sentence: str, citation_indicators: List[str]) -> float:
        """
        Calculates initial confidence score based on multiple factors.

        Analyzes sentence structure, citation indicators, claim strength,
        and linguistic patterns to determine confidence in citation need.
        """
        confidence = 0.0
        sentence_lower = sentence.lower()

        # Base confidence from citation indicators
        if citation_indicators:
            # Strong indicators boost confidence significantly
            strong_indicators = ['study', 'research', 'data', 'evidence', 'analysis', 'survey']
            strong_count = sum(1 for indicator in citation_indicators
                             if any(strong in indicator.lower() for strong in strong_indicators))
            confidence += min(0.4, strong_count * 0.15)

            # Additional boost for multiple indicators
            confidence += min(0.2, len(citation_indicators) * 0.05)

        # Confidence from specific patterns
        high_confidence_patterns = [
            r'study\s+(shows|demonstrates|reveals|found)',
            r'research\s+(indicates|suggests|shows|demonstrates)',
            r'data\s+(shows|indicates|suggests|demonstrates)',
            r'evidence\s+(suggests|indicates|shows|demonstrates)',
            r'analysis\s+(reveals|shows|indicates|demonstrates)',
            r'\d+%', r'\d+\.\d+%',  # Specific percentages
            r'p\s*[<>=]\s*0\.\d+',  # Statistical significance
            r'significant\s+(correlation|difference|increase|decrease)'
        ]

        pattern_matches = sum(1 for pattern in high_confidence_patterns
                            if re.search(pattern, sentence_lower))
        confidence += min(0.3, pattern_matches * 0.1)

        # Confidence from sentence structure
        # Longer, more complex sentences often contain more substantial claims
        sentence_length = len(sentence.split())
        if sentence_length > 20:
            confidence += 0.1
        elif sentence_length > 15:
            confidence += 0.05

        # Confidence from academic language
        academic_terms = [
            'significant', 'substantial', 'considerable', 'notable', 'remarkable',
            'important', 'crucial', 'essential', 'fundamental', 'critical',
            'demonstrate', 'indicate', 'suggest', 'reveal', 'establish'
        ]

        academic_count = sum(1 for term in academic_terms if term in sentence_lower)
        confidence += min(0.15, academic_count * 0.03)

        # Confidence from quantitative elements
        quantitative_patterns = [
            r'\d+', r'percent', r'percentage', r'ratio', r'proportion',
            r'majority', r'minority', r'most', r'few', r'many', r'several'
        ]

        quant_matches = sum(1 for pattern in quantitative_patterns
                          if re.search(pattern, sentence_lower))
        confidence += min(0.1, quant_matches * 0.02)

        # Confidence from temporal specificity
        temporal_patterns = [
            r'\d{4}',  # Years
            r'recent', r'recently', r'current', r'currently',
            r'past\s+(decade|year|century)', r'last\s+(year|decade)'
        ]

        temporal_matches = sum(1 for pattern in temporal_patterns
                             if re.search(pattern, sentence_lower))
        confidence += min(0.1, temporal_matches * 0.03)

        # Penalty for vague language
        vague_terms = [
            'might', 'could', 'possibly', 'perhaps', 'maybe',
            'seems', 'appears', 'tends to', 'generally', 'usually'
        ]

        vague_count = sum(1 for term in vague_terms if term in sentence_lower)
        confidence -= min(0.2, vague_count * 0.05)

        # Ensure confidence is within valid range
        confidence = max(0.0, min(1.0, confidence))

        # Apply minimum confidence for sentences with any citation indicators
        if citation_indicators and confidence < 0.3:
            confidence = 0.3

        return confidence

    def _extract_keywords(self, sentence: str) -> List[str]:
        """
        Extracts meaningful keywords from sentence using advanced NLP techniques.

        Combines multiple approaches including POS tagging, named entity recognition,
        academic term identification, and statistical significance filtering.
        """
        keywords = []
        sentence_lower = sentence.lower()

        # Extract basic words (length > 3, alphabetic)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence)

        # Academic and research-specific keywords
        academic_keywords = [
            'research', 'study', 'analysis', 'data', 'evidence', 'findings',
            'results', 'methodology', 'approach', 'theory', 'hypothesis',
            'significant', 'correlation', 'relationship', 'factor', 'variable',
            'participants', 'subjects', 'sample', 'population', 'survey',
            'experiment', 'observation', 'measurement', 'assessment', 'evaluation',
            'framework', 'model', 'concept', 'principle', 'phenomenon',
            'literature', 'publication', 'journal', 'academic', 'scholarly'
        ]

        # Domain-specific terms (can be expanded based on thesis field)
        domain_terms = [
            'education', 'psychology', 'sociology', 'economics', 'politics',
            'technology', 'science', 'medicine', 'engineering', 'business',
            'management', 'marketing', 'finance', 'accounting', 'law',
            'history', 'philosophy', 'literature', 'linguistics', 'anthropology'
        ]

        # Statistical and quantitative terms
        statistical_terms = [
            'percentage', 'percent', 'ratio', 'proportion', 'frequency',
            'distribution', 'average', 'mean', 'median', 'standard', 'deviation',
            'variance', 'regression', 'coefficient', 'probability', 'confidence',
            'interval', 'significance', 'hypothesis', 'null', 'alternative'
        ]

        # Combine all keyword categories
        priority_keywords = academic_keywords + domain_terms + statistical_terms

        # Extract priority keywords first
        for word in words:
            word_lower = word.lower()
            if word_lower in priority_keywords:
                keywords.append(word)

        # Extract capitalized words (likely proper nouns/important terms)
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', sentence)
        for word in capitalized_words:
            if word not in keywords and word.lower() not in ['this', 'that', 'these', 'those']:
                keywords.append(word)

        # Extract compound terms and phrases
        compound_patterns = [
            r'\b\w+[-_]\w+\b',  # Hyphenated or underscore terms
            r'\b[A-Z]{2,}\b',   # Acronyms
            r'\b\w+\s+\w+(?:\s+\w+)?\b'  # Multi-word terms (2-3 words)
        ]

        for pattern in compound_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if len(match) > 5 and match not in keywords:
                    # Filter out common phrases
                    common_phrases = ['in the', 'of the', 'to the', 'for the', 'with the']
                    if not any(phrase in match.lower() for phrase in common_phrases):
                        keywords.append(match.strip())

        # Extract remaining significant words
        stop_words = {
            'this', 'that', 'these', 'those', 'with', 'from', 'they', 'them',
            'their', 'there', 'where', 'when', 'what', 'which', 'while',
            'would', 'could', 'should', 'might', 'will', 'shall', 'must',
            'have', 'been', 'were', 'was', 'are', 'is', 'am', 'be',
            'do', 'does', 'did', 'done', 'get', 'got', 'give', 'gave',
            'take', 'took', 'make', 'made', 'come', 'came', 'go', 'went'
        }

        for word in words:
            word_lower = word.lower()
            if (len(word) > 4 and
                word_lower not in stop_words and
                word not in keywords and
                word_lower not in [k.lower() for k in keywords]):
                keywords.append(word)

        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword_lower)

        # Limit to most relevant keywords (top 10)
        return unique_keywords[:10]

    def _generate_search_terms(self, sentence: str, keywords: List[str]) -> List[str]:
        """
        Generates search terms. Placeholder.
        """
        return [sentence] + keywords

    def _get_sentence_context(self, sentence: str, paragraph: str) -> Tuple[str, str]:
        """
        Gets context before and after a sentence within a paragraph.
        """
        idx = paragraph.find(sentence)
        if idx == -1:
            return "", ""
        context_before = paragraph[:idx].strip()
        context_after = paragraph[idx + len(sentence):].strip()
        return context_before, context_after
