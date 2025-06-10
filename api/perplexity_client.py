"""
Perplexity API Client - Production-Grade Real-Time Research Intelligence.

This module provides complete integration with the Perplexity API for real-time
research, fact-checking, and academic information retrieval. Implements full
API coverage with advanced query optimization and response validation.

Features:
    - Complete Perplexity API integration
    - Real-time research and fact-checking
    - Academic query optimization
    - Citation extraction and validation
    - Multi-model support (sonar-small, sonar-medium, sonar-large)
    - Response quality assessment
    - Source verification and ranking
    - Rate limiting and error handling
    - Response caching for performance
    - Batch processing capabilities

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
import re

from core.exceptions import APIError, ValidationError
from core.types import DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class PerplexitySource:
    """Source information from Perplexity response."""
    url: str
    title: str
    snippet: str
    domain: str
    relevance_score: float
    publication_date: Optional[datetime] = None
    author: Optional[str] = None
    source_type: str = "web"  # web, academic, news, etc.
    credibility_score: float = 0.0


@dataclass
class PerplexityResponse:
    """Complete Perplexity API response."""
    query: str
    answer: str
    sources: List[PerplexitySource]
    model_used: str
    response_time: float
    
    # Quality metrics
    confidence_score: float
    factual_accuracy: float
    source_quality: float
    citation_count: int
    
    # Metadata
    response_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    usage_tokens: int = 0
    
    # Academic analysis
    academic_sources: List[PerplexitySource] = field(default_factory=list)
    peer_reviewed_count: int = 0
    recent_sources_count: int = 0


class PerplexityClient:
    """
    Production-grade Perplexity API client.
    
    Provides complete access to Perplexity's real-time research capabilities
    with enterprise-level reliability, performance, and academic optimization.
    """
    
    def __init__(self, 
                 api_key: str,
                 cache_directory: str = "perplexity_cache",
                 rate_limit_requests_per_minute: float = 60.0,
                 max_retries: int = 3,
                 timeout_seconds: int = 60,
                 default_model: str = "sonar-medium-online"):
        """
        Initialize Perplexity client.
        
        Args:
            api_key: Perplexity API key (required)
            cache_directory: Directory for response caching
            rate_limit_requests_per_minute: Rate limiting configuration
            max_retries: Maximum retry attempts for failed requests
            timeout_seconds: Request timeout in seconds
            default_model: Default model to use for queries
        """
        if not api_key:
            raise ValueError("Perplexity API key is required")
        
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.rate_limit = rate_limit_requests_per_minute
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.default_model = default_model
        
        # Available models
        self.available_models = {
            "sonar-small-online": {"max_tokens": 4096, "cost_per_1k": 0.0005, "speed": "fast"},
            "sonar-medium-online": {"max_tokens": 4096, "cost_per_1k": 0.001, "speed": "medium"},
            "sonar-large-online": {"max_tokens": 4096, "cost_per_1k": 0.003, "speed": "slow"},
            "sonar-small-chat": {"max_tokens": 16384, "cost_per_1k": 0.0002, "speed": "fast"},
            "sonar-medium-chat": {"max_tokens": 16384, "cost_per_1k": 0.0006, "speed": "medium"},
            "sonar-large-chat": {"max_tokens": 16384, "cost_per_1k": 0.002, "speed": "slow"}
        }
        
        # Cache setup
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=6)  # Cache for 6 hours (research changes)
        
        # Rate limiting
        self.last_request_time = datetime.now()
        self.request_interval = 60.0 / rate_limit_requests_per_minute
        
        # Session will be created when needed
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Academic domain patterns
        self.academic_domains = {
            'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
            'jstor.org', 'springer.com', 'sciencedirect.com', 'ieee.org',
            'acm.org', 'nature.com', 'science.org', 'cell.com', 'plos.org',
            'bmj.com', 'nejm.org', 'thelancet.com', 'wiley.com', 'elsevier.com'
        }
        
        logger.info(f"Perplexity client initialized with model: {default_model}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if not self.session:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'AI-Powered-Thesis-Assistant/2.0'
            }
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=self.timeout
            )
    
    async def research_query(self, 
                           query: str,
                           model: Optional[str] = None,
                           max_tokens: Optional[int] = None,
                           temperature: float = 0.2,
                           return_citations: bool = True,
                           return_images: bool = False,
                           search_domain_filter: Optional[List[str]] = None,
                           search_recency_filter: Optional[str] = None) -> PerplexityResponse:
        """
        Perform research query using Perplexity.
        
        Args:
            query: Research question or query
            model: Model to use (defaults to default_model)
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-2.0)
            return_citations: Whether to return source citations
            return_images: Whether to return images
            search_domain_filter: Filter sources by domains
            search_recency_filter: Filter by recency (day, week, month, year)
            
        Returns:
            Complete Perplexity response with sources and analysis
        """
        try:
            start_time = datetime.now()
            
            # Use default model if not specified
            if model is None:
                model = self.default_model
            
            # Validate model
            if model not in self.available_models:
                raise ValueError(f"Invalid model: {model}. Available: {list(self.available_models.keys())}")
            
            # Set max_tokens based on model if not specified
            if max_tokens is None:
                max_tokens = min(2048, self.available_models[model]["max_tokens"])
            
            # Check cache first
            cache_key = self._generate_cache_key(query, model, max_tokens, temperature)
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                logger.debug(f"Using cached research result for query: {query}")
                return self._parse_cached_response(cached_result, query, start_time)
            
            # Prepare request
            request_data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise academic research assistant. Provide accurate, well-sourced information with proper citations. Focus on peer-reviewed sources when available."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "return_citations": return_citations,
                "return_images": return_images
            }
            
            # Add search filters if specified
            if search_domain_filter:
                request_data["search_domain_filter"] = search_domain_filter
            if search_recency_filter:
                request_data["search_recency_filter"] = search_recency_filter
            
            # Make API request
            response_data = await self._make_request("chat/completions", request_data)
            
            # Parse response
            perplexity_response = await self._parse_research_response(
                response_data, query, model, start_time
            )
            
            # Cache response
            await self._cache_response(cache_key, {
                'response_data': response_data,
                'perplexity_response': perplexity_response.__dict__
            })
            
            logger.info(f"Research query completed: {len(perplexity_response.sources)} sources found")
            return perplexity_response
            
        except Exception as e:
            error_msg = f"Research query failed for '{query}': {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    async def fact_check_claim(self, 
                             claim: str,
                             context: Optional[str] = None,
                             require_academic_sources: bool = True) -> PerplexityResponse:
        """
        Fact-check a specific claim using Perplexity.
        
        Args:
            claim: The claim to fact-check
            context: Additional context for the claim
            require_academic_sources: Whether to prioritize academic sources
            
        Returns:
            Fact-checking response with verification and sources
        """
        try:
            # Construct fact-checking query
            if context:
                query = f"Fact-check this claim with academic sources: '{claim}'. Context: {context}. Provide verification with peer-reviewed sources if available."
            else:
                query = f"Fact-check this claim with academic sources: '{claim}'. Provide verification with peer-reviewed sources if available."
            
            # Use academic domain filter if required
            domain_filter = None
            if require_academic_sources:
                domain_filter = list(self.academic_domains)
            
            response = await self.research_query(
                query=query,
                model="sonar-medium-online",  # Good balance for fact-checking
                temperature=0.1,  # Low temperature for factual accuracy
                search_domain_filter=domain_filter,
                search_recency_filter="year"  # Recent sources for current facts
            )
            
            # Enhance response with fact-checking analysis
            response.factual_accuracy = await self._assess_factual_accuracy(response)
            response.source_quality = await self._assess_source_quality(response)
            
            logger.info(f"Fact-check completed for claim: {claim[:100]}...")
            return response
            
        except Exception as e:
            error_msg = f"Fact-checking failed for claim '{claim}': {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    async def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract academic metadata from text using Perplexity.
        
        Args:
            text: Text to extract metadata from
            
        Returns:
            Extracted metadata dictionary
        """
        try:
            query = f"""Extract academic metadata from this text in JSON format:

{text[:2000]}

Extract:
- title
- authors (list)
- year
- journal/venue
- doi
- abstract
- keywords
- publication_type

Return only valid JSON."""
            
            response = await self.research_query(
                query=query,
                model="sonar-small-chat",  # Fast model for extraction
                temperature=0.0,  # Deterministic for metadata
                max_tokens=1000
            )
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response.answer, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group())
                    logger.info("Metadata extraction completed successfully")
                    return metadata
                else:
                    logger.warning("No JSON found in metadata extraction response")
                    return {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse metadata JSON: {e}")
                return {}
                
        except Exception as e:
            error_msg = f"Metadata extraction failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    async def batch_research_queries(self, queries: List[str], model: Optional[str] = None) -> List[PerplexityResponse]:
        """
        Process multiple research queries in batch with rate limiting.

        Args:
            queries: List of research queries
            model: Model to use for all queries

        Returns:
            List of Perplexity responses
        """
        try:
            responses = []

            for i, query in enumerate(queries):
                logger.info(f"Processing batch query {i+1}/{len(queries)}: {query[:50]}...")

                response = await self.research_query(query, model=model)
                responses.append(response)

                # Rate limiting between requests
                if i < len(queries) - 1:  # Don't wait after last query
                    await asyncio.sleep(self.request_interval)

            logger.info(f"Batch research completed: {len(responses)} queries processed")
            return responses

        except Exception as e:
            error_msg = f"Batch research failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make rate-limited API request with retry logic.

        Args:
            endpoint: API endpoint
            data: Request data

        Returns:
            API response data
        """
        await self._ensure_session()

        # Rate limiting
        await self._enforce_rate_limit()

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(url, json=data) as response:
                    return await self._handle_response(response)

            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    raise APIError(f"Request failed after {self.max_retries} retries: {e}")

                # Exponential backoff
                wait_time = (2 ** attempt) * 2.0
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Handle API response with proper error checking.

        Args:
            response: aiohttp response object

        Returns:
            Parsed response data
        """
        if response.status == 200:
            return await response.json()
        elif response.status == 429:
            # Rate limit exceeded
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds")
            await asyncio.sleep(retry_after)
            raise aiohttp.ClientError("Rate limit exceeded")
        elif response.status == 401:
            raise APIError("Invalid API key")
        elif response.status == 400:
            error_text = await response.text()
            raise APIError(f"Bad request: {error_text}")
        elif response.status >= 500:
            raise aiohttp.ClientError(f"Server error: {response.status}")
        else:
            error_text = await response.text()
            raise APIError(f"API error {response.status}: {error_text}")

    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        now = datetime.now()
        time_since_last = (now - self.last_request_time).total_seconds()

        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = datetime.now()

    def _generate_cache_key(self, query: str, model: str, max_tokens: int, temperature: float) -> str:
        """Generate cache key for request."""
        cache_data = f"{query}:{model}:{max_tokens}:{temperature}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if not cache_file.exists():
                return None

            # Check if cache is expired
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time > self.cache_ttl:
                cache_file.unlink()  # Remove expired cache
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    async def _cache_response(self, cache_key: str, data: Dict[str, Any]):
        """Cache API response."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def _parse_cached_response(self, cached_data: Dict[str, Any], query: str, start_time: datetime) -> PerplexityResponse:
        """Parse cached response data."""
        response_data = cached_data.get('perplexity_response', {})

        # Reconstruct PerplexityResponse from cached data
        sources = []
        for source_data in response_data.get('sources', []):
            source = PerplexitySource(**source_data)
            sources.append(source)

        academic_sources = []
        for source_data in response_data.get('academic_sources', []):
            source = PerplexitySource(**source_data)
            academic_sources.append(source)

        response = PerplexityResponse(
            query=query,
            answer=response_data.get('answer', ''),
            sources=sources,
            model_used=response_data.get('model_used', ''),
            response_time=(datetime.now() - start_time).total_seconds(),
            confidence_score=response_data.get('confidence_score', 0.0),
            factual_accuracy=response_data.get('factual_accuracy', 0.0),
            source_quality=response_data.get('source_quality', 0.0),
            citation_count=response_data.get('citation_count', 0),
            response_id=response_data.get('response_id', ''),
            usage_tokens=response_data.get('usage_tokens', 0),
            academic_sources=academic_sources,
            peer_reviewed_count=response_data.get('peer_reviewed_count', 0),
            recent_sources_count=response_data.get('recent_sources_count', 0)
        )

        return response

    async def _parse_research_response(self, response_data: Dict[str, Any], query: str, model: str, start_time: datetime) -> PerplexityResponse:
        """Parse API response into PerplexityResponse object."""
        try:
            # Extract main response
            choice = response_data.get('choices', [{}])[0]
            message = choice.get('message', {})
            answer = message.get('content', '')

            # Extract citations/sources
            citations = response_data.get('citations', [])
            sources = []
            academic_sources = []

            for i, citation in enumerate(citations):
                source = PerplexitySource(
                    url=citation.get('url', ''),
                    title=citation.get('title', f'Source {i+1}'),
                    snippet=citation.get('snippet', ''),
                    domain=self._extract_domain(citation.get('url', '')),
                    relevance_score=citation.get('relevance_score', 0.5),
                    source_type=self._classify_source_type(citation.get('url', ''))
                )

                sources.append(source)

                # Check if academic source
                if self._is_academic_source(source.domain):
                    academic_sources.append(source)

            # Calculate metrics
            response_time = (datetime.now() - start_time).total_seconds()
            confidence_score = await self._calculate_confidence_score(answer, sources)

            # Count peer-reviewed and recent sources
            peer_reviewed_count = len([s for s in academic_sources if self._is_peer_reviewed_domain(s.domain)])
            recent_sources_count = len([s for s in sources if self._is_recent_source(s)])

            response = PerplexityResponse(
                query=query,
                answer=answer,
                sources=sources,
                model_used=model,
                response_time=response_time,
                confidence_score=confidence_score,
                factual_accuracy=0.0,  # Will be calculated if needed
                source_quality=await self._assess_source_quality_from_sources(sources),
                citation_count=len(sources),
                response_id=response_data.get('id', ''),
                usage_tokens=response_data.get('usage', {}).get('total_tokens', 0),
                academic_sources=academic_sources,
                peer_reviewed_count=peer_reviewed_count,
                recent_sources_count=recent_sources_count
            )

            return response

        except Exception as e:
            logger.error(f"Failed to parse research response: {e}")
            # Return minimal response on parse failure
            return PerplexityResponse(
                query=query,
                answer="Response parsing failed",
                sources=[],
                model_used=model,
                response_time=(datetime.now() - start_time).total_seconds(),
                confidence_score=0.0,
                factual_accuracy=0.0,
                source_quality=0.0,
                citation_count=0,
                response_id=""
            )

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""

    def _classify_source_type(self, url: str) -> str:
        """Classify source type based on URL."""
        domain = self._extract_domain(url)

        if domain in self.academic_domains:
            return "academic"
        elif any(news in domain for news in ['news', 'times', 'post', 'guardian', 'reuters', 'bbc']):
            return "news"
        elif any(gov in domain for gov in ['.gov', '.edu']):
            return "institutional"
        else:
            return "web"

    def _is_academic_source(self, domain: str) -> bool:
        """Check if domain is academic."""
        return domain in self.academic_domains or '.edu' in domain

    def _is_peer_reviewed_domain(self, domain: str) -> bool:
        """Check if domain typically hosts peer-reviewed content."""
        peer_reviewed_domains = {
            'pubmed.ncbi.nlm.nih.gov', 'springer.com', 'sciencedirect.com',
            'ieee.org', 'acm.org', 'nature.com', 'science.org', 'cell.com',
            'plos.org', 'bmj.com', 'nejm.org', 'thelancet.com'
        }
        return domain in peer_reviewed_domains

    def _is_recent_source(self, source: PerplexitySource) -> bool:
        """Check if source is recent (within last 2 years)."""
        if source.publication_date:
            return (datetime.now() - source.publication_date).days <= 730
        return False  # Assume not recent if no date

    async def _calculate_confidence_score(self, answer: str, sources: List[PerplexitySource]) -> float:
        """Calculate confidence score based on answer and sources."""
        score = 0.5  # Base score

        # Boost for academic sources
        academic_ratio = len([s for s in sources if self._is_academic_source(s.domain)]) / max(len(sources), 1)
        score += academic_ratio * 0.3

        # Boost for multiple sources
        if len(sources) >= 3:
            score += 0.1
        if len(sources) >= 5:
            score += 0.1

        # Penalize if very few sources
        if len(sources) < 2:
            score -= 0.2

        return min(1.0, max(0.0, score))

    async def _assess_factual_accuracy(self, response: PerplexityResponse) -> float:
        """Assess factual accuracy of response."""
        # Simple heuristic based on source quality and academic sources
        academic_ratio = len(response.academic_sources) / max(len(response.sources), 1)
        peer_reviewed_ratio = response.peer_reviewed_count / max(len(response.sources), 1)

        accuracy = 0.6 + (academic_ratio * 0.2) + (peer_reviewed_ratio * 0.2)
        return min(1.0, accuracy)

    async def _assess_source_quality(self, response: PerplexityResponse) -> float:
        """Assess overall source quality."""
        return await self._assess_source_quality_from_sources(response.sources)

    async def _assess_source_quality_from_sources(self, sources: List[PerplexitySource]) -> float:
        """Assess source quality from source list."""
        if not sources:
            return 0.0

        quality_score = 0.0
        for source in sources:
            if self._is_academic_source(source.domain):
                quality_score += 0.8
            elif self._is_peer_reviewed_domain(source.domain):
                quality_score += 1.0
            elif source.source_type == "institutional":
                quality_score += 0.6
            elif source.source_type == "news":
                quality_score += 0.4
            else:
                quality_score += 0.2

        return min(1.0, quality_score / len(sources))
