"""
Semantic Scholar API Client - Production-Grade Academic Paper Discovery.

This module provides complete integration with the Semantic Scholar API for
academic paper discovery, metadata extraction, and citation network analysis.
Implements full API coverage with rate limiting, error handling, and caching.

Features:
    - Complete Semantic Scholar API integration
    - Academic paper search and discovery
    - Citation network analysis
    - Author and venue information
    - PDF availability detection
    - Rate limiting and retry logic
    - Comprehensive error handling
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

from core.exceptions import APIError, ValidationError
from core.types import DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class SemanticScholarPaper:
    """Complete representation of a Semantic Scholar paper."""
    paper_id: str
    title: str
    abstract: Optional[str]
    authors: List[Dict[str, Any]]
    venue: Optional[str]
    year: Optional[int]
    citation_count: int
    reference_count: int
    influential_citation_count: int
    
    # URLs and identifiers
    url: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    pubmed_id: Optional[str]
    corpus_id: Optional[str]
    
    # Content information
    pdf_urls: List[str] = field(default_factory=list)
    open_access_pdf: Optional[str] = None
    fields_of_study: List[str] = field(default_factory=list)
    
    # Citation information
    citations: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    # Metadata
    retrieved_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result from Semantic Scholar."""
    papers: List[SemanticScholarPaper]
    total_results: int
    query: str
    search_time: float
    next_offset: Optional[int] = None


class SemanticScholarClient:
    """
    Production-grade Semantic Scholar API client.
    
    Provides complete access to Semantic Scholar's academic database with
    enterprise-level reliability, performance, and error handling.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 cache_directory: str = "semantic_scholar_cache",
                 rate_limit_requests_per_second: float = 10.0,
                 max_retries: int = 3,
                 timeout_seconds: int = 30):
        """
        Initialize Semantic Scholar client.
        
        Args:
            api_key: Optional API key for higher rate limits
            cache_directory: Directory for response caching
            rate_limit_requests_per_second: Rate limiting configuration
            max_retries: Maximum retry attempts for failed requests
            timeout_seconds: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.rate_limit = rate_limit_requests_per_second
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        
        # Cache setup
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=24)  # Cache for 24 hours
        
        # Rate limiting
        self.last_request_time = datetime.now()
        self.request_interval = 1.0 / rate_limit_requests_per_second
        
        # Session will be created when needed
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Semantic Scholar client initialized with rate limit: {rate_limit_requests_per_second} req/s")
    
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
                'User-Agent': 'AI-Powered-Thesis-Assistant/2.0',
                'Accept': 'application/json'
            }
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=self.timeout
            )
    
    async def search_papers(self, 
                          query: str,
                          limit: int = 100,
                          offset: int = 0,
                          fields: Optional[List[str]] = None,
                          publication_types: Optional[List[str]] = None,
                          min_citation_count: Optional[int] = None,
                          year_range: Optional[Tuple[int, int]] = None,
                          venues: Optional[List[str]] = None,
                          fields_of_study: Optional[List[str]] = None) -> SearchResult:
        """
        Search for academic papers using Semantic Scholar.
        
        Args:
            query: Search query string
            limit: Maximum number of results (1-100)
            offset: Offset for pagination
            fields: Specific fields to retrieve
            publication_types: Filter by publication types
            min_citation_count: Minimum citation count filter
            year_range: Tuple of (min_year, max_year)
            venues: Filter by specific venues
            fields_of_study: Filter by fields of study
            
        Returns:
            SearchResult with papers and metadata
        """
        try:
            start_time = datetime.now()
            
            # Build search parameters
            params = {
                'query': query,
                'limit': min(limit, 100),  # API maximum is 100
                'offset': offset
            }
            
            # Add field selection
            if fields is None:
                fields = [
                    'paperId', 'title', 'abstract', 'authors', 'venue', 'year',
                    'citationCount', 'referenceCount', 'influentialCitationCount',
                    'url', 'doi', 'arxivId', 'pubmedId', 'corpusId',
                    'openAccessPdf', 'fieldsOfStudy', 'embedding'
                ]
            params['fields'] = ','.join(fields)
            
            # Add filters
            if publication_types:
                params['publicationTypes'] = ','.join(publication_types)
            if min_citation_count is not None:
                params['minCitationCount'] = min_citation_count
            if year_range:
                params['year'] = f"{year_range[0]}-{year_range[1]}"
            if venues:
                params['venue'] = ','.join(venues)
            if fields_of_study:
                params['fieldsOfStudy'] = ','.join(fields_of_study)
            
            # Check cache first
            cache_key = self._generate_cache_key('search', params)
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                logger.debug(f"Using cached search result for query: {query}")
                return self._parse_search_response(cached_result, query, start_time)
            
            # Make API request
            response_data = await self._make_request('paper/search', params)
            
            # Cache response
            await self._cache_response(cache_key, response_data)
            
            # Parse and return results
            result = self._parse_search_response(response_data, query, start_time)
            
            logger.info(f"Search completed: {result.total_results} papers found for '{query}'")
            return result
            
        except Exception as e:
            error_msg = f"Paper search failed for query '{query}': {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    async def get_paper_details(self, paper_id: str, fields: Optional[List[str]] = None) -> SemanticScholarPaper:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID, DOI, or ArXiv ID
            fields: Specific fields to retrieve
            
        Returns:
            Complete paper information
        """
        try:
            # Default fields for complete paper information
            if fields is None:
                fields = [
                    'paperId', 'title', 'abstract', 'authors', 'venue', 'year',
                    'citationCount', 'referenceCount', 'influentialCitationCount',
                    'url', 'doi', 'arxivId', 'pubmedId', 'corpusId',
                    'openAccessPdf', 'fieldsOfStudy', 'embedding',
                    'citations', 'references'
                ]
            
            params = {'fields': ','.join(fields)}
            
            # Check cache
            cache_key = self._generate_cache_key('paper', {'id': paper_id, **params})
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                logger.debug(f"Using cached paper details for: {paper_id}")
                return self._parse_paper_data(cached_result)
            
            # Make API request
            response_data = await self._make_request(f'paper/{paper_id}', params)
            
            # Cache response
            await self._cache_response(cache_key, response_data)
            
            # Parse and return paper
            paper = self._parse_paper_data(response_data)
            
            logger.info(f"Retrieved paper details: {paper.title}")
            return paper
            
        except Exception as e:
            error_msg = f"Failed to get paper details for {paper_id}: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    async def get_paper_citations(self, 
                                paper_id: str,
                                limit: int = 100,
                                offset: int = 0,
                                fields: Optional[List[str]] = None) -> List[SemanticScholarPaper]:
        """
        Get papers that cite the specified paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to retrieve
            offset: Offset for pagination
            fields: Specific fields to retrieve for citing papers
            
        Returns:
            List of papers that cite the specified paper
        """
        try:
            if fields is None:
                fields = [
                    'paperId', 'title', 'authors', 'venue', 'year',
                    'citationCount', 'influentialCitationCount', 'doi'
                ]
            
            params = {
                'fields': ','.join(fields),
                'limit': min(limit, 1000),  # API maximum
                'offset': offset
            }
            
            # Check cache
            cache_key = self._generate_cache_key('citations', {'id': paper_id, **params})
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                return [self._parse_paper_data(paper) for paper in cached_result.get('data', [])]
            
            # Make API request
            response_data = await self._make_request(f'paper/{paper_id}/citations', params)
            
            # Cache response
            await self._cache_response(cache_key, response_data)
            
            # Parse citations
            citations = []
            for citation_data in response_data.get('data', []):
                if 'citingPaper' in citation_data:
                    paper = self._parse_paper_data(citation_data['citingPaper'])
                    citations.append(paper)
            
            logger.info(f"Retrieved {len(citations)} citations for paper {paper_id}")
            return citations
            
        except Exception as e:
            error_msg = f"Failed to get citations for paper {paper_id}: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    async def get_paper_references(self, 
                                 paper_id: str,
                                 limit: int = 100,
                                 offset: int = 0,
                                 fields: Optional[List[str]] = None) -> List[SemanticScholarPaper]:
        """
        Get papers referenced by the specified paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references to retrieve
            offset: Offset for pagination
            fields: Specific fields to retrieve for referenced papers
            
        Returns:
            List of papers referenced by the specified paper
        """
        try:
            if fields is None:
                fields = [
                    'paperId', 'title', 'authors', 'venue', 'year',
                    'citationCount', 'influentialCitationCount', 'doi'
                ]
            
            params = {
                'fields': ','.join(fields),
                'limit': min(limit, 1000),  # API maximum
                'offset': offset
            }
            
            # Check cache
            cache_key = self._generate_cache_key('references', {'id': paper_id, **params})
            cached_result = await self._get_cached_response(cache_key)
            if cached_result:
                return [self._parse_paper_data(paper) for paper in cached_result.get('data', [])]
            
            # Make API request
            response_data = await self._make_request(f'paper/{paper_id}/references', params)
            
            # Cache response
            await self._cache_response(cache_key, response_data)
            
            # Parse references
            references = []
            for ref_data in response_data.get('data', []):
                if 'citedPaper' in ref_data:
                    paper = self._parse_paper_data(ref_data['citedPaper'])
                    references.append(paper)
            
            logger.info(f"Retrieved {len(references)} references for paper {paper_id}")
            return references
            
        except Exception as e:
            error_msg = f"Failed to get references for paper {paper_id}: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    async def get_author_papers(self,
                              author_id: str,
                              limit: int = 100,
                              offset: int = 0,
                              fields: Optional[List[str]] = None) -> List[SemanticScholarPaper]:
        """
        Get papers by a specific author.

        Args:
            author_id: Semantic Scholar author ID
            limit: Maximum number of papers to retrieve
            offset: Offset for pagination
            fields: Specific fields to retrieve

        Returns:
            List of papers by the author
        """
        try:
            if fields is None:
                fields = [
                    'paperId', 'title', 'authors', 'venue', 'year',
                    'citationCount', 'influentialCitationCount', 'doi'
                ]

            params = {
                'fields': ','.join(fields),
                'limit': min(limit, 1000),
                'offset': offset
            }

            response_data = await self._make_request(f'author/{author_id}/papers', params)

            papers = []
            for paper_data in response_data.get('data', []):
                paper = self._parse_paper_data(paper_data)
                papers.append(paper)

            logger.info(f"Retrieved {len(papers)} papers for author {author_id}")
            return papers

        except Exception as e:
            error_msg = f"Failed to get papers for author {author_id}: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    async def batch_paper_lookup(self, paper_ids: List[str], fields: Optional[List[str]] = None) -> List[SemanticScholarPaper]:
        """
        Retrieve multiple papers in a single batch request.

        Args:
            paper_ids: List of paper IDs to retrieve
            fields: Specific fields to retrieve

        Returns:
            List of papers
        """
        try:
            if not paper_ids:
                return []

            if fields is None:
                fields = [
                    'paperId', 'title', 'abstract', 'authors', 'venue', 'year',
                    'citationCount', 'referenceCount', 'influentialCitationCount',
                    'doi', 'openAccessPdf'
                ]

            # Semantic Scholar batch API supports up to 500 papers per request
            batch_size = 500
            all_papers = []

            for i in range(0, len(paper_ids), batch_size):
                batch_ids = paper_ids[i:i + batch_size]

                request_data = {
                    'ids': batch_ids,
                    'fields': fields
                }

                response_data = await self._make_request('paper/batch', request_data, method='POST')

                for paper_data in response_data:
                    if paper_data:  # Skip None entries
                        paper = self._parse_paper_data(paper_data)
                        all_papers.append(paper)

            logger.info(f"Batch lookup completed: {len(all_papers)} papers retrieved")
            return all_papers

        except Exception as e:
            error_msg = f"Batch paper lookup failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    async def _make_request(self, endpoint: str, params: Dict[str, Any], method: str = 'GET') -> Dict[str, Any]:
        """
        Make rate-limited API request with retry logic.

        Args:
            endpoint: API endpoint
            params: Request parameters
            method: HTTP method

        Returns:
            API response data
        """
        await self._ensure_session()

        # Rate limiting
        await self._enforce_rate_limit()

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                if method == 'GET':
                    async with self.session.get(url, params=params) as response:
                        return await self._handle_response(response)
                elif method == 'POST':
                    async with self.session.post(url, json=params) as response:
                        return await self._handle_response(response)
                else:
                    raise APIError(f"Unsupported HTTP method: {method}")

            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    raise APIError(f"Request failed after {self.max_retries} retries: {e}")

                # Exponential backoff
                wait_time = (2 ** attempt) * 1.0
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
        elif response.status == 404:
            raise APIError("Resource not found")
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

    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        cache_data = f"{operation}:{json.dumps(params, sort_keys=True)}"
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
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def _parse_search_response(self, response_data: Dict[str, Any], query: str, start_time: datetime) -> SearchResult:
        """Parse search response into SearchResult object."""
        papers = []
        for paper_data in response_data.get('data', []):
            paper = self._parse_paper_data(paper_data)
            papers.append(paper)

        search_time = (datetime.now() - start_time).total_seconds()

        return SearchResult(
            papers=papers,
            total_results=response_data.get('total', len(papers)),
            query=query,
            search_time=search_time,
            next_offset=response_data.get('next')
        )

    def _parse_paper_data(self, paper_data: Dict[str, Any]) -> SemanticScholarPaper:
        """Parse paper data into SemanticScholarPaper object."""
        # Extract PDF URLs
        pdf_urls = []
        open_access_pdf = None
        if 'openAccessPdf' in paper_data and paper_data['openAccessPdf']:
            open_access_pdf = paper_data['openAccessPdf'].get('url')
            if open_access_pdf:
                pdf_urls.append(open_access_pdf)

        # Extract authors
        authors = []
        for author_data in paper_data.get('authors', []):
            authors.append({
                'authorId': author_data.get('authorId'),
                'name': author_data.get('name'),
                'url': author_data.get('url')
            })

        # Extract citations and references
        citations = []
        references = []
        if 'citations' in paper_data:
            citations = [cite.get('paperId') for cite in paper_data['citations'] if cite.get('paperId')]
        if 'references' in paper_data:
            references = [ref.get('paperId') for ref in paper_data['references'] if ref.get('paperId')]

        return SemanticScholarPaper(
            paper_id=paper_data.get('paperId', ''),
            title=paper_data.get('title', ''),
            abstract=paper_data.get('abstract'),
            authors=authors,
            venue=paper_data.get('venue'),
            year=paper_data.get('year'),
            citation_count=paper_data.get('citationCount', 0),
            reference_count=paper_data.get('referenceCount', 0),
            influential_citation_count=paper_data.get('influentialCitationCount', 0),
            url=paper_data.get('url'),
            doi=paper_data.get('doi'),
            arxiv_id=paper_data.get('arxivId'),
            pubmed_id=paper_data.get('pubmedId'),
            corpus_id=paper_data.get('corpusId'),
            pdf_urls=pdf_urls,
            open_access_pdf=open_access_pdf,
            fields_of_study=paper_data.get('fieldsOfStudy', []),
            citations=citations,
            references=references,
            embedding=paper_data.get('embedding', {}).get('vector') if paper_data.get('embedding') else None
        )
