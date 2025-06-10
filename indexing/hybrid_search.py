"""
Hybrid search combining vector and keyword search.
"""
from typing import List, Dict, Any
from core.types import SearchResult
from indexing.vector_index import VectorIndex
from indexing.keyword_index import KeywordIndex


class HybridSearch:
    """Combines vector and keyword search for better results."""
    
    def __init__(self, vector_index: VectorIndex, keyword_index: KeywordIndex):
        """Initialize hybrid search.
        
        Args:
            vector_index: Vector search index
            keyword_index: Keyword search index
        """
        self.vector_index = vector_index
        self.keyword_index = keyword_index
    
    def search(self, query: str, k: int = 10, 
               vector_weight: float = 0.6, 
               keyword_weight: float = 0.4,
               threshold: float = 0.0) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword results.
        
        Args:
            query: Search query
            k: Number of results to return
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            threshold: Minimum combined score threshold
            
        Returns:
            List of search results sorted by combined score
        """
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight
        else:
            vector_weight = 0.5
            keyword_weight = 0.5
        
        # Get results from both indexes
        vector_results = self.vector_index.search(query, k * 2)  # Get more to allow for fusion
        keyword_results = self.keyword_index.search(query, k * 2)
        
        # Create combined results dictionary
        combined_results: Dict[str, SearchResult] = {}
        
        # Process vector results
        for result in vector_results:
            chunk_id = result.chunk.chunk_id
            combined_score = result.score * vector_weight
            
            combined_results[chunk_id] = SearchResult(
                chunk=result.chunk,
                score=combined_score,
                rank=0,  # Will be set later
                search_type="hybrid"
            )
        
        # Process keyword results and combine scores
        for result in keyword_results:
            chunk_id = result.chunk.chunk_id
            keyword_score = result.score * keyword_weight
            
            if chunk_id in combined_results:
                # Combine scores
                combined_results[chunk_id].score += keyword_score
            else:
                # Add new result
                combined_results[chunk_id] = SearchResult(
                    chunk=result.chunk,
                    score=keyword_score,
                    rank=0,
                    search_type="hybrid"
                )
        
        # Sort by combined score and apply threshold
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        # Filter by threshold and limit results
        final_results = []
        for i, result in enumerate(sorted_results[:k]):
            if result.score >= threshold:
                result.rank = i + 1
                final_results.append(result)
        
        return final_results
    
    def search_vector_only(self, query: str, k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Perform vector-only search.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum score threshold
            
        Returns:
            List of vector search results
        """
        return self.vector_index.search(query, k, threshold)
    
    def search_keyword_only(self, query: str, k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Perform keyword-only search.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum score threshold
            
        Returns:
            List of keyword search results
        """
        return self.keyword_index.search(query, k, threshold)
    
    def add_chunks(self, chunks: List) -> None:
        """Add chunks to both indexes.
        
        Args:
            chunks: List of text chunks to add
        """
        self.vector_index.add_chunks(chunks)
        self.keyword_index.add_chunks(chunks)
    
    def remove_document(self, document_id: str) -> None:
        """Remove document from both indexes.
        
        Args:
            document_id: ID of document to remove
        """
        self.vector_index.remove_document(document_id)
        self.keyword_index.remove_document(document_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both indexes.
        
        Returns:
            Dictionary with combined statistics
        """
        return {
            "vector_index": self.vector_index.get_stats(),
            "keyword_index": self.keyword_index.get_stats()
        }


class HybridSearchEngine:
    """
    Complete hybrid search engine with automatic index management.

    This class provides a high-level interface for hybrid search
    with automatic initialization and document management.
    """

    def __init__(self, index_directory: str):
        """
        Initialize hybrid search engine.

        Args:
            index_directory: Directory to store index files
        """
        import os
        os.makedirs(index_directory, exist_ok=True)

        # Initialize indexes
        self.vector_index = VectorIndex(
            model_name="all-MiniLM-L6-v2",
            index_dir=os.path.join(index_directory, "vector")
        )

        self.keyword_index = KeywordIndex(
            index_dir=os.path.join(index_directory, "keyword")
        )

        # Initialize hybrid search
        self.hybrid_search = HybridSearch(self.vector_index, self.keyword_index)

    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum score threshold

        Returns:
            List of search results
        """
        return self.hybrid_search.search(query, k, threshold=threshold)

    def add_documents(self, chunks: List) -> None:
        """
        Add document chunks to the search engine.

        Args:
            chunks: List of text chunks to add
        """
        self.hybrid_search.add_chunks(chunks)

    def remove_document(self, document_id: str) -> None:
        """
        Remove document from search engine.

        Args:
            document_id: ID of document to remove
        """
        self.hybrid_search.remove_document(document_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get search engine statistics.

        Returns:
            Dictionary with statistics
        """
        return self.hybrid_search.get_stats()

