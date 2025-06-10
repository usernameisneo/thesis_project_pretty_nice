"""
Keyword-based indexing using TF-IDF.

Features lazy loading of ML libraries to improve startup performance.
"""
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from core.types import TextChunk, SearchResult
from core.exceptions import IndexingError
from core.lazy_imports import lazy_import_sklearn


class KeywordIndex:
    """TF-IDF based keyword index for traditional search."""
    
    def __init__(self, index_dir: Optional[str] = None):
        """Initialize keyword index.

        Args:
            index_dir: Directory to store index files
        """
        if index_dir is None:
            home_dir = Path.home()
            index_dir = home_dir / ".thesis_assistant" / "indexes" / "keyword"

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Lazy initialization - vectorizer loaded on first use
        self.vectorizer = None
        self.chunks: List[TextChunk] = []
        self.tfidf_matrix = None
        self.is_fitted = False
        self._initialized = False

        # Load existing index if available
        self._load_index()

    def _ensure_initialized(self) -> None:
        """Ensure sklearn libraries are loaded."""
        if self._initialized:
            return

        # Lazy import sklearn components
        TfidfVectorizer, _ = lazy_import_sklearn()

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=1.0  # Changed from 0.95 to 1.0 to handle small collections
        )
        self._initialized = True
    
    def _load_index(self) -> None:
        """Load existing index from disk."""
        vectorizer_file = self.index_dir / "vectorizer.pkl"
        chunks_file = self.index_dir / "chunks.pkl"
        matrix_file = self.index_dir / "tfidf_matrix.npz"
        
        if all(f.exists() for f in [vectorizer_file, chunks_file, matrix_file]):
            try:
                # Load vectorizer
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                # Load chunks
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                # Load TF-IDF matrix
                import scipy.sparse
                self.tfidf_matrix = scipy.sparse.load_npz(matrix_file)
                self.is_fitted = True
                
                print(f"Loaded keyword index with {len(self.chunks)} chunks")
            except Exception as e:
                print(f"Warning: Could not load existing keyword index: {e}")
                self._reset_index()
        else:
            self._reset_index()
    
    def _reset_index(self) -> None:
        """Reset index to empty state."""
        self.chunks = []
        self.tfidf_matrix = None
        self.is_fitted = False
    
    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            vectorizer_file = self.index_dir / "vectorizer.pkl"
            chunks_file = self.index_dir / "chunks.pkl"
            matrix_file = self.index_dir / "tfidf_matrix.npz"
            
            # Save vectorizer
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save chunks
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save TF-IDF matrix
            if self.tfidf_matrix is not None:
                import scipy.sparse
                scipy.sparse.save_npz(matrix_file, self.tfidf_matrix)
            
        except Exception as e:
            raise IndexingError(f"Failed to save keyword index: {e}")
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Add text chunks to the index.

        Args:
            chunks: List of text chunks to add
        """
        if not chunks:
            return

        # Ensure sklearn is loaded
        self._ensure_initialized()

        try:
            # Add chunks to collection
            self.chunks.extend(chunks)

            # Rebuild TF-IDF matrix with all chunks
            texts = [chunk.text for chunk in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.is_fitted = True

            # Save updated index
            self._save_index()

            print(f"Added {len(chunks)} chunks to keyword index. Total: {len(self.chunks)}")

        except Exception as e:
            raise IndexingError(f"Failed to add chunks to keyword index: {e}")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Search for relevant chunks using TF-IDF.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of search results
        """
        if not self.chunks or not self.is_fitted:
            return []

        # Ensure sklearn is loaded
        self._ensure_initialized()
        _, cosine_similarity = lazy_import_sklearn()

        try:
            # Transform query using fitted vectorizer
            query_vector = self.vectorizer.transform([query])

            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]

            # Convert to SearchResult objects
            results = []
            for i, idx in enumerate(top_indices):
                score = similarities[idx]
                if score >= threshold:
                    result = SearchResult(
                        chunk=self.chunks[idx],
                        score=float(score),
                        rank=i + 1,
                        search_type="keyword"
                    )
                    results.append(result)

            return results

        except Exception as e:
            raise IndexingError(f"Keyword search failed: {e}")
    
    def remove_document(self, document_id: str) -> None:
        """Remove all chunks for a document from the index.
        
        Args:
            document_id: ID of document to remove
        """
        # Find chunks to keep
        chunks_to_keep = [chunk for chunk in self.chunks 
                         if chunk.document_id != document_id]
        
        removed_count = len(self.chunks) - len(chunks_to_keep)
        if removed_count == 0:
            return
        
        self.chunks = chunks_to_keep
        
        # Rebuild TF-IDF matrix
        if self.chunks:
            # Ensure sklearn is loaded
            self._ensure_initialized()
            texts = [chunk.text for chunk in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            self.tfidf_matrix = None
            self.is_fitted = False
        
        # Save updated index
        self._save_index()
        
        print(f"Removed {removed_count} chunks for document {document_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        stats = {
            "total_chunks": len(self.chunks),
            "is_fitted": self.is_fitted,
            "documents": len(set(chunk.document_id for chunk in self.chunks))
        }
        
        if self.is_fitted and self.tfidf_matrix is not None:
            stats.update({
                "vocabulary_size": len(self.vectorizer.vocabulary_),
                "matrix_shape": self.tfidf_matrix.shape,
                "matrix_nnz": self.tfidf_matrix.nnz
            })
        
        return stats
    
    def get_top_terms(self, chunk_idx: int, n_terms: int = 10) -> List[tuple]:
        """Get top TF-IDF terms for a specific chunk.
        
        Args:
            chunk_idx: Index of the chunk
            n_terms: Number of top terms to return
            
        Returns:
            List of (term, score) tuples
        """
        if not self.is_fitted or chunk_idx >= len(self.chunks):
            return []
        
        try:
            # Get TF-IDF scores for the chunk
            chunk_vector = self.tfidf_matrix[chunk_idx]
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get non-zero scores
            scores = chunk_vector.toarray().flatten()
            term_scores = [(feature_names[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
            
            # Sort by score and return top n
            term_scores.sort(key=lambda x: x[1], reverse=True)
            return term_scores[:n_terms]
            
        except Exception as e:
            print(f"Warning: Could not get top terms: {e}")
            return []

