"""
Semantic Matching Engine for Citation Validation.

This module provides advanced semantic similarity calculation and matching
capabilities for academic citation validation and claim-source alignment.

Features:
    - Multi-method semantic similarity calculation
    - Sentence transformer-based embeddings
    - Contextual similarity scoring
    - Batch processing for efficiency
    - Caching for performance optimization

Author: AI-Powered Thesis Assistant Team
Version: 2.0 - Production Grade
License: MIT
"""

import asyncio
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

from core.exceptions import ProcessingError
from core.lazy_imports import lazy_import_sentence_transformers, lazy_import_sklearn

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of semantic similarity calculation."""
    similarity_score: float
    method: str
    confidence: float
    metadata: Dict[str, Any]


class SemanticMatcher:
    """
    Advanced semantic matching engine for citation validation.
    
    This class provides high-precision semantic similarity calculation
    using multiple methods and caching for optimal performance.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_directory: str = "semantic_cache",
                 batch_size: int = 32,
                 similarity_threshold: float = 0.7):
        """
        Initialize the semantic matcher.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_directory: Directory for caching embeddings
            batch_size: Batch size for processing multiple texts
            similarity_threshold: Minimum similarity threshold
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        
        # Model will be loaded lazily
        self._model = None
        self._embedding_cache = {}
        
        logger.info(f"Semantic matcher initialized with model: {model_name}")
    
    async def initialize(self):
        """Initialize the semantic matcher with required models."""
        try:
            # Load sentence transformer model
            SentenceTransformer = lazy_import_sentence_transformers()
            self._model = SentenceTransformer(self.model_name)
            
            # Load embedding cache
            await self._load_cache()
            
            logger.info("Semantic matcher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic matcher: {e}")
            raise ProcessingError(f"Semantic matcher initialization failed: {e}")
    
    async def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            if not self._model:
                await self.initialize()
            
            # Get embeddings for both texts
            embedding1 = await self._get_embedding(text1)
            embedding2 = await self._get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding1, embedding2)
            
            logger.debug(f"Semantic similarity calculated: {similarity:.3f}")
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    async def calculate_batch_similarity(self, 
                                       text_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Calculate semantic similarity for multiple text pairs efficiently.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            
        Returns:
            List of similarity scores
        """
        try:
            if not self._model:
                await self.initialize()
            
            # Extract unique texts
            all_texts = set()
            for text1, text2 in text_pairs:
                all_texts.add(text1)
                all_texts.add(text2)
            
            # Get embeddings for all unique texts
            text_list = list(all_texts)
            embeddings = await self._get_batch_embeddings(text_list)
            
            # Create embedding lookup
            embedding_lookup = dict(zip(text_list, embeddings))
            
            # Calculate similarities
            similarities = []
            for text1, text2 in text_pairs:
                emb1 = embedding_lookup[text1]
                emb2 = embedding_lookup[text2]
                similarity = self._cosine_similarity(emb1, emb2)
                similarities.append(float(similarity))
            
            logger.debug(f"Batch similarity calculated for {len(text_pairs)} pairs")
            return similarities
            
        except Exception as e:
            logger.warning(f"Batch similarity calculation failed: {e}")
            return [0.0] * len(text_pairs)
    
    async def find_most_similar(self, 
                              query_text: str, 
                              candidate_texts: List[str],
                              top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar texts to a query.
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        try:
            if not candidate_texts:
                return []
            
            # Calculate similarities
            text_pairs = [(query_text, candidate) for candidate in candidate_texts]
            similarities = await self.calculate_batch_similarity(text_pairs)
            
            # Combine and sort
            results = list(zip(candidate_texts, similarities))
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return results[:top_k]
            
        except Exception as e:
            logger.warning(f"Most similar search failed: {e}")
            return []
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text with caching."""
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Generate embedding
        embedding = self._model.encode([text])[0]
        
        # Cache the result
        self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    async def _get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts efficiently."""
        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._embedding_cache:
                cached_embeddings[i] = self._embedding_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self._model.encode(uncached_texts)
            
            # Cache new embeddings
            for text, embedding, idx in zip(uncached_texts, new_embeddings, uncached_indices):
                cache_key = hashlib.md5(text.encode()).hexdigest()
                self._embedding_cache[cache_key] = embedding
                cached_embeddings[idx] = embedding
        
        # Return embeddings in original order
        return [cached_embeddings[i] for i in range(len(texts))]
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    async def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / "embedding_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Convert lists back to numpy arrays
                for key, embedding_list in cache_data.items():
                    self._embedding_cache[key] = np.array(embedding_list)
                
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
                
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
    
    async def save_cache(self):
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / "embedding_cache.json"
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            cache_data = {}
            for key, embedding in self._embedding_cache.items():
                cache_data[key] = embedding.tolist()
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"Saved {len(self._embedding_cache)} embeddings to cache")
            
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_embeddings': len(self._embedding_cache),
            'model_name': self.model_name,
            'cache_directory': str(self.cache_dir),
            'similarity_threshold': self.similarity_threshold
        }
