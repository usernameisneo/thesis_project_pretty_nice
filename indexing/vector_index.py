"""
Vector indexing using FAISS and sentence transformers.

Features lazy loading of ML libraries to improve startup performance.
"""
import pickle
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from core.types import TextChunk, SearchResult
from core.exceptions import IndexingError
from core.lazy_imports import lazy_import_faiss, lazy_import_sentence_transformers


class VectorIndex:
    """FAISS-based vector index for semantic search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: Optional[str] = None):
        """Initialize vector index.

        Args:
            model_name: Name of the sentence transformer model
            index_dir: Directory to store index files
        """
        self.model_name = model_name

        if index_dir is None:
            home_dir = Path.home()
            index_dir = home_dir / ".thesis_assistant" / "indexes" / model_name

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Lazy initialization - models loaded on first use
        self.model = None
        self.embedding_dim = None
        self.index = None
        self.chunks: List[TextChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self._initialized = False

        # Load existing index if available
        self._load_index()

    def _ensure_initialized(self) -> None:
        """Ensure ML libraries and models are loaded."""
        if self._initialized:
            return

        # Lazy import libraries
        faiss_lib = lazy_import_faiss()
        SentenceTransformerClass = lazy_import_sentence_transformers()

        # Initialize model
        try:
            self.model = SentenceTransformerClass(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise IndexingError(f"Failed to load model {self.model_name}: {e}")

        # Initialize FAISS index
        self.index = faiss_lib.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self._initialized = True

    def _load_index(self) -> None:
        """Load existing index from disk with integrity verification."""
        index_file = self.index_dir / "faiss_index.bin"
        chunks_file = self.index_dir / "chunks.pkl"
        embeddings_file = self.index_dir / "embeddings.npy"
        metadata_file = self.index_dir / "index_metadata.json"

        # Check if all required files exist
        if all(f.exists() for f in [index_file, chunks_file, embeddings_file]):
            try:
                # Load and verify metadata first
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                # Load chunks
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)

                # Load embeddings
                self.chunk_embeddings = np.load(embeddings_file)

                # Verify integrity
                expected_chunks = metadata.get('chunk_count', len(self.chunks))
                expected_embeddings = metadata.get('embedding_count', len(self.chunk_embeddings) if self.chunk_embeddings is not None else 0)

                if len(self.chunks) != expected_chunks or (self.chunk_embeddings is not None and len(self.chunk_embeddings) != expected_embeddings):
                    raise IndexingError(f"Index integrity check failed: chunks={len(self.chunks)}, expected={expected_chunks}, embeddings={len(self.chunk_embeddings) if self.chunk_embeddings is not None else 0}, expected_emb={expected_embeddings}")

                # Load FAISS index immediately for consistency
                if self._initialized:
                    faiss_lib = lazy_import_faiss()
                    self.index = faiss_lib.read_index(str(index_file))

                    # Verify FAISS index size matches embeddings
                    if self.index.ntotal != len(self.chunk_embeddings):
                        print(f"Warning: FAISS index size ({self.index.ntotal}) doesn't match embeddings ({len(self.chunk_embeddings)}). Rebuilding...")
                        self.index = faiss_lib.IndexFlatIP(self.embedding_dim)
                        if self.chunk_embeddings is not None:
                            self.index.add(self.chunk_embeddings)
                        self._save_index()

                print(f"Loaded index with {len(self.chunks)} chunks and verified integrity")

            except Exception as e:
                print(f"Warning: Could not load existing index: {e}")
                print("Resetting to empty index...")
                self._reset_index()
        else:
            print("No existing index found, starting fresh")
            self.chunks = []
            self.chunk_embeddings = None
    
    def _reset_index(self) -> None:
        """Reset index to empty state."""
        if self._initialized and self.index is not None:
            faiss_lib = lazy_import_faiss()
            self.index = faiss_lib.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        self.chunk_embeddings = None

    def _save_index(self) -> None:
        """Save index to disk with atomic operations and integrity verification."""
        try:
            # Use temporary files for atomic operations
            temp_dir = self.index_dir / "temp"
            temp_dir.mkdir(exist_ok=True)

            index_file = self.index_dir / "faiss_index.bin"
            chunks_file = self.index_dir / "chunks.pkl"
            embeddings_file = self.index_dir / "embeddings.npy"
            metadata_file = self.index_dir / "index_metadata.json"

            temp_index_file = temp_dir / "faiss_index.bin"
            temp_chunks_file = temp_dir / "chunks.pkl"
            temp_embeddings_file = temp_dir / "embeddings.npy"
            temp_metadata_file = temp_dir / "index_metadata.json"

            # Create metadata
            metadata = {
                'chunk_count': len(self.chunks),
                'embedding_count': len(self.chunk_embeddings) if self.chunk_embeddings is not None else 0,
                'embedding_dimension': self.embedding_dim,
                'model_name': self.model_name,
                'last_updated': datetime.now().isoformat(),
                'index_version': '2.0'
            }

            # Save to temporary files first
            # Save FAISS index (only if initialized)
            if self._initialized and self.index is not None:
                faiss_lib = lazy_import_faiss()
                faiss_lib.write_index(self.index, str(temp_index_file))

            # Save chunks
            with open(temp_chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)

            # Save embeddings
            if self.chunk_embeddings is not None:
                np.save(temp_embeddings_file, self.chunk_embeddings)

            # Save metadata
            with open(temp_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Atomic move operations (replace existing files)
            if temp_index_file.exists():
                shutil.move(str(temp_index_file), str(index_file))

            shutil.move(str(temp_chunks_file), str(chunks_file))

            if temp_embeddings_file.exists():
                shutil.move(str(temp_embeddings_file), str(embeddings_file))

            shutil.move(str(temp_metadata_file), str(metadata_file))

            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        except Exception as e:
            # Clean up temp files on failure
            temp_dir = self.index_dir / "temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise IndexingError(f"Failed to save index: {e}")
    
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Add text chunks to the index with duplicate detection and atomic operations.

        Args:
            chunks: List of text chunks to add
        """
        if not chunks:
            return

        # Filter out duplicates based on chunk_id and document_id
        existing_chunk_ids = set()
        existing_doc_chunks = set()

        for chunk in self.chunks:
            if hasattr(chunk, 'chunk_id'):
                existing_chunk_ids.add(chunk.chunk_id)
            existing_doc_chunks.add((chunk.document_id, chunk.text[:100]))  # Use first 100 chars as fingerprint

        new_chunks = []
        for chunk in chunks:
            # Check for exact chunk_id match
            if hasattr(chunk, 'chunk_id') and chunk.chunk_id in existing_chunk_ids:
                continue

            # Check for document + text fingerprint match
            fingerprint = (chunk.document_id, chunk.text[:100])
            if fingerprint in existing_doc_chunks:
                continue

            new_chunks.append(chunk)

        if not new_chunks:
            print("No new chunks to add (all were duplicates)")
            return

        # Ensure ML libraries are loaded
        self._ensure_initialized()
        faiss_lib = lazy_import_faiss()

        try:
            # Extract text for embedding
            texts = [chunk.text for chunk in new_chunks]

            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Normalize embeddings for cosine similarity
            faiss_lib.normalize_L2(embeddings)

            # Backup current state for rollback
            chunks_backup = self.chunks.copy()
            embeddings_backup = self.chunk_embeddings.copy() if self.chunk_embeddings is not None else None
            index_backup = None

            if self._initialized and self.index is not None:
                # Create backup of FAISS index
                temp_backup_file = self.index_dir / "temp_backup.bin"
                faiss_lib.write_index(self.index, str(temp_backup_file))

            try:
                # Add to FAISS index
                self.index.add(embeddings)

                # Store chunks and embeddings
                self.chunks.extend(new_chunks)

                if self.chunk_embeddings is None:
                    self.chunk_embeddings = embeddings
                else:
                    self.chunk_embeddings = np.vstack([self.chunk_embeddings, embeddings])

                # Save updated index (atomic operation)
                self._save_index()

                # Clean up backup
                if temp_backup_file.exists():
                    temp_backup_file.unlink()

                print(f"Added {len(new_chunks)} new chunks to index. Total: {len(self.chunks)}")

            except Exception as save_error:
                # Rollback on failure
                print(f"Error during save, rolling back: {save_error}")
                self.chunks = chunks_backup
                self.chunk_embeddings = embeddings_backup

                if temp_backup_file.exists():
                    # Restore FAISS index from backup
                    self.index = faiss_lib.read_index(str(temp_backup_file))
                    temp_backup_file.unlink()

                raise save_error

        except Exception as e:
            raise IndexingError(f"Failed to add chunks to index: {e}")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar chunks.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of search results
        """
        if not self.chunks:
            return []

        # Ensure ML libraries are loaded
        self._ensure_initialized()
        faiss_lib = lazy_import_faiss()

        # Load FAISS index if not already loaded
        if self.index.ntotal == 0 and self.chunk_embeddings is not None:
            self.index.add(self.chunk_embeddings)

        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss_lib.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding, min(k, len(self.chunks)))

            # Convert to SearchResult objects
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score >= threshold and idx < len(self.chunks):
                    result = SearchResult(
                        chunk=self.chunks[idx],
                        score=float(score),
                        rank=i + 1,
                        search_type="vector"
                    )
                    results.append(result)

            return results

        except Exception as e:
            raise IndexingError(f"Search failed: {e}")
    
    def remove_document(self, document_id: str) -> None:
        """Remove all chunks for a document from the index.

        Args:
            document_id: ID of document to remove
        """
        # Find indices of chunks to remove
        indices_to_remove = []
        for i, chunk in enumerate(self.chunks):
            if chunk.document_id == document_id:
                indices_to_remove.append(i)

        if not indices_to_remove:
            return

        # Remove chunks and embeddings
        self.chunks = [chunk for i, chunk in enumerate(self.chunks)
                      if i not in indices_to_remove]

        if self.chunk_embeddings is not None:
            mask = np.ones(len(self.chunk_embeddings), dtype=bool)
            mask[indices_to_remove] = False
            self.chunk_embeddings = self.chunk_embeddings[mask]

        # Rebuild FAISS index if initialized
        if self._initialized:
            faiss_lib = lazy_import_faiss()
            self.index = faiss_lib.IndexFlatIP(self.embedding_dim)
            if self.chunk_embeddings is not None and len(self.chunk_embeddings) > 0:
                self.index.add(self.chunk_embeddings)

        # Save updated index
        self._save_index()

        print(f"Removed {len(indices_to_remove)} chunks for document {document_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        # Get embedding dimension (initialize if needed for stats)
        if self.embedding_dim is None and len(self.chunks) > 0:
            self._ensure_initialized()

        embedding_dim = self.embedding_dim or 384  # Default dimension
        index_size = 0
        if self._initialized and self.index is not None:
            index_size = self.index.ntotal * embedding_dim * 4 / (1024 * 1024)

        return {
            "model_name": self.model_name,
            "total_chunks": len(self.chunks),
            "embedding_dimension": embedding_dim,
            "index_size_mb": index_size,
            "documents": len(set(chunk.document_id for chunk in self.chunks)),
            "initialized": self._initialized
        }

