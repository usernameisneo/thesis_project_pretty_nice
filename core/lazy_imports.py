"""
Lazy import utilities for heavy ML libraries.

This module provides utilities for lazy loading of heavy machine learning
libraries to improve application startup performance.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

# Type checking imports
if TYPE_CHECKING:
    import faiss
    from sentence_transformers import SentenceTransformer
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
else:
    faiss = None
    SentenceTransformer = None
    spacy = None
    TfidfVectorizer = None
    cosine_similarity = None

logger = logging.getLogger(__name__)

# Global cache for loaded modules
_loaded_modules = {}


def lazy_import_faiss():
    """
    Lazy import of FAISS library.
    
    Returns:
        faiss module
        
    Raises:
        ImportError: If FAISS is not available
    """
    global faiss
    if faiss is None:
        if 'faiss' not in _loaded_modules:
            try:
                import faiss as _faiss
                _loaded_modules['faiss'] = _faiss
                logger.info("FAISS library loaded successfully")
            except ImportError as e:
                error_msg = f"FAISS library not available: {e}. Install with: pip install faiss-cpu"
                logger.error(error_msg)
                raise ImportError(error_msg)
        faiss = _loaded_modules['faiss']
    return faiss


def lazy_import_sentence_transformers():
    """
    Lazy import of SentenceTransformers library.
    
    Returns:
        SentenceTransformer class
        
    Raises:
        ImportError: If SentenceTransformers is not available
    """
    global SentenceTransformer
    if SentenceTransformer is None:
        if 'sentence_transformers' not in _loaded_modules:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer
                _loaded_modules['sentence_transformers'] = _SentenceTransformer
                logger.info("SentenceTransformers library loaded successfully")
            except ImportError as e:
                error_msg = f"SentenceTransformers library not available: {e}. Install with: pip install sentence-transformers"
                logger.error(error_msg)
                raise ImportError(error_msg)
        SentenceTransformer = _loaded_modules['sentence_transformers']
    return SentenceTransformer


def lazy_import_spacy():
    """
    Lazy import of spaCy library.
    
    Returns:
        spacy module
        
    Raises:
        ImportError: If spaCy is not available
    """
    global spacy
    if spacy is None:
        if 'spacy' not in _loaded_modules:
            try:
                import spacy as _spacy
                _loaded_modules['spacy'] = _spacy
                logger.info("spaCy library loaded successfully")
            except ImportError as e:
                error_msg = f"spaCy library not available: {e}. Install with: pip install spacy"
                logger.error(error_msg)
                raise ImportError(error_msg)
        spacy = _loaded_modules['spacy']
    return spacy


def lazy_import_sklearn():
    """
    Lazy import of scikit-learn components.
    
    Returns:
        Tuple of (TfidfVectorizer, cosine_similarity)
        
    Raises:
        ImportError: If scikit-learn is not available
    """
    global TfidfVectorizer, cosine_similarity
    if TfidfVectorizer is None or cosine_similarity is None:
        if 'sklearn' not in _loaded_modules:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
                _loaded_modules['sklearn'] = (_TfidfVectorizer, _cosine_similarity)
                logger.info("scikit-learn components loaded successfully")
            except ImportError as e:
                error_msg = f"scikit-learn library not available: {e}. Install with: pip install scikit-learn"
                logger.error(error_msg)
                raise ImportError(error_msg)
        TfidfVectorizer, cosine_similarity = _loaded_modules['sklearn']
    return TfidfVectorizer, cosine_similarity


def is_ml_library_available(library_name: str) -> bool:
    """
    Check if a ML library is available without importing it.
    
    Args:
        library_name: Name of the library to check
        
    Returns:
        True if library is available
    """
    import importlib.util
    
    library_specs = {
        'faiss': ['faiss', 'faiss-cpu', 'faiss-gpu'],
        'sentence_transformers': ['sentence_transformers'],
        'spacy': ['spacy'],
        'sklearn': ['sklearn', 'scikit-learn'],
        'torch': ['torch'],
        'transformers': ['transformers']
    }
    
    if library_name not in library_specs:
        return False
    
    for spec_name in library_specs[library_name]:
        if importlib.util.find_spec(spec_name) is not None:
            return True
    
    return False


def get_ml_library_status() -> dict:
    """
    Get status of all ML libraries.
    
    Returns:
        Dictionary with library availability status
    """
    libraries = ['faiss', 'sentence_transformers', 'spacy', 'sklearn', 'torch', 'transformers']
    status = {}
    
    for lib in libraries:
        status[lib] = {
            'available': is_ml_library_available(lib),
            'loaded': lib in _loaded_modules
        }
    
    return status


def preload_ml_libraries(libraries: Optional[list] = None) -> dict:
    """
    Preload specified ML libraries.
    
    Args:
        libraries: List of libraries to preload. If None, preloads all available.
        
    Returns:
        Dictionary with loading results
    """
    if libraries is None:
        libraries = ['faiss', 'sentence_transformers', 'spacy', 'sklearn']
    
    results = {}
    
    for lib in libraries:
        try:
            if lib == 'faiss':
                lazy_import_faiss()
            elif lib == 'sentence_transformers':
                lazy_import_sentence_transformers()
            elif lib == 'spacy':
                lazy_import_spacy()
            elif lib == 'sklearn':
                lazy_import_sklearn()
            
            results[lib] = {'success': True, 'error': None}
            
        except ImportError as e:
            results[lib] = {'success': False, 'error': str(e)}
            logger.warning(f"Failed to preload {lib}: {e}")
    
    return results


def clear_ml_cache():
    """Clear the ML library cache."""
    global _loaded_modules, faiss, SentenceTransformer, spacy, TfidfVectorizer, cosine_similarity
    
    _loaded_modules.clear()
    faiss = None
    SentenceTransformer = None
    spacy = None
    TfidfVectorizer = None
    cosine_similarity = None
    
    logger.info("ML library cache cleared")


def lazy_import(module_name: str, package: Optional[str] = None):
    """
    Generic lazy import function.

    Args:
        module_name: Name of the module to import
        package: Package name for relative imports

    Returns:
        Imported module

    Raises:
        ImportError: If module cannot be imported
    """
    try:
        import importlib
        return importlib.import_module(module_name, package)
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        raise


class LazyImporter:
    """
    Lazy importer class for managing module imports.

    This class provides a convenient interface for lazy loading of modules,
    particularly useful for optional dependencies and heavy libraries.
    """

    def __init__(self):
        """Initialize the lazy importer."""
        self._module_cache = {}

    def import_module(self, module_name: str, package: Optional[str] = None):
        """
        Import a module lazily with caching.

        Args:
            module_name: Name of the module to import
            package: Package name for relative imports

        Returns:
            Imported module

        Raises:
            ImportError: If module cannot be imported
        """
        cache_key = f"{package}.{module_name}" if package else module_name

        if cache_key not in self._module_cache:
            try:
                import importlib
                module = importlib.import_module(module_name, package)
                self._module_cache[cache_key] = module
                logger.debug(f"Lazily imported module: {cache_key}")
            except ImportError as e:
                logger.error(f"Failed to import {module_name}: {e}")
                raise

        return self._module_cache[cache_key]

    def is_available(self, module_name: str, package: Optional[str] = None) -> bool:
        """
        Check if a module is available without importing it.

        Args:
            module_name: Name of the module to check
            package: Package name for relative imports

        Returns:
            True if module is available
        """
        try:
            import importlib.util
            if package:
                spec = importlib.util.find_spec(f"{package}.{module_name}")
            else:
                spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ValueError):
            return False

    def clear_cache(self):
        """Clear the module cache."""
        self._module_cache.clear()
        logger.info("Module cache cleared")
