from .chroma import ChromaClient
from .embeddings import GeminiEmbeddings
from .exceptions import (
    DocumentProcessingError,
    RetrievalError,
    GenerationError
)
from typing import Optional
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Service instances with lazy initialization
_chroma_client: Optional[ChromaClient] = None
_embeddings: Optional[GeminiEmbeddings] = None

def initialize_services():
    """Initialize all core services"""
    global _chroma_client, _embeddings
    
    try:
        logger.info("Initializing ChromaDB client...")
        _chroma_client = ChromaClient()
        
        logger.info("Initializing Gemini embeddings...")
        _embeddings = GeminiEmbeddings()
        
        logger.info("Core services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def get_chroma_client() -> ChromaClient:
    """Get the initialized ChromaDB client instance"""
    if _chroma_client is None:
        raise RuntimeError("Chroma client not initialized. Call initialize_services() first.")
    return _chroma_client

def get_embeddings() -> GeminiEmbeddings:
    """Get the initialized embeddings instance"""
    if _embeddings is None:
        raise RuntimeError("Embeddings not initialized. Call initialize_services() first.")
    return _embeddings

# Public API
__all__ = [
    'ChromaClient',
    'GeminiEmbeddings',
    'DocumentProcessingError',
    'RetrievalError',
    'GenerationError',
    'initialize_services',
    'get_chroma_client',
    'get_embeddings'
]