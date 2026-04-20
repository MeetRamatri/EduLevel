"""Embedding generation and similarity search services."""

import logging
import json
import os
import numpy as np
from pathlib import Path
from typing import List
import google.generativeai as genai

from models import Chunk, ChunkWithEmbedding, SearchResult
from config import EMBEDDINGS_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def _validate_google_api_key() -> str:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY must be set in the environment to generate Gemini embeddings."
        )
    return google_api_key


def _get_embeddings(inputs: List[str]) -> List[List[float]]:
    google_api_key = _validate_google_api_key()
    genai.configure(api_key=google_api_key)
    embeddings = []
    for text in inputs:
        response = genai.embed_content(model="models/text-embedding-004", content=text)
        embeddings.append(response["embedding"])
    return embeddings


def generate_embeddings(chunks: List[Chunk]) -> List[ChunkWithEmbedding]:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        List of ChunkWithEmbedding objects
    """
    texts = [chunk.content for chunk in chunks]
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = _get_embeddings(texts)
    
    chunks_with_embeddings = [
        ChunkWithEmbedding(
            chunk_id=chunk.chunk_id,
            text=chunk.content,
            embedding=emb
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    
    logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
    return chunks_with_embeddings


def save_embeddings(filename: str, chunks_with_embeddings: List[ChunkWithEmbedding]) -> str:
    """
    Save embeddings to JSON file.
    
    Args:
        filename: Original filename (used to create storage path)
        chunks_with_embeddings: List of chunks with embeddings
        
    Returns:
        Path to saved embeddings file
    """
    base_filename = Path(filename).stem
    embedding_file = EMBEDDINGS_DIR / f"{base_filename}_embeddings.json"
    
    embedding_data = [
        {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "embedding": chunk.embedding
        }
        for chunk in chunks_with_embeddings
    ]
    
    with open(embedding_file, "w") as f:
        json.dump(embedding_data, f, indent=2)
    
    logger.info(f"Embeddings saved to {embedding_file}")
    return str(embedding_file)


def cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between query and chunk embeddings.
    
    Args:
        query_embedding: Query embedding vector
        chunk_embeddings: Array of chunk embedding vectors
        
    Returns:
        Array of similarity scores
    """
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    
    query_norm = max(query_norm, 1e-8)
    chunk_norms = np.maximum(chunk_norms, 1e-8)
    
    similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_norms * query_norm)
    return similarities


def similarity_search(query: str, embedding_file_path: str, top_k: int = 3) -> List[SearchResult]:
    """
    Search for similar chunks based on query.
    
    Args:
        query: Search query
        embedding_file_path: Path to embeddings JSON file
        top_k: Number of top results to return
        
    Returns:
        List of SearchResult objects
    """
    with open(embedding_file_path, "r") as f:
        embedding_data = json.load(f)
    
    logger.info(f"Loaded {len(embedding_data)} embeddings from {embedding_file_path}")
    
    query_embedding = np.array(_get_embeddings([query])[0], dtype=float)
    logger.info(f"Generated query embedding for: '{query}'")
    
    chunk_embeddings = np.array([item["embedding"] for item in embedding_data], dtype=float)
    chunk_ids = [item["id"] for item in embedding_data]
    chunk_texts = [item["text"] for item in embedding_data]
    
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [
        SearchResult(
            chunk_id=chunk_ids[idx],
            text=chunk_texts[idx],
            similarity_score=float(similarities[idx])
        )
        for idx in top_indices
    ]
    
    logger.info(f"Found {len(results)} top results for query: '{query}'")
    return results
