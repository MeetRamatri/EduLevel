"""Pydantic models for request/response validation."""

from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    """Request model for asking the AI tutor."""
    query: str
    filename: str = "document"


class QueryResponse(BaseModel):
    """Response model for AI tutor answers."""
    answer: str


class Chunk(BaseModel):
    """Model for a text chunk."""
    chunk_id: str
    content: str
    word_count: int


class PDFUploadResponse(BaseModel):
    """Response model for PDF upload and text extraction."""
    filename: str
    text: str
    page_count: int
    status: str = "success"


class ChunkWithEmbedding(BaseModel):
    """Model for a chunk with its embedding."""
    chunk_id: str
    text: str
    embedding: List[float]


class PDFChunkedResponse(BaseModel):
    """Response model for PDF upload with chunking."""
    filename: str
    page_count: int
    chunks: List[Chunk]
    total_chunks: int
    status: str = "success"


class EmbeddingStorageResponse(BaseModel):
    """Response model for PDF upload with embeddings."""
    filename: str
    total_chunks: int
    embeddings_saved: int
    storage_path: str
    status: str = "success"


class SearchResult(BaseModel):
    """Model for similarity search results."""
    chunk_id: str
    text: str
    similarity_score: float


class SimilaritySearchResponse(BaseModel):
    """Response model for similarity search."""
    query: str
    embedding_file: str
    results: List[SearchResult]
    num_results: int
    status: str = "success"


class ImageMetadata(BaseModel):
    """Model for image metadata."""
    id: str
    filename: str
    title: str
    keywords: List[str]
    description: str
    embedding: List[float] = []  # Embedding vector for semantic search

    def get_embedding_text(self) -> str:
        """Combines description and keywords for embedding generation."""
        return f"{self.description} " + " ".join(self.keywords)


class ImageUploadResponse(BaseModel):
    """Response model for image upload with metadata."""
    filename: str
    metadata: ImageMetadata
    status: str = "success"


class ImageEmbeddingResponse(BaseModel):
    """Response model for image embedding generation."""
    filename: str
    metadata_id: str
    embedding_generated: bool
    embedding_dimensions: int
    status: str = "success"


class ImageSearchResult(BaseModel):
    """Model for image similarity search results."""
    metadata_id: str
    filename: str
    title: str
    keywords: List[str]
    description: str
    similarity_score: float


class ImageSimilaritySearchResponse(BaseModel):
    """Response model for image similarity search."""
    query: str
    results: List[ImageSearchResult]
    num_results: int
    status: str = "success"
