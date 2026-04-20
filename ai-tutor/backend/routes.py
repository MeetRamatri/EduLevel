"""API route definitions."""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File

from models import (
    QueryRequest, QueryResponse, PDFUploadResponse, PDFChunkedResponse,
    EmbeddingStorageResponse, SimilaritySearchResponse, ImageUploadResponse
)
from pdf_service import extract_text_from_pdf, chunk_text
from embedding_service import generate_embeddings, save_embeddings, similarity_search
from llm_service import generate_rag_answer
from image_service import generate_image_metadata, save_image_metadata
from image_embeddings import retrieve_relevant_image, generate_and_store_image_embeddings
from config import EMBEDDINGS_DIR, ALLOWED_IMAGE_TYPES, ALLOWED_IMAGE_EXTENSIONS, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check():
    """Check if the API is healthy."""
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy", "service": "RAG AI Tutor Backend"}


@router.post("/api/ask", response_model=QueryResponse, summary="Ask the AI Tutor")
async def ask_tutor(request: QueryRequest):
    """
    Ask the AI tutor a question about uploaded documents.
    
    The response is grounded in the document content using RAG.
    """
    logger.info(f"Received query: {request.query}")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        base_filename = Path(request.filename).stem
        embedding_file = EMBEDDINGS_DIR / f"{base_filename}_embeddings.json"
        
        if not embedding_file.exists():
            return QueryResponse(answer="No document embeddings found. Please upload a PDF first.")
        
        chunks = similarity_search(request.query, str(embedding_file), top_k=3)
        if not chunks:
            return QueryResponse(answer="No relevant information found in the document.")
        
        answer = generate_rag_answer(request.query, chunks)
        
        # Retrieve relevant image based on the generated answer
        images = retrieve_relevant_image(query=answer, top_k=1)
        relevant_image = images[0] if images else None
        
        return QueryResponse(answer=answer, image=relevant_image)
    
    except Exception as e:
        logger.error(f"Error in ask_tutor: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/upload", response_model=PDFUploadResponse, summary="Upload and Extract Text from PDF")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and extract text from all pages."""
    logger.info(f"Received file: {file.filename}")
    
    if file.content_type != "application/pdf" or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        contents = await file.read()
        extracted_text, page_count = extract_text_from_pdf(contents)
        
        logger.info(f"Successfully extracted text from {file.filename} ({page_count} pages)")
        
        return PDFUploadResponse(
            filename=file.filename,
            text=extracted_text,
            page_count=page_count,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/upload/chunked", response_model=PDFChunkedResponse, summary="Upload PDF and Return Text Chunks")
async def upload_pdf_chunked(
    file: UploadFile = File(...),
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = PDF_CHUNK_OVERLAP
):
    """Upload a PDF and receive text chunks with configurable size and overlap."""
    logger.info(f"Received file: {file.filename} (chunk_size={chunk_size}, overlap={overlap})")
    
    if file.content_type != "application/pdf" or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        contents = await file.read()
        extracted_text, page_count = extract_text_from_pdf(contents)
        chunks = chunk_text(extracted_text, chunk_size=chunk_size, overlap=overlap)
        
        logger.info(f"Successfully processed {file.filename}: {page_count} pages -> {len(chunks)} chunks")
        
        return PDFChunkedResponse(
            filename=file.filename,
            page_count=page_count,
            chunks=chunks,
            total_chunks=len(chunks),
            status="success"
        )
    
    except ValueError as e:
        logger.error(f"Invalid parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/upload/embeddings", response_model=EmbeddingStorageResponse, summary="Upload PDF, Generate Embeddings, and Store Locally")
async def upload_pdf_with_embeddings(
    file: UploadFile = File(...),
    chunk_size: int = PDF_CHUNK_SIZE,
    overlap: int = PDF_CHUNK_OVERLAP
):
    """Upload a PDF, generate embeddings for chunks, and store them locally."""
    logger.info(f"Received file: {file.filename} (generating embeddings)")
    
    if file.content_type != "application/pdf" or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        contents = await file.read()
        extracted_text, page_count = extract_text_from_pdf(contents)
        chunks = chunk_text(extracted_text, chunk_size=chunk_size, overlap=overlap)
        chunks_with_embeddings = generate_embeddings(chunks)
        storage_path = save_embeddings(file.filename, chunks_with_embeddings)
        
        logger.info(f"Successfully processed {file.filename}: {page_count} pages -> {len(chunks)} chunks with embeddings")
        
        return EmbeddingStorageResponse(
            filename=file.filename,
            total_chunks=len(chunks),
            embeddings_saved=len(chunks_with_embeddings),
            storage_path=storage_path,
            status="success"
        )
    
    except ValueError as e:
        logger.error(f"Invalid parameters: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/search", response_model=SimilaritySearchResponse, summary="Search Similar Chunks Using Query")
async def search_chunks(query: str, filename: str, top_k: int = 3):
    """Search for similar chunks based on a query."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10.")
    
    try:
        base_filename = Path(filename).stem
        embedding_file = EMBEDDINGS_DIR / f"{base_filename}_embeddings.json"
        
        if not embedding_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Embeddings file not found for '{filename}'. Please upload the PDF first."
            )
        
        results = similarity_search(query, str(embedding_file), top_k=top_k)
        
        logger.info(f"Similarity search completed for query: '{query}' with {len(results)} results")
        
        return SimilaritySearchResponse(
            query=query,
            embedding_file=str(embedding_file),
            results=results,
            num_results=len(results),
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during similarity search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during similarity search: {str(e)}")


@router.post("/upload/image", response_model=ImageUploadResponse, summary="Upload Image and Generate Metadata")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and generate AI-powered metadata."""
    logger.info(f"Received image file: {file.filename}")
    
    if file.content_type not in ALLOWED_IMAGE_TYPES or not any(file.filename.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, GIF, WebP, BMP, TIFF)."
        )
    
    try:
        contents = await file.read()
        image_size = len(contents)
        image_info = f"File size: {image_size} bytes, Content type: {file.content_type}"
        
        metadata = generate_image_metadata(file.filename, image_info)
        save_image_metadata(metadata)
        
        # Generate and store embedding for semantic search
        generate_and_store_image_embeddings([metadata])
        
        logger.info(f"Successfully processed image {file.filename} and generated metadata")
        
        return ImageUploadResponse(
            filename=file.filename,
            metadata=metadata,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
