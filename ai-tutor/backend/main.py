from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import fitz
import io
import json
import numpy as np
import os
from typing import List
from sentence_transformers import SentenceTransformer
from pathlib import Path
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDINGS_DIR = Path("./embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)

def get_embedding_model():
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully.")
    return embedding_model

app = FastAPI(
    title="RAG AI Tutor API",
    description="Backend for a RAG-based AI tutor.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    filename: str = "document"

class QueryResponse(BaseModel):
    answer: str

class PDFUploadResponse(BaseModel):
    filename: str
    text: str
    page_count: int
    status: str = "success"

class ChunkWithEmbedding(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]

class EmbeddingStorageResponse(BaseModel):
    filename: str
    total_chunks: int
    embeddings_saved: int
    storage_path: str
    status: str = "success"

class Chunk(BaseModel):
    chunk_id: str
    content: str
    word_count: int

class PDFChunkedResponse(BaseModel):
    filename: str
    page_count: int
    chunks: List[Chunk]
    total_chunks: int
    status: str = "success"

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    similarity_score: float

class SimilaritySearchResponse(BaseModel):
    query: str
    embedding_file: str
    results: List[SearchResult]
    num_results: int
    status: str = "success"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    words = text.split()
    chunks = []
    chunk_id = 0
    step_size = chunk_size - overlap
    
    if step_size <= 0:
        raise ValueError("Chunk size must be greater than overlap")
    
    start_idx = 0
    while start_idx < len(words):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)
        
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{chunk_id:04d}",
                content=chunk_text,
                word_count=len(chunk_words)
            )
        )
        
        chunk_id += 1
        start_idx += step_size
        
        if end_idx == len(words):
            break
    
    return chunks

def generate_embeddings(chunks: List[Chunk]) -> List[ChunkWithEmbedding]:
    model = get_embedding_model()
    
    texts = [chunk.content for chunk in chunks]
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=False)
    
    chunks_with_embeddings = [
        ChunkWithEmbedding(
            chunk_id=chunk.chunk_id,
            text=chunk.content,
            embedding=emb.tolist() if hasattr(emb, 'tolist') else emb
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    
    logger.info(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
    return chunks_with_embeddings

def save_embeddings(filename: str, chunks_with_embeddings: List[ChunkWithEmbedding]) -> str:

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
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    
    query_norm = max(query_norm, 1e-8)
    chunk_norms = np.maximum(chunk_norms, 1e-8)
    
    similarities = np.dot(chunk_embeddings, query_embedding) / (chunk_norms * query_norm)
    return similarities

def similarity_search(query: str, embedding_file_path: str, top_k: int = 3) -> List[SearchResult]:

    with open(embedding_file_path, "r") as f:
        embedding_data = json.load(f)
    
    logger.info(f"Loaded {len(embedding_data)} embeddings from {embedding_file_path}")
    
    model = get_embedding_model()
    query_embedding = model.encode(query, convert_to_numpy=True)
    logger.info(f"Generated query embedding for: '{query}'")
    
    chunk_embeddings = np.array([item["embedding"] for item in embedding_data])
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

def generate_rag_prompt(query: str, chunks: List[SearchResult]) -> str:
    context = "\n\n".join([f"Chunk {i+1}: {chunk.text}" for i, chunk in enumerate(chunks)])
    return f"""You are an AI tutor. Answer ONLY from the context below. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

def call_llm(prompt: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        return "I apologize, but I'm unable to generate a response at the moment. Please try again later."

def generate_rag_answer(query: str, filename: str, top_k: int = 3) -> str:
    try:
        base_filename = Path(filename).stem
        embedding_file = EMBEDDINGS_DIR / f"{base_filename}_embeddings.json"
        
        if not embedding_file.exists():
            return "No document embeddings found. Please upload a PDF first."
        
        chunks = similarity_search(query, str(embedding_file), top_k=top_k)
        if not chunks:
            return "No relevant information found in the document."
        
        prompt = generate_rag_prompt(query, chunks)
        answer = call_llm(prompt)
        return answer
    except Exception as e:
        logger.error(f"RAG answer generation failed: {str(e)}")
        return "An error occurred while processing your question. Please try again."

@app.get("/health", summary="Health Check")
async def health_check():
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy", "service": "RAG AI Tutor Backend"}

@app.post("/api/ask", response_model=QueryResponse, summary="Ask the AI Tutor")
async def ask_tutor(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    answer = generate_rag_answer(request.query, request.filename)
    
    return QueryResponse(answer=answer)

@app.post("/upload", response_model=PDFUploadResponse, summary="Upload and Extract Text from PDF")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    
    # Validate file type
    if file.content_type != "application/pdf" or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        # Read file contents
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        # Open and extract text from PDF
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        extracted_text = ""
        page_count = doc.page_count
        
        # Extract text from all pages
        for page_num in range(page_count):
            page = doc[page_num]
            extracted_text += page.get_text()
        
        doc.close()
        
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

@app.post("/upload/chunked", response_model=PDFChunkedResponse, summary="Upload PDF and Return Text Chunks")
async def upload_pdf_chunked(file: UploadFile = File(...), chunk_size: int = 500, overlap: int = 50):

    logger.info(f"Received file: {file.filename} (chunk_size={chunk_size}, overlap={overlap})")
    
    if file.content_type != "application/pdf" or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        extracted_text = ""
        page_count = doc.page_count
        
        for page_num in range(page_count):
            page = doc[page_num]
            extracted_text += page.get_text()
        
        doc.close()
        
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

@app.post("/upload/embeddings", response_model=EmbeddingStorageResponse, summary="Upload PDF, Generate Embeddings, and Store Locally")
async def upload_pdf_with_embeddings(file: UploadFile = File(...), chunk_size: int = 500, overlap: int = 50):

    logger.info(f"Received file: {file.filename} (generating embeddings)")
    
    if file.content_type != "application/pdf" or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    
    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)

        doc = fitz.open(stream=pdf_file, filetype="pdf")
        extracted_text = ""
        page_count = doc.page_count
        
        for page_num in range(page_count):
            page = doc[page_num]
            extracted_text += page.get_text()
        
        doc.close()
        
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

@app.post("/search", response_model=SimilaritySearchResponse, summary="Search Similar Chunks Using Query")
async def search_chunks(query: str, filename: str, top_k: int = 3):

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
