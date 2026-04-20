from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import fitz
import io
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class QueryResponse(BaseModel):
    answer: str

class PDFUploadResponse(BaseModel):
    filename: str
    text: str
    page_count: int
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

@app.get("/health", summary="Health Check")
async def health_check():
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy", "service": "RAG AI Tutor Backend"}

@app.post("/api/ask", response_model=QueryResponse, summary="Ask the AI Tutor")
async def ask_tutor(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    response_text = f"You asked: '{request.query}'. This is a placeholder response from the AI tutor. RAG pipeline goes here."
    return QueryResponse(answer=response_text)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
