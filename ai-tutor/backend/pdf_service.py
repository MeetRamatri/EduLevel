"""PDF processing and text extraction services."""

import logging
import fitz
import io
from pathlib import Path
from typing import List

from models import Chunk

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_contents: bytes) -> tuple[str, int]:
    """
    Extract text from PDF file contents.
    
    Args:
        file_contents: PDF file contents as bytes
        
    Returns:
        Tuple of (extracted_text, page_count)
    """
    pdf_file = io.BytesIO(file_contents)
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    extracted_text = ""
    page_count = doc.page_count
    
    for page_num in range(page_count):
        page = doc[page_num]
        extracted_text += page.get_text()
    
    doc.close()
    logger.info(f"Successfully extracted text from PDF ({page_count} pages)")
    return extracted_text, page_count


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of Chunk objects
    """
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
    
    logger.info(f"Text split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks
