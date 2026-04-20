"""LLM integration and prompt generation services."""

import logging
import os
from typing import List
from groq import Groq

from models import SearchResult
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


def call_llm(prompt: str) -> str:
    """
    Call the Groq LLM API with a prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        Response text from the LLM
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        return "I apologize, but I'm unable to generate a response at the moment. Please try again later."


def generate_rag_prompt(query: str, chunks: List[SearchResult]) -> str:
    """
    Generate a RAG prompt with context from retrieved chunks.
    
    Args:
        query: User query
        chunks: Retrieved similar chunks
        
    Returns:
        Formatted prompt for the LLM
    """
    context = "\n\n".join([f"Chunk {i+1}: {chunk.text}" for i, chunk in enumerate(chunks)])
    return f"""You are an AI tutor. Answer ONLY from the context below. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."

Context:
{context}

Question: {query}

Answer:"""


def generate_rag_answer(query: str, chunks: List[SearchResult]) -> str:
    """
    Generate an answer using RAG (Retrieval Augmented Generation).
    
    Args:
        query: User query
        chunks: Retrieved similar chunks
        
    Returns:
        Generated answer
    """
    if not chunks:
        return "No relevant information found in the document."
    
    prompt = generate_rag_prompt(query, chunks)
    answer = call_llm(prompt)
    return answer


def generate_metadata_prompt(filename: str, additional_info: str = "") -> str:
    """
    Generate a prompt for image metadata generation.
    
    Args:
        filename: Image filename
        additional_info: Additional information about the image
        
    Returns:
        Formatted prompt for metadata generation
    """
    return f"""You are an AI assistant that creates metadata for images. Based on the filename and any additional information provided, generate appropriate metadata.

Filename: {filename}
Additional Info: {additional_info}

Please provide metadata in the following JSON format:
{{
    "title": "A descriptive title for the image",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "description": "A detailed description of what the image likely contains or represents"
}}

Make the title engaging and descriptive. Keywords should be relevant tags that would help in searching. Description should be comprehensive but concise.

Output only the JSON object, no additional text."""
