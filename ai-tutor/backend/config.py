"""Configuration and constants for the RAG AI Tutor backend."""

import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory paths
EMBEDDINGS_DIR = Path("./embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)

METADATA_DIR = Path("./metadata")
METADATA_DIR.mkdir(exist_ok=True)

# API Configuration
APP_TITLE = "RAG AI Tutor API"
APP_DESCRIPTION = "Backend for a RAG-based AI tutor."
APP_VERSION = "1.0.0"

# CORS Configuration
CORS_ORIGINS = ["*"]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# PDF Configuration
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50

# Embedding Configuration
EMBEDDING_MODEL = "gemini-1.5-mini"

# LLM Configuration
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1000

# Image Upload Configuration
ALLOWED_IMAGE_TYPES = [
    "image/jpeg", "image/jpg", "image/png", "image/gif",
    "image/webp", "image/bmp", "image/tiff"
]
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]
