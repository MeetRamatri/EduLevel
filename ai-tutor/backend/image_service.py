"""Image processing and metadata generation services."""

import logging
import json
import uuid
from pathlib import Path
from typing import List

from models import ImageMetadata
from config import METADATA_DIR
from llm_service import call_llm, generate_metadata_prompt

logger = logging.getLogger(__name__)


def generate_image_metadata(filename: str, image_info: str = "") -> ImageMetadata:
    """
    Generate metadata for an image using LLM.
    
    Args:
        filename: Image filename
        image_info: Additional image information (e.g., file size, content type)
        
    Returns:
        ImageMetadata object with AI-generated metadata
    """
    prompt = generate_metadata_prompt(filename, image_info)
    
    try:
        metadata_json = call_llm(prompt)
        metadata_dict = json.loads(metadata_json.strip())
        
        metadata = ImageMetadata(
            id=str(uuid.uuid4()),
            filename=filename,
            title=metadata_dict.get("title", filename),
            keywords=metadata_dict.get("keywords", []),
            description=metadata_dict.get("description", "")
        )
        
        logger.info(f"Generated metadata for image: {filename}")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to generate image metadata: {str(e)}")
        # Return basic metadata if LLM fails
        return ImageMetadata(
            id=str(uuid.uuid4()),
            filename=filename,
            title=filename,
            keywords=["image"],
            description=f"Image file: {filename}"
        )


def save_image_metadata(metadata: ImageMetadata) -> str:
    """
    Save image metadata to JSON file.
    
    Args:
        metadata: ImageMetadata object to save
        
    Returns:
        Path to saved metadata file
    """
    metadata_file = METADATA_DIR / "images_metadata.json"
    
    # Load existing metadata or create empty list
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                metadata_list = json.load(f)
        except:
            metadata_list = []
    else:
        metadata_list = []
    
    # Add new metadata
    metadata_dict = {
        "id": metadata.id,
        "filename": metadata.filename,
        "title": metadata.title,
        "keywords": metadata.keywords,
        "description": metadata.description,
        "created_at": str(Path(metadata_file).stat().st_mtime) if metadata_file.exists() else str(Path.cwd())
    }
    
    metadata_list.append(metadata_dict)
    
    # Save back to file
    with open(metadata_file, "w") as f:
        json.dump(metadata_list, f, indent=2)
    
    logger.info(f"Image metadata saved to {metadata_file}")
    return str(metadata_file)


def get_all_image_metadata() -> List[dict]:
    """
    Retrieve all stored image metadata.
    
    Returns:
        List of metadata dictionaries
    """
    metadata_file = METADATA_DIR / "images_metadata.json"
    
    if not metadata_file.exists():
        return []
    
    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load image metadata: {str(e)}")
        return []
