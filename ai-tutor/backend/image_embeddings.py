import json
import os
from typing import List
from sentence_transformers import SentenceTransformer
from models import ImageMetadata

# Initialize the embedding model 
# (all-MiniLM-L6-v2 is fast, lightweight, and great for semantic search)
MODEL_NAME = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(MODEL_NAME)

def generate_and_store_image_embeddings(
    metadata_list: List[ImageMetadata], 
    json_filepath: str = "image_embeddings.json"
) -> None:
    """
    Generates embeddings from description + keywords and stores them in a JSON file.
    """
    # 1. Load existing data if the file exists so we don't overwrite it
    existing_models = {}
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Store in a dict by ID for easy updating
                existing_models = {item['id']: ImageMetadata(**item) for item in existing_data}
        except json.JSONDecodeError:
            print(f"Warning: {json_filepath} was empty or invalid. Starting fresh.")

    # 2. Generate embeddings and update models
    for metadata in metadata_list:
        # Combine description and keywords using our new schema helper method
        text_to_embed = metadata.get_embedding_text()
        
        # Generate embedding and convert numpy array to list for JSON serialization
        embedding = embedding_model.encode(text_to_embed).tolist()
        metadata.embedding = embedding
        
        # Update or add to our collection
        existing_models[metadata.id] = metadata

    # 3. Save everything back to the JSON file
    with open(json_filepath, 'w', encoding='utf-8') as f:
        # Use .model_dump() for Pydantic v2 (use .dict() if on Pydantic v1)
        json.dump([m.model_dump() for m in existing_models.values()], f, indent=4)
        
    print(f"Successfully saved {len(metadata_list)} embeddings to {json_filepath}")

# Example Usage
if __name__ == "__main__":
    sample_img = ImageMetadata(
        id="img_001",
        filename="cell_structure.png",
        title="Plant Cell",
        keywords=["biology", "plant", "cell", "chloroplast", "wall"],
        description="A diagram outlining the structural components of a plant cell."
    )
    
    generate_and_store_image_embeddings([sample_img])