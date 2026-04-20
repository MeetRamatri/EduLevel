import json
import os
from typing import List
from sentence_transformers import SentenceTransformer, util
from models import ImageMetadata, ImageSearchResult

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

def retrieve_relevant_image(
    query: str, 
    json_filepath: str = "image_embeddings.json",
    top_k: int = 1
) -> List[ImageSearchResult]:
    """
    Converts a query or generated answer to an embedding, matches it against 
    stored image embeddings, and returns the most relevant image(s).
    """
    if not os.path.exists(json_filepath):
        print(f"Warning: {json_filepath} not found.")
        return []
        
    with open(json_filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
            
    # Load valid models that have an embedding
    valid_items = [ImageMetadata(**item) for item in data if item.get("embedding")]
    if not valid_items:
        return []
        
    corpus_embeddings = [item.embedding for item in valid_items]
    
    # Generate embedding for the input query
    query_embedding = embedding_model.encode(query)
    
    # Calculate cosine similarity using sentence_transformers util
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Pair scores with items and sort descending by score
    scored_items = list(zip(cos_scores.tolist(), valid_items))
    scored_items.sort(key=lambda x: x[0], reverse=True)
    
    # Build the result payload based on our models
    results = []
    for score, item in scored_items[:top_k]:
        results.append(
            ImageSearchResult(
                metadata_id=item.id,
                filename=item.filename,
                title=item.title,
                keywords=item.keywords,
                description=item.description,
                similarity_score=float(score)
            )
        )
        
    return results

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
    
    # Test retrieval
    search_query = "What does the inside of a plant cell look like?"
    matches = retrieve_relevant_image(search_query)
    if matches:
        print(f"Found match: {matches[0].filename} (Score: {matches[0].similarity_score:.4f})")