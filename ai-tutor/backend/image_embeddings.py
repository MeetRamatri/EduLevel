import json
import os
from typing import List

import numpy as np
import google.generativeai as genai

from models import ImageMetadata, ImageSearchResult

# Use Gemini embeddings via Google Gemini.
MODEL_NAME = "gemini-1.5-mini"


def _validate_google_api_key() -> str:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY must be set in the environment to generate Gemini embeddings."
        )
    return google_api_key


def _get_embedding(text: str) -> List[float]:
    google_api_key = _validate_google_api_key()
    genai.configure(api_key=google_api_key)
    response = genai.embed_content(model="models/embedding-001", content=text)
    return response["embedding"]


def _cosine_similarity(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    query_norm = max(query_norm, 1e-8)
    corpus_norms = np.maximum(corpus_norms, 1e-8)
    return np.dot(corpus_embeddings, query_embedding) / (corpus_norms * query_norm)


def generate_and_store_image_embeddings(
    metadata_list: List[ImageMetadata], 
    json_filepath: str = "image_embeddings.json"
) -> None:
    """
    Generates embeddings from description + keywords and stores them in a JSON file.
    """
    existing_models = {}
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_models = {item['id']: ImageMetadata(**item) for item in existing_data}
        except json.JSONDecodeError:
            print(f"Warning: {json_filepath} was empty or invalid. Starting fresh.")

    for metadata in metadata_list:
        text_to_embed = metadata.get_embedding_text()
        metadata.embedding = _get_embedding(text_to_embed)
        existing_models[metadata.id] = metadata

    with open(json_filepath, 'w', encoding='utf-8') as f:
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

    valid_items = [ImageMetadata(**item) for item in data if item.get("embedding")]
    if not valid_items:
        return []

    corpus_embeddings = np.array([item.embedding for item in valid_items], dtype=float)
    query_embedding = np.array(_get_embedding(query), dtype=float)
    cos_scores = _cosine_similarity(query_embedding, corpus_embeddings)

    scored_items = list(zip(cos_scores.tolist(), valid_items))
    scored_items.sort(key=lambda x: x[0], reverse=True)

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

    search_query = "What does the inside of a plant cell look like?"
    matches = retrieve_relevant_image(search_query)
    if matches:
        print(f"Found match: {matches[0].filename} (Score: {matches[0].similarity_score:.4f})")