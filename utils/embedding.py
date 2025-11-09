"""Gemini embedding generation utilities."""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np


def embed_titles_gemini(model_id: str, titles: List[str], api_key: Optional[str]) -> np.ndarray:
    """Generate embeddings for titles using Gemini API.

    Args:
        model_id: Gemini model identifier
        titles: List of title strings to embed
        api_key: Gemini API key (or None to use env var)

    Returns:
        Array of embeddings with shape (len(titles), embedding_dim)
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not provided. Supply --gemini-api-key or set GEMINI_API_KEY.")
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError("google-generativeai package is required for Gemini embeddings.") from exc

    genai.configure(api_key=api_key)
    embeddings: List[List[float]] = []
    logging.info("Requesting embeddings from Gemini model %s", model_id)
    for title in titles:
        response = genai.embed_content(model=model_id, content=title)
        vector = response.get("embedding")
        if vector is None:
            raise ValueError("Gemini response missing 'embedding' field.")
        embeddings.append(vector)
    return np.asarray(embeddings, dtype=float)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length.

    Args:
        embeddings: Array of embeddings

    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms
