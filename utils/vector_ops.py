"""Vector operations and conversions."""
from __future__ import annotations

import json
from typing import Any, List

import numpy as np


def ensure_vector_list(value: Any) -> List[float]:
    """Convert various vector representations to a list of floats.

    Args:
        value: Vector in various formats (list, ndarray, string, etc.)

    Returns:
        List of float values
    """
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            from ast import literal_eval

            try:
                parsed = literal_eval(value)
            except (SyntaxError, ValueError):
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    inner = stripped[1:-1]
                    parts = [part for part in inner.replace("\n", " ").split(" ") if part]
                    if parts:
                        return [float(part) for part in parts]
                raise
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected list-like string, got {type(parsed)}")
        return [float(v) for v in parsed]
    if isinstance(value, tuple):
        return [float(v) for v in value]
    raise TypeError(f"Unsupported vector type: {type(value)}")


def serialize_vector(vec: np.ndarray) -> List[float]:
    """Convert numpy array to serializable list.

    Args:
        vec: Numpy array

    Returns:
        List of floats
    """
    return vec.astype(float).tolist()


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length.

    Args:
        vectors: Array of vectors

    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
