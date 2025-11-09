"""K-Means clustering utilities."""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .data_io import dataframe_to_matrix
from .vector_ops import ensure_vector_list


@dataclass
class ClusterMetrics:
    k: int
    inertia: float
    silhouette: Optional[float]
    cluster_sizes: Dict[int, int]
    mismatch_rate: float
    majority_mapping: Dict[int, str]


def compute_majority_mapping(groups: Iterable[str]) -> Tuple[str, float]:
    """Compute majority group and mismatch rate for a cluster.

    Args:
        groups: Iterable of group labels

    Returns:
        Tuple of (majority_group, mismatch_rate)
    """
    counter = Counter(groups)
    majority_group, count = counter.most_common(1)[0]
    total = sum(counter.values())
    mismatch_rate = 1.0 - (count / total)
    return majority_group, mismatch_rate


def build_metrics(df: pd.DataFrame, assignments: np.ndarray, kmeans: KMeans) -> ClusterMetrics:
    """Build clustering metrics from K-Means results.

    Args:
        df: DataFrame with original data
        assignments: Cluster assignments
        kmeans: Fitted KMeans object

    Returns:
        ClusterMetrics object
    """
    df = df.copy()
    df["cluster"] = assignments
    cluster_sizes = df["cluster"].value_counts().sort_index().to_dict()

    mismatches = []
    mapping: Dict[int, str] = {}
    for cluster_id, group_df in df.groupby("cluster"):
        majority_group, mismatch_rate = compute_majority_mapping(group_df["original_group"])
        mapping[cluster_id] = majority_group
        mismatches.append(mismatch_rate * len(group_df))
    total_points = len(df)
    mismatch_rate = sum(mismatches) / total_points if total_points else math.nan

    silhouette: Optional[float] = None
    if total_points >= 2 * kmeans.n_clusters:
        try:
            silhouette = float(silhouette_score(
                dataframe_to_matrix(df, "normalized_embedding"), assignments, metric="cosine"
            ))
        except Exception as exc:
            logging.warning("Unable to compute silhouette score: %s", exc)

    return ClusterMetrics(
        k=kmeans.n_clusters,
        inertia=float(kmeans.inertia_),
        silhouette=silhouette,
        cluster_sizes={int(k): int(v) for k, v in cluster_sizes.items()},
        mismatch_rate=mismatch_rate,
        majority_mapping={int(k): str(v) for k, v in mapping.items()},
    )


def save_record_level(df: pd.DataFrame, assignments: np.ndarray, distances: np.ndarray, path: Path) -> Path:
    """Save detailed cluster assignments to CSV.

    Args:
        df: Original DataFrame
        assignments: Cluster assignments
        distances: Distances to centroids
        path: Output path

    Returns:
        Path to saved file
    """
    logging.info("Writing detailed assignments to %s", path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    result = df.copy()

    def serialize_column(value: Any) -> str:
        vector = ensure_vector_list(value)
        return json.dumps(vector)

    if "embedding" in result.columns:
        result["embedding"] = result["embedding"].apply(serialize_column)
    if "normalized_embedding" in result.columns:
        result["normalized_embedding"] = result["normalized_embedding"].apply(serialize_column)
    result["cluster_id"] = assignments.astype(int)
    result["distance_to_centroid"] = distances.astype(float)
    result.to_csv(path, index=False)
    return path


def render_markdown_report(metrics: ClusterMetrics) -> str:
    """Render clustering metrics as markdown.

    Args:
        metrics: ClusterMetrics object

    Returns:
        Markdown-formatted string
    """
    lines = [
        "# K-Means Report",
        "",
        f"- Clusters (k): {metrics.k}",
        f"- Inertia: {metrics.inertia:.4f}",
        f"- Silhouette (cosine): {metrics.silhouette:.4f}" if metrics.silhouette is not None else "- Silhouette (cosine): N/A",
        f"- Mismatch rate: {metrics.mismatch_rate:.4f}",
        "",
        "| Cluster | Size | Majority Original Group |",
        "| --- | ---: | --- |",
    ]
    for cluster_id, size in sorted(metrics.cluster_sizes.items()):
        majority = metrics.majority_mapping.get(cluster_id, "N/A")
        lines.append(f"| {cluster_id} | {size} | {majority} |")
    return "\n".join(lines) + "\n"


def save_report(metrics: ClusterMetrics, path: Path) -> Path:
    """Save clustering report.

    Args:
        metrics: ClusterMetrics object
        path: Output path

    Returns:
        Path to saved file
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".md", ".markdown"}:
        path.write_text(render_markdown_report(metrics), encoding="utf-8")
    else:
        rows = []
        for cluster_id, size in sorted(metrics.cluster_sizes.items()):
            rows.append({
                "cluster_id": cluster_id,
                "size": size,
                "majority_original_group": metrics.majority_mapping.get(cluster_id, ""),
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
    logging.info("Wrote report to %s", path)
    return path


def save_centroids(path: Path, kmeans: KMeans, manifest: Optional[Dict[str, Any]]) -> Path:
    """Save cluster centroids to JSON.

    Args:
        path: Output path
        kmeans: Fitted KMeans object
        manifest: Optional manifest with metadata

    Returns:
        Path to saved file
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "k": int(kmeans.n_clusters),
        "centroids": kmeans.cluster_centers_.tolist(),
        "embedding_dim": int(kmeans.cluster_centers_.shape[1]),
        "model_id": manifest.get("model_id") if manifest else None,
        "normalization": manifest.get("normalization") if manifest else "l2",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Persisted centroids to %s", path)
    return path
