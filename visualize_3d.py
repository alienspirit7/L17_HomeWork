#!/usr/bin/env python3
"""
Generate three 3D visualizations of embeddings:
1. Manual PCA implementation
2. Sklearn PCA (for validation)
3. t-SNE
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import plotly.graph_objects as go

from utils.data_io import load_dataset, dataframe_to_matrix
from utils.pca_manual import manual_pca_3d, validate_pca_with_sklearn
from utils.visualization_3d import create_3d_scatter, generate_sklearn_pca, generate_tsne


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D visualizations (Manual PCA, Sklearn PCA, t-SNE).")
    parser.add_argument("--data", required=True, help="Path to embeddings dataset.")
    parser.add_argument("--assignments", required=True, help="Path to clustering results with cluster_id.")
    parser.add_argument("--output-dir", required=True, help="Directory for output visualizations.")
    parser.add_argument("--perplexity", type=int, default=10, help="t-SNE perplexity (default: 10).")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity."
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading embeddings from %s", args.data)
    df_embeddings = load_dataset(Path(args.data), required_cols={"title", "original_group", "normalized_embedding"})
    X = dataframe_to_matrix(df_embeddings, "normalized_embedding")

    # Load cluster assignments
    logging.info("Loading cluster assignments from %s", args.assignments)
    df_assignments = load_dataset(Path(args.assignments), required_cols={"title", "cluster_id"})

    # Merge cluster_id into main dataframe
    cluster_map = dict(zip(df_assignments["title"], df_assignments["cluster_id"]))
    df_embeddings["cluster_id"] = df_embeddings["title"].map(cluster_map)

    if df_embeddings["cluster_id"].isna().any():
        missing = df_embeddings[df_embeddings["cluster_id"].isna()]["title"].tolist()
        raise ValueError(f"Missing cluster assignments for {len(missing)} titles")

    original_dims = X.shape[1]
    logging.info("Data loaded: %d samples, %d dimensions", X.shape[0], original_dims)

    # 1. Manual PCA
    logging.info("=" * 60)
    logging.info("Generating Manual PCA visualization")
    logging.info("=" * 60)
    start_time = time.time()
    manual_result = manual_pca_3d(X)
    X_manual, variance_manual, eigenvalues, P = manual_result
    manual_elapsed = time.time() - start_time
    logging.info("Manual PCA completed in %.3fs", manual_elapsed)

    subtitle_manual = f"Variance Explained: {variance_manual*100:.2f}%"
    fig_manual = create_3d_scatter(
        X_manual, df_embeddings,
        "Manual PCA: High-Dimensional → 3D Reduction",
        subtitle_manual,
        original_dims,
        manual_elapsed
    )
    manual_path = output_dir / "manual_pca_3d.html"
    fig_manual.write_html(str(manual_path))
    logging.info("Saved manual PCA visualization to %s", manual_path)

    # 2. Sklearn PCA
    logging.info("=" * 60)
    logging.info("Generating Sklearn PCA visualization")
    logging.info("=" * 60)
    sklearn_path = output_dir / "sklearn_pca_3d.html"
    sklearn_result = generate_sklearn_pca(X, df_embeddings, sklearn_path)

    # 3. Validate Manual vs Sklearn
    logging.info("=" * 60)
    logging.info("Validating Manual PCA against Sklearn")
    logging.info("=" * 60)
    validation_report = validate_pca_with_sklearn(X, manual_result)
    logging.info(
        "Validation metrics — eigenvalue_diff=%.2e variance_diff=%.2e coords_match=%s",
        validation_report["eigenvalue_max_diff"],
        validation_report["variance_explained_diff"],
        validation_report["coordinates_match"]
    )

    # 4. t-SNE
    logging.info("=" * 60)
    logging.info("Generating t-SNE visualization")
    logging.info("=" * 60)
    tsne_path = output_dir / "tsne_3d.html"
    tsne_result = generate_tsne(X, df_embeddings, tsne_path, args.perplexity)

    # Summary
    logging.info("=" * 60)
    logging.info("3D Visualization Generation Complete")
    logging.info("=" * 60)
    logging.info("Outputs:")
    logging.info("  - Manual PCA: %s", manual_path)
    logging.info("  - Sklearn PCA: %s", sklearn_path)
    logging.info("  - t-SNE: %s", tsne_path)
    logging.info("PCA Validation: %s",
                 "PASSED ✓" if validation_report['all_checks_passed'] else "FAILED ✗")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Failed: %s", exc)
        sys.exit(1)
