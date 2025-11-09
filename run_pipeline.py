#!/usr/bin/env python3
"""
Full pipeline orchestrator: Embedding → Clustering (k=3) → 3D Visualizations.
"""
from __future__ import annotations

import argparse
import logging
import sys
import subprocess
from pathlib import Path

from utils.data_io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: Gemini embeddings → K-Means (k=3) → 3D visualizations."
    )
    parser.add_argument("--input", required=True, help="Path to CSV with 'title' and 'group' columns.")
    parser.add_argument(
        "--output-dir", default="pipeline_output",
        help="Directory for all output artifacts (default: pipeline_output)."
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Format for embeddings dataset (default: parquet)."
    )
    parser.add_argument("--gemini-model", help="Gemini embedding model (default: models/text-embedding-004).")
    parser.add_argument("--gemini-api-key", help="Gemini API key (or use GEMINI_API_KEY env var).")
    parser.add_argument("--config", help="Optional JSON config file for Gemini settings.")
    parser.add_argument("--perplexity", type=int, default=10, help="t-SNE perplexity (default: 10).")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity."
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(message)s")


def run_command(cmd: list[str]) -> None:
    """Execute a subprocess command."""
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)

    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gemini_model = config.get("gemini_model", args.gemini_model)
    gemini_api_key = config.get("gemini_api_key", args.gemini_api_key)

    # Step 1: Prepare embeddings
    logging.info("=" * 60)
    logging.info("STEP 1: Generating Embeddings")
    logging.info("=" * 60)
    embeddings_base = output_dir / "embeddings"
    prepare_cmd = [sys.executable, str(base_dir / "prepare_embeddings.py"),
                   "--input", str(Path(args.input).resolve()),
                   "--output", str(embeddings_base),
                   "--format", args.format]
    if args.config:
        prepare_cmd.extend(["--config", args.config])
    if gemini_model:
        prepare_cmd.extend(["--gemini-model", gemini_model])
    if gemini_api_key:
        prepare_cmd.extend(["--gemini-api-key", gemini_api_key])
    run_command(prepare_cmd)

    dataset_path = embeddings_base.with_suffix(f".{args.format}")
    manifest_path = embeddings_base.with_suffix(".manifest.json")

    # Step 2: K-Means clustering (k=3)
    logging.info("=" * 60)
    logging.info("STEP 2: K-Means Clustering (k=3)")
    logging.info("=" * 60)
    clustering_output = output_dir / "clustering_results.csv"
    report_path = output_dir / "kmeans_report.md"
    centroids_path = output_dir / "centroids.json"

    kmeans_cmd = [sys.executable, str(base_dir / "run_kmeans.py"),
                  "--data", str(dataset_path),
                  "--output", str(clustering_output),
                  "--report", str(report_path),
                  "--centroids", str(centroids_path)]
    if manifest_path.exists():
        kmeans_cmd.extend(["--manifest", str(manifest_path)])
    run_command(kmeans_cmd)

    # Step 3: 3D Visualizations
    logging.info("=" * 60)
    logging.info("STEP 3: Generating 3D Visualizations")
    logging.info("=" * 60)
    viz_dir = output_dir / "visualizations"
    viz_cmd = [sys.executable, str(base_dir / "visualize_3d.py"),
               "--data", str(dataset_path),
               "--assignments", str(clustering_output),
               "--output-dir", str(viz_dir),
               "--perplexity", str(args.perplexity)]
    run_command(viz_cmd)

    # Summary
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)
    logging.info("Outputs in: %s", output_dir)
    logging.info("  - Embeddings: %s", dataset_path)
    logging.info("  - Clustering: %s", clustering_output)
    logging.info("  - Visualizations: %s/", viz_dir)
    logging.info("    • manual_pca_3d.html")
    logging.info("    • sklearn_pca_3d.html")
    logging.info("    • tsne_3d.html")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        sys.exit(1)
