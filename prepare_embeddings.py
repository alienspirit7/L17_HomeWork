#!/usr/bin/env python3
"""
Prepare embeddings for title data using the Gemini embedding API.
Reads a two-column CSV (title, group), produces raw and normalized embeddings,
and persists outputs alongside a metadata manifest.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

from utils.data_io import load_config, read_input_csv, write_dataset
from utils.embedding import embed_titles_gemini, normalize_embeddings
from utils.vector_ops import serialize_vector

SUPPORTED_FORMATS = {"parquet", "jsonl", "csv"}
DEFAULT_GEMINI_MODEL = "models/text-embedding-004"


@dataclass
class RunManifest:
    script: str
    version: str
    timestamp_utc: str
    input_path: str
    input_rows: int
    processed_rows: int
    skipped_rows: int
    provider: str
    model_id: str
    embedding_dim: int
    normalization: str
    seed: Optional[int]
    elapsed_seconds: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings and normalized vectors for CSV title data using Gemini."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV with columns 'title' and 'group'.")
    parser.add_argument("--output", required=True, help="Base path (without extension) for processed dataset.")
    parser.add_argument(
        "--format", choices=sorted(SUPPORTED_FORMATS), default="parquet", help="Output format for processed dataset."
    )
    parser.add_argument("--manifest", help="Optional path to write manifest JSON. Defaults to <output>.manifest.json.")
    parser.add_argument("--config", help="Optional path to JSON config file providing model/API defaults.")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini embedding model identifier.")
    parser.add_argument("--gemini-api-key", help="Gemini API key. Falls back to GEMINI_API_KEY environment variable.")
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate titles in the input data.")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity."
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(message)s")


def enforce_uniqueness(df: pd.DataFrame, allow_duplicates: bool) -> pd.DataFrame:
    if allow_duplicates:
        return df
    duplicate_titles = df[df.duplicated("title", keep=False)]["title"].unique()
    if duplicate_titles.size > 0:
        raise ValueError(
            f"Duplicate titles detected. Re-run with --allow-duplicates to proceed. Examples: {duplicate_titles[:5]!r}"
        )
    return df


def write_manifest(manifest_path: Path, manifest: RunManifest) -> None:
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    logging.info("Wrote manifest to %s", manifest_path)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    configure_logging(args.log_level)

    start_time = time.time()
    input_path = Path(args.input)
    output_base = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_base.with_suffix(".manifest.json")

    logging.info("Reading input %s", input_path)
    raw_df = read_input_csv(input_path)
    input_rows = len(raw_df)
    logging.info("Loaded %d rows", input_rows)
    logging.info("Embedding provider: Gemini")

    filtered_df = enforce_uniqueness(raw_df, args.allow_duplicates)
    processed_rows = len(filtered_df)
    skipped_rows = input_rows - processed_rows

    if processed_rows == 0:
        raise ValueError("No valid rows to process after applying filters.")

    titles = filtered_df["title"].tolist()
    gemini_model = config.get("gemini_model", args.gemini_model)
    gemini_key = args.gemini_api_key or config.get("gemini_api_key")
    embeddings = embed_titles_gemini(gemini_model, titles, gemini_key)
    model_identifier = gemini_model
    normalized = normalize_embeddings(embeddings)

    embedding_dim = embeddings.shape[1]
    logging.info("Embedding dimension: %d", embedding_dim)

    dataset = filtered_df.copy()
    dataset["embedding"] = [serialize_vector(vec) for vec in embeddings]
    dataset["normalized_embedding"] = [serialize_vector(vec) for vec in normalized]

    output_path = write_dataset(dataset, output_base, args.format)

    manifest = RunManifest(
        script="prepare_embeddings.py",
        version="1.0.0",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        input_path=str(input_path.resolve()),
        input_rows=input_rows,
        processed_rows=processed_rows,
        skipped_rows=skipped_rows,
        provider="gemini",
        model_id=model_identifier,
        embedding_dim=embedding_dim,
        normalization="l2",
        seed=None,
        elapsed_seconds=round(time.time() - start_time, 3),
    )
    write_manifest(manifest_path, manifest)

    logging.info(
        "Processing complete. processed=%d skipped=%d elapsed=%.2fs output=%s",
        processed_rows, skipped_rows, manifest.elapsed_seconds, output_path,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Failed: %s", exc)
        sys.exit(1)
