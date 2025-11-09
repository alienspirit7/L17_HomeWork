"""Data loading and saving utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .vector_ops import ensure_vector_list


SUPPORTED_FORMATS = {"parquet", "jsonl", "csv"}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        path: Path to config file

    Returns:
        Configuration dictionary
    """
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_input_csv(path: Path) -> pd.DataFrame:
    """Read and validate input CSV with title and group columns.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with 'title' and 'original_group' columns
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    normalized_map = {col.strip().lower(): col for col in df.columns}
    expected_cols = {"title", "group"}
    missing = [col for col in expected_cols if col not in normalized_map]
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    title_col = normalized_map["title"]
    group_col = normalized_map["group"]
    df = df[[title_col, group_col]]
    df.rename(columns={title_col: "title", group_col: "original_group"}, inplace=True)
    df["title"] = df["title"].astype(str).str.strip()
    df["original_group"] = df["original_group"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["original_group"] != "")]
    return df.reset_index(drop=True)


def load_dataset(path: Path, required_cols: Optional[set] = None) -> pd.DataFrame:
    """Load dataset from parquet, CSV, or JSONL.

    Args:
        path: Path to dataset file
        required_cols: Set of required column names

    Returns:
        DataFrame with loaded data
    """
    ext = path.suffix.lower()
    # Remove leading dot for comparison
    fmt = ext.lstrip('.')
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Expected one of {sorted(SUPPORTED_FORMATS)}")

    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:  # .jsonl
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)

    if required_cols:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df.reset_index(drop=True)


def dataframe_to_matrix(df: pd.DataFrame, column: str) -> np.ndarray:
    """Convert DataFrame column of vectors to numpy matrix.

    Args:
        df: DataFrame containing vector column
        column: Name of column with vectors

    Returns:
        Numpy array of shape (n_rows, vector_dim)
    """
    vectors = [ensure_vector_list(v) for v in df[column]]
    return np.asarray(vectors, dtype=float)


def write_dataset(df: pd.DataFrame, output_path: Path, fmt: str) -> Path:
    """Write dataset to file in specified format.

    Args:
        df: DataFrame to write
        output_path: Base output path
        fmt: Format ('parquet', 'csv', or 'jsonl')

    Returns:
        Actual path written
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        path = output_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        path = output_path.with_suffix(".csv")
        df.to_csv(path, index=False)
    elif fmt == "jsonl":
        path = output_path.with_suffix(".jsonl")
        with path.open("w", encoding="utf-8") as fh:
            for record in df.to_dict(orient="records"):
                fh.write(json.dumps(record) + "\n")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logging.info("Wrote dataset to %s", path)
    return path
