"""3D visualization utilities using Plotly."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Cluster color scheme
CLUSTER_COLORS = {
    0: 'red',
    1: 'blue',
    2: 'green'
}

# Shape markers for original groups
SHAPE_MARKERS = [
    'circle', 'square', 'diamond', 'cross',
    'x', 'circle-open', 'square-open', 'diamond-open'
]


def create_3d_scatter(
    data_3d: np.ndarray,
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    original_dims: int,
    elapsed_time: float = None,
) -> go.Figure:
    """
    Create a 3D scatter plot with cluster colors and group shapes.

    Args:
        data_3d: 3D transformed data (n_samples, 3)
        df: DataFrame with 'cluster_id' and 'original_group' columns
        title: Main plot title
        subtitle: Subtitle (variance info)
        original_dims: Number of original dimensions
        elapsed_time: Time taken for transformation in seconds

    Returns:
        Plotly Figure object
    """
    # Map groups to shapes
    unique_groups = sorted(df['original_group'].unique())
    group_to_shape = {
        group: SHAPE_MARKERS[i % len(SHAPE_MARKERS)]
        for i, group in enumerate(unique_groups)
    }

    # Create figure
    fig = go.Figure()

    # Add traces for each group
    for group in unique_groups:
        group_mask = df['original_group'] == group
        group_df = df[group_mask].copy()
        group_df['x'] = data_3d[group_mask, 0]
        group_df['y'] = data_3d[group_mask, 1]
        group_df['z'] = data_3d[group_mask, 2]

        # Get colors for this group based on cluster_id
        colors = [CLUSTER_COLORS.get(int(cid), 'gray') for cid in group_df['cluster_id']]

        # Create hover text
        hover_text = [
            f"Title: {row['title']}<br>Group: {row['original_group']}<br>Cluster: {int(row['cluster_id'])}"
            for _, row in group_df.iterrows()
        ]

        fig.add_trace(go.Scatter3d(
            x=group_df['x'],
            y=group_df['y'],
            z=group_df['z'],
            mode='markers',
            name=group,
            marker=dict(
                size=8,
                color=colors,
                symbol=group_to_shape[group],
                line=dict(color='black', width=0.5)
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))

    # Build title text
    title_parts = [title, f"<sub>{subtitle}</sub>", f"<sub>Dimensions: {original_dims} → 3</sub>"]
    if elapsed_time is not None:
        title_parts.append(f"<sub>Processing Time: {elapsed_time:.3f}s</sub>")
    title_text = "<br>".join(title_parts)

    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
        ),
        legend=dict(
            title="Original Groups",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1000,
        height=800
    )

    return fig


def generate_sklearn_pca(X: np.ndarray, df: pd.DataFrame, output_path: Path) -> Dict:
    """Generate sklearn PCA 3D visualization."""
    logging.info("Generating sklearn PCA visualization")
    start_time = time.time()

    pca = PCA(n_components=3, random_state=42)
    X_transformed = pca.fit_transform(X)
    variance_explained = pca.explained_variance_ratio_.sum()

    elapsed = time.time() - start_time
    logging.info("Sklearn PCA completed in %.3fs, variance explained: %.4f%%", elapsed, variance_explained * 100)

    subtitle = f"Variance Explained: {variance_explained*100:.2f}%"
    fig = create_3d_scatter(
        X_transformed, df,
        "Sklearn PCA: High-Dimensional → 3D Reduction",
        subtitle,
        X.shape[1],
        elapsed
    )

    fig.write_html(str(output_path))
    logging.info("Saved sklearn PCA visualization to %s", output_path)

    return {
        "variance_explained": float(variance_explained),
        "transformed_shape": X_transformed.shape,
        "elapsed_time": elapsed
    }


def generate_tsne(X: np.ndarray, df: pd.DataFrame, output_path: Path, perplexity: int = 10) -> Dict:
    """Generate t-SNE 3D visualization."""
    logging.info("Generating t-SNE visualization with perplexity=%d", perplexity)
    start_time = time.time()

    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    X_transformed = tsne.fit_transform(X)

    elapsed = time.time() - start_time
    logging.info("t-SNE completed in %.3fs", elapsed)

    subtitle = f"Perplexity: {perplexity} (t-SNE does not compute variance explained)"
    fig = create_3d_scatter(
        X_transformed, df,
        "t-SNE: High-Dimensional → 3D Reduction",
        subtitle,
        X.shape[1],
        elapsed
    )

    fig.write_html(str(output_path))
    logging.info("Saved t-SNE visualization to %s", output_path)

    return {
        "perplexity": perplexity,
        "transformed_shape": X_transformed.shape,
        "elapsed_time": elapsed
    }
