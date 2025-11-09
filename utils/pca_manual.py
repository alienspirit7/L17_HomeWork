"""Manual PCA implementation for educational and validation purposes."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np


def manual_pca_3d(X: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Perform manual PCA reduction to 3 dimensions.

    Algorithm:
    1. Center the data (subtract mean)
    2. Compute covariance matrix: C = (X^T × X) / (n-1)
    3. Calculate eigenvalues and eigenvectors of C
    4. Sort eigenvectors by eigenvalues (descending order)
    5. Create transformation matrix P from top 3 eigenvectors
    6. Transform data: Y = X × P
    7. Calculate variance explained

    Args:
        X: Input data matrix (n_samples, n_features)

    Returns:
        - Transformed data (n_samples, 3)
        - Variance explained ratio (float)
        - Eigenvalues (sorted descending)
        - Transformation matrix P (n_features, 3)
    """
    logging.info("Starting manual PCA: input shape %s", X.shape)

    # 1. Center the data
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    logging.debug("Data centered: mean shape %s", X_mean.shape)

    # 2. Compute covariance matrix
    n = X_centered.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
    logging.debug("Covariance matrix computed: shape %s", cov_matrix.shape)

    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    logging.debug("Eigendecomposition complete: %d eigenvalues", len(eigenvalues))

    # 4. Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Log top eigenvalues
    logging.info("Top 5 eigenvalues: %s", eigenvalues[:5])

    # 5. Select top 3 eigenvectors
    P = eigenvectors[:, :3]
    logging.debug("Transformation matrix P: shape %s", P.shape)

    # 6. Transform data
    X_transformed = X_centered @ P
    logging.info("Data transformed: output shape %s", X_transformed.shape)

    # 7. Calculate variance explained
    total_variance = eigenvalues.sum()
    variance_top3 = eigenvalues[:3].sum()
    variance_explained = variance_top3 / total_variance

    logging.info("Variance explained by 3 components: %.4f%%", variance_explained * 100)

    return X_transformed, variance_explained, eigenvalues, P


def validate_pca_with_sklearn(X: np.ndarray, manual_result: Tuple, tolerance: float = 1e-6) -> dict:
    """
    Validate manual PCA against sklearn implementation.

    Args:
        X: Original input data
        manual_result: Tuple from manual_pca_3d
        tolerance: Numerical tolerance for comparison

    Returns:
        Validation report dictionary
    """
    from sklearn.decomposition import PCA

    manual_transformed, manual_variance, manual_eigenvalues, manual_P = manual_result

    # Run sklearn PCA
    pca_sklearn = PCA(n_components=3)
    sklearn_transformed = pca_sklearn.fit_transform(X)
    sklearn_variance = pca_sklearn.explained_variance_ratio_.sum()
    sklearn_eigenvalues = pca_sklearn.explained_variance_

    # Compare eigenvalues (sklearn returns explained variance, not eigenvalues directly)
    # Explained variance = eigenvalue / (n-1) for covariance matrix
    n = X.shape[0]
    manual_explained_var = manual_eigenvalues[:3] / (n - 1)
    eigenvalue_diff = np.abs(manual_explained_var - sklearn_eigenvalues).max()

    # Compare variance explained
    variance_diff = abs(manual_variance - sklearn_variance)

    # Compare transformed data (allow sign flips)
    coords_match = True
    for i in range(3):
        manual_col = manual_transformed[:, i]
        sklearn_col = sklearn_transformed[:, i]
        # Check if columns match or are negatives of each other
        if not (np.allclose(manual_col, sklearn_col, atol=tolerance) or
                np.allclose(manual_col, -sklearn_col, atol=tolerance)):
            coords_match = False
            break

    report = {
        "eigenvalue_max_diff": float(eigenvalue_diff),
        "variance_explained_diff": float(variance_diff),
        "eigenvalues_match": eigenvalue_diff < tolerance,
        "variance_match": variance_diff < tolerance,
        "coordinates_match": coords_match,
        "manual_variance": float(manual_variance),
        "sklearn_variance": float(sklearn_variance),
        "all_checks_passed": (eigenvalue_diff < tolerance and
                             variance_diff < tolerance and
                             coords_match)
    }

    logging.info("PCA Validation: eigenvalue_diff=%.2e, variance_diff=%.2e, coords_match=%s",
                 eigenvalue_diff, variance_diff, coords_match)

    return report
