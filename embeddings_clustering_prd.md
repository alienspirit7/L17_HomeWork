# Title Embedding, Clustering & 3D Visualization PRD

## 1. Product Overview
A Python-based toolkit that transforms article titles into semantic embeddings using Google Gemini, clusters them with K-Means (k=3), and generates comprehensive 3D visualizations comparing original groupings with algorithmic clusters. The system provides both manual and scikit-learn implementations of dimensionality reduction algorithms to validate mathematical correctness.

**Core Value Proposition:**
- Convert textual article titles into high-dimensional semantic vectors using Gemini embeddings
- Discover thematic groupings via K-Means clustering while preserving original group labels
- Generate three distinct 3D visualizations (manual PCA, sklearn PCA, t-SNE) for multi-perspective analysis
- Validate mathematical implementations through cross-verification of manual vs library PCA results

## 2. Goals & Success Metrics

### Primary Goals
1. **Embedding Generation**: 100% of valid CSV rows produce normalized Gemini embeddings (768-dim) with original group metadata preserved
2. **Clustering Accuracy**: K-Means (k=3) completes with clear cluster assignments and metrics (silhouette score, inertia, variance)
3. **Visualization Quality**: Three distinct 3D plots generated showing:
   - Original vs algorithmic groupings (color = K-Means cluster, shape = original group)
   - Variance explained by top 3 principal components
   - Dimensionality reduction from N-dim → 3-dim clearly labeled
4. **Mathematical Validation**: Manual PCA implementation matches sklearn PCA results within 1e-6 tolerance

### Success Metrics
- Processing time: < 5 minutes for 100 titles (including Gemini API calls)
- Variance explained: Display percentage of total variance captured by 3D representation
- Visualization clarity: All plots include legends, dimension info, and variance metrics
- Code correctness: Manual PCA eigenvalues/eigenvectors match sklearn decomposition

## 3. Primary Users & Key Flows

### User: Data Analyst / Content Strategist
**Flow 1: Full Pipeline Execution**
1. Provides CSV with columns `title` and `group`
2. Runs pipeline script with Gemini API key
3. System generates embeddings, clusters with K-Means (k=3)
4. Receives 3 interactive 3D visualizations (HTML or static PNG)
5. Analyzes how original groupings align with algorithmic clusters
6. Examines variance explained to assess dimensionality reduction quality

**Flow 2: Manual Verification**
1. Compares manual PCA vs sklearn PCA visualizations
2. Validates transformation matrices match
3. Confirms eigenvalue sorting and variance calculations
4. Uses as educational tool to understand PCA mechanics

## 4. Functional Requirements

### 4.1 Data Ingestion (`prepare_embeddings.py`)
**Input:**
- CSV file with columns: `title` (string), `group` (string)
- Path: `pipeline_input/<filename>.csv`

**Processing:**
- Validate CSV schema (case-insensitive column matching)
- Trim whitespace, remove empty rows
- Preserve original group labels for each title
- Detect and report duplicate titles (optional flag `--allow-duplicates`)

**Output:**
- DataFrame with columns: `title`, `original_group`, `embedding`, `normalized_embedding`
- Manifest JSON with metadata: model, dimensions, row counts, timestamp

**CLI Arguments:**
```bash
--input <path>              # Path to input CSV
--output <path>             # Base path for output files
--format [parquet|csv]      # Output format (default: parquet)
--gemini-api-key <key>      # Gemini API key (or GEMINI_API_KEY env var)
--gemini-model <model>      # Gemini model (default: models/text-embedding-004)
--config <path>             # Optional JSON config file
--allow-duplicates          # Allow duplicate titles
--log-level [debug|info]    # Logging verbosity
```

**Error Handling:**
- Missing API key → clear error with setup instructions
- Invalid CSV → report missing columns with examples
- Gemini API failures → retry once, then fail with error details
- Empty dataset after filtering → abort with row count stats

### 4.2 K-Means Clustering (`run_kmeans.py`)
**Input:**
- Normalized embeddings dataset (parquet/CSV)
- Fixed k=3 (hardcoded, not configurable)

**Processing:**
- Load normalized embeddings from dataset
- Run scikit-learn K-Means with k=3, fixed random seed (42)
- Calculate cluster assignments for each title
- Compute metrics:
  - Cluster sizes (count per cluster)
  - Inertia (within-cluster sum of squares)
  - Silhouette score (cosine metric)
  - Distance to centroid for each point
  - Majority original group per cluster
  - Mismatch rate (cluster vs original group disagreement)

**Output:**
- Enhanced dataset with new column: `cluster_id` (0, 1, or 2)
- Cluster metrics report (markdown):
  - Table: Cluster ID | Size | Majority Original Group | Avg Distance
  - Overall silhouette score
  - Total variance / inertia
- Centroids JSON file for reference

**CLI Arguments:**
```bash
--data <path>               # Path to embeddings dataset
--output <path>             # Path for clustering results
--report <path>             # Path for metrics report (markdown)
--centroids <path>          # Path for centroids JSON
--seed <int>                # Random seed (default: 42)
--log-level [debug|info]    # Logging verbosity
```

**Constraints:**
- k is hardcoded to 3 (not configurable via CLI)
- Random seed default: 42 for reproducibility

### 4.3 3D Visualization Generation (`visualize_3d.py`)

#### 4.3.1 Manual PCA Implementation
**Algorithm Steps:**
1. Center the data (subtract mean)
2. Compute covariance matrix: C = (X^T × X) / (n-1)
3. Calculate eigenvalues and eigenvectors of C
4. Sort eigenvectors by eigenvalues (descending order)
5. Create transformation matrix P from top 3 eigenvectors (as columns)
6. Transform data: Y = X × P (keep only first 3 dimensions)
7. Calculate variance explained: (sum of top 3 eigenvalues) / (sum of all eigenvalues)

**Output:**
- 3D scatter plot (interactive HTML via Plotly)
- Title: "Manual PCA: N-dim → 3-dim Reduction"
- Subtitle: "Variance Explained: XX.XX%"
- Legend:
  - Colors: K-Means clusters (0=red, 1=blue, 2=green)
  - Shapes: Original groups (circle, square, triangle, etc.)
- Axes labels: PC1, PC2, PC3

**Validation:**
- Log eigenvalues and transformation matrix
- Compare with sklearn PCA eigenvalues (assert close within 1e-6)

#### 4.3.2 Sklearn PCA Implementation
**Algorithm:**
1. Use `sklearn.decomposition.PCA(n_components=3)`
2. Fit and transform normalized embeddings
3. Extract explained variance ratio

**Output:**
- 3D scatter plot (interactive HTML via Plotly)
- Title: "Sklearn PCA: N-dim → 3-dim Reduction"
- Subtitle: "Variance Explained: XX.XX%"
- Same legend and color/shape scheme as manual PCA

**Validation Check:**
- Compare transformed coordinates with manual PCA (allow sign flips)
- Compare explained variance percentages
- Log comparison results for verification

#### 4.3.3 t-SNE Implementation
**Algorithm:**
1. Use `sklearn.manifold.TSNE(n_components=3, random_state=42, perplexity=30)`
2. Fit and transform normalized embeddings
3. t-SNE does not provide variance explained (note this in visualization)

**Output:**
- 3D scatter plot (interactive HTML via Plotly)
- Title: "t-SNE: N-dim → 3-dim Reduction"
- Subtitle: "Perplexity: 30 (t-SNE does not compute variance explained)"
- Same legend and color/shape scheme

**Parameters:**
- Perplexity: 30 (configurable via CLI)
- Random state: 42
- Early exaggeration: 12 (default)
- Learning rate: 200 (default)

#### 4.3.4 Common Visualization Elements
**Required Information on Each Plot:**
1. **Dimension Info:**
   - Original dimensions: {N} (from embedding size)
   - Reduced dimensions: 3
2. **Variance Info:**
   - PCA plots: "Variance Explained: XX.XX%"
   - t-SNE plot: "t-SNE does not compute variance explained"
3. **Legend:**
   - **Colors**: K-Means Cluster ID (0, 1, 2)
     - Cluster 0: Red
     - Cluster 1: Blue
     - Cluster 2: Green
   - **Shapes**: Original Group
     - Map unique groups to shapes: circle, square, diamond, triangle, etc.
4. **Interactive Features:**
   - Hover tooltips: show title, original group, cluster ID
   - Rotation controls (3D interactive)
   - Zoom and pan

**CLI Arguments:**
```bash
--data <path>               # Path to embeddings dataset
--assignments <path>        # Path to clustering results
--output-dir <path>         # Directory for output plots
--perplexity <int>          # t-SNE perplexity (default: 30)
--format [html|png]         # Output format (default: html for interactivity)
--log-level [debug|info]    # Logging verbosity
```

**Outputs:**
- `manual_pca_3d.html` - Manual PCA visualization
- `sklearn_pca_3d.html` - Sklearn PCA visualization
- `tsne_3d.html` - t-SNE visualization
- `pca_validation_report.txt` - Comparison of manual vs sklearn PCA
- Optional: PNG versions if `--format png` specified

### 4.4 Pipeline Orchestrator (`run_pipeline.py`)
**Purpose:** Execute full workflow in one command

**Steps:**
1. Prepare embeddings (Gemini API)
2. Run K-Means clustering (k=3)
3. Generate all 3 visualizations
4. Save validation report

**CLI Arguments:**
```bash
--input <path>              # Input CSV path
--output-dir <path>         # Output directory (default: pipeline_output)
--gemini-api-key <key>      # Gemini API key
--config <path>             # Optional config JSON
--viz-format [html|png]     # Visualization format
--log-level [debug|info]    # Logging verbosity
```

**Output Structure:**
```
pipeline_output/
├── embeddings.parquet
├── embeddings.manifest.json
├── clustering_results.csv
├── kmeans_report.md
├── centroids.json
├── visualizations/
│   ├── manual_pca_3d.html
│   ├── sklearn_pca_3d.html
│   ├── tsne_3d.html
│   └── pca_validation_report.txt
```

## 5. Data & Model Specifications

### 5.1 Input Data Schema
**CSV Columns:**
- `title` (string, required): Article title text
- `group` (string, required): Original group/category label

**Validation Rules:**
- Both columns must exist (case-insensitive matching)
- Empty or whitespace-only values removed
- Duplicate titles: warn or halt based on `--allow-duplicates` flag

**Example:**
```csv
title,group
"Introduction to Machine Learning",Technology
"Python for Data Science",Technology
"Quantum Computing Basics",Science
"Climate Change Impact",Environment
```

### 5.2 Gemini Embeddings
**Model:** `models/text-embedding-004`
**Dimensions:** 768 (output dimension)
**Normalization:** L2 normalization applied to all vectors
**API:** Google Generative AI Python SDK

**Embedding Process:**
1. Call `genai.embed_content(model=model_id, content=title)`
2. Extract embedding vector
3. Apply L2 normalization: v_norm = v / ||v||₂
4. Store both raw and normalized embeddings

### 5.3 K-Means Configuration
- **Algorithm:** Lloyd's algorithm (sklearn default)
- **k:** 3 (fixed)
- **Initialization:** k-means++ (sklearn default)
- **Random State:** 42 (reproducible)
- **Max Iterations:** 300 (sklearn default)
- **Distance Metric:** Euclidean (on normalized vectors ≈ cosine)

### 5.4 Dimensionality Reduction Algorithms

#### PCA (Principal Component Analysis)
- **Components:** 3
- **Explained Variance:** Computed and displayed
- **Implementation:** Manual + sklearn validation

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Components:** 3
- **Perplexity:** 30 (configurable)
- **Random State:** 42
- **Iterations:** 1000 (sklearn default)

## 6. Non-Functional Requirements

### 6.1 Performance
- **Embedding Generation:** ~1-2 sec per title (Gemini API latency)
- **K-Means:** < 1 sec for 100 titles
- **PCA:** < 1 sec for 768-dim → 3-dim reduction
- **t-SNE:** ~5-10 sec for 100 titles (more intensive)
- **Total Pipeline:** < 5 min for 100 titles

### 6.2 Compatibility
- **Python:** 3.10+
- **OS:** macOS, Linux (Windows with WSL)
- **Dependencies:**
  - numpy >= 1.23
  - pandas >= 1.5
  - scikit-learn >= 1.2
  - plotly >= 5.14 (for 3D interactive plots)
  - google-generativeai >= 0.5
  - pyarrow >= 11.0 (for parquet support)

### 6.3 Code Quality
- All `.py` files < 200 lines
- Modular design with `utils/` package
- Type hints for function signatures
- Docstrings for public functions
- Logging at INFO level (DEBUG available)

### 6.4 Reproducibility
- Fixed random seeds (42) throughout
- Manifest files track all parameters
- Validation reports for PCA cross-checking

## 7. Error Handling & Validation

### 7.1 Input Validation
- **Missing Columns:** Fail with clear error listing expected columns
- **Empty Dataset:** Abort with row count summary
- **Duplicate Titles:** Warn by default, halt if `--allow-duplicates` not set
- **Invalid CSV:** Report parse errors with line numbers

### 7.2 API & Network Errors
- **Missing Gemini API Key:** Clear error with setup instructions
- **Gemini API Failures:** Retry once with exponential backoff, then fail with error details
- **Rate Limiting:** Implement backoff strategy, log warnings

### 7.3 Mathematical Validation
- **PCA Eigenvalue Check:** Log if manual vs sklearn differ by > 1e-6
- **Negative Eigenvalues:** Warn and clip to zero (numerical precision issues)
- **Variance Sum Check:** Ensure explained variance ≤ 100%

### 7.4 Visualization Errors
- **Empty Clusters:** Handle gracefully, show warning in plot
- **Too Few Points:** Warn if < 10 points per cluster
- **Shape Assignment:** Cycle through shapes if more groups than available shapes

## 8. Testing Strategy

### 8.1 Unit Tests
- CSV validation logic
- L2 normalization correctness
- Manual PCA eigenvalue/eigenvector computation
- Variance calculation formulas
- Color/shape assignment logic

### 8.2 Integration Tests
- End-to-end pipeline with sample CSV (10 titles)
- Verify all outputs generated
- Check PCA validation report shows match
- Ensure visualizations render without errors

### 8.3 Validation Tests
- Compare manual PCA vs sklearn PCA:
  - Eigenvalues match (within tolerance)
  - Transformed coordinates match (allowing sign flips)
  - Explained variance matches
- Golden file tests for deterministic outputs with seed=42

### 8.4 Visual Inspection Tests
- Manual review of plots for:
  - Correct color assignments (cluster 0=red, 1=blue, 2=green)
  - Correct shape assignments per original group
  - Legend clarity and completeness
  - Dimension and variance info displayed

## 9. Implementation Notes

### 9.1 Manual PCA Implementation
```python
def manual_pca_3d(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Perform manual PCA reduction to 3 dimensions.

    Returns:
        - Transformed data (N x 3)
        - Variance explained ratio (float)
    """
    # 1. Center the data
    X_centered = X - X.mean(axis=0)

    # 2. Compute covariance matrix
    n = X_centered.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)

    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Select top 3 eigenvectors
    P = eigenvectors[:, :3]

    # 6. Transform data
    X_transformed = X_centered @ P

    # 7. Calculate variance explained
    variance_explained = eigenvalues[:3].sum() / eigenvalues.sum()

    return X_transformed, variance_explained
```

### 9.2 Visualization Color/Shape Scheme
```python
CLUSTER_COLORS = {
    0: 'red',
    1: 'blue',
    2: 'green'
}

SHAPE_MARKERS = [
    'circle', 'square', 'diamond', 'triangle-up',
    'triangle-down', 'cross', 'x', 'pentagon', 'hexagon'
]

# Map original groups to shapes
unique_groups = sorted(df['original_group'].unique())
group_to_shape = {
    group: SHAPE_MARKERS[i % len(SHAPE_MARKERS)]
    for i, group in enumerate(unique_groups)
}
```

## 10. Out of Scope
- Interactive classification of new titles (removed from scope)
- Automated cluster interpretation or labeling
- Hyperparameter tuning for K-Means (k is fixed at 3)
- Multiple k values comparison
- 2D visualizations (only 3D required)
- Real-time streaming processing
- Web UI or dashboard
- Model fine-tuning or custom embeddings

## 11. Future Enhancements (Post-V1)
- Support for k values other than 3
- Additional dimensionality reduction techniques (UMAP, LDA)
- Cluster stability analysis (silhouette plots per cluster)
- Export to 3D formats (OBJ, PLY) for external tools
- Batch processing for multiple CSV files
- Comparison mode for different embedding models

## 12. Open Questions
- Should we save intermediate PCA transformation matrices for reuse?
- Static PNG vs interactive HTML preference for reports?
- Maximum dataset size before memory issues (current target: 10k titles)?
- Perplexity range recommendations for t-SNE based on dataset size?
- Should validation tolerance (1e-6) be configurable?
