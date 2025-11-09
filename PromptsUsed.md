# Session Prompts & Outcomes

## Session 2: Refactoring to 3D Visualizations

### User Prompts (Verbatim)

1. **"Analyse the code in folder. Keep only embedding version with gemini. Clean the rest and update the documentation."**
   - Removed all SentenceTransformer code
   - Updated all scripts to Gemini-only approach
   - Removed "provider" field from config
   - Updated requirements.txt and README.md

2. **"Make all the .py files less than 200 lines. Split into multiple files if needed."**
   - Created modular utils/ package (7 modules)
   - Refactored all main scripts to <200 lines
   - Final counts: prepare_embeddings.py (155), classify_title.py (165), run_kmeans.py (103), run_pipeline.py (110), visualize_clusters.py (85)

3. **"Move .csv files to pipeline_input folder, correct the code where necessary"**
   - Created pipeline_input/ folder
   - Moved TitlesforL16HomeWork.csv and new_titles.csv
   - Updated all code references and README.md

4. **"Now re-write PRD such that the whole point of the script is the following: 1) Get articles titles and their original groups from the csv file provided by user. 2) Convert titles to embeddings (normalised vectors) using gemini. 3) Once titles are loaded, run K-Means with k=3 and get new groups for each title (while keep information about the old group). 4) Take the vectors and create 3 visualizations in 3D: a) with the help of PCA algorythm, but not using scilearn: create covariance matrix, find eigen values and eigen vectors, sort and place the eigen vectors in matrix by their relevant eigen values from highest to lowest, transpose the matrix (resulting in transformation matrix), move initial data set of embeddings into a new coordinate system using transformation matrix, keep top 3 features only. b) with PCS algorythm using skilearn (to check if our manual version was correct. c) with t-SNE algorythm. All visualisations should have: 1) info about initial dimantions number and reduced dimensions number; 2) % of variance covered with 3 dimensions left in the visualisation; 3) Color of the dots should be of K-Means groups and form of the dots should be per original group in the input file which should be explained in the legend."**
   - New PRD requirements acknowledged
   - Removed classification/KNN from scope
   - Added manual PCA implementation requirement
   - Specified 3 visualization types with detailed requirements

5. **"Now change the code to be according to the PRD. Keep all the code files under 200 lines. In the end - update ReadMe and clean all the not needed files/folders"**
   - Updated run_kmeans.py: hardcoded k=3
   - Created utils/pca_manual.py (132 lines): manual PCA implementation with validation
   - Created utils/visualization_3d.py (164 lines): 3D Plotly visualizations
   - Created visualize_3d.py (155 lines): main visualization script
   - Updated run_pipeline.py (130 lines): new 3-step workflow
   - Updated requirements.txt: removed matplotlib, added plotly>=5.14
   - Removed: classify_title.py, visualize_clusters.py, utils/knn_classifier.py, utils/visualization.py, utils/pipeline_runner.py
   - Updated README.md with comprehensive 3D visualization documentation

6. **"re-write PromptsUsed file completely and include all the prompts used from this and previous sessions"**
   - Rewrote PromptsUsed.md with full session history

7. **"Rewrite promptsUsed file with exact prompts used in this session and short summary of what was done"**
   - Simplified PromptsUsed.md with concise format

8. **"Keep only Session 2 prompts"**
   - Removed Session 1 content, kept only current session

9. **"For each of the method of dimensionality reduction start timer when start the process and end timer when finish and add the time it took to run to the visualisation"**
   - Added `import time` to utils/visualization_3d.py and visualize_3d.py
   - Added timing to manual PCA, sklearn PCA, and t-SNE functions
   - Updated `create_3d_scatter()` to accept optional `elapsed_time` parameter
   - Added processing time display to visualization titles (e.g., "Processing Time: 0.045s")
   - Updated return dictionaries to include `elapsed_time` field

10. **"Re-write tasks.json file according to a new PRD"**
    - Completely rewrote tasks.json to reflect new 3D visualization workflow
    - Removed all SentenceTransformer and classification tasks
    - Added detailed sections: visualization features, manual PCA details, output structure
    - Documented color scheme (Red/Blue/Green) and shape scheme
    - Added timing feature documentation
    - Updated CLI examples for new pipeline

11. **"Add to ReadMe explanation on why to do PCA or t-SNE (for visualisation of multidimensional data set, etc) and what are the differences between the 2 methods (pros and cons of both... compare in table)"**
    - Added "Why Dimensionality Reduction?" section explaining the need for visualization
    - Created comprehensive comparison table with 12 aspects (Type, Goal, Speed, etc.)
    - Added "When to Use Each Method" with practical guidance for PCA vs t-SNE
    - Explained "Why We Use Both" - complementary views of data
    - Highlighted speed difference now visible via timing feature

12. **"remove myenv, deactivate myenv, recreate new virtual env and run requirements"**
    - Removed old myenv directory (had symlink issues)
    - Created fresh Python 3.9 virtual environment
    - Upgraded pip from 21.2.4 to 25.3
    - Installed all requirements including plotly 6.4.0
    - Verified plotly installation successful
    - Fixed parquet file loading bug in utils/data_io.py (ext.lstrip('.'))

13. **"create gitignor"**
    - Created comprehensive .gitignore file
    - Excluded: Python artifacts, virtual environments, IDE files, sensitive configs
    - Initially excluded pipeline_output/ and CSV files
    - Kept config/embedding.json excluded (contains API key)

14. **"allow pipeline output to be copied to git"**
    - Removed pipeline_output/ exclusion from .gitignore
    - Removed *.parquet and *.manifest.json exclusions
    - Added exception for pipeline_output CSV files

15. **"remove new_titles.csv file and allow csv files to be copied to git"**
    - Deleted pipeline_input/new_titles.csv
    - Removed all CSV exclusion rules from .gitignore
    - Now all CSV files are tracked in git

16. **"remove all the files from pipeline_output folder"**
    - Cleared all contents from pipeline_output/
    - Folder now empty and ready for fresh pipeline runs

17. **"update prompts file with missing prompts"**
    - Current task: adding prompts 9-17 to PromptsUsed.md

---

## Session Summary

**Phase 1**: Removed SentenceTransformer, kept Gemini-only approach
**Phase 2**: Modularized code into utils/ package, all files <200 lines
**Phase 3**: Organized CSV files into pipeline_input/ folder
**Phase 4**: Complete redesign - implemented manual PCA from scratch, created 3D interactive visualizations (Plotly), hardcoded k=3, removed classification features

**Result**: Clean, educational codebase with manual PCA validation, three 3D visualization types (Manual PCA, Sklearn PCA, t-SNE), all files <200 lines

---

## Final Architecture

### File Structure
```
L17_HomeWork/
├── prepare_embeddings.py (155 lines)
├── run_kmeans.py (104 lines)
├── visualize_3d.py (155 lines)
├── run_pipeline.py (130 lines)
└── utils/
    ├── embedding.py (53 lines)
    ├── vector_ops.py (70 lines)
    ├── data_io.py (141 lines)
    ├── clustering.py (197 lines)
    ├── pca_manual.py (132 lines)
    └── visualization_3d.py (164 lines)
```

### Pipeline Workflow
1. **Embeddings**: Gemini API → L2 normalized vectors
2. **Clustering**: K-Means with k=3 (hardcoded)
3. **Visualization**: Three 3D visualizations
   - Manual PCA (custom implementation)
   - Sklearn PCA (validation)
   - t-SNE (perplexity=10)

### Key Features
- Manual PCA: covariance matrix → eigendecomposition → transformation
- Validation: Manual vs sklearn PCA (1e-6 tolerance)
- Interactive: Plotly HTML with hover info
- Color scheme: Red/Blue/Green for clusters
- Shape scheme: Different markers for original groups
- Metrics: Dimension info, variance explained percentages

### Technologies
- Google Gemini API (text-embedding-004, 768-dim)
- Scikit-learn (K-Means, PCA, t-SNE)
- Plotly (3D interactive visualizations)
- NumPy, Pandas, PyArrow
