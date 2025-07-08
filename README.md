![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white)
![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![License: MIT](https://img.shields.io/badge/MIT-green?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/PhD-Brown/astro-spectro-classification)
![Repo Size](https://img.shields.io/github/repo-size/PhD-Brown/astro-spectro-classification)
![Build](https://img.shields.io/github/actions/workflow/status/PhD-Brown/astro-spectro-classification/python-app.yml?branch=main)
![Project Status](https://img.shields.io/badge/status-active-brightgreen)


# 🌌 Astro-Spectro Classification Pipeline — LAMOST DR5

> **A modular end-to-end pipeline for automated stellar spectral classification using LAMOST DR5 data. Built for robustness, transparency, and scientific impact.**

---

## Overview

This project implements a **fully reproducible workflow** for the automatic classification of stellar types from raw LAMOST DR5 spectra.
It combines state-of-the-art data handling, feature engineering, and machine learning techniques tailored for large astronomical datasets.

* **Automated robust data download and management**
* **Advanced preprocessing:** normalization, continuum fitting, denoising
* **Physical feature extraction:** line identification (Hα, Hβ, CaII K\&H, G-band, etc.), FWHM, equivalent width
* **Flexible ML models:** Random Forest, SVM, CNN 1D (to do), etc.
* **Evaluation and reporting:** confusion matrix, ROC, PDF/HTML reports
* **Open source, modular, and designed for extension**

> **Status:** *Active development — core pipeline functional, major improvements and features planned. See [ROADMAP.md](ROADMAP.md) for details.*

---

## Pipeline Structure

[1] Data Download & Catalog Parsing  
   │  
   ▼  
[2] Preprocessing & Quality Checks  
   │  
   ▼  
[3] Feature Extraction (Spectral Lines)  
   │  
   ▼  
[4] Dataset Assembly (Features + Labels)  
   │  
   ▼  
[5] Machine Learning: Training & Validation  
   │  
   ▼  
[6] Results: Evaluation, Visualization, Reporting  
   │  
   ▼  
[7] Documentation & Deployment  

---

## Quickstart

### 1. **Clone the Repository**

```bash
git clone https://github.com/PhD-Brown/astro-spectro-classification.git
cd astro-spectro-classification
```

### 2. **Set Up Your Python Environment**

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare and Download the Data
Recommended:
Use the notebook [01_download_spectra](notebooks/01_download_spectra.ipynb)
to automatically download raw LAMOST spectra and build local catalogs.

Data will be placed in ``data/raw/``

Catalogs/metadata will be created in ``data/catalog/``

_Alternatively_, you may manually place FITS files in ``data/raw/`` and your catalog CSV in ``data/catalog/``.

### 4. **Run the Pipeline**

* Jupyter notebooks for step-by-step demonstration:

  * [00_master_pipeline](notebooks/00_master_pipeline.ipynb) — Full workflow (recommended starting point)

* Python scripts for modular/automated runs:

  * `src/processing.py`, `src/feature_engineering.py`, etc.
    
* For visualization and additional tools, see:
  
  * [02_tools_and_visuals](notebooks/02_tools_and_visuals.ipynb)

### 5. **Results**

* Processed features, model outputs, and reports are generated in `reports/` or `data/processed/`.
* Evaluation plots and logs are available for each run.

---

## Project Structure

```
astro_spectro_git/
│
├── .gitignore              # Exclude raw data, logs, outputs, venv, etc.
├── main.py                 # Main script (entry point)
├── requirements.txt        # Python dependencies
│
├── data/                   # Data folder (ignored by git)
│   ├── archive/            # Old/archived data or intermediate files
│   ├── catalog/            # Catalogs (CSV, metadata, external labels)
│   ├── models/             # (Optionnel) Saved models or intermediate model files
│   ├── processed/          # Processed features/ML-ready datasets
│   └── raw/                # Raw LAMOST FITS (HUGE, many subfolders)
│       ├── B6202/
│       ├── B6210/
│       ├── B6212/
│       ├── GAC_105N29_B1/
│       ├── M31_011N40_B1/
│       └── M31_011N40_M1/
│
├── images/                 # Images, visualizations, pipeline diagrams
│   └── spectre_visualisation/
│
├── logs/                   # Log files, output from processing
│
├── notebooks/              # Jupyter Notebooks
│   ├── 00_master_pipeline.ipynb
│   ├── 01_download_spectra.ipynb
│   ├── 02_tools_and_visuals.ipynb
│   └── archive/            # Old notebooks, experimental code
│
├── reports/                # Generated reports, evaluation results, figures
│
└── src/                    # Main source code
    ├── pipeline/           # Main pipeline logic (modular Python scripts)
    │   ├── classifier.py
    │   ├── feature_engineering.py
    │   ├── peak_detector.py
    │   ├── preprocessor.py
    │   └── processing.py
    │
    └── tools/              # Utility scripts/tools
       ├── dataset_builder.py
       ├── dr5_downloader.py
       └── generate_catalog_from_fits.py
```

### Legend

- **data/raw/**: All raw LAMOST FITS spectra. Subfolders are organized by field/plan_id. (Never versioned — excluded by `.gitignore`.)
- **data/catalog/**: CSV metadata and external catalogs (e.g., labels, cross-matches).
- **data/processed/**: Processed/cleaned datasets, ML-ready features.
- **data/models/**: Saved model files, checkpoints, or pickled objects (optional).
- **data/archive/**: Old or intermediate data, deprecated versions.
- **src/pipeline/**: Main pipeline code (each file = a major pipeline stage: preprocessing, feature engineering, etc.).
- **src/tools/**: Utility scripts for building catalogs, downloading data, or other helper tools.
- **notebooks/**: Jupyter notebooks for prototyping, exploratory analysis, demos.
- **notebooks/archive/**: Deprecated or experimental notebooks.
- **reports/**: Automatically generated reports, visualizations, and evaluation results.
- **logs/**: Log files and process outputs for debugging/tracing.
- **images/**: All project illustrations, plots, or pipeline diagrams for documentation and reporting.
- **venv/**: Python virtual environment — *local only*, always ignored by git.
- **README.md, ROADMAP.md, LICENSE**: Project documentation, roadmap, and open-source license. *(Omitted from tree for clarity, but always present at the root.)*

---

## Core Features

### **Data Handling & Preprocessing**

* Robust automated LAMOST spectrum download (FITS, multi-format parsing)
* Deduplication, error handling, and catalog cross-matching
* Adaptive normalization, local segment-wise continuum correction
* Denoising (Savitzky-Golay, median filter, outlier removal)

### **Feature Engineering**

* Extraction of main astrophysical lines: Hα, Hβ, CaII K/H, G-band, He II, NaI D, Mg b, TiO, CaII IR triplet, etc.
* Physical measurements: depth, FWHM, equivalent width, line ratios
* Pseudo-color indices: continuum slope, curvature, PCA features
* Integration of catalog metadata: SNR, magnitude, redshift, RA/Dec

### **Machine Learning**

* **Random Forest** baseline
* SVM, KNN, Gradient Boosting (LightGBM, XGBoost)
* Model ensembling (stacking/voting)
* Cross-validation, hyperparameter tuning (GridSearchCV, Optuna)
* Automatic class balancing (SMOTE/undersampling)

### **Evaluation & Reporting**

* Automated confusion matrices, ROC/PR curves, feature importance
* PDF/HTML report generation (Jinja2, nbconvert)
* Experiment logging (MLflow, TensorBoard, Weights & Biases supported)
* Reproducible notebooks and output artifacts

---

## Example Usage

**Train and Evaluate a Classifier (Random Forest):**

```python
from src.classifier import SpectraClassifier

clf = SpectraClassifier(
    input_features='data/processed/features_20250701.csv',
    labels_file='data/catalog/dr5_v3_plan_clean.csv'
)
clf.train()
clf.evaluate()
clf.plot_confusion_matrix()
```

**Run End-to-End in Jupyter Notebook:**

 See notebooks/00_master_pipeline.ipynb for full examples
 
---

## Roadmap (2025+)

* [ ] **Feature expansion:** Add new lines (He II, CaII triplet, TiO), advanced indices
* [ ] **Physics-driven augmentation:** Synthetic spectra, noise simulation
* [ ] **Deep Learning:** Benchmark 1D CNN, autoencoders, self-supervised models (SimCLR, BYOL)
* [ ] **Explainable AI:** SHAP/LIME visualization, per-spectrum explanation
* [ ] **RAG/LLM Integration:** Mini RAG assistant for pipeline documentation and troubleshooting
* [ ] **Real-time & batch mode:** Automated pipeline orchestration (Snakemake, Prefect)
* [ ] **Multi-survey support:** Extend to SDSS, Gaia XP, cross-matched datasets
* [ ] **Dashboard/demo:** Streamlit/Gradio web interface for interactive exploration
* [ ] **Open science:** Prepare mock datasets, reproducible Colab demos, detailed contribution guide

*See [ROADMAP.md](ROADMAP.md) (in french) for the full list of planned and in-progress features.*

---

## How to Contribute

* **Bug reports:** Please use the [Issues](https://github.com/PhD-Brown/astro-spectro-classification/issues) tab.
* **Pull requests:** Welcome! Fork the repo, create a branch, and submit a PR with clear description.
* **Feature suggestions:** Open a discussion or PR.
* **Documentation improvements:** All improvements to docstrings, notebooks, and guides are appreciated.

### **Best Practices**

* Follow PEP8 (for Python), use clear variable names, comment code blocks.
* Write modular, testable code (see `tests/` if present).
* Document major changes in the CHANGELOG (or Issues).

---

## License

Distributed under the MIT License.
See [LICENSE](LICENSE) for details.

---

## Credits & Contact

**Author:** [Alex Baker (PhD-Brown)](https://github.com/PhD-Brown)

**Acknowledgments:** LAMOST Collaboration, Astropy, scikit-learn, community contributors

**Contact:**
Open an issue, or contact via alex.baker.1@ulaval.ca

---

## Star the repo and watch for updates!

Stay tuned for major upgrades and public releases.

