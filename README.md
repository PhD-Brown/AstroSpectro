# 🌌 Astro-Spectro Classification Pipeline — LAMOST DR5

> **A modular end-to-end pipeline for automated stellar spectral classification using LAMOST DR5 data. Built for robustness, transparency, and scientific impact.**

---

## Overview

This project implements a **fully reproducible workflow** for the automatic classification of stellar types from raw LAMOST DR5 spectra.
It combines state-of-the-art data handling, feature engineering, and machine learning techniques tailored for large astronomical datasets.

* **Automated robust data download and management**
* **Advanced preprocessing:** normalization, continuum fitting, denoising
* **Physical feature extraction:** line identification (Hα, Hβ, CaII K\&H, G-band, etc.), FWHM, equivalent width
* **Flexible ML models:** Random Forest, SVM, CNN 1D, etc.
* **Evaluation and reporting:** confusion matrix, ROC, PDF/HTML reports
* **Open source, modular, and designed for extension**

> **Status:** *Active development — core pipeline functional, major improvements and features planned. See [ROADMAP.md](ROADMAP.md) for details.*

---

## Pipeline Structure

```mermaid
graph TD
    A[Data Download & Catalog Parsing]
    B[Preprocessing & Quality Checks]
    C[Feature Extraction (Spectral Lines)]
    D[Dataset Assembly (Features + Labels)]
    E[Machine Learning: Training & Validation]
    F[Results: Evaluation, Visualization, Reporting]
    G[Documentation & Deployment]

    A --> B --> C --> D --> E --> F --> G
```

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

### 3. **Prepare the Data**

* Raw LAMOST spectra (FITS) should be placed in `data/raw/` (this folder is **ignored by git**).
* Example catalogs (CSV) go in `data/catalog/`.
* See [notebooks/00\_master\_pipeline.ipynb](notebooks/00_master_pipeline.ipynb) for end-to-end examples.

### 4. **Run the Pipeline**

* Jupyter notebooks for step-by-step demonstration:

  * `notebooks/00_master_pipeline.ipynb` — Full workflow (recommended starting point)
* Python scripts for modular/automated runs:

  * `src/processing.py`, `src/feature_engineering.py`, etc.

### 5. **Results**

* Processed features, model outputs, and reports are generated in `reports/` or `data/processed/`.
* Evaluation plots and logs are available for each run.

---

## Project Structure

```
astro-spectro-classification/
│
├── src/                  # Main pipeline source code (modular Python scripts)
│   ├── preprocessor.py
│   ├── peak_detector.py
│   ├── feature_engineering.py
│   ├── classifier.py
│   └── ...
├── notebooks/            # Jupyter Notebooks for exploration, demo, and analysis
│   └── 00_master_pipeline.ipynb
├── data/
│   ├── raw/              # Raw LAMOST FITS spectra (**NOT versioned!**)
│   ├── catalog/          # External catalogs (CSV, metadata, etc.)
│   └── processed/        # Processed features and ML-ready datasets
├── reports/              # Auto-generated evaluation reports, figures
├── images/               # Figures, pipeline diagrams, README illustrations
├── requirements.txt      # Python dependencies
├── LICENSE               # License (MIT)
├── README.md             # This file
└── .gitignore            # To avoid leaking data/models/logs
```

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
* 1D CNN on raw or processed spectra (deep learning support)
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

```python
# See notebooks/00_master_pipeline.ipynb for full examples
```

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

*See [ROADMAP.md](ROADMAP.md) for the full list of planned and in-progress features.*

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
Open an issue, or contact via [Mail](alex.baker.1@ulaval.ca/)

---

## Star the repo and watch for updates!

Stay tuned for major upgrades and public releases.

