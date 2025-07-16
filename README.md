<!-- TECHNO BADGES -->
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/)
[![Scikit-learn 1.5.0](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Astropy 6.0.0](https://img.shields.io/badge/astropy-6.0.0-6D6D6D?style=for-the-badge)](https://www.astropy.org/)
[![LAMOST DR5](https://img.shields.io/badge/LAMOST-DR5-0099FF?style=for-the-badge)](http://www.lamost.org/public/)
[![Docusaurus](https://img.shields.io/badge/Docusaurus-3ECC5F?style=for-the-badge&logo=Docusaurus&logoColor=white)](https://docusaurus.io/)

<!-- STATUS / META BADGES -->
[![status](https://img.shields.io/badge/status-active-brightgreen?style=for-the-badge)](https://github.com/PhD-Brown/AstroSpectro)
[![GitHub release](https://img.shields.io/github/v/release/PhD-Brown/AstroSpectro?include_prereleases&style=for-the-badge)](https://github.com/PhD-Brown/AstroSpectro/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
![GitHub last commit](https://img.shields.io/github/last-commit/PhD-Brown/AstroSpectro?style=for-the-badge)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://phd-brown.github.io/AstroSpectro/)
![Repo Size](https://img.shields.io/github/repo-size/PhD-Brown/AstroSpectro)
[![Deploy Docusaurus to GitHub Pages](https://github.com/PhD-Brown/AstroSpectro/actions/workflows/deploy.yml/badge.svg)](https://github.com/PhD-Brown/AstroSpectro/actions/workflows/deploy.yml)
<br>

---

<!-- LOGO & MAIN LINK -->
<p align="center">
  <a href="https://phd-brown.github.io/AstroSpectro/">
    <img src="https://raw.githubusercontent.com/PhD-Brown/AstroSpectro/main/website/static/img/logo.png" alt="Astro-Spectro Logo" width="120">
  </a>
</p>

<h1 align="center">AstroSpectro Classification Pipeline</h1>

<p align="center">
  <strong>A modular end-to-end pipeline for automated stellar spectral classification using LAMOST DR5 data. Built for robustness, transparency, and scientific impact.</strong>
  <br><br>
  <a href="https://phd-brown.github.io/AstroSpectro/"><strong>Explore the Full Documentation »</strong></a>
  <br><br>
  <a href="https://github.com/PhD-Brown/AstroSpectro/issues">Report Bug</a>
  ·
  <a href="https://github.com/PhD-Brown/AstroSpectro/issues">Request Feature</a>
</p>

---

## About The Project

This project implements a **fully reproducible workflow** for the automatic classification of stellar types from raw LAMOST DR5 spectra. It combines state-of-the-art data handling, feature engineering, and machine learning techniques tailored for large astronomical datasets.

**Core features include:**
*   **Automated Workflow:** From data download to model training and reporting.
*   **Advanced Preprocessing:** Normalization, continuum fitting, denoising.
*   **Physical Feature Extraction:** Line identification (Hα, Hβ, CaII K&H), FWHM, etc.
*   **Flexible ML Models:** Random Forest, SVM, with a path to Deep Learning.
*    modular, and designed for extension.

> **Status:** *Active development. See the [Roadmap](https://phd-brown.github.io/AstroSpectro/docs/community/roadmap) in our documentation for details.*

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps. For detailed explanations, please refer to our **[full documentation](https://phd-brown.github.io/AstroSpectro/docs/getting-started/installation)**.

### Prerequisites

*   Python 3.9+
*   Git

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/PhD-Brown/AtroSpectro.git
    cd AstroSpectro
    ```
2.  **Set up your Python environment**
    ```sh
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    
    # Install dependencies
    pip install -r requirements.txt
    ```
3.  **Run the pipeline**
    *   The main entry point is the `notebooks/00_master_pipeline.ipynb` notebook.
    *   For a detailed step-by-step guide, see the **[First Run Tutorial](https://phd-brown.github.io/AstroSpectro/docs/getting-started/first-run)**.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please refer to the **[Contribution Guide](https://phd-brown.github.io/AstroSpectro/docs/community/contributing)** for details on our code of conduct, and the process for submitting pull requests to us.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Contact

Alex Baker - alex.baker.1@ulaval.ca
