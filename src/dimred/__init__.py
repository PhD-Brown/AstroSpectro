"""
AstroSpectro — Module de réduction de dimension (PHY-3500)
===========================================================

Modules
-------
- data_loader       : chargement features + catalog Gaia
- pca_analyzer      : ACP + interprétation physique des axes
- embedding         : UMAP / t-SNE wrappers reproductibles
- dimred_visualizer : figures qualité publication
- autoencoder       : autoencodeur spectral PyTorch
- run_reporter      : sauvegarde et rapports de run (JSON/TXT/joblib)
- hdbscan_analyzer  : analyse HDBSCAN + profils physiques
- xgboost_bridge    : pont XGBoost ↔ espace UMAP
"""

from .data_loader import DimRedDataLoader
from .pca_analyzer import PCAAnalyzer
from .embedding import EmbeddingEngine
from .dimred_visualizer import DimRedVisualizer
from .autoencoder import SpectralAutoencoder
from .run_reporter import save_pca_run, save_umap_tsne_run, save_autoencoder_run
from .hdbscan_analyzer import (
    HDBSCANAnalyzer,
    compute_feature_profiles,
    compute_sensitivity,
)
from .xgboost_bridge import load_and_predict as xgboost_predict
from .autoencoder import tester_candidat, latent_arithmetic

__all__ = [
    "DimRedDataLoader",
    "PCAAnalyzer",
    "EmbeddingEngine",
    "DimRedVisualizer",
    "SpectralAutoencoder",
    "save_pca_run",
    "save_umap_tsne_run",
    "save_autoencoder_run",
    "HDBSCANAnalyzer",
    "compute_feature_profiles",
    "compute_sensitivity",
    "xgboost_predict",
    "tester_candidat",
    "latent_arithmetic",
]

__version__ = "0.3.0"
__author__ = "AstroSpectro — PHY-3500"
