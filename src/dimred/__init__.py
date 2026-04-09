"""
AstroSpectro — API publique du sous-package dimred (PHY-3500)
==============================================================

Pourquoi ce fichier existe
--------------------------
Ce __init__.py joue le role de "façade" pour tout le sous-package
de réduction de dimension:

1) Il expose un point d'entrée unique et stable:
   `from dimred import ...`

2) Il masque l'organisation interne (noms de fichiers et détails
    d'implémentation), afin de réduire le couplage avec les notebooks
   et scripts du cours.

3) Il documente explicitement ce qui fait partie de l'API publique,
   c'est-à-dire les objets qui peuvent être utilisés sans dépendre
   des détails internes du projet.

Contenu fonctionnel exposé
--------------------------
- data_loader       : chargement des features + métadonnées Gaia
- pca_analyzer      : ACP, variance expliquée, loadings, correlations
- embedding         : wrappers UMAP / t-SNE reproductibles
- dimred_visualizer : figures standardisées pour analyse et rapport
- autoencoder       : autoencodeur spectral (entraînement + inférence)
- run_reporter      : sauvegarde des artefacts (joblib, JSON, TXT)
- hdbscan_analyzer  : clustering dense + profils physiques
- xgboost_bridge    : lien modèle supervisé <-> espace UMAP

Note de maintenabilité
----------------------
Chaque symbole importé ici doit être considéré comme public. Renommer
ou retirer un symbole exposé peut casser les notebooks PHY-3500; ces
changements doivent donc être versionnés et annoncés clairement.
"""

# ---------------------------------------------------------------------------
# Blocs "core": chargement des données et méthodes de réduction de dimension
# ---------------------------------------------------------------------------
from .data_loader import DimRedDataLoader
from .pca_analyzer import PCAAnalyzer
from .embedding import EmbeddingEngine
from .dimred_visualizer import DimRedVisualizer
from .autoencoder import SpectralAutoencoder

# ---------------------------------------------------------------------------
# Bloc "reporting": fonctions de sauvegarde standardisées des runs
# ---------------------------------------------------------------------------
from .run_reporter import save_pca_run, save_umap_tsne_run, save_autoencoder_run

# ---------------------------------------------------------------------------
# Bloc "clustering": outils HDBSCAN et analyses de sensibilité
# ---------------------------------------------------------------------------
from .hdbscan_analyzer import (
    HDBSCANAnalyzer,
    compute_feature_profiles,
    compute_sensitivity,
)

# ---------------------------------------------------------------------------
# Bloc "bridges" et utilitaires d'inférence
# - Alias xgboost_predict: nom court et explicite côté utilisateur.
# - tester_candidat / latent_arithmetic: outils pratiques utilisés dans NB03.
# ---------------------------------------------------------------------------
from .xgboost_bridge import load_and_predict as xgboost_predict
from .autoencoder import tester_candidat, latent_arithmetic


# Contrat explicite de l'API publique.
# Cette liste contrôle ce qui est exporté via `from dimred import *`
# et sert de référence de "surface stable" pour les notebooks.
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


# Métadonnées minimales du package dimred.
# __version__ est utile pour tracer les notebooks/runs qui dépendent de cette
# API; __author__ garde le contexte pédagogique du sous-projet PHY-3500.
__version__ = "0.3.0"
__author__ = "AstroSpectro — PHY-3500"
