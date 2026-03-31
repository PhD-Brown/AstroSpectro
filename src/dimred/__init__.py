"""
AstroSpectro — Module de réduction de dimension (PHY-3500)
===========================================================

Sous-module dédié à l'analyse en composantes principales (PCA),
aux embeddings non-linéaires (UMAP, t-SNE) et (stretch) à l'autoencodeur.

Objectif scientifique
---------------------
Explorer la structure de l'espace des spectres LAMOST DR5 en dimension
réduite et valider que cet espace encode la physique stellaire (T_eff,
log g, [Fe/H], type MK).

Modules
-------
- data_loader      : chargement features + catalog Gaia
- pca_analyzer     : ACP + interprétation physique des axes
- embedding        : UMAP / t-SNE wrappers reproductibles
- dimred_visualizer: figures qualité publication

Usage rapide
------------
>>> from dimred import DimRedDataLoader, PCAAnalyzer, EmbeddingEngine
>>> loader = DimRedDataLoader("data/reports/features.csv",
...                           "data/catalog/master_catalog_gaia.csv")
>>> X, y, meta = loader.load()
>>> pca = PCAAnalyzer()
>>> pca.fit(X)
>>> pca.plot_variance_explained()
"""

from .data_loader import DimRedDataLoader
from .pca_analyzer import PCAAnalyzer
from .embedding import EmbeddingEngine
from .dimred_visualizer import DimRedVisualizer
from .autoencoder import SpectralAutoencoder

__all__ = [
    "DimRedDataLoader",
    "PCAAnalyzer",
    "EmbeddingEngine",
    "DimRedVisualizer",
    "SpectralAutoencoder",
]

__version__ = "0.1.0"
__author__ = "AstroSpectro — PHY-3500"
