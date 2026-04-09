"""
AstroSpectro — dimred.autoencoder
===================================

Autoencodeur neuronal pour la réduction de dimension non-linéaire
appliquée aux spectres stellaires LAMOST DR5 (PHY-3500).

Architecture
------------
Autoencodeur symétrique entièrement connecté (Fully-Connected / MLP) :

    X (D) → Encodeur → z (latent_dim) → Décodeur → X_recon (D)

    Encodeur : D → 256 → 128 → 64 → latent_dim
    Décodeur : latent_dim → 64 → 128 → 256 → D

    Activations  : ReLU (couches cachées), Linear (sortie)
    Régularisation: BatchNorm1d + Dropout (p=0.1)

Pourquoi un autoencodeur ici ?
-------------------------------
- PCA est linéaire : capture uniquement les axes de variance maximale.
- UMAP/t-SNE sont non-paramétriques : pas de transform() sur nouveaux points.
- L'autoencodeur est **non-linéaire + paramétrique** : il apprend une
  représentation compressée et peut projeter de nouveaux spectres sans
  recalcul complet.

Pour PHY-3500, l'objectif est de comparer les espaces latents PCA,
UMAP et autoencodeur et de montrer si la non-linéarité apporte
de l'information supplémentaire.

Références
----------
- Hinton & Salakhutdinov (2006). Reducing the Dimensionality of Data
  with Neural Networks. Science 313:504.
- Portillo et al. (2020). AJ 160:45 — autoencodeur sur spectres SDSS.
- Kruse et al. (2019). Variational Autoencoders for Stellar Spectra.

Exemple
-------
>>> ae = SpectralAutoencoder(input_dim=196, latent_dim=2)
>>> ae.fit(X_train, epochs=100, lr=1e-3)
>>> Z = ae.encode(X)
>>> X_recon = ae.decode(Z)
>>> print(ae.reconstruction_mse(X))
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vérification PyTorch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch non trouvé. Installer : pip install torch")


def _check_torch():
    """Valide la disponibilité de PyTorch avant toute opération AE."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch requis pour SpectralAutoencoder.\n" "Installer : pip install torch"
        )


# ---------------------------------------------------------------------------
# Architecture réseau
# ---------------------------------------------------------------------------


class _Encoder(nn.Module):
    """Encodeur MLP symétrique."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Construction séquentielle des blocs: Linear -> BatchNorm -> ReLU -> Dropout.
        layers = []
        prev = input_dim
        for h in hidden_dims:
            # Chaque itération réduit progressivement la dimension vers l'espace latent.
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev = h
        # Dernière couche linéaire: projection finale vers z (sans activation).
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passage avant complet dans l'encodeur.
        return self.net(x)


class _Decoder(nn.Module):
    """Décodeur MLP symétrique (miroir de l'encodeur)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Architecture miroir de l'encodeur pour reconstruire l'entrée originale.
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev = h
        # Couche de sortie linéaire vers la dimension d'origine D.
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Reconstruction à partir d'un point latent.
        return self.net(z)


class _AutoencoderNet(nn.Module):
    """Réseau autoencodeur complet (encoder + decoder)."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Encodeur direct puis décodeur miroir.
        self.encoder = _Encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = _Decoder(
            latent_dim, list(reversed(hidden_dims)), input_dim, dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retourne à la fois reconstruction et représentation latente.
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------


class SpectralAutoencoder:
    """
    Autoencodeur MLP pour la réduction de dimension spectrale.

    Entraîne un autoencodeur sur les features standardisées, expose
    un espace latent de dimension `latent_dim` comparable aux scores PCA
    et aux embeddings UMAP/t-SNE.

    Parameters
    ----------
    input_dim : int
        Dimension de l'entrée (nombre de features). Requis.
    latent_dim : int
        Dimension de l'espace latent (2 pour visualisation 2D, 8-16 pour
        représentation plus riche).
    hidden_dims : list[int]
        Dimensions des couches cachées de l'encodeur.
        Défaut : [256, 128, 64] — adapté pour ~100-200 features.
    dropout : float
        Taux de dropout pour la régularisation (0.0 = désactivé).
    device : str | None
        'cuda', 'cpu', ou None (auto-détection).

    Exemple
    -------
    >>> ae = SpectralAutoencoder(input_dim=196, latent_dim=2)
    >>> history = ae.fit(X_train, epochs=150, lr=1e-3, batch_size=512)
    >>> Z = ae.encode(X)
    >>> print(f"MSE reconstruction : {ae.reconstruction_mse(X):.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        _check_torch()

        # Hyperparamètres d'architecture exposés publiquement.
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout

        # Sélection device
        if device is None:
            # Auto-détection: CUDA si disponible, sinon CPU.
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # État entraînement (initialement vide).
        self._model: Optional[_AutoencoderNet] = None
        self.history_: Optional[Dict[str, List[float]]] = None
        self.fit_time_: Optional[float] = None

        logger.info(
            "SpectralAutoencoder initialisé | D=%d → z=%d | device=%s",
            input_dim,
            latent_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 150,
        lr: float = 1e-3,
        batch_size: int = 512,
        val_fraction: float = 0.10,
        weight_decay: float = 1e-5,
        lr_scheduler: bool = True,
        early_stopping_patience: int = 20,
        verbose: bool = True,
        wandb_run=None,
    ) -> Dict[str, List[float]]:
        """
        Entraîne l'autoencodeur sur la matrice X.

        Parameters
        ----------
        X : np.ndarray (N, D)
            Matrice d'entrée standardisée (StandardScaler recommandé).
        epochs : int
            Nombre maximum d'epochs.
        lr : float
            Taux d'apprentissage initial (Adam).
        batch_size : int
            Taille des mini-batchs.
        val_fraction : float
            Fraction des données réservée à la validation.
        weight_decay : float
            Régularisation L2 sur les poids (Adam).
        lr_scheduler : bool
            Si True, applique ReduceLROnPlateau sur la val_loss.
        early_stopping_patience : int
            Arrêt anticipé si val_loss ne s'améliore pas après N epochs.
            0 = désactivé.
        verbose : bool
            Si True, affiche la progression.
        wandb_run : wandb.Run | None
            Si fourni, loggue les métriques dans W&B.

        Returns
        -------
        dict avec clés 'train_loss', 'val_loss', 'lr_history'.
        """
        _check_torch()

        # Construction du modèle
        # 1) Instancie un nouveau réseau pour ce run d'entraînement.
        self._model = _AutoencoderNet(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Compte des paramètres pour suivi de complexité.
        n_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info("Modèle construit : %d paramètres entraînables", n_params)

        # Split train / val
        # 2) Crée une validation aléatoire reproductible (seed fixe).
        N = len(X)
        n_val = max(1, int(N * val_fraction))
        rng = np.random.default_rng(42)
        val_idx = rng.choice(N, size=n_val, replace=False)
        train_idx = np.setdiff1d(np.arange(N), val_idx)

        # 3) Conversion NumPy -> tenseurs float et transfert vers device.
        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)

        # Dataset autoencodeur: cible = entrée (apprentissage auto-supervisé).
        train_loader = DataLoader(
            TensorDataset(X_train, X_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # 4) Optimisation: Adam + MSE, scheduler optionnel sur val_loss.
        optimizer = optim.Adam(
            self._model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
            )
            if lr_scheduler
            else None
        )
        criterion = nn.MSELoss()

        # Structures de suivi run (courbes + early stopping).
        history = {"train_loss": [], "val_loss": [], "lr_history": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        # Chronométrage global du fit.
        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            # ── Train ───────────────────────────────────────────────
            # Mode entraînement: active Dropout/BatchNorm en mode train.
            self._model.train()
            train_losses = []
            for X_batch, _ in train_loader:
                # Reset gradients de l'itération précédente.
                optimizer.zero_grad()
                # Forward: reconstruction du batch courant.
                X_recon, _ = self._model(X_batch)
                # Perte MSE entre entrée et reconstruction.
                loss = criterion(X_recon, X_batch)
                # Backpropagation des gradients.
                loss.backward()
                # Clip de gradient pour stabiliser l'entraînement.
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Moyenne des pertes train sur tous les mini-batchs.
            train_loss = np.mean(train_losses)

            # ── Validation ──────────────────────────────────────────
            # Mode évaluation: désactive Dropout, BatchNorm en mode eval.
            self._model.eval()
            with torch.no_grad():
                X_val_recon, _ = self._model(X_val)
                val_loss = criterion(X_val_recon, X_val).item()

            # LR effectivement utilisée à cette epoch.
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr_history"].append(current_lr)

            if scheduler is not None:
                # ReduceLROnPlateau observe la val_loss pour ajuster lr.
                scheduler.step(val_loss)

            # ── Early stopping ──────────────────────────────────────
            if val_loss < best_val_loss:
                # Nouveau meilleur modèle: snapshot complet des poids.
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                # Aucun progrès: incrémente le compteur de patience.
                patience_counter += 1

            if (
                early_stopping_patience > 0
                and patience_counter >= early_stopping_patience
            ):
                logger.info(
                    "Early stopping à l'epoch %d (patience=%d)",
                    epoch,
                    early_stopping_patience,
                )
                if verbose:
                    print(f"  Early stopping epoch {epoch} | val_loss={val_loss:.6f}")
                break

            # ── Logging ─────────────────────────────────────────────
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"train={train_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.2e}"
                )

            if wandb_run is not None:
                # Logging externe pour suivi expérimental (dashboard W&B).
                wandb_run.log(
                    {
                        "ae/train_loss": train_loss,
                        "ae/val_loss": val_loss,
                        "ae/lr": current_lr,
                        "ae/epoch": epoch,
                    }
                )

        # Restauration du meilleur modèle
        if best_state is not None:
            # On restaure l'état optimal, pas nécessairement la dernière epoch.
            self._model.load_state_dict(best_state)
            logger.info("Meilleur état restauré (val_loss=%.6f)", best_val_loss)

        # Métadonnées de fin d'entraînement conservées sur l'objet.
        self.fit_time_ = time.perf_counter() - t0
        self.history_ = history

        logger.info(
            "Autoencodeur entraîné en %.1f s | best val_loss=%.6f",
            self.fit_time_,
            best_val_loss,
        )
        return history

    # ------------------------------------------------------------------
    # Encode / Decode / Reconstruct
    # ------------------------------------------------------------------

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode X dans l'espace latent.

        Parameters
        ----------
        X : np.ndarray (N, D)

        Returns
        -------
        np.ndarray (N, latent_dim)
        """
        self._check_fitted()
        self._model.eval()
        with torch.no_grad():
            # Conversion + transfert device puis retour NumPy sur CPU.
            X_t = torch.FloatTensor(X).to(self.device)
            Z = self._model.encode(X_t).cpu().numpy()
        return Z

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Décode des coordonnées latentes en features reconstruites.

        Parameters
        ----------
        Z : np.ndarray (N, latent_dim)

        Returns
        -------
        np.ndarray (N, D)
        """
        self._check_fitted()
        self._model.eval()
        with torch.no_grad():
            # Pipeline inverse: latent -> tenseur -> reconstruction -> NumPy.
            Z_t = torch.FloatTensor(Z).to(self.device)
            X_recon = self._model.decode(Z_t).cpu().numpy()
        return X_recon

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode puis décode X. Retourne X_recon (N, D)."""
        # Compose explicitement les deux opérations pour garder une API claire.
        return self.decode(self.encode(X))

    # ------------------------------------------------------------------
    # Métriques de qualité
    # ------------------------------------------------------------------

    def reconstruction_mse(
        self, X: np.ndarray, per_sample: bool = False
    ) -> np.ndarray | float:
        """
        Erreur quadratique moyenne de reconstruction.

        Parameters
        ----------
        X : np.ndarray (N, D)
        per_sample : bool
            Si True, retourne un vecteur (N,) MSE par échantillon.
            Si False, retourne la MSE scalaire moyenne.

        Returns
        -------
        float ou np.ndarray (N,)
        """
        # Reconstruit d'abord chaque échantillon.
        X_recon = self.reconstruct(X)
        # Puis calcule la MSE ligne par ligne.
        mse_per = np.mean((X - X_recon) ** 2, axis=1)
        if per_sample:
            return mse_per
        # Valeur globale agrégée utile pour comparer des modèles.
        return float(mse_per.mean())

    def reconstruction_summary(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Résumé des erreurs de reconstruction par classe.

        Parameters
        ----------
        X : np.ndarray (N, D)
        y : np.ndarray (N,) | None
            Étiquettes de classe. Si None, une seule ligne 'all'.

        Returns
        -------
        pd.DataFrame avec colonnes ['classe', 'n', 'mse_mean', 'mse_std',
                                    'mse_median', 'mse_q95'].
        """
        # MSE individuelle servant de base à tous les agrégats.
        mse = self.reconstruction_mse(X, per_sample=True)
        rows = []

        if y is None:
            # Mode global: une seule ligne de synthèse.
            rows.append(
                {
                    "classe": "all",
                    "n": len(X),
                    "mse_mean": mse.mean(),
                    "mse_std": mse.std(),
                    "mse_median": np.median(mse),
                    "mse_q95": np.percentile(mse, 95),
                }
            )
        else:
            # Mode par classe: agrégation indépendante pour chaque label.
            for cls in np.unique(y):
                mask = y == cls
                m = mse[mask]
                rows.append(
                    {
                        "classe": cls,
                        "n": int(mask.sum()),
                        "mse_mean": m.mean(),
                        "mse_std": m.std(),
                        "mse_median": np.median(m),
                        "mse_q95": np.percentile(m, 95),
                    }
                )

                # Retour tabulaire exploitable dans les rapports NB03.
        return pd.DataFrame(rows)

    def compare_with_pca(
        self,
        X: np.ndarray,
        pca_analyzer,
        n_pcs_list: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compare la MSE de reconstruction autoencodeur vs PCA
        pour différents nombres de composantes.

        Parameters
        ----------
        X : np.ndarray (N, D)
        pca_analyzer : PCAAnalyzer
            Objet PCAAnalyzer ajusté.
        n_pcs_list : list[int] | None
            Nombres de PCs à tester. Défaut : [1, 2, 3, 5, 10, 20, 50].

        Returns
        -------
        pd.DataFrame ['method', 'n_components', 'mse_mean']
        """
        if n_pcs_list is None:
            n_pcs_list = [1, 2, 3, 5, 10, 20, 50]

        rows = []

        # Autoencodeur
        # Point de référence non-linéaire (dimension latente fixée).
        ae_mse = self.reconstruction_mse(X)
        rows.append(
            {
                "method": f"Autoencodeur (z={self.latent_dim})",
                "n_components": self.latent_dim,
                "mse_mean": ae_mse,
            }
        )

        # PCA pour différents nombres de PCs
        for n in n_pcs_list:
            # Garde-fou: ignore les n supérieurs aux composantes disponibles.
            if n > pca_analyzer.sklearn_pca.n_components_:
                continue
            mse = pca_analyzer.reconstruction_error(X, n_components=n).mean()
            rows.append(
                {
                    "method": "PCA",
                    "n_components": n,
                    "mse_mean": float(mse),
                }
            )

        # Tableau final directement plotable/comparable.
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Interpolation dans l'espace latent
    # ------------------------------------------------------------------

    def latent_interpolation(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray,
        n_steps: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpole linéairement entre deux points dans l'espace latent.

        Utile pour visualiser que l'espace latent est continu et
        physiquement cohérent (ex : interpoler entre une étoile froide
        et une étoile chaude).

        Parameters
        ----------
        X_a, X_b : np.ndarray (1, D) ou (D,)
            Deux points de départ et d'arrivée.
        n_steps : int
            Nombre de points intermédiaires.

        Returns
        -------
        Z_interp : np.ndarray (n_steps, latent_dim) — trajectoire latente.
        X_interp : np.ndarray (n_steps, D) — reconstructions.
        """
        # Normalise les entrées pour accepter vecteur (D,) ou matrice (1, D).
        if X_a.ndim == 1:
            X_a = X_a[np.newaxis]
        if X_b.ndim == 1:
            X_b = X_b[np.newaxis]

        # Encodage des deux extrémités de l'interpolation.
        z_a = self.encode(X_a)[0]
        z_b = self.encode(X_b)[0]

        # Mélange linéaire entre z_a et z_b avec n_steps points.
        alphas = np.linspace(0, 1, n_steps)
        Z_interp = np.array([(1 - a) * z_a + a * z_b for a in alphas])
        # Décodage de toute la trajectoire en espace des features.
        X_interp = self.decode(Z_interp)

        return Z_interp, X_interp

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle et les hyperparamètres.

        Parameters
        ----------
        path : str
            Chemin du fichier (.pt ou .pth recommandé).
        """
        _check_torch()
        self._check_fitted()
        # Checkpoint complet: poids + hyperparamètres + métadonnées de fit.
        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "history": self.history_,
            "fit_time": self.fit_time_,
        }
        torch.save(checkpoint, path)
        logger.info("SpectralAutoencoder sauvegardé : %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SpectralAutoencoder":
        """
        Charge un SpectralAutoencoder préalablement sauvegardé.

        Parameters
        ----------
        path : str
            Chemin du fichier .pt.
        device : str | None
            Device cible. None = auto-détection.
        """
        _check_torch()
        # PyTorch>=2.6 defaults to weights_only=True, which can fail for
        # checkpoints containing non-tensor metadata (history, numpy scalars).
        # On force weights_only=False pour relire aussi l'historique de training.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Réinstancie l'objet avec la même configuration architecturale.
        ae = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
            device=device,
        )
        # Reconstruit le réseau et recharge les poids appris.
        ae._model = _AutoencoderNet(
            input_dim=ae.input_dim,
            latent_dim=ae.latent_dim,
            hidden_dims=ae.hidden_dims,
            dropout=ae.dropout,
        ).to(ae.device)
        ae._model.load_state_dict(checkpoint["model_state_dict"])
        ae._model.eval()
        # Métadonnées optionnelles (présentes selon la version du checkpoint).
        ae.history_ = checkpoint.get("history")
        ae.fit_time_ = checkpoint.get("fit_time")
        logger.info("SpectralAutoencoder chargé depuis : %s", path)
        return ae

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        # Garde-fou centralisé pour toutes les méthodes dépendantes du modèle.
        if self._model is None:
            raise RuntimeError(
                "SpectralAutoencoder non entraîné — appeler fit() d'abord."
            )

    def __repr__(self) -> str:
        # Représentation compacte utile en console/notebook.
        arch = f"{self.input_dim} → {self.hidden_dims} → {self.latent_dim}"
        return f"SpectralAutoencoder({arch})"


# ── Fonctions utilitaires de niveau module ───────────────────────────────────


def tester_candidat(
    x_raw: np.ndarray,
    ae_model,
    loader_ref,
    feature_names_ref: list,
    Z_train: np.ndarray,
    cluster_labels_train=None,
    meta_train=None,
    y_train=None,
    teff_train=None,
    label: str = "Candidat",
    save_path=None,
) -> dict:
    """
    Projette un nouveau spectre dans l'espace latent AE et produit
    une fiche diagnostique complète (3 panneaux).

    Provient de la cellule 57 du notebook phy3500_03_autoencoder.ipynb.

    Parameters
    ----------
    x_raw : np.ndarray (1, n_features) ou (n_features,)
        Features brutes (non standardisées) du candidat.
    ae_model : SpectralAutoencoder
        Modèle entraîné (z=2).
    loader_ref : DimRedDataLoader
        Loader ayant servi à l'entraînement (contient le scaler).
    feature_names_ref : list[str]
        Noms des features dans l'ordre du modèle.
    Z_train : np.ndarray (N, 2)
        Espace latent du dataset d'entraînement.
    label : str
        Étiquette du candidat pour la figure.
    save_path : Path | str | None
        Chemin de sauvegarde de la figure.

    Returns
    -------
    dict : z (coords latentes), mse, cluster, verdict, x_scaled, x_recon.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Force un format (1, D) pour harmoniser le flux de calcul.
    x = np.atleast_2d(x_raw).astype(float)

    # 1. Standardisation
    # Si le loader possède un scaler entraîné, on applique exactement la
    # même transformation que pour le set d'entraînement AE.
    if hasattr(loader_ref, "scaler_") and loader_ref.scaler_ is not None:
        x_scaled = loader_ref.scaler_.transform(x)
    else:
        # Fallback: on continue sans scaling pour ne pas bloquer le diagnostic.
        x_scaled = x.copy()

    # 2. Encodage
    # Coordonnées latentes du candidat (forme attendue: (1, 2)).
    z = ae_model.encode(x_scaled)  # (1, 2)

    # 3. Reconstruction
    # Mesure d'anomalie primaire: erreur de reconstruction sur espace standardisé.
    x_recon_scaled = ae_model.reconstruct(x_scaled)
    mse = float(np.mean((x_scaled - x_recon_scaled) ** 2))

    # 4. Cluster k-NN
    # Approximation locale du cluster via les 50 voisins latents les plus proches.
    dists = np.linalg.norm(Z_train - z, axis=1)
    nn_idx = np.argsort(dists)[:50]
    cluster_pred = None
    if cluster_labels_train is not None:
        nn_cl = cluster_labels_train[nn_idx]
        # On ignore le bruit HDBSCAN (-1) pour le vote majoritaire.
        nn_cl_valid = nn_cl[nn_cl >= 0]
        cluster_pred = (
            int(np.bincount(nn_cl_valid).argmax()) if len(nn_cl_valid) > 0 else -1
        )

    # 5. Verdict anomalie
    # Seuils heuristiques issus du notebook (q95/q99 approximatifs).
    MSE_Q95, MSE_Q99 = 0.97, 3.0
    if mse < MSE_Q95:
        verdict = "🟢 BANAL — spectre typique du dataset"
    elif mse < MSE_Q99:
        verdict = "🟡 INHABITUEL — dans le top 5% des erreurs"
    else:
        verdict = "🔴 ANOMALIE — objet très atypique"

    # ── Figure 3 panneaux ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6), dpi=150)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.4, 1.3, 1.3], wspace=0.35)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]

    # Panneau 1 : espace latent
    if teff_train is not None:
        # Coloration continue par température si disponible.
        valid_t = np.isfinite(teff_train)
        sc = ax1.scatter(
            Z_train[valid_t, 0],
            Z_train[valid_t, 1],
            c=teff_train[valid_t],
            cmap="plasma",
            s=0.5,
            alpha=0.2,
            rasterized=True,
            zorder=1,
        )
        plt.colorbar(sc, ax=ax1, label="T_eff (K)", fraction=0.03)
    else:
        # Sinon, nuage neutre en gris.
        ax1.scatter(
            Z_train[:, 0], Z_train[:, 1], c="#4A4A4A", s=0.5, alpha=0.2, rasterized=True
        )
    # Mise en évidence du candidat par une étoile dorée.
    ax1.scatter(
        z[0, 0],
        z[0, 1],
        s=300,
        c="#FFD700",
        marker="*",
        edgecolors="white",
        linewidths=1.5,
        zorder=6,
    )
    # Annotation texte avec coordonnées latentes exactes.
    ax1.annotate(
        f"{label}\n({z[0,0]:.3f}, {z[0,1]:.3f})",
        (z[0, 0], z[0, 1]),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=9,
        fontweight="bold",
        color="#FFD700",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75),
        zorder=7,
    )
    # Titre dynamique selon prédiction de cluster disponible ou non.
    title_str = (
        f"Position latente — Cluster C{cluster_pred}"
        if cluster_pred is not None and cluster_pred >= 0
        else "Position latente — Bruit/outlier"
    )
    ax1.set_title(title_str, fontsize=11, fontweight="bold")
    ax1.set_xlabel("AE axe 1")
    ax1.set_ylabel("AE axe 2")
    ax1.grid(True, alpha=0.15)

    # Panneau 2 : profil features clés
    KEY_F = [
        ("Hα EW", "feature_Hα_eq_width", "#E8593C"),
        ("Hβ EW", "feature_Hβ_eq_width", "#E8593C"),
        ("Ca II K", "feature_CaIIK_eq_width", "#3B8BD4"),
        ("Mg b", "feature_Mg_b_eq_width", "#4C9B6F"),
        ("FeH proxy", "feature_FeH_proxy", "#B07DB8"),
        ("BV synthét.", "feature_synthetic_BV", "#7F8FA6"),
    ]
    # Ne conserve que les features présentes dans le référentiel courant.
    valid_kf = [
        (label_name, feat_name, color)
        for label_name, feat_name, color in KEY_F
        if feat_name in feature_names_ref
    ]
    # Extrait les valeurs originales et reconstruites pour comparaison visuelle.
    kf_orig = [x_scaled[0, feature_names_ref.index(n)] for _, n, _ in valid_kf]
    kf_recon = [x_recon_scaled[0, feature_names_ref.index(n)] for _, n, _ in valid_kf]
    kf_labels = [label_name for label_name, _, _ in valid_kf]
    kf_colors = [c for _, _, c in valid_kf]
    y_kf = np.arange(len(kf_labels))
    w_kf = 0.35
    ax2.barh(
        y_kf - w_kf / 2,
        kf_orig,
        w_kf,
        color=kf_colors,
        alpha=0.90,
        edgecolor="white",
        linewidth=0.4,
        label="Original",
    )
    ax2.barh(
        y_kf + w_kf / 2,
        kf_recon,
        w_kf,
        color=kf_colors,
        alpha=0.35,
        edgecolor=kf_colors,
        linewidth=1.2,
        label="Reconstruit",
    )
    ax2.set_yticks(y_kf)
    ax2.set_yticklabels(kf_labels, fontsize=9)
    ax2.axvline(0, color="gray", lw=0.7, ls="--")
    ax2.set_xlabel("Score standardisé (σ)")
    ax2.set_title("Profil spectroscopique clé", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(axis="x", alpha=0.2)

    # Panneau 3 : jauge MSE
    # Fond coloré type jauge (vert -> rouge) pour lecture immédiate du risque.
    cmap_gauge = plt.cm.RdYlGn_r
    for j in range(100):
        ax3.barh(0, 0.1, left=j * 0.1, height=0.4, color=cmap_gauge(j / 100), alpha=0.7)
    # Curseur noir = MSE du candidat (tronquée visuellement à 10).
    ax3.axvline(min(mse, 10.0), color="black", lw=3, ymin=0.1, ymax=0.9)
    ax3.axvline(MSE_Q95, color="orange", lw=1.5, ls="--", label=f"q95 = {MSE_Q95:.2f}")
    ax3.axvline(MSE_Q99, color="red", lw=1.5, ls="--", label=f"q99 ≈ {MSE_Q99:.2f}")
    ax3.set_xlim(0, 10)
    ax3.set_ylim(-0.5, 1)
    ax3.set_xlabel("MSE de reconstruction")
    ax3.set_yticks([])
    ax3.set_title(f"Score d'anomalie\nMSE = {mse:.4f}", fontsize=11)
    ax3.text(
        0.5,
        0.75,
        verdict,
        transform=ax3.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#AAA", alpha=0.9),
    )
    ax3.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Fiche diagnostique AE — {label}", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    if save_path is not None:
        # Export facultatif de la fiche diagnostique.
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    return {
        "z": z[0],
        "mse": mse,
        "cluster": cluster_pred,
        "verdict": verdict,
        "x_scaled": x_scaled[0],
        "x_recon": x_recon_scaled[0],
    }


def latent_arithmetic(
    Z_ae: np.ndarray,
    meta: "pd.DataFrame",
    y: np.ndarray,
    save_path=None,
) -> dict:
    """
    Arithmétique dans l'espace latent (word2vec stellaire).

    Test : K_géante − K_naine + G_naine ≈ G_géante ?

    Provient de la cellule 46 du notebook phy3500_03_autoencoder.ipynb.

    Returns
    -------
    dict avec les populations sélectionnées, les résultats des tests
    et la figure matplotlib.
    """
    import matplotlib.pyplot as plt

    # Récupération robuste des paramètres physiques requis.
    teff_v = (
        meta["teff_gspphot"].values.astype(float)
        if "teff_gspphot" in meta.columns
        else None
    )
    logg_v = (
        meta["logg_gspphot"].values.astype(float)
        if "logg_gspphot" in meta.columns
        else None
    )
    if teff_v is None or logg_v is None:
        raise ValueError(
            "latent_arithmetic requiert teff_gspphot et logg_gspphot dans meta."
        )

    # Filtre de base: étoiles avec teff/logg valides.
    valid_all = np.isfinite(teff_v) & np.isfinite(logg_v) & (y == "STAR")
    rng = np.random.default_rng(42)

    def select_pop(teff_range, logg_range, n=200, label=""):
        # Fenêtre astrophysique ciblée (type spectral + classe gravité).
        t_lo, t_hi = teff_range
        g_lo, g_hi = logg_range
        mask = (
            valid_all
            & (teff_v >= t_lo)
            & (teff_v < t_hi)
            & (logg_v >= g_lo)
            & (logg_v < g_hi)
        )
        n_found = mask.sum()
        if n_found < 20:
            print(f"  ⚠  {label} : seulement {n_found} étoiles — résultat peu fiable")
        # Échantillonnage plafonné à n pour garder des populations comparables.
        n_sel = min(n, n_found)
        if n_sel == 0:
            return None, 0
        # Barycentre latent de la population sélectionnée.
        idx_sel = rng.choice(np.where(mask)[0], n_sel, replace=False)
        z_mean = Z_ae[idx_sel].mean(axis=0)
        print(f"  {label:20s} : n={n_found:5d}  z̄=({z_mean[0]:+.3f}, {z_mean[1]:+.3f})")
        return z_mean, n_found

    print("Sélection des populations de référence :")
    print("-" * 65)
    pops = {
        "K_naine": select_pop((3800, 5200), (4.0, 5.5), label="K naine"),
        "G_naine": select_pop((5200, 6200), (4.0, 5.5), label="G naine"),
        "F_naine": select_pop((6200, 7500), (4.0, 5.5), label="F naine"),
        "K_geante": select_pop((3800, 5200), (0.5, 3.0), label="K géante"),
        "G_geante": select_pop((5200, 6200), (0.5, 3.0), label="G géante"),
        "F_geante": select_pop((6200, 7500), (0.5, 3.0), label="F géante"),
    }
    # On garde uniquement les barycentres latents z (sans les effectifs).
    z = {k: v[0] for k, v in pops.items()}

    # Construit les équations testées seulement si les populations nécessaires existent.
    TESTS = []
    if all(z[k] is not None for k in ["K_geante", "K_naine", "G_naine", "G_geante"]):
        TESTS.append(
            {
                "eq": "K_géante − K_naine + G_naine ≈ G_géante ?",
                "pred": z["K_geante"] - z["K_naine"] + z["G_naine"],
                "tgt": z["G_geante"],
                "col_p": "#E8593C",
                "col_t": "#3B8BD4",
            }
        )
    if all(z[k] is not None for k in ["G_geante", "G_naine", "K_naine", "K_geante"]):
        TESTS.append(
            {
                "eq": "G_géante − G_naine + K_naine ≈ K_géante ?",
                "pred": z["G_geante"] - z["G_naine"] + z["K_naine"],
                "tgt": z["K_geante"],
                "col_p": "#4C9B6F",
                "col_t": "#B07DB8",
            }
        )

    print("\n" + "=" * 65 + "\n  RÉSULTATS DE L'ARITHMÉTIQUE LATENTE\n" + "=" * 65)
    results = []
    # Distance de référence pour normaliser l'erreur des analogies.
    dist_ref = (
        float(np.linalg.norm(z["K_naine"] - z["G_geante"]))
        if (z["K_naine"] is not None and z["G_geante"] is not None)
        else 1.0
    )
    for t in TESTS:
        # Compare position prédite vs barycentre réel attendu.
        dist = float(np.linalg.norm(t["pred"] - t["tgt"]))
        ratio = dist / max(dist_ref, 1e-8) * 100
        verdict = (
            "✓ ANALOGIE FONCTIONNELLE"
            if ratio < 30
            else ("~ Partielle" if ratio < 60 else "✗ Pas d'analogie")
        )
        print(f"\n  {t['eq']}")
        print(f"    Prédit  : ({t['pred'][0]:+.4f}, {t['pred'][1]:+.4f})")
        print(f"    Réel    : ({t['tgt'][0]:+.4f}, {t['tgt'][1]:+.4f})")
        print(f"    Distance: {dist:.4f} (= {ratio:.1f}% de K_naine↔G_géante)")
        print(f"    Verdict : {verdict}")
        results.append(
            {
                "equation": t["eq"],
                "distance": dist,
                "ratio_pct": ratio,
                "verdict": verdict,
            }
        )

    # Figure
    # Fond global: étoiles de référence colorées par température.
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sc = ax.scatter(
        Z_ae[valid_all, 0],
        Z_ae[valid_all, 1],
        c=teff_v[valid_all],
        cmap="plasma",
        s=0.5,
        alpha=0.2,
        rasterized=True,
        zorder=1,
    )
    plt.colorbar(sc, ax=ax, label="T_eff Gaia (K)", fraction=0.025)

    POPS_PLOT = [
        ("K_naine", "#B07DB8", "o"),
        ("G_naine", "#4C9B6F", "o"),
        ("F_naine", "#F5A623", "o"),
        ("K_geante", "#B07DB8", "^"),
        ("G_geante", "#4C9B6F", "^"),
        ("F_geante", "#F5A623", "^"),
    ]
    # Marqueurs pleins pour barycentres des populations de référence.
    for name, col, mrk in POPS_PLOT:
        if z[name] is not None:
            ax.scatter(
                z[name][0],
                z[name][1],
                s=200,
                color=col,
                marker=mrk,
                edgecolors="white",
                linewidths=1.5,
                zorder=4,
            )
            ax.annotate(
                name.replace("_", " "),
                z[name],
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
                color=col,
                fontweight="bold",
            )

    # Flèches: relient le point prédit (étoile) à la cible réelle.
    for t in TESTS:
        ax.annotate(
            "",
            xy=t["tgt"],
            xytext=t["pred"],
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
        )
        ax.scatter(
            t["pred"][0],
            t["pred"][1],
            s=150,
            color=t["col_p"],
            marker="*",
            edgecolors="white",
            linewidths=1,
            zorder=5,
        )

    ax.set_xlabel("AE axe 1")
    ax.set_ylabel("AE axe 2")
    ax.set_title(
        "Arithmétique latente — word2vec stellaire\nLAMOST DR5 × Gaia DR3",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path is not None:
        # Sauvegarde optionnelle de la figure de synthèse.
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    return {
        "populations": {
            k: (v.tolist() if v is not None else None) for k, v in z.items()
        },
        "tests": results,
        "fig": fig,
    }
