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

Pour PHY-3500, l'objectif est de **comparer** les espaces latents PCA,
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
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
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
        self.encoder = _Encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = _Decoder(
            latent_dim, list(reversed(hidden_dims)), input_dim, dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout

        # Sélection device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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
        self._model = _AutoencoderNet(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info("Modèle construit : %d paramètres entraînables", n_params)

        # Split train / val
        N = len(X)
        n_val = max(1, int(N * val_fraction))
        rng = np.random.default_rng(42)
        val_idx = rng.choice(N, size=n_val, replace=False)
        train_idx = np.setdiff1d(np.arange(N), val_idx)

        X_train = torch.FloatTensor(X[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train, X_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

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

        history = {"train_loss": [], "val_loss": [], "lr_history": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            # ── Train ───────────────────────────────────────────────
            self._model.train()
            train_losses = []
            for X_batch, _ in train_loader:
                optimizer.zero_grad()
                X_recon, _ = self._model(X_batch)
                loss = criterion(X_recon, X_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            # ── Validation ──────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                X_val_recon, _ = self._model(X_val)
                val_loss = criterion(X_val_recon, X_val).item()

            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr_history"].append(current_lr)

            if scheduler is not None:
                scheduler.step(val_loss)

            # ── Early stopping ──────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
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
            self._model.load_state_dict(best_state)
            logger.info("Meilleur état restauré (val_loss=%.6f)", best_val_loss)

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
            Z_t = torch.FloatTensor(Z).to(self.device)
            X_recon = self._model.decode(Z_t).cpu().numpy()
        return X_recon

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode puis décode X. Retourne X_recon (N, D)."""
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
        X_recon = self.reconstruct(X)
        mse_per = np.mean((X - X_recon) ** 2, axis=1)
        if per_sample:
            return mse_per
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
        mse = self.reconstruction_mse(X, per_sample=True)
        rows = []

        if y is None:
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
        if X_a.ndim == 1:
            X_a = X_a[np.newaxis]
        if X_b.ndim == 1:
            X_b = X_b[np.newaxis]

        z_a = self.encode(X_a)[0]
        z_b = self.encode(X_b)[0]

        alphas = np.linspace(0, 1, n_steps)
        Z_interp = np.array([(1 - a) * z_a + a * z_b for a in alphas])
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
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        ae = cls(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
            device=device,
        )
        ae._model = _AutoencoderNet(
            input_dim=ae.input_dim,
            latent_dim=ae.latent_dim,
            hidden_dims=ae.hidden_dims,
            dropout=ae.dropout,
        ).to(ae.device)
        ae._model.load_state_dict(checkpoint["model_state_dict"])
        ae._model.eval()
        ae.history_ = checkpoint.get("history")
        ae.fit_time_ = checkpoint.get("fit_time")
        logger.info("SpectralAutoencoder chargé depuis : %s", path)
        return ae

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "SpectralAutoencoder non entraîné — appeler fit() d'abord."
            )

    def __repr__(self) -> str:
        arch = f"{self.input_dim} → {self.hidden_dims} → {self.latent_dim}"
        return f"SpectralAutoencoder({arch})"
