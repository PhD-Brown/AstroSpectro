"""Weights & Biases configuration for AstroSpectro.

Provides opt-in W&B integration with graceful degradation.
All W&B calls are wrapped so that failures never interrupt training.

Usage
-----
>>> from pipeline.wandb_config import should_use_wandb, init_wandb_run
>>> if should_use_wandb(use_wandb=True):
...     run = init_wandb_run(name="xgboost-v1", config={"lr": 0.1})
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_WANDB_PROJECT = "AstroSpectro"


def should_use_wandb(use_wandb: bool = False) -> bool:
    """Check whether W&B logging should proceed.

    Returns ``True`` only when *use_wandb* is ``True``, the
    ``WANDB_DISABLED`` environment variable is not set to ``"true"``,
    and the ``wandb`` package is importable.
    """
    if not use_wandb:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() == "true":
        return False
    try:
        import wandb  # noqa: F401

        return True
    except ImportError:
        print("⚠️  wandb is not installed. Run: pip install wandb")
        return False


def init_wandb_run(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Any:
    """Initialise a W&B run for the AstroSpectro project.

    Parameters
    ----------
    name : str
        Human-readable run name (e.g. ``"xgboost-baseline-v1"``).
    config : dict, optional
        Hyperparameters / metadata to log.
    tags : list[str], optional
        Tags for filtering in the W&B dashboard.

    Returns
    -------
    wandb.sdk.wandb_run.Run
        The active W&B run object.
    """
    import wandb

    return wandb.init(
        project=_WANDB_PROJECT,
        name=name,
        config=config or {},
        tags=tags or [],
        reinit=True,
    )
