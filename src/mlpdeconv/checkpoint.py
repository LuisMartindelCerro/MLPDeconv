"""Checkpoint loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mlpdeconv.model import MLPDeconvRegressor, build_model_from_checkpoint

REQUIRED_CHECKPOINT_KEYS = {
    "state_dict",
    "model_config",
    "genes",
    "cell_types",
    "gene_mean",
    "gene_std",
    "normalization",
}


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Convert a user-friendly device option into a torch.device."""

    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def torch_load_checkpoint(path: str | Path, device: str | torch.device = "auto") -> dict:
    """Load a PyTorch checkpoint, supporting older and newer torch versions."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
    torch_device = resolve_device(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    validate_checkpoint(checkpoint, checkpoint_path)
    return checkpoint


def validate_checkpoint(checkpoint: dict, checkpoint_path: str | Path | None = None) -> None:
    """Check that a checkpoint contains the fields needed for prediction."""

    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(checkpoint))
    if missing:
        source = f" in {checkpoint_path}" if checkpoint_path is not None else ""
        raise ValueError(f"Checkpoint{source} is missing required keys: {', '.join(missing)}")

    n_genes = len(checkpoint["genes"])
    if len(checkpoint["gene_mean"]) != n_genes or len(checkpoint["gene_std"]) != n_genes:
        raise ValueError("Checkpoint gene_mean/gene_std lengths must match the training gene list length.")
    if len(checkpoint["cell_types"]) == 0:
        raise ValueError("Checkpoint must contain at least one cell type.")

    std = np.asarray(checkpoint["gene_std"], dtype=np.float32)
    if np.any(std <= 0):
        raise ValueError("Checkpoint gene_std values must be positive.")


def load_trained_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[MLPDeconvRegressor, dict, torch.device]:
    """Load a trained model and its checkpoint metadata."""

    torch_device = resolve_device(device)
    checkpoint = torch_load_checkpoint(checkpoint_path, device=torch_device)
    model = build_model_from_checkpoint(checkpoint, device=torch_device)
    return model, checkpoint, torch_device
