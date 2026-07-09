"""Public prediction API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mlpdeconv.checkpoint import load_trained_model
from mlpdeconv.preprocessing import (
    GeneAlignmentReport,
    align_bulk_dataframe,
    apply_standardizer,
    normalize_bulk_counts,
    read_bulk_count_csv,
)
from mlpdeconv.registry import find_model_checkpoint


@dataclass(frozen=True)
class PredictionResult:
    """Predictions plus useful metadata about the run."""

    predictions: pd.DataFrame
    alignment_report: GeneAlignmentReport
    checkpoint_path: Path
    species: str | None


def _predict_array(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            X_batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(X_batch).cpu().numpy())
    return np.vstack(predictions).astype(np.float32)


def predict_bulk_dataframe(
    bulk_df: pd.DataFrame,
    species: str | None = None,
    checkpoint_path: str | Path | None = None,
    genes_are_rows: bool = False,
    device: str | torch.device = "auto",
    batch_size: int = 256,
    return_result: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | PredictionResult:
    """Predict cell fractions from a bulk count DataFrame.

    The DataFrame should normally have samples as rows and genes as columns.
    Set ``genes_are_rows=True`` when genes are rows and samples are columns.
    """

    resolved_checkpoint = find_model_checkpoint(
        species=species,
        checkpoint_path=checkpoint_path,
    )
    trained_model, checkpoint, torch_device = load_trained_model(resolved_checkpoint, device=device)
    aligned_bulk, alignment_report = align_bulk_dataframe(
        bulk_df,
        training_genes=checkpoint["genes"],
        genes_are_rows=genes_are_rows,
        return_report=True,
        verbose=verbose,
    )

    X_counts = aligned_bulk.to_numpy(dtype=np.float32)
    scale_factor = float(checkpoint.get("normalization", {}).get("scale_factor", 1_000_000.0))
    X_normalized = normalize_bulk_counts(X_counts, scale_factor=scale_factor)
    X_scaled = apply_standardizer(
        X_normalized,
        np.asarray(checkpoint["gene_mean"], dtype=np.float32),
        np.asarray(checkpoint["gene_std"], dtype=np.float32),
    )
    predictions = _predict_array(trained_model, X_scaled, batch_size=batch_size, device=torch_device)
    prediction_df = pd.DataFrame(predictions, index=aligned_bulk.index, columns=checkpoint["cell_types"])

    if return_result:
        return PredictionResult(
            predictions=prediction_df,
            alignment_report=alignment_report,
            checkpoint_path=resolved_checkpoint,
            species=species,
        )
    return prediction_df


def predict_bulk(
    bulk: str | Path | pd.DataFrame,
    species: str | None = None,
    checkpoint_path: str | Path | None = None,
    genes_are_rows: bool = True,
    device: str | torch.device = "auto",
    batch_size: int = 256,
    return_result: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | PredictionResult:
    """Predict cell fractions from a CSV path or a pandas DataFrame."""

    if isinstance(bulk, pd.DataFrame):
        bulk_df = bulk
    else:
        bulk_df = read_bulk_count_csv(bulk, genes_are_rows=genes_are_rows)
        genes_are_rows = False

    return predict_bulk_dataframe(
        bulk_df,
        species=species,
        checkpoint_path=checkpoint_path,
        genes_are_rows=genes_are_rows,
        device=device,
        batch_size=batch_size,
        return_result=return_result,
        verbose=verbose,
    )