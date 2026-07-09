"""Input reading, gene alignment, and count preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneAlignmentReport:
    """Summary of how a bulk matrix was aligned to a model gene list."""

    bulk_gene_count: int
    training_gene_count: int
    overlapping_gene_count: int
    dropped_gene_count: int
    added_gene_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "bulk_gene_count": self.bulk_gene_count,
            "training_gene_count": self.training_gene_count,
            "overlapping_gene_count": self.overlapping_gene_count,
            "dropped_gene_count": self.dropped_gene_count,
            "added_gene_count": self.added_gene_count,
        }


def read_bulk_count_csv(file_path: str | Path, genes_are_rows: bool = True) -> pd.DataFrame:
    """Read a count CSV and return samples as rows and genes as columns."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing bulk count matrix: {path}")
    bulk_df = pd.read_csv(path, index_col=0)
    if genes_are_rows:
        bulk_df = bulk_df.T
    bulk_df.index = bulk_df.index.astype(str)
    bulk_df.columns = bulk_df.columns.astype(str)
    return bulk_df


def align_bulk_dataframe(
    bulk_df: pd.DataFrame,
    training_genes: Sequence[str],
    genes_are_rows: bool = False,
    return_report: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, GeneAlignmentReport]:
    """Align a bulk matrix to the exact gene order used by a trained model.

    Extra bulk genes are dropped. Training genes missing from the bulk matrix
    are added as zero-filled columns.
    """

    if genes_are_rows:
        bulk_df = bulk_df.T
    bulk_df = bulk_df.copy()
    bulk_df.index = bulk_df.index.astype(str)
    bulk_df.columns = bulk_df.columns.astype(str)
    if not bulk_df.columns.is_unique:
        raise ValueError("Bulk matrix has duplicated gene columns. Collapse duplicates before prediction.")

    training_genes = [str(gene) for gene in training_genes]
    if len(training_genes) != len(set(training_genes)):
        raise ValueError("Training gene list contains duplicated genes.")

    training_gene_set = set(training_genes)
    bulk_gene_set = set(bulk_df.columns)
    extra_bulk_genes = [gene for gene in bulk_df.columns if gene not in training_gene_set]
    missing_genes = [gene for gene in training_genes if gene not in bulk_gene_set]
    overlapping_genes = [gene for gene in bulk_df.columns if gene in training_gene_set]

    if not overlapping_genes:
        raise ValueError("No genes overlap between the bulk matrix and the model training genes.")

    report = GeneAlignmentReport(
        bulk_gene_count=bulk_df.shape[1],
        training_gene_count=len(training_genes),
        overlapping_gene_count=len(overlapping_genes),
        dropped_gene_count=len(extra_bulk_genes),
        added_gene_count=len(missing_genes),
    )
    if verbose:
        if extra_bulk_genes:
            print(f"Dropping {len(extra_bulk_genes):,} bulk genes that were not used by the trained model.")
        if missing_genes:
            print(f"Adding {len(missing_genes):,} model genes missing from the bulk matrix with zero counts.")

    aligned_bulk = bulk_df.reindex(columns=training_genes, fill_value=0.0)
    if return_report:
        return aligned_bulk, report
    return aligned_bulk


def normalize_bulk_counts(X: np.ndarray, scale_factor: float = 1_000_000.0) -> np.ndarray:
    """CPM-normalize each bulk sample and apply log1p."""

    X = np.asarray(X, dtype=np.float32)
    library_sizes = X.sum(axis=1, keepdims=True)
    if np.any(library_sizes <= 0):
        raise ValueError("At least one bulk sample has zero total counts and cannot be normalized.")
    normalized = (X / library_sizes) * scale_factor
    return np.log1p(normalized).astype(np.float32)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply gene-wise standardization values stored in the checkpoint."""

    return ((X - mean) / std).astype(np.float32)
