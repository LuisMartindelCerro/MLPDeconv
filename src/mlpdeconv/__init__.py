"""MLPDeconv inference package.

The package exposes a small Python API for predicting cell fractions from brain bulk
RNA-seq count matrices with a trained MLPDeconv checkpoint.
"""

from importlib.metadata import PackageNotFoundError, version

from mlpdeconv.predict import PredictionResult, predict_bulk, predict_bulk_dataframe
from mlpdeconv.registry import find_model_checkpoint, list_available_models

try:
    __version__ = version("mlpdeconv")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "PredictionResult",
    "__version__",
    "find_model_checkpoint",
    "list_available_models",
    "predict_bulk",
    "predict_bulk_dataframe",
]
