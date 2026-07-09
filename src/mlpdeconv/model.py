"""Neural-network architecture used by MLPDeconv checkpoints."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPDeconvRegressor(nn.Module):
    """Small feed-forward neural network that outputs cell-type fractions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.network(X)
        return torch.softmax(logits, dim=1)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> MLPDeconvRegressor:
    """Create a model object and load weights from a saved checkpoint."""

    model_config = checkpoint["model_config"]
    model = MLPDeconvRegressor(
        input_dim=len(checkpoint["genes"]),
        output_dim=len(checkpoint["cell_types"]),
        hidden_sizes=tuple(model_config["hidden_sizes"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
