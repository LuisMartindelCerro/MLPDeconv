"""Find species-specific pretrained checkpoints."""

from __future__ import annotations

from pathlib import Path


def normalize_species(species: str) -> str:
    """Normalize species names for folder/file lookup."""
    return str(species).strip().lower().replace("-", "_").replace(" ", "_")


def packaged_models_dir() -> Path:
    """Return the package models directory."""
    return Path(__file__).resolve().parent / "models"


def checkpoint_name_for_species(species: str) -> str:
    """Return the expected checkpoint filename for a species."""
    species_key = normalize_species(species)
    return f"{species_key}_mlp_deconv_checkpoint.pt"


def find_model_checkpoint(
    species: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Find a checkpoint either by explicit path or by species name."""

    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {path}")
        return path

    if species is None or str(species).strip() == "":
        raise ValueError("Pass either species or checkpoint_path.")

    species_key = normalize_species(species)
    checkpoint_path = packaged_models_dir() / species_key / checkpoint_name_for_species(species_key)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint was found for species '{species}'. Expected:\n"
            f"{checkpoint_path}\n"
            f"You can also pass --checkpoint /path/to/checkpoint.pt."
        )

    return checkpoint_path


def list_available_models() -> dict[str, Path]:
    """List species checkpoints available inside src/mlpdeconv/models."""

    models_dir = packaged_models_dir()
    found: dict[str, Path] = {}

    if not models_dir.exists():
        return found

    for species_dir in models_dir.iterdir():
        if not species_dir.is_dir():
            continue

        species_key = normalize_species(species_dir.name)
        checkpoint_path = species_dir / checkpoint_name_for_species(species_key)

        if checkpoint_path.exists():
            found[species_key] = checkpoint_path.resolve()

    return dict(sorted(found.items()))