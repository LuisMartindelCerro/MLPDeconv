from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in this Python environment") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlpdeconv.cli import main as cli_main
from mlpdeconv.model import MLPDeconvRegressor
from mlpdeconv.predict import PredictionResult, predict_bulk
from mlpdeconv.preprocessing import align_bulk_dataframe


def write_test_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model = MLPDeconvRegressor(input_dim=3, output_dim=2, hidden_sizes=(4,), dropout=0.0)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": {"hidden_sizes": (4,), "dropout": 0.0},
        "genes": ["g1", "g2", "g3"],
        "cell_types": ["cell_a", "cell_b"],
        "gene_mean": np.zeros(3, dtype=np.float32),
        "gene_std": np.ones(3, dtype=np.float32),
        "normalization": {"method": "CPM_log1p", "scale_factor": 1_000_000.0},
        "loss": "MSELoss",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


class PredictionTests(unittest.TestCase):
    def test_align_bulk_dataframe_drops_extra_and_adds_missing_genes(self) -> None:
        bulk = pd.DataFrame(
            {
                "g2": [10.0, 4.0],
                "extra": [99.0, 99.0],
            },
            index=["sample_1", "sample_2"],
        )

        aligned, report = align_bulk_dataframe(
            bulk,
            training_genes=["g1", "g2", "g3"],
            return_report=True,
        )

        expected = pd.DataFrame(
            {
                "g1": [0.0, 0.0],
                "g2": [10.0, 4.0],
                "g3": [0.0, 0.0],
            },
            index=["sample_1", "sample_2"],
        )
        pd.testing.assert_frame_equal(aligned, expected)
        self.assertEqual(report.dropped_gene_count, 1)
        self.assertEqual(report.added_gene_count, 2)
        self.assertEqual(report.overlapping_gene_count, 1)

    def test_predict_bulk_uses_species_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "rat" / "mlp_deconv_checkpoint.pt"
            write_test_checkpoint(checkpoint_path)

            bulk_path = tmp_path / "bulk.csv"
            bulk = pd.DataFrame(
                {
                    "sample_1": [10.0, 2.0],
                    "sample_2": [5.0, 8.0],
                },
                index=["g2", "extra"],
            )
            bulk.to_csv(bulk_path)

            result = predict_bulk(
                bulk_path,
                species="rat",
                model_dir=tmp_path,
                genes_are_rows=True,
                return_result=True,
                device="cpu",
            )

            self.assertIsInstance(result, PredictionResult)
            self.assertEqual(result.predictions.shape, (2, 2))
            self.assertEqual(result.predictions.columns.tolist(), ["cell_a", "cell_b"])
            np.testing.assert_allclose(result.predictions.sum(axis=1).to_numpy(), np.ones(2), rtol=1e-6)
            self.assertEqual(result.alignment_report.dropped_gene_count, 1)
            self.assertEqual(result.alignment_report.added_gene_count, 2)

    def test_cli_writes_prediction_and_report_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "rat" / "mlp_deconv_checkpoint.pt"
            write_test_checkpoint(checkpoint_path)

            bulk_path = tmp_path / "bulk.csv"
            pd.DataFrame(
                {
                    "sample_1": [10.0],
                    "sample_2": [7.0],
                },
                index=["g2"],
            ).to_csv(bulk_path)

            out_path = tmp_path / "predictions.csv"
            report_path = tmp_path / "report.json"
            exit_code = cli_main(
                [
                    "predict",
                    "--bulk",
                    str(bulk_path),
                    "--species",
                    "rat",
                    "--model-dir",
                    str(tmp_path),
                    "--out",
                    str(out_path),
                    "--report-json",
                    str(report_path),
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(exit_code, 0)
            predictions = pd.read_csv(out_path, index_col=0)
            self.assertEqual(predictions.shape, (2, 2))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["species"], "rat")
            self.assertEqual(report["alignment"]["added_gene_count"], 2)


if __name__ == "__main__":
    unittest.main()
