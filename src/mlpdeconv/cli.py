"""Command-line interface for MLPDeconv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlpdeconv import __version__
from mlpdeconv.predict import PredictionResult, predict_bulk
from mlpdeconv.registry import list_available_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlpdeconv",
        description="Predict cell fractions from bulk RNA-seq counts with a trained MLPDeconv model.",
    )
    parser.add_argument("--version", action="version", version=f"mlpdeconv {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict cell fractions for a bulk count matrix.")
    predict_parser.add_argument("--bulk", required=True, help="Input CSV count matrix.")
    predict_parser.add_argument("--species", help="Species/model name. Allowed options are 'rat' or 'mouse'.")
    predict_parser.add_argument("--checkpoint", help="Explicit checkpoint path. Overrides species lookup.")
    predict_parser.add_argument("--out", help="Output CSV path. If omitted, predictions are printed to stdout.")
    predict_parser.add_argument("--report-json", help="Optional JSON path for gene-alignment metadata.")
    predict_parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    predict_parser.add_argument("--batch-size", type=int, default=256, help="Prediction batch size.")
    orientation = predict_parser.add_mutually_exclusive_group()
    orientation.add_argument(
        "--genes-are-rows",
        dest="genes_are_rows",
        action="store_true",
        default=True,
        help="Input CSV has genes as rows and samples as columns. This is the default.",
    )
    orientation.add_argument(
        "--samples-are-rows",
        dest="genes_are_rows",
        action="store_false",
        help="Input CSV has samples as rows and genes as columns.",
    )

    subparsers.add_parser("list-models", help="List checkpoints discovered on disk.")

    return parser


def _run_predict(args: argparse.Namespace) -> int:
    if not args.species and not args.checkpoint:
        raise SystemExit("Use --species for a pretrained model, or --checkpoint for a specific checkpoint file.")

    result = predict_bulk(
        args.bulk,
        species=args.species,
        checkpoint_path=args.checkpoint,
        genes_are_rows=args.genes_are_rows,
        device=args.device,
        batch_size=args.batch_size,
        return_result=True,
        verbose=False,
    )
    assert isinstance(result, PredictionResult)

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(output_path, index_label="sample_id")
    else:
        sys.stdout.write(result.predictions.to_csv(index_label="sample_id"))

    report = {
        "species": result.species,
        "checkpoint_path": str(result.checkpoint_path),
        "alignment": result.alignment_report.to_dict(),
    }
    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2), file=sys.stderr)

    return 0


def _run_list_models(args: argparse.Namespace) -> int:
    models = list_available_models()
    if not models:
        print("No model checkpoints found.")
        return 0
    for name, path in models.items():
        print(f"{name}\t{path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "list-models":
        return _run_list_models(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())