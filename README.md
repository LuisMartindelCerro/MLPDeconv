# MLPDeconv

MLPDeconv predicts cell-type fractions from bulk RNA-seq count matrices using a trained PyTorch MLP model.

The repository now has two layers:

1. **Package code** in `src/mlpdeconv/`
   This is the reusable tool. It can be installed and called from Python, from the command line, or from an R pipeline.
2. **Notebooks** in the repository root
   These are examples and development workflows. They show how pseudobulks are created, how a model is trained, and how validation plots are made.

For production prediction, use the package or CLI. For learning, retraining, or experimenting with new methods, use the notebooks.

## Install For Local Use

From the repository root:

```bash
pip install -e .
```

`-e` means editable install. Python imports the package directly from this folder, so code changes are visible without reinstalling.

If you use `uv`, you can also run:

```bash
uv pip install -e .
```

## Pretrained Model Location

The prediction tool needs a trained checkpoint file. The checkpoint produced by `02_train_eval_model.ipynb` is named:

```text
mlp_deconv_checkpoint.pt
```

The easiest species-specific layout is:

```text
models/
  rat/
    rat_mlp_deconv_checkpoint.pt
  mouse/
    mouse_mlp_deconv_checkpoint.pt
```

Then prediction can use `--species rat` or `species="rat"`.
You can always bypass species lookup by passing a checkpoint path directly.

## Command-Line Prediction

Most useful for an R pipeline:

```bash
mlpdeconv predict \
  --bulk raw_counts.csv \
  --species rat \
  --genes-are-rows \
  --out predictions.csv
```

By default, the CLI expects the input CSV to have genes as rows and samples as columns. Use `--samples-are-rows` if your file has samples as rows and genes as columns.

To use a specific checkpoint:

```bash
mlpdeconv predict \
  --bulk raw_counts.csv \
  --checkpoint models/rat/mlp_deconv_checkpoint.pt \
  --out predictions.csv
```

To see which models are found:

```bash
mlpdeconv list-models
```

## Python Prediction

```python
from mlpdeconv import predict_bulk

predictions = predict_bulk(
    "raw_counts.csv",
    species="rat",
    genes_are_rows=True,
)

predictions.to_csv("predictions.csv", index_label="sample_id")
```

You can also pass a pandas DataFrame:

```python
import pandas as pd
from mlpdeconv import predict_bulk_dataframe

bulk_df = pd.read_csv("raw_counts.csv", index_col=0).T

predictions = predict_bulk_dataframe(
    bulk_df,
    species="rat",
)
```

## R Pipeline Usage

R can call the command-line tool and then read the CSV:

```r
status <- system2(
  "mlpdeconv",
  args = c(
    "predict",
    "--bulk", "raw_counts.csv",
    "--species", "rat",
    "--genes-are-rows",
    "--out", "predictions.csv"
  )
)

if (status != 0) {
  stop("MLPDeconv prediction failed")
}

predictions <- read.csv("predictions.csv", row.names = 1)
```

This is usually cleaner than trying to load Python objects directly inside R.

## What The Checkpoint Contains

The trained checkpoint is more than neural-network weights. It also stores:

- the exact training gene list and order
- cell-type names and order
- gene-wise mean and standard deviation from training
- normalization settings
- model architecture settings
- trained PyTorch weights

This is why exporting only raw weights is not enough. Prediction must repeat the same preprocessing used during training.

## Gene Matching Behavior

Before prediction, the package aligns the input bulk matrix to the model gene list:

- bulk genes not used during training are dropped
- model genes missing from the bulk matrix are added as zero-filled columns
- columns are reordered to exactly match the checkpoint gene order

This makes external bulk matrices easier to use safely.

## Package File Map

```text
src/mlpdeconv/
  __init__.py       Public Python imports
  model.py          MLP neural-network architecture
  checkpoint.py     Checkpoint loading and validation
  preprocessing.py  CSV reading, gene alignment, normalization
  registry.py       Species/model checkpoint lookup
  predict.py        Python prediction API
  cli.py            Command-line interface
```

## Notebooks

- `01_create_pseudobulks.ipynb`: creates train/test pseudobulk datasets
- `02_train_eval_model.ipynb`: trains and saves the model checkpoint
- `03_predict_real_bulk.ipynb`: notebook-style validation/prediction workflow
- `04_hyperparameter_tuning_optuna.ipynb`: hyperparameter exploration

The package is the stable tool. The notebooks are examples and development workflows.
