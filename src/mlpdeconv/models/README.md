# Packaged Models

This folder is reserved for checkpoints that should ship inside an installed Python package.

Use this layout if you later decide to distribute a wheel that includes pretrained models:

```text
src/mlpdeconv/models/
  rat/
    mlp_deconv_checkpoint.pt
  mouse/
    mlp_deconv_checkpoint.pt
```

For local work, the top-level `models/` directory is usually easier.
