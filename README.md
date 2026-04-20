# Dswin_Dyconv

Minimal repository for IXI medical image registration experiments based on TransMorph, including:

- DSwin3D + DynConv matched variant
- DSwin3D + Attention-Guided Dynamic Convolution variant
- B-spline lightweight DSwin3D + DynConv variant
- B-spline lightweight DSwin3D + Attention-Guided Dynamic Convolution variant
- Expanded B-spline lightweight AGDynConv variant (`embed_dim=64`, `resize_channels=(24,24)`)

## Structure

- `IXI/TransMorph/models`: model and config files
- `IXI/TransMorph/data`: IXI dataset and transform pipeline
- `IXI/TransMorph/train_*.py`: training scripts
- `IXI/TransMorph/infer_*.py`: inference scripts
- `EXPERIMENT_ANALYSIS.md`: experiment results, comparisons, and debugging notes

## Notes

- Dataset paths in train/infer scripts are placeholders and should be updated before running.
- This repository intentionally contains only the files needed to reproduce the implemented variants.
