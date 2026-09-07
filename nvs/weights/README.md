# Instrument-segmentation weights

`convnext-unet-best.pth` (~1.1 GB) is **not tracked in git**. Place it here before
running `../build.sh`; the Dockerfile copies it to `/opt/weights/` so the container
runs with `--network=none`.

It is a U-Net with a ConvNeXt-Base encoder (`tu-convnext_base.dinov3_lvd1689m`,
via `timm`), trained for binary surgical-instrument segmentation on the public
**Cholec80** dataset. No iMED challenge data was used to train it, and it is used
inference-only.

If the file is absent the pipeline still runs: `predict.py` falls back to the
supplied `endoscope2/toolL` masks, and if those are missing too, to reprojecting
the instrument (the pre-exclusion behaviour). Both fallbacks score lower.
