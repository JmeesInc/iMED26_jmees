# iMED Challenge 2026 — Team Jmees

Submission code for both tasks of the [iMED Challenge](https://imed-challenge.github.io/)
(EndoVis 2026 @ MICCAI 2026): pose estimation (PE) and novel view synthesis (NVS)
on stereo endoscopic surgical frames.

| Task | Method | Submitted image | Development set | Validation LB |
|---|---|---|---|---|
| PE  | **Tri3D-GridBA** | `jmees26-pe:v1`  | **0.953 mm** mean ATE (19 seq) vs 2.540 baseline | 2.296 mm ATE-RMSE |
| NVS | **SDW** (Static-rig Depth Warp) | `jmees26-nvs:v12` | **22.39 dB** PSNR (20 seq / 3,953 frames) | 21.156 PSNR / 0.621 SSIM / 0.200 LPIPS |

The code in `pe/` and `nvs/` is the code that ran inside the submitted containers:
`nvs/predict.py` and `nvs/tool_seg.py` are byte-identical to `jmees26-nvs:v12`, and
`pe/predict.py` was extracted from `jmees26-pe:v1`.

A full description of both methods, the ablations behind them and the negative
results is in [`report/imed_structured_report.tex`](report/imed_structured_report.tex)
(the challenge's Structured Method Report).

---

## Pose estimation — Tri3D-GridBA

The target is the **same-time cross-camera** relative pose `T(e2 <- e1)` per frame,
frame 0 = identity, scored by Sim(3)-aligned ATE. Two facts drive the design: ATE
sees only the **translation**, and Sim(3) alignment absorbs one global scale — so an
absolute scale is unnecessary but the *relative* scale between frames matters. The
official baseline recovers the pose from an essential matrix and therefore normalises
the translation to unit norm, discarding exactly that. Recovering it is our main gain.

The pipeline is **fully geometric — nothing is trained**:

1. ALIKED + LightGlue features, extracted once per image and shared across three
   matches per frame: `e1L↔e1R`, `e2L↔e2R`, `e1L↔e2L`.
2. Each scope's left–right extrinsic is constant, so it is estimated once and
   aggregated over **all** frames — one fixed triangulation scale per sequence.
3. Both stereo clouds are triangulated and paired through the cross match, then
   aligned by a **Tukey-IRLS weighted Umeyama Sim(3)** fit. For a same-time rigid
   pose any correct correspondence already determines the transform, so whatever
   disagrees with the dominant model is a mismatch, a non-rigidly moving point or
   inter-camera sync jitter — all handled as outliers.
4. **Grid BA.** The exact rigid relation `T(t+1) = M_e2(t) T(t) M_e1(t)^-1` is
   *linear in the translations* once the rotations are fixed, so the per-frame
   (absolute) terms, the temporal ego-motion (relative) terms and a second-difference
   smoothness prior fuse into a single linear least-squares solve.
5. Frame-to-initial, then frame 0 re-anchored to exact identity.

**Ablation** (19 released test sequences with ground truth, mean ATE):

| Configuration | mean ATE | vs baseline |
|---|---|---|
| Official baseline (ALIKED+LightGlue, essential matrix, unit `t`) | 2.540 mm | — |
| Tri3D: two-sided 3D–3D Umeyama, consistent stereo scale | 1.034 mm | −59% |
| + Tukey IRLS robust rigid fit | 0.998 mm | −61% |
| **+ grid BA (K=1, λ_rel=0.3)** — *submitted* | **0.953 mm** | **−62.5%** |

Learned alternatives all lost to the classical pipeline: VGGT cross-camera 1.79 mm;
Depth-Anything-3 cross-camera 1.39 vs 1.27 (same subset); DA3 metric depth instead of
triangulation 1.084 vs 1.014; a learned per-frame BA confidence 1.092 vs 1.067 (LOSO).

## Novel view synthesis — SDW

`pose.txt` holds exactly two entries per sequence — endoscope 2 at identity,
endoscope 1 at a fixed relative pose. **The cameras never move; only the scene
deforms.** So this is not trajectory novel-view synthesis but a per-frame
re-projection between two static viewpoints, and the deformation needs no model at
all: it is fully observed in the source frame and its depth map. That is why a
training-free warp matches a 4D Gaussian Splatting baseline that optimises for hours
per sequence:

| Method (`session_004_scene_2_tool_1`, native resolution) | PSNR | SSIM | LPIPS | Cost |
|---|---|---|---|---|
| Official Endo-4DGS (coarse 2000 + fine 6000 it.) | 21.14 | 0.699 | 0.0190 | ~2 h optimisation / seq |
| Ours, depth-reprojection warp | **21.45** | **0.721** | 0.0198 | inference only |

Per frame: segment the instrument out of the source image and drop those pixels
(the official metric excludes endoscope-1 instrument pixels, so a reprojected
instrument can only paint metal over tissue *inside* the scored region) → back-project
with the source intrinsics → transform by the constant relative pose → forward-splat
with an exact GPU z-buffer (depth and source index packed into one int64, resolved by
a single `scatter_reduce(amin)`) → fill disocclusions with a push–pull pyramid → apply
a frozen per-channel affine tone curve.

**2×2 factorial** (all 20 sequences / 3,953 frames, measured from the containers):

| Instrument excluded | Hole filler | PSNR | SSIM (gray) |
|---|---|---|---|
| no | push–pull | 22.3625 | 0.6892 |
| **yes** | **push–pull** | **22.4880** | **0.6939** |
| no | Fast-Marching inpainting | 22.4473 | 0.6449 |
| yes | Fast-Marching inpainting | 22.5602 | 0.6499 |

Main effects are additive: instrument exclusion is +0.126 PSNR / +0.005 SSIM (a clean
win on both), inpainting is +0.085 PSNR but **−0.044 SSIM** — and the submission that
bundled the two lost on the leaderboard. Hence push–pull is the deployed filler, and
hence one factor per submission.

---

## Build and run

Both containers follow the challenge I/O contract: `/input` is mounted read-only and
results are written under `/output`. Neither reads the target view (`endoscope1`) at
inference, and neither needs network access at runtime — weights are baked in at build
time so the container runs with `--network=none`.

```bash
# --- PE ---
cd pe
./build.sh imed-pe-jmees:dev
./scripts/local_test.sh imed-pe-jmees:dev /path/to/sequences /tmp/pe_out
# -> /tmp/pe_out/<seq>/pose_predictions.txt   (8 cols: k tx ty tz qx qy qz qw)

# --- NVS ---
cd nvs
# place convnext-unet-best.pth in nvs/weights/ first (see nvs/weights/README.md)
./build.sh imed-nvs-jmees:dev
./scripts/local_test.sh imed-nvs-jmees:dev /path/to/sequences /tmp/nvs_out
# -> /tmp/nvs_out/<seq>/renders/00000.png ...  (1280x1024 8-bit RGB)
```

The NVS container **requires CUDA**: the evaluator rejects CPU-only inference
regardless of wall-clock time. If your host lacks `nvidia-container-runtime`,
`--gpus all` will fail; you can wire the GPU up by hand instead by passing the
`/dev/nvidia*` device nodes with `--device` and bind-mounting `libcuda.so.<driver>`,
`libnvidia-ml.so.<driver>` and `libnvidia-ptxjitcompiler.so.<driver>` over the
container's stubs at `/usr/lib/x86_64-linux-gnu/*.so.1`.

Every NVS knob is an environment variable (`NVS_EXCLUDE_TOOL`, `NVS_TOOL_SOURCE`,
`NVS_TOOL_THR`, `NVS_HOLE_FILL`, `NVS_PP_LEVELS`, `NVS_COLOUR`, …), so ablations run
on a single image with no rebuild. `scripts/local_test.sh` forwards any `NVS_*` set in
the calling shell.

**Runtime.** PE ≈ 0.43 s/frame (11 min for 19 sequences on a Quadro RTX 8000; the
evaluator allows 10 min total on an RTX 4090). NVS ≈ 59 s/sequence.

## Pretrained models and external data

- **PE** uses ALIKED (`aliked-n16`) and LightGlue (`aliked` weights) from the public
  [cvg/LightGlue](https://github.com/cvg/LightGlue) release, off the shelf, no
  fine-tuning. **No external datasets, no training.**
- **NVS** uses one learned component: a U-Net with a ConvNeXt-Base encoder
  (`tu-convnext_base.dinov3_lvd1689m` via `timm`) trained for binary instrument
  segmentation on the public **Cholec80** dataset. The checkpoint predates this
  challenge, contains no iMED data, and is used inference-only. See
  [`nvs/weights/README.md`](nvs/weights/README.md).
- The NVS photometric constants are the only other fitted quantity; they are fitted
  offline on the released public iMED NVS sequences and frozen. No ground truth is
  read at inference.

## Repository layout

```
pe/     predict.py, Dockerfile, build + local-test scripts   (= jmees26-pe:v1)
nvs/    predict.py, tool_seg.py, Dockerfile, build + local-test scripts (= jmees26-nvs:v12)
report/ imed_structured_report.tex   — the challenge Structured Method Report
```

## License

MIT — see [LICENSE](LICENSE). The challenge data itself is **not** included and is
distributed by the organisers under their own terms.
