# AV1 3DGS Pipeline

Prepare a calibrated, metric-aligned COLMAP dataset once, then choose a trainer.
Keep `av1_colmap.py`, `av1_to_3dgs.py`, `av1_train.py`, and
`av1_spirula_sfm.py` together. These
pipeline scripts use `uv` to install their Python dependencies automatically.

## Get the scripts

```bash
git clone https://github.com/arcturus-vision/3dgs-scripts.git
cd 3dgs-scripts
```

This repository contains the scripts published on the
[Arcturus Vision apps page](https://arcturus.vision/apps#create-gaussian-model).
Trainer executables, including Spirula Studio, are installed separately; no
custom trainer build is bundled here.

The apps page documents the command-line workflows for Linux. The macOS
source-build and M-Splat instructions below are experimental; for the desktop
Spirula workflow on Windows or macOS, use the apps page.

| Trainer | Compute platform | Output | Gaussian limit |
| --- | --- | --- | --- |
| LichtFeld Studio (default/reference) | Supported NVIDIA GPU | `scene.sog`, plus `scene.ply` when emitted | 1.5 million by default |
| [M-Splat](https://github.com/rayanht/msplat) native C++ CLI | Apple Silicon, macOS 14+ / Metal | `msplat/scene.ply` | No hard cap in upstream CLI |
| [Spirula Studio](https://github.com/arcturus-vision/spirula-studio) | macOS / MoltenVK; other Vulkan or CUDA hosts | `spirula/scene.ply` | 1.5 million by default |

Native trainers run non-interactively. PLY is the interchange artifact; optional
SOG conversion is described below. The existing Linux LichtFeld command remains:

```bash
uv run av1_to_3dgs.py recording.mp4 out-lichtfeld --headless
```

Omit `--headless` to open LichtFeld's GUI. Its training schedule is controlled
by LichtFeld; the wrapper's `--iterations` applies only to M-Splat and Spirula.

Use `--max-gaussians 500000` (or another positive integer) to change the
LichtFeld or Spirula cap from its default of 1,500,000. M-Splat has no hard cap.
To retrain with a different cap, use a new output directory; `--resume` reuses
completed training rather than changing its Gaussian count.

## Set up Spirula on macOS or Linux

The same Vulkan build supports training and native SfM on both platforms.
Install [uv](https://docs.astral.sh/uv/) for the AV1 3DGS Pipeline scripts, then
install the platform prerequisites below.

**Apple Silicon macOS:** install Xcode command-line tools (or select an installed
Xcode), Homebrew, and:

```bash
brew install cmake ninja libomp
# Only for the default COLMAP reconstruction route:
brew install colmap
```

**Linux (Ubuntu/Debian):** install a C++17 compiler, CMake, Ninja, Vulkan headers
and loader, and a working GPU Vulkan driver:

```bash
sudo apt-get update
sudo apt-get install build-essential git cmake ninja-build libvulkan-dev vulkan-tools
# Only for the default COLMAP reconstruction route:
sudo apt-get install colmap
vulkaninfo --summary
```

`vulkaninfo` should list the intended physical GPU. Install the appropriate
NVIDIA driver or AMD/Intel Mesa Vulkan driver for the host; the loader package
alone does not provide GPU acceleration. In WSL, verify GPU Vulkan availability
inside the distribution; working CUDA alone does not establish Vulkan support.
The unified Vulkan route does not require the CUDA toolkit. Native Spirula SfM
needs no COLMAP executable; `uv` supplies `pycolmap` for model I/O and alignment.

**Build on either platform:** clone Spirula into a separate directory and use
the pinned fork revision and build command:

```bash
git clone --branch sfm-calibrated-manifest https://github.com/arcturus-vision/spirula-studio.git
git -C spirula-studio checkout 09383f75029fd9e0f34847e50d8514c3932db984
(cd spirula-studio && bash build_develop.bash -DSS_BACKEND=vulkan \
  -DSS_BUILD_GUI=OFF -DSS_BUILD_SFM=ON -DSS_BUILD_SAM=OFF)

# Keep this absolute path for the training commands below.
case "$(uname -s)" in
  Darwin) SPIRULA_BIN="$PWD/spirula-studio/build/spirula" ;;
  Linux)  SPIRULA_BIN="$PWD/spirula-studio/build_vulkan/spirula" ;;
esac
"$SPIRULA_BIN" sfm auto --help
```

Spirula fetches its pinned Slang compiler during configuration and MoltenVK on
macOS. Linux uses the installed Vulkan loader and GPU driver. This CLI build
includes native SfM and training; it can also train a shared COLMAP dataset.
For training only, `-DSS_BUILD_SFM=OFF` omits native reconstruction. The commands
below use `"$SPIRULA_BIN"`; set it again to the appropriate absolute binary path
if using a new shell. No separate calibration patch is needed.

### Optional M-Splat setup (Apple Silicon macOS only)

Install Xcode with its Metal compiler and select it as the active developer
directory, then build the native CLI separately:

```bash
git clone https://github.com/rayanht/msplat.git
cmake -S msplat -B msplat/build -DCMAKE_BUILD_TYPE=Release
cmake --build msplat/build -j 4
```

Keep M-Splat's `default.metallib` beside its binary. Use the **C++ `msplat`
binary**, not the separately packaged `msplat-train` CLI.

### Validation history

Record `git rev-parse HEAD` in each checkout for repeatable experiments.
The native adapters were exercised on Apple M3 Max/macOS 15.7.5 with M-Splat
1.1.4 (`6b819711fa7c90f054567cb4fb4937743367afdd`), Spirula
2026.9.13 (`d579cf8c755f475d43eebaf7e2f0874eb633e595`), COLMAP 4.2.0,
and SplatTransform 3.4.2. These are the original experiment revisions. The
Spirula fork revision pinned above includes calibrated native SfM support; its
macOS build and added calibration checks passed. The full manifest suite still
has two existing dual-fisheye rig expectation failures. A full reconstruction
has not been rerun at this newer revision, and the Linux build recipe has not
been rerun at this pin. Newer revisions may change flags or output layout.

## Prepare the recording once

Run these commands from this script directory, substituting your paths:

```bash
uv run av1_colmap.py "/path/to/recording.mp4" "/path/to/experiment/dataset" \
  --frames 100 --no-guided-matching
```

A capture must contain the Recorder app's two video tracks and embedded
`gpmd` metadata (`AVmf` / `AVmd`, JSON v1). An ordinary MP4 without calibration
and poses is not equivalent. No remote upload is needed.

`--frames` counts **stereo capture instants**: 100 gives 200 images, selected
across the full clip. Omit it to target two pairs per second (100–500 pairs).
The scripts read calibration and poses, rectify each eye to PINHOLE, convert
HLG to SDR where indicated by the stream metadata, reconstruct with COLMAP,
and align the reconstruction to the recorded poses. They retain the feature
and match database, raw reconstruction, and alignment report under `dataset/`.
Check `dataset/alignment.json`, registration coverage, and images before training.
Moving subjects and sky can still produce reconstruction artifacts.

On macOS, automatic COLMAP selection uses CPU extraction/matching; Metal
training does not make COLMAP use Metal. The default CPU feature-worker limit
is four; change it with `--colmap-threads`. For CPU processing, start with
`--no-guided-matching`: geometric verification remains enabled, but the extra
guided rematching pass is skipped. That pass can allocate large per-pair
descriptor matrices for these high-resolution images; raising the worker count
also raises memory demand. The compatibility default remains guided matching.
If registration or point coverage is poor, try the guided pass in a separate
dataset and compare alignment/coverage before choosing the shared dataset.
PyAV tries VideoToolbox on macOS
and falls back to software. Use `--hardware-decode none` to force software or
`--hardware-decode videotoolbox` to require hardware decoding.

Keep the default undistortion for cross-trainer comparisons. M-Splat does not
support the unrectified OPENCV_FISHEYE model and the adapter rejects it.
`--refine-focal-length` is an optional reconstruction experiment, not a trainer
setting: changing it requires a separate dataset.

Use a local volume with enough free space for extracted images, COLMAP's
SQLite database, resized training images, trainer builds and checkpoints.
Point output paths at an external disk when the system disk is tight; the
original recording is read in place.

## Train from that dataset

Use a fresh output directory for each experiment:

```bash
uv run av1_to_3dgs.py "/path/to/experiment/dataset" "/path/to/experiment/run-msplat" \
  --prepared-dataset --backend msplat --msplat "/path/to/msplat/build/msplat" \
  --iterations 7000 --training-image-scale 0.5

uv run av1_to_3dgs.py "/path/to/experiment/dataset" "/path/to/experiment/run-spirula" \
  --prepared-dataset --backend spirula --spirula "$SPIRULA_BIN" \
  --iterations 7000 --training-image-scale 0.5
```

For a single command starting from MP4, replace the dataset argument with the
recording and omit `--prepared-dataset`; add `--no-guided-matching` for the CPU
preparation described above. Override `--colmap` when it is not on
PATH. All trainers receive a temporary half-resolution dataset by default;
`--training-image-scale 1` uses original resolution. M-Splat's progressive
resolution schedule is disabled and Spirula's additional image downscaling
is disabled so this setting has one meaning across both native trainers.

M-Splat exports with `--keep-crs`. Spirula retains the input point frame with
scene centering disabled. Preserve this metric frame for the Arcturus XR
viewer instead of manually rotating or normalizing each trained model.

Each native run retains its PLY, Spirula checkpoints/config when applicable,
and `run.json` with the exact command, executable/data hashes, elapsed time,
Gaussian count and PLY hash. Redirect stdout/stderr to a log when collecting
performance evidence. PLY headers and payload sizes are checked after training.
M-Splat does **not** enforce the 1.5M delivery target: inspect its measured count
before device testing; do not silently prune it for a quality comparison.
The tested M-Splat version also limits contributions per image tile to 2,048.
If training reports per-tile overflow, some contributions were dropped during
rasterization; retain that warning as a quality caveat in the comparison log.

`--resume` reuses extraction/alignment checkpoints and verifies completed native
training against its recorded input/settings/output hashes. It is **not** a
native optimizer resume. Interrupted native training is retained for diagnosis;
use a fresh output directory with `--prepared-dataset` to restart training.
To change scale, iterations, executable or dataset, use a new output directory.
Extraction resume uses its saved image selection and calibration; use a new
dataset directory when changing the recording or extraction settings.
A SOG conversion failure can be retried with `--resume` without retraining.

## Spirula with its own SfM

`--sfm-backend colmap` remains the default, including the original LichtFeld
baseline. Select `--sfm-backend spirula --backend spirula --no-undistort` to use
Spirula's native feature extraction, matching, mapping and bundle adjustment.
The pipeline retains native calibrated fisheye images, then aligns the largest
reconstruction to recorded tracking. `pycolmap` only reads/writes the interchange
model and applies that alignment; no COLMAP reconstruction commands run.

Use the SfM-enabled Vulkan binary from the shared macOS/Linux setup above.
The pinned [Arcturus Vision fork](https://github.com/arcturus-vision/spirula-studio/tree/sfm-calibrated-manifest)
includes full-calibration manifest support. If previously built with
`-DSS_BUILD_SFM=OFF`, rerun that build command with `-DSS_BUILD_SFM=ON`.

The fork preserves separate `fx/fy`, the actual principal point and all four
fisheye distortion coefficients in manifest `params`.
The wrapper validates exported fixed intrinsics and fails if they were ignored.
Focal length is refined by native SfM; principal point and distortion remain fixed.
The tested CPU bundle-adjustment path is selected explicitly on all hosts.

Run separate experiments from the same recording and sampling count:

```bash
# Original reference: COLMAP preparation, then LichtFeld on an NVIDIA host.
uv run av1_to_3dgs.py recording.mp4 out-lfs --backend lichtfeld \
  --sfm-backend colmap --frames 100 --headless

# Native Spirula SfM and training on macOS or Linux.
uv run av1_to_3dgs.py recording.mp4 out-spirula-sfm --backend spirula \
  --sfm-backend spirula --no-undistort --frames 100 \
  --spirula "$SPIRULA_BIN" --iterations 30000 \
  --training-image-scale 0.5 --splat-transform /path/to/splat-transform
```

`--spirula-sfm /path/to/another/spirula` can select a separate SfM-enabled binary
while `--spirula` selects the trainer. `--prepared-dataset` bypasses reconstruction
and therefore cannot be combined with `--sfm-backend spirula`. COLMAP tuning flags
apply only to the COLMAP route; native SfM uses the tested high-quality video /
prefilter settings. No image-circle mask is added by this option.

Native SfM retains `dataset/.spirula-sfm/`, `spirula-manifest.json`,
`spirula-sfm.log`, `spirula-sfm-run.json`, aligned `sparse/0/`, and `alignment.json`.
`spirula-dataset.json` records source/binary/settings and output hashes.
At least half the extracted views must register, with a nonempty sparse cloud.
Exit code 4 (native metric frame unavailable) is accepted only after these checks,
because the wrapper performs tracking alignment itself; other nonzero codes fail.

`--resume` verifies and reuses a completed native-SfM dataset and completed training;
it does not restart partial SfM or restore a training optimizer. Partial outputs
and logs are retained. Changed source, settings, binary or dataset bytes require a
new output directory. Spirula training explicitly disables pinhole warping.

This comparison changes reconstruction, camera estimates and potentially the
registered-image set as well as the trainer. A fixed original-camera PSNR score
can penalize camera/alignment differences, even when local detail looks better.
For a trainer-only comparison, use one shared `--prepared-dataset` instead.

## Optional SOG for the XR viewer

Install [PlayCanvas SplatTransform](https://github.com/playcanvas/splat-transform)
with Node.js/npm:

```bash
npm install -g @playcanvas/splat-transform
```

Add `--splat-transform /path/to/splat-transform` to either native training command
(or use the executable name when it is on PATH). The wrapper retains PLY and
writes `msplat/scene.sog` or `spirula/scene.sog`, checking the ZIP's integrity and
Gaussian count and recording its hash.

The validated SOG is also copied atomically to `scene.sog` at the output root,
matching the LichtFeld output path. Resuming with SOG conversion enabled
recreates this file without repeating completed training. Backend-specific
PLY files, checkpoints, and logs remain in their existing folders.
Alternatively convert a PLY yourself:

```bash
splat-transform scene.ply scene.sog
```

SOG encoding is lossy. Compare trainer PLY renders first; compare converted SOG
and on-device behavior separately. A valid SOG archive is not proof of XR
runtime performance or visual quality. The current av1-apps Gaussian XR SOG
loader renders DC color only and skips higher-order SH payloads; retain full-SH
PLY for desktop evaluation and report DC-only XR results separately. LichtFeld
retains its built-in SOG conversion path and now also keeps `scene.ply` when
the trainer emits one. If the selected LichtFeld mode exports only SOG, export
PLY separately there for a comparison before compression.

## Later LichtFeld reference comparison

Copy the **same** dataset to the NVIDIA host, then run:

```bash
uv run av1_to_3dgs.py "/path/to/experiment/dataset" "/path/to/experiment/run-lichtfeld" \
  --prepared-dataset --backend lichtfeld --headless --training-image-scale 0.5
```

Freeze image selection, rectification, intrinsics, poses, sparse points, image
scale and color processing. Record trainer revision, settings, hardware,
wall time, Gaussian count, output size, and artifact hashes. Equal iteration
counts alone do not equalize training work or quality. These production
commands train on all registered images; they do not report held-out quality.
For quantitative evaluation, define one common split by **capture instant**
(keep both eyes together) before any trainer runs, keep held-out images out of
training, and evaluate all outputs with the same renderer/views. Per-trainer
PSNR/SSIM numbers and independently selected eval splits are not directly
comparable. Retain the full PLY and distinguish compression effects, rendering
conventions, scale/orientation, and XR frame-time results from trainer quality.
When rendering previews with SplatTransform 3.4.2, its PLY importer maps source
coordinates into engine space with a 180-degree rotation around Z. Apply the
same transform to COLMAP-derived camera position, target and up vectors;
otherwise a correct model can appear to render from the wrong viewpoint. Its
CLI camera assumes a centered principal point, so use a calibrated renderer
for pixel-aligned metrics.

## PyAV compatibility

The scripts support PyAV 18.1 and 19.x (`av>=18.1,<20`). PyAV 19 requires
Python 3.12 or later; Python 3.11 can use PyAV 18.1. Container metadata decoding
uses each version's default (strict UTF-8 in 18.1, UTF-8 with surrogate escape in
19). Embedded camera JSON is still decoded and validated separately.

## Script checks

To check syntax and command-line options from this directory:

```bash
python3 -m py_compile av1_colmap.py av1_to_3dgs.py av1_train.py av1_spirula_sfm.py
uv run av1_colmap.py --help
uv run av1_to_3dgs.py --help
```

These checks do not validate GPU execution or reconstruction quality. Verify
native backends with a real recording or prepared dataset on the target host.

## Gaussian XR starting view

The LichtFeld and Spirula workflows write `scene.xrSceneTransform.json` beside
the final `scene.sog`. If Spirula runs without `--splat-transform`, the sidecar
is beside `spirula/scene.ply` instead.
Copy both files to Gaussian XR. The sidecar selects a registered camera near
the median camera position, applies the source track's MP4 Display rotation,
and preserves COLMAP's +Z-forward, +Y-down camera axes, as required by Gaussian
XR's `worldAdjustment` transform (the viewer handles the XR axis conversion).
The starting view keeps the camera's position and horizontal heading, but removes
pitch and roll using the tracking-aligned world's up axis. For a camera pointing
straight up or down, its display-oriented right axis supplies the heading instead.
This does not change scene geometry or metric scale. `--resume` backfills a missing sidecar without
rerunning LichtFeld or reconstruction. Spirula also exports the sidecar when
resuming completed training. Both COLMAP and Spirula SfM use the aligned
reconstruction frame; no additional world rotation is applied for SOG.

For `--prepared-dataset`, provide `--xr-recording /path/to/original.mp4` to
recover the display orientation. Without that original recording, training
still works, but the script warns and skips starting-view export.

## License

These scripts are provided under the [MIT License](LICENSE). Separately
installed trainers and dependencies remain subject to their own licenses.
