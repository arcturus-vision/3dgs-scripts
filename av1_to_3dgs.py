#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "av>=18.1,<20",
#   "numpy",
#   "opencv-python-headless",
#   "pycolmap",
# ]
# ///
"""Prepare AV Cam images with COLMAP or Spirula SfM and select a 3DGS trainer."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import av
import cv2
import numpy as np
import pycolmap
from av1_colmap import (
    _add_dataset_arguments,
    _resolve_executable,
    _run,
    prepare_dataset,
)
from av1_train import train_native

MAX_GAUSSIANS = 1_500_000
DEFAULT_TRAINING_IMAGE_SCALE = 0.5


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _latest(paths: Iterable[Path]) -> Path | None:
    paths = [path for path in paths if path.is_file() and path.stat().st_size]
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None


def _display_rotation(recording: Path, camera_index: int) -> float:
    with av.open(str(recording)) as container:
        streams = list(container.streams.video)
        if camera_index >= len(streams):
            raise ValueError(f"Video stream {camera_index} is absent")
        frame = next(container.decode(streams[camera_index]), None)
        if frame is None:
            raise ValueError(f"Video stream {camera_index} has no decodable frame")
        # Named lookup works with both PyAV 18 values and PyAV 19 key iteration.
        has_display_matrix = frame.side_data.get("DISPLAYMATRIX") is not None
        rotation = float(frame.rotation if has_display_matrix else 0.0)
    if not has_display_matrix:
        print(
            f"Warning: video stream {camera_index} has no Display Matrix; "
            "assuming no display rotation.",
            file=sys.stderr,
        )
    quarter_turn = round(rotation / 90.0)
    if not math.isclose(rotation, quarter_turn * 90.0, abs_tol=1e-3):
        raise ValueError(
            f"Video stream {camera_index} has unsupported Display rotation "
            f"{rotation:g} degrees"
        )
    return float(quarter_turn * 90)


def _rotation_z(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.array([
        [cosine, -sine, 0.0, 0.0],
        [sine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _level_xr_starting_pose(pose: np.ndarray) -> np.ndarray:
    """Keep position and heading in the tracking-aligned, Y-down GS world."""
    if not np.all(np.isfinite(pose)):
        raise ValueError("Gaussian XR initial pose is not finite")
    down = np.array([0.0, 1.0, 0.0])
    forward = pose[:3, 2].copy()
    forward[1] = 0.0
    if np.linalg.norm(forward) < 1e-6:
        # Looking straight up/down has no horizontal optical-axis heading.
        # Use the display-oriented right axis to choose a stable level view.
        right = pose[:3, 0].copy()
        right[1] = 0.0
        forward = np.cross(right, down)
        print("Warning: starting camera points vertically; using its right axis "
              "for the level heading.", file=sys.stderr)
    length = np.linalg.norm(forward)
    if length < 1e-6:
        raise ValueError("Cannot determine Gaussian XR starting heading")
    forward /= length
    result = pose.copy()
    result[:3, :3] = np.column_stack((np.cross(down, forward), down, forward))
    return result


def write_xr_scene_transform(recording: Path | None, dataset: Path, scene: Path) -> Path | None:
    if recording is None:
        print("Warning: no XR starting pose exported for prepared dataset; "
              "pass --xr-recording with the original MP4.", file=sys.stderr)
        return None
    reconstruction = pycolmap.Reconstruction(dataset / "sparse" / "0")
    candidates: list[tuple[str, int, np.ndarray]] = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        name = image.name.replace("\\", "/")
        camera_dir = name.partition("/")[0]
        if not camera_dir.startswith("cam") or not camera_dir[3:].isdigit():
            raise ValueError(f"Cannot determine video stream from COLMAP image {name!r}")
        pose = np.eye(4)
        pose[:3, :] = np.asarray(image.cam_from_world().inverse().matrix(), dtype=np.float64)
        candidates.append((name, int(camera_dir[3:]), pose))
    if not candidates:
        raise ValueError("Cannot create Gaussian XR pose: COLMAP registered no images")

    centers = np.asarray([candidate[2][:3, 3] for candidate in candidates])
    median = np.median(centers, axis=0)
    name, camera_index, world_from_cv_camera = min(
        candidates,
        key=lambda candidate: (
            float(np.linalg.norm(candidate[2][:3, 3] - median)), candidate[0]
        ),
    )
    rotation_ccw = _display_rotation(recording, camera_index)
    # Gaussian XR loads this as worldAdjustment and renders with
    # worldAdjustment @ flipYZ @ xr_pose @ flipYZ. At the reference XR pose,
    # the two flips cancel: the sidecar must use COLMAP's +Z-forward,
    # +Y-down camera axes, not an OpenGL/OpenXR camera basis. An extra
    # right-hand flip here turns the starting view upside down and backward.
    world_from_gs_camera = _level_xr_starting_pose(
        world_from_cv_camera @ _rotation_z(rotation_ccw)
    )

    sidecar = scene.with_suffix(".xrSceneTransform.json")
    sidecar.write_text(
        json.dumps({
            "transform": world_from_gs_camera.tolist(),
            "worldScale": 1.0,
            "worldScaleOffset": [0.0, 0.0, 0.0],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Gaussian XR initial view: {name} "
        f"(level horizon; Display rotation {rotation_ccw:g}° counter-clockwise)."
    )
    print(f"Gaussian XR scene transform: {sidecar}")
    return sidecar


def _lichtfeld_help(lichtfeld: str) -> str:
    result = _run([lichtfeld, "--help"], capture=True)
    text = f"{result.stdout}\n{result.stderr}"
    if "--max-cap" not in text:
        raise RuntimeError(
            "Installed LichtFeld Studio cannot enforce the Gaussian cap: "
            "--max-cap is absent from --help."
        )
    if "--data-path" not in text and not re.search(r"(?:^|\s)-d(?=[\s\[=,])", text):
        raise RuntimeError("Installed LichtFeld Studio --help has no dataset-path option")
    if "--output-path" not in text and not re.search(r"(?:^|\s)-o(?=[\s\[=,])", text):
        raise RuntimeError("Installed LichtFeld Studio --help has no output-path option")
    return text


def prepare_training_dataset(dataset: Path, output_dir: Path, image_scale: float) -> Path:
    if not math.isfinite(image_scale) or not 0 < image_scale <= 1:
        raise ValueError("Training image scale must be greater than 0 and at most 1")
    training_dataset = output_dir / ".training-input"
    if dataset.resolve().is_relative_to(training_dataset.resolve()):
        raise ValueError("Prepared dataset overlaps the temporary training directory")
    if training_dataset.exists():
        shutil.rmtree(training_dataset)
    if image_scale == 1:
        print("Training with full-resolution COLMAP images.", flush=True)
        return dataset

    images_dir = training_dataset / "images"
    sparse_dir = training_dataset / "sparse" / "0"
    sparse_dir.mkdir(parents=True)

    reconstruction = pycolmap.Reconstruction(dataset / "sparse" / "0")
    camera_sizes: dict[int, tuple[int, int, int, int]] = {}
    for camera_id, camera in reconstruction.cameras.items():
        width, height = camera.width, camera.height
        scaled_width = max(1, round(width * image_scale))
        scaled_height = max(1, round(height * image_scale))
        camera_sizes[camera_id] = (width, height, scaled_width, scaled_height)
        camera.rescale(scaled_width, scaled_height)

    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    progress_interval = max(1, len(images) // 10)
    for index, image in enumerate(images, start=1):
        source_path = dataset / "images" / image.name
        pixels = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if pixels is None:
            raise RuntimeError(f"Could not read registered image: {source_path}")
        width, height, scaled_width, scaled_height = camera_sizes[image.camera_id]
        if pixels.shape[1] != width or pixels.shape[0] != height:
            raise RuntimeError(
                f"Image dimensions do not match COLMAP camera for {image.name}: "
                f"{pixels.shape[1]}x{pixels.shape[0]} versus {width}x{height}"
            )
        for point in image.points2D:
            point.xy = point.xy * [scaled_width / width, scaled_height / height]
        destination_path = images_dir / image.name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        resized = cv2.resize(
            pixels, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA
        )
        if not cv2.imwrite(str(destination_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write resized image: {destination_path}")
        if index == len(images) or index % progress_interval == 0:
            print(f"Prepared training images: {index}/{len(images)}", flush=True)

    reconstruction.write(sparse_dir)
    print(
        f"Training image scale: {image_scale:g} "
        f"({len(images)} registered images beneath {images_dir}).",
        flush=True,
    )
    return training_dataset


def train_and_convert(
    dataset: Path,
    output_dir: Path,
    *,
    lichtfeld: str,
    headless: bool,
    max_gaussians: int = MAX_GAUSSIANS,
) -> Path:
    help_text = _lichtfeld_help(lichtfeld)
    if headless and "--headless" not in help_text:
        raise RuntimeError("Installed LichtFeld Studio does not advertise --headless")

    training_dir = output_dir / ".lichtfeld-training"
    if training_dir.exists():
        shutil.rmtree(training_dir)
    training_dir.mkdir()
    command: list[object] = [
        lichtfeld,
        "--data-path" if "--data-path" in help_text else "-d", dataset,
        "--output-path" if "--output-path" in help_text else "-o", training_dir,
        f"--max-cap={max_gaussians}",
    ]
    if headless:
        command.append("--headless")
    _run(command)

    final_sog = output_dir / "scene.sog"
    ply = _latest(training_dir.rglob("splat_*.ply"))
    if ply is not None:
        shutil.copy2(ply, output_dir / "scene.ply")
    direct_sog = _latest(training_dir.rglob("*.sog"))
    if direct_sog is not None:
        shutil.copy2(direct_sog, final_sog)
    else:
        if ply is None:
            mode = "Headless training" if headless else "LichtFeld Studio exited"
            raise RuntimeError(
                f"{mode} without producing a trained splat in {training_dir}."
            )
        _run([lichtfeld, "convert", ply, final_sog])

    if not final_sog.is_file() or final_sog.stat().st_size == 0:
        raise RuntimeError(f"LichtFeld conversion did not produce a non-empty SOG: {final_sog}")
    shutil.rmtree(training_dir)
    print(f"Final SOG: {final_sog}")
    return final_sog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AV Cam MP4 → COLMAP or Spirula SfM → aligned 3DGS training."
    )
    parser.add_argument("recording", type=Path, help="AV Cam stereo MP4, or COLMAP directory with --prepared-dataset")
    parser.add_argument("output_directory", type=Path, help="new or empty output directory")
    parser.add_argument("--backend", choices=("lichtfeld", "msplat", "spirula"), default="lichtfeld")
    parser.add_argument("--prepared-dataset", action="store_true",
                        help="input is an existing aligned COLMAP dataset; skip MP4/SfM")
    parser.add_argument("--msplat", default="msplat", help="native M-Splat C++ CLI executable")
    parser.add_argument("--spirula", default="spirula", help="Spirula executable")
    parser.add_argument("--sfm-backend", choices=("colmap", "spirula"), default="colmap",
                        help="MP4 reconstruction engine (default: original COLMAP baseline)")
    parser.add_argument("--spirula-sfm", help="optional separate SfM-enabled Spirula executable")
    parser.add_argument("--iterations", type=int, default=7000,
                        help="M-Splat/Spirula iterations (default: 7000)")
    parser.add_argument("--splat-transform", help="optional splat-transform executable for PLY → SOG")
    parser.add_argument(
        "--headless", action="store_true",
        help="run LichtFeld Studio non-interactively (GUI is the default)",
    )
    _add_dataset_arguments(parser, include_ncams=False)
    parser.add_argument(
        "--lichtfeld", default="LichtFeld-Studio",
        help="LichtFeld Studio executable (default: PATH)",
    )
    parser.add_argument(
        "--training-image-scale", type=float, default=DEFAULT_TRAINING_IMAGE_SCALE,
        metavar="SCALE",
        help="Training image scale; use 1 for full resolution (default: 0.5)",
    )
    parser.add_argument(
        "--max-gaussians", type=_positive_int, default=MAX_GAUSSIANS, metavar="N",
        help=f"maximum LichtFeld/Spirula Gaussian count (default: {MAX_GAUSSIANS}); M-Splat has no hard cap",
    )
    parser.add_argument("--xr-recording", type=Path,
                        help="original MP4 for XR starting pose when using --prepared-dataset")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_directory.expanduser().resolve()
    xr_recording = args.xr_recording or (None if args.prepared_dataset else args.recording)
    if xr_recording is not None:
        xr_recording = xr_recording.expanduser().resolve()
    try:
        if args.sfm_backend == "spirula" and (
            args.prepared_dataset or args.undistort or args.backend != "spirula"
        ):
            raise ValueError("--sfm-backend spirula requires --backend spirula, an MP4 and --no-undistort")
        if args.spirula_sfm and args.sfm_backend != "spirula":
            raise ValueError("--spirula-sfm requires --sfm-backend spirula")
        if args.iterations < 1 or args.iterations > 1_000_000:
            raise ValueError("Iterations must be between 1 and 1000000")
        if not math.isfinite(args.training_image_scale) or not 0 < args.training_image_scale <= 1:
            raise ValueError("Training image scale must be greater than 0 and at most 1")
        if args.backend == "msplat" and not args.undistort and not args.prepared_dataset:
            raise ValueError("M-Splat requires rectified images; omit --no-undistort")
        if args.prepared_dataset and output_dir.is_relative_to(args.recording.expanduser().resolve()):
            raise ValueError("Training output must be outside the prepared dataset")
        populated = output_dir.exists() and any(output_dir.iterdir())
        if populated and not args.resume:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        final_sog = output_dir / "scene.sog"
        if args.backend == "lichtfeld" and args.resume and final_sog.is_file() and final_sog.stat().st_size:
            print(f"Final SOG already exists: {final_sog}")
            sidecar = final_sog.with_suffix(".xrSceneTransform.json")
            if not sidecar.is_file() or not sidecar.stat().st_size:
                dataset = args.recording.expanduser().resolve() if args.prepared_dataset else output_dir / "dataset"
                write_xr_scene_transform(xr_recording, dataset, final_sog)
            return 0
        if args.resume and populated and not args.prepared_dataset and not (output_dir / "dataset").is_dir():
            raise ValueError(f"Cannot resume: dataset checkpoint is missing beneath {output_dir}")
        if args.headless and "QT_QPA_PLATFORM" not in os.environ:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            print("Headless Qt platform: offscreen", flush=True)
        executable = _resolve_executable(getattr(args, args.backend), args.backend)
        converter = (_resolve_executable(args.splat_transform, "splat-transform")
                     if args.splat_transform else None)
        dataset = output_dir / "dataset"
        if args.prepared_dataset:
            dataset = args.recording.expanduser().resolve()
            if not (dataset / "sparse" / "0" / "cameras.bin").is_file():
                raise ValueError(f"Expected COLMAP images/ and sparse/0/*.bin beneath {dataset}")
        elif args.sfm_backend == "spirula":
            from av1_spirula_sfm import prepare_spirula_dataset

            prepare_spirula_dataset(
                args.recording, dataset, executable=args.spirula_sfm or args.spirula,
                fps=args.fps, frames=args.frames, hardware_decode=args.hardware_decode,
                resume=args.resume,
            )
        else:
            prepare_dataset(
                args.recording, dataset,
                fps=args.fps, frames=args.frames, undistort=args.undistort,
                ncams=2, focal_scale=args.ph2kb_focal_scale, colmap=args.colmap,
                colmap_gpu=args.colmap_gpu, colmap_threads=args.colmap_threads,
                refine_focal_length=args.refine_focal_length,
                guided_matching=args.guided_matching,
                hardware_decode=args.hardware_decode, resume=args.resume,
            )
        training_dataset = prepare_training_dataset(
            dataset, output_dir, args.training_image_scale,
        )
        if args.backend == "lichtfeld":
            final_sog = train_and_convert(
                training_dataset, output_dir, lichtfeld=executable, headless=args.headless,
                max_gaussians=args.max_gaussians,
            )
            write_xr_scene_transform(xr_recording, dataset, final_sog)
        else:
            if args.backend == "msplat":
                print("M-Splat has no hard Gaussian cap; check run.json before XR delivery.", flush=True)
            final_scene = train_native(
                training_dataset, output_dir, backend=args.backend, executable=executable,
                iterations=args.iterations, max_gaussians=args.max_gaussians,
                image_scale=args.training_image_scale, resume=args.resume,
                splat_transform=converter,
            )
            if args.backend == "spirula":
                write_xr_scene_transform(xr_recording, dataset, final_scene)
        if training_dataset != dataset:
            shutil.rmtree(training_dataset)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
