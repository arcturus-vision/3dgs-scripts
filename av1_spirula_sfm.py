"""Calibrated Spirula-native SfM; pycolmap is only an interchange/alignment reader."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pycolmap
from av1_colmap import (
    _largest_reconstruction,
    _resolve_executable,
    align_reconstruction,
    extract_selected_frames,
    load_extraction_checkpoint,
    read_av1_metadata,
    select_records,
    write_scene,
)
from av1_train import sha256, write_manifest


def camera_manifest(intrinsics: list[dict]) -> dict:
    cameras = []
    for index, camera in enumerate(intrinsics):
        if camera["model"] != "OPENCV_FISHEYE":
            raise ValueError(
                "Spirula native SfM requires native fisheye extraction (--no-undistort)"
            )
        params = [camera[k] for k in ("fx", "fy", "cx", "cy")] + list(camera["dist"])
        if len(params) != 8 or not np.isfinite(params).all() or min(params[:2]) <= 0:
            raise ValueError("Incomplete OPENCV_FISHEYE calibration")
        cameras.append(
            {"prefix": f"cam{index}", "model": "opencv-fisheye", "params": params}
        )
    return {"image_dir": "images", "camera_mode": "folder", "cameras": cameras}


def validate_calibration(reconstruction, intrinsics: list[dict], poses: dict) -> None:
    """Fail closed if an older binary ignored complete manifest intrinsics."""
    if (
        reconstruction.num_reg_images() < max(3, (len(poses) + 1) // 2)
        or reconstruction.num_points3D() < 1
    ):
        raise RuntimeError(
            "Spirula SfM registered fewer than half the images or produced no sparse points"
        )
    for image in reconstruction.images.values():
        if image.name not in poses:
            raise ValueError(f"Unexpected reconstructed image: {image.name}")
        index = int(Path(image.name).parts[0].removeprefix("cam"))
        expected = intrinsics[index]
        camera = reconstruction.cameras[image.camera_id]
        fixed = [expected["cx"], expected["cy"], *expected["dist"]]
        if (
            camera.model_name != "OPENCV_FISHEYE"
            or (camera.width, camera.height) != (expected["w"], expected["h"])
            or not np.allclose(camera.params[2:], fixed, rtol=0, atol=1e-7)
        ):
            raise RuntimeError(
                "Spirula did not preserve calibrated principal point/distortion. "
                "Use a build with full manifest camera params support; see README."
            )


def run_spirula_sfm(
    output: Path, intrinsics: list[dict], poses: dict, executable: str
) -> Path:
    workspace = output / ".spirula-sfm"
    if workspace.exists():
        raise FileExistsError(
            f"Incomplete SfM retained at {workspace}; use a fresh output directory"
        )
    workspace.mkdir()
    manifest = output / "spirula-manifest.json"
    write_manifest(manifest, camera_manifest(intrinsics))
    command = [
        executable,
        "sfm",
        "auto",
        str(output / "images"),
        "-o",
        str(workspace),
        "--manifest",
        str(manifest),
        "--quality",
        "high",
        "--data-type",
        "video",
        "--pairs",
        "prefilter",
        "--no-refine-extra-params",
        "--no-final-extra-params",
        "--exif-orientation",
        "none",
        "--no-orient",
        "--ba-real",
        "cpu",
        "--ba-real-coarse",
        "cpu",
        "--lang",
        "en",
    ]
    record = {
        "command": command,
        "status": "running",
        "executable_sha256": sha256(Path(executable)),
        "manifest_sha256": sha256(manifest),
    }
    path = output / "spirula-sfm-run.json"
    write_manifest(path, record)
    start = time.monotonic()
    try:
        with (output / "spirula-sfm.log").open("w") as log:
            print(f"Running native Spirula SfM; log: {log.name}", flush=True)
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        record.update(exit_code=result.returncode, seconds=time.monotonic() - start)
        # Code 4 only reports unavailable native metric alignment; we align to tracking below.
        if result.returncode not in (0, 4):
            raise RuntimeError(
                f"Spirula SfM exited {result.returncode}; see {output / 'spirula-sfm.log'}"
            )
        model = _largest_reconstruction(workspace / "sparse")
        reconstruction = pycolmap.Reconstruction(model)
        validate_calibration(reconstruction, intrinsics, poses)
        aligned = align_reconstruction(reconstruction, poses, output)
        record.update(
            status="complete",
            model=str(model),
            registered_images=reconstruction.num_reg_images(),
            sparse_points=reconstruction.num_points3D(),
            unregistered_images=sorted(
                set(poses) - {im.name for im in reconstruction.images.values()}
            ),
        )
        write_manifest(path, record)
        return aligned
    except Exception as error:
        record.update(status="failed", error=str(error))
        write_manifest(path, record)
        raise


def prepare_spirula_dataset(
    recording: Path,
    output: Path,
    *,
    executable: str,
    fps: float,
    frames: int | None,
    hardware_decode: str,
    resume: bool,
) -> Path:
    recording = recording.expanduser().resolve()
    executable = _resolve_executable(executable, "Spirula SfM")
    # Check SfM was compiled in before decoding the recording.
    probe = subprocess.run(
        [executable, "sfm", "auto", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode or "--manifest" not in probe.stdout + probe.stderr:
        raise RuntimeError(
            "Spirula binary lacks native SfM/manifest support; build with SS_BUILD_SFM=ON"
        )
    identity = {
        "sfm_engine": "spirula",
        "recording": str(recording),
        "recording_sha256": sha256(recording),
        "fps": fps,
        "frames": frames,
        "hardware_decode": hardware_decode,
        "executable_sha256": sha256(Path(executable)),
    }
    state_path = output / "spirula-dataset.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not state_path.is_file():
            raise ValueError(
                "Spirula SfM output already exists without a matching resume manifest"
            )
        state = json.loads(state_path.read_text())
        if state["input"] != identity:
            raise ValueError(
                "Spirula SfM inputs/settings changed; use a fresh output directory"
            )
        if state.get("status") != "complete":
            raise RuntimeError(
                "Incomplete native SfM retained for diagnosis; use a fresh output directory"
            )
        for name, expected in state["files_sha256"].items():
            if sha256(output / name) != expected:
                raise ValueError(f"Spirula dataset changed: {name}")
        return output / "sparse/0"
    output.mkdir(parents=True, exist_ok=True)
    state = {"input": identity, "status": "running"}
    write_manifest(state_path, state)
    try:
        records, calibration = read_av1_metadata(recording, ncams=2)
        selected = select_records(records, fps=fps, frames=frames)
        intrinsics = extract_selected_frames(
            recording,
            output,
            selected,
            calibration,
            ncams=2,
            undistort=False,
            hardware_decode=hardware_decode,
        )
        write_scene(output, selected, intrinsics, ncams=2)
        intrinsics, poses = load_extraction_checkpoint(output, ncams=2)
        aligned = run_spirula_sfm(output, intrinsics, poses, executable)
        files = [
            output / "main.aiscene",
            output / "alignment.json",
            output / "spirula-manifest.json",
        ]
        files += sorted((output / "images").rglob("*.jpg")) + sorted(
            aligned.glob("*.bin")
        )
        state.update(
            status="complete",
            files_sha256={str(p.relative_to(output)): sha256(p) for p in files},
        )
        write_manifest(state_path, state)
        return aligned
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_manifest(state_path, state)
        raise
