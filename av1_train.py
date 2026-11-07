"""Native trainers consuming the same aligned COLMAP dataset."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from av1_colmap import _run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def validate_ply(path: Path) -> int:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY: {path}")
        header = []
        for _ in range(256):
            line = stream.readline().decode("ascii").strip()
            if line == "end_header":
                break
            header.append(line)
        else:
            raise ValueError(f"Missing PLY header terminator: {path}")
        required = {
            "x",
            "y",
            "z",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        }
        properties = [
            line.split()[-1] for line in header if line.startswith("property ")
        ]
        vertices = next(
            (
                int(line.split()[2])
                for line in header
                if line.startswith("element vertex ")
            ),
            0,
        )
        if vertices < 1 or not required.issubset(properties):
            raise ValueError(f"PLY has no Gaussians or lacks 3DGS properties: {path}")
        if "format binary_little_endian 1.0" not in header or any(
            not line.startswith("property float ")
            for line in header
            if line.startswith("property ")
        ):
            raise ValueError(f"Expected binary float 3DGS PLY: {path}")
        if path.stat().st_size - stream.tell() != vertices * len(properties) * 4:
            raise ValueError(f"Truncated or unexpected PLY payload: {path}")
    return vertices


def native_command(
    backend: str,
    executable: str,
    dataset: Path,
    run_dir: Path,
    iterations: int,
    max_gaussians: int,
) -> list[object]:
    if backend == "msplat":
        return [
            executable,
            dataset,
            "-o",
            run_dir / "scene.ply",
            "-n",
            iterations,
            "--num-downscales",
            "0",
            "--keep-crs",
            "--bg-color",
            "0",
            "0",
            "0",
        ]
    if backend == "spirula":
        return [
            executable,
            "train",
            "--data",
            dataset,
            "--data-format",
            "colmap",
            "--output-dir-prefix",
            run_dir,
            "--output-dir-name",
            "training",
            "--num-iterations",
            iterations,
            "--cap-max",
            max_gaussians,
            "--train-resolution-divisor",
            "1",
            "--eval-mode",
            "all",
            "--warp-spherical-to-pinhole",
            "0",
            "--warp-to-pinhole",
            "0",
            "--scene-center",
            "none",
            "--disable-viewer",
            "1",
            "--keep-viewer-alive",
            "0",
        ]
    raise ValueError(f"Unknown native backend: {backend}")


def train_native(
    dataset: Path,
    output: Path,
    *,
    backend: str,
    executable: str,
    iterations: int,
    max_gaussians: int,
    image_scale: float,
    resume: bool,
    splat_transform: str | None,
) -> Path:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(dataset / "sparse" / "0")
    if reconstruction.num_reg_images() < 3 or reconstruction.num_points3D() < 1:
        raise ValueError(
            "Training requires registered cameras and a sparse point cloud"
        )
    if backend == "msplat" and any(
        c.model_name not in ("PINHOLE", "SIMPLE_PINHOLE")
        for c in reconstruction.cameras.values()
    ):
        raise ValueError(
            "M-Splat requires rectified PINHOLE images; omit --no-undistort"
        )

    run_dir = output / backend
    command = native_command(
        backend, executable, dataset, run_dir, iterations, max_gaussians
    )
    identity = {
        "backend": backend,
        "command": [str(v) for v in command],
        "executable_sha256": sha256(Path(executable)),
        "image_scale": image_scale,
        "sparse_sha256": {
            p.name: sha256(p) for p in sorted((dataset / "sparse" / "0").glob("*.bin"))
        },
        "images_sha256": {
            im.name: sha256(dataset / "images" / im.name)
            for im in sorted(reconstruction.images.values(), key=lambda im: im.name)
        },
    }
    if backend == "msplat":
        shader = Path(executable).parent / "default.metallib"
        if shader.is_file():
            identity["shader_sha256"] = sha256(shader)
    manifest_path = run_dir / "run.json"
    final_ply = run_dir / "scene.ply"
    if run_dir.exists():
        if not resume or not manifest_path.is_file():
            raise FileExistsError(f"Use a new output directory or --resume: {run_dir}")
        previous = json.loads(manifest_path.read_text())
        if not isinstance(previous, dict) or previous.get("input") != identity:
            raise ValueError(
                "Training inputs/settings changed; use a new output directory"
            )
        if previous.get("status") == "complete":
            validate_ply(final_ply)
            if sha256(final_ply) != previous.get("ply_sha256"):
                raise ValueError(f"Completed PLY changed: {final_ply}")
            print(f"Reusing completed {backend} training: {final_ply}", flush=True)
        else:
            raise RuntimeError(
                f"Incomplete training retained at {run_dir}; use a new output directory to restart"
            )
    else:
        run_dir.mkdir()
        manifest = {
            "input": identity,
            "status": "running",
            "platform": platform.platform(),
            "gaussian_limit": max_gaussians if backend == "spirula" else None,
        }
        write_manifest(manifest_path, manifest)
        start = time.monotonic()
        try:
            _run(command)
        except RuntimeError as error:
            manifest.update(status="failed", error=str(error))
            write_manifest(manifest_path, manifest)
            raise
        if backend == "spirula":
            candidates = sorted((run_dir / "training").glob("step-*.ckpt/splat.ply"))
            if not candidates:
                raise RuntimeError(
                    f"Spirula produced no checkpoint PLY beneath {run_dir}"
                )
            shutil.copy2(candidates[-1], final_ply)
        count = validate_ply(final_ply)
        manifest.update(
            status="complete",
            training_seconds=time.monotonic() - start,
            gaussian_count=count,
            ply_sha256=sha256(final_ply),
        )
        write_manifest(manifest_path, manifest)
        print(f"{backend}: {count:,} Gaussians in {final_ply}", flush=True)
    if splat_transform:
        sog = run_dir / "scene.sog"
        temporary = run_dir / "scene.pending.sog"
        _run([splat_transform, "--overwrite", final_ply, temporary])
        try:
            with zipfile.ZipFile(temporary) as archive:
                if (
                    "meta.json" not in archive.namelist()
                    or archive.testzip() is not None
                ):
                    raise ValueError(f"Invalid SOG archive: {temporary}")
                metadata = json.loads(archive.read("meta.json"))
                if not isinstance(metadata, dict) or metadata.get(
                    "count"
                ) != validate_ply(final_ply):
                    raise ValueError(
                        f"SOG Gaussian count does not match PLY: {temporary}"
                    )
        except zipfile.BadZipFile as error:
            raise ValueError(f"Invalid SOG archive: {temporary}") from error
        temporary.replace(sog)
        manifest = json.loads(manifest_path.read_text())
        manifest["sog_sha256"] = sha256(sog)
        manifest["conversion_command"] = [
            splat_transform,
            "--overwrite",
            str(final_ply),
            str(temporary),
        ]
        write_manifest(manifest_path, manifest)
        final_sog = output / "scene.sog"
        with tempfile.NamedTemporaryFile(
            dir=output, suffix=".pending.sog", delete=False
        ) as stream:
            delivery = Path(stream.name)
        try:
            shutil.copyfile(sog, delivery)
            delivery.replace(final_sog)
        finally:
            delivery.unlink(missing_ok=True)
        print(f"Final SOG: {final_sog}", flush=True)
        return final_sog
    return final_ply
