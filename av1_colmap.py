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
"""Prepare an aligned COLMAP dataset from an stereo Arcturus Vision Camera MP4."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Iterable

import av
from av.codec.hwaccel import HWAccel, hwdevices_available
import cv2
import numpy as np
import pycolmap


MIN_FRAMES = 100
MAX_FRAMES = 500
DEFAULT_CPU_COLMAP_THREADS = 4
EXTRACTION_FORMAT = "Arcturus Vision Camera extraction v3"
HARDWARE_DECODE_PREFERENCES = {
    "darwin": ("videotoolbox",),
    "win32": ("d3d11va", "d3d12va", "qsv"),
}

# FFmpeg AVColor* enum values used by the AV Cam recordings.
AVCOL_PRI_BT2020 = 9
AVCOL_TRC_ARIB_STD_B67 = 18

# Match Video Player XR's default HLG presentation in av1-apps:
# BT.2100 OOTF gamma 1.2 followed by neutral extended Reinhard.
HLG_SYSTEM_GAMMA = 1.2
EXTENDED_REINHARD_WHITE_SCALE = 5.0
BT2020_TO_BT709 = np.asarray([
    [1.660491, -0.587641, -0.072850],
    [-0.124550, 1.132900, -0.008349],
    [-0.018151, -0.100579, 1.118730],
], dtype=np.float32)
BT709_LUMINANCE = np.asarray([[0.2126, 0.7152, 0.0722]], dtype=np.float32)

# Recorded poses and velocities use the OpenXR/OpenVR world convention
# (X right, Y up, Z backward). The Gaussian scene convention flips Y and Z.
XR_WORLD_TO_GS_WORLD = np.diag([1.0, -1.0, -1.0, 1.0])


def _hlg_to_linear_lut() -> np.ndarray:
    encoded = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    linear = np.empty_like(encoded)
    low = encoded <= 0.5
    linear[low] = encoded[low] * encoded[low] / 3.0
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    linear[~low] = (np.exp((encoded[~low] - c) / a) + b) / 12.0
    return linear


def _linear_to_srgb8_lut() -> np.ndarray:
    linear = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
    encoded = np.empty_like(linear)
    low = linear <= 0.0031308
    encoded[low] = 12.92 * linear[low]
    encoded[~low] = 1.055 * np.power(linear[~low], 1.0 / 2.4) - 0.055
    return np.rint(encoded * 255.0).astype(np.uint8)


HLG_TO_LINEAR = _hlg_to_linear_lut()
LINEAR_TO_SRGB8 = _linear_to_srgb8_lut()


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float]

    def scaled(self, width: int, height: int) -> "Calibration":
        sx, sy = width / self.width, height / self.height
        return Calibration(
            width, height, self.fx * sx, self.fy * sy,
            self.cx * sx, self.cy * sy, self.distortion,
        )


def _run(command: Iterable[object], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [str(value) for value in command]
    print(f"+ {shlex.join(cmd)}", flush=True)
    result = subprocess.run(cmd, text=True, capture_output=capture, check=False)
    if result.returncode:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        raise RuntimeError(
            f"Command exited with status {result.returncode}: {shlex.join(cmd)}{detail}"
        )
    return result


def _resolve_executable(value: str, label: str) -> str:
    if "/" in value or "\\" in value:
        path = Path(value).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"{label} executable not found or not executable: {path}")
        return str(path)
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(
            f"{label} executable '{value}' was not found on PATH; pass its path explicitly."
        )
    return resolved


def _gpmf_records(data: bytes) -> Iterable[tuple[str, bytes]]:
    """Yield the flat records used by the AVmf json/v1 + AVmd envelope."""
    offset = 0
    while offset + 8 <= len(data):
        key = data[offset : offset + 4].decode("ascii", "replace")
        element_size = data[offset + 5]
        repeat = struct.unpack_from(">H", data, offset + 6)[0]
        length = element_size * repeat
        end = offset + 8 + length
        if end > len(data):
            raise ValueError(f"Truncated {key!r} metadata record")
        yield key, data[offset + 8 : end]
        offset = offset + 8 + ((length + 3) & ~3)


def _decode_metadata_packet(packet: av.Packet) -> dict[str, Any]:
    values = dict(_gpmf_records(bytes(packet)))
    metadata_format = values.get("AVmf", b"").rstrip(b"\0")
    if metadata_format != b"json/v1":
        raise ValueError(f"Unsupported AV Cam metadata format: {metadata_format!r}")
    payload = values.get("AVmd")
    if payload is None:
        raise ValueError("AV Cam metadata packet has no AVmd payload")
    return json.loads(payload.rstrip(b"\0"))


def _pose_matrix(value: Any, field: str) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.size == 16:
            matrix = matrix.reshape(4, 4)
        if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)):
            return matrix
    if not isinstance(value, dict):
        raise ValueError(f"Metadata field {field} is not a pose")

    try:
        x = np.asarray(value.get("plus_x", value.get("plusX")), dtype=np.float64)
        z = np.asarray(value.get("plus_z", value.get("plusZ")), dtype=np.float64)
        position = np.asarray(value["position"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Metadata field {field} is missing pose axes") from error
    if x.shape != (3,) or z.shape != (3,) or position.shape != (3,):
        raise ValueError(f"Metadata field {field} has invalid pose dimensions")

    x /= np.linalg.norm(x)
    z -= x * np.dot(x, z)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((x, y, z))
    matrix[:3, 3] = position
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Metadata field {field} contains non-finite values")
    return matrix


def _vector(value: Any, field: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Metadata field {field} is not a finite 3-vector")
    return vector.tolist()


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    timestamp = raw.get("playbackTimestampSec", raw.get("pts"))
    if timestamp is None or not np.isfinite(float(timestamp)):
        raise ValueError("Metadata record has no finite playback timestamp")
    extrinsics = raw.get("extrinsics", {})
    velocity = raw.get("headliveVelocityInWorld", {})
    linear = velocity.get("linearMps", velocity.get("linear"))
    angular = velocity.get("angularRadps", velocity.get("angular"))
    world_from_camera = XR_WORLD_TO_GS_WORLD @ _pose_matrix(
        extrinsics.get("avcam0InWorld"), "extrinsics.avcam0InWorld"
    )
    world_rotation = XR_WORLD_TO_GS_WORLD[:3, :3]
    result: dict[str, Any] = {
        "playbackTimestampSec": float(timestamp),
        "avcam0InWorld": world_from_camera.tolist(),
        "linearVelocityMps": (
            world_rotation @ _vector(linear, "headliveVelocityInWorld.linear")
        ).tolist(),
        "angularVelocityRadps": (
            world_rotation @ _vector(angular, "headliveVelocityInWorld.angular")
        ).tolist(),
    }
    stereo = extrinsics.get("avcam1InAvcam0")
    if stereo is not None:
        result["avcam1InAvcam0"] = _pose_matrix(
            stereo, "extrinsics.avcam1InAvcam0"
        ).tolist()
    return result


def _public_calibrations(raw_cameras: Any, count: int) -> list[Calibration]:
    if not isinstance(raw_cameras, list):
        raise ValueError("Metadata has no cameras array")
    ordered = sorted(raw_cameras, key=lambda camera: int(camera.get("track", 0)))
    if len(ordered) < count:
        raise ValueError(f"Metadata contains {len(ordered)} cameras, but {count} were requested")

    calibrations: list[Calibration] = []
    for camera_index, camera in enumerate(ordered[:count]):
        candidates = camera.get("intrinsics", camera)
        if isinstance(candidates, dict):
            candidates = [candidates]
        public = next(
            (
                intrinsics for intrinsics in candidates
                if str(intrinsics.get("cameraModel", "")).lower() in {"kb", "kb4"}
            ),
            None,
        )
        if public is None:
            raise ValueError(
                f"Camera {camera_index} has no public kb calibration; "
                "the proprietary polar-spline model is intentionally unsupported."
            )
        try:
            calibrations.append(Calibration(
                int(public.get("width", public.get("w"))),
                int(public.get("height", public.get("h"))),
                float(public["fx"]), float(public["fy"]),
                float(public["cx"]), float(public["cy"]),
                tuple(float(public[f"k{i}"]) for i in range(1, 5)),
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Camera {camera_index} has an incomplete kb calibration") from error
    return calibrations


def read_av1_metadata(
    mp4_path: Path, *, ncams: int = 2, limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[Calibration]]:
    records: list[dict[str, Any]] = []
    calibrations: list[Calibration] | None = None
    with av.open(str(mp4_path)) as container:
        videos = list(container.streams.video)
        data_streams = [stream for stream in container.streams if stream.type == "data"]
        gpmd = next((
            stream for stream in data_streams
            if "videoplayerxr" in " ".join(map(str, stream.metadata.values())).lower()
        ), data_streams[0] if len(data_streams) == 1 else None)
        if len(videos) < ncams:
            raise ValueError(f"Expected {ncams} video streams, found {len(videos)}")
        if gpmd is None:
            raise ValueError("No gpmd metadata stream found")
        print(
            f"Detected {len(videos)} video streams and gpmd stream {gpmd.index}.",
            flush=True,
        )
        for packet in container.demux(gpmd):
            if not packet.size:
                continue
            raw = _decode_metadata_packet(packet)
            if calibrations is None:
                calibrations = _public_calibrations(raw.get("cameras"), ncams)
            try:
                records.append(_normalize_record(raw))
            except ValueError as error:
                print(f"Warning: skipping metadata packet: {error}", file=sys.stderr)
            if limit is not None and len(records) >= limit:
                break
    if calibrations is None or not records:
        raise ValueError("No usable AV Cam JSON metadata records were decoded")
    records.sort(key=lambda record: record["playbackTimestampSec"])
    return records, calibrations


def select_records(
    records: list[dict[str, Any]], *, fps: float, frames: int | None,
) -> list[dict[str, Any]]:
    if fps <= 0:
        raise ValueError("--fps must be positive")
    first, last = records[0]["playbackTimestampSec"], records[-1]["playbackTimestampSec"]
    if last <= first:
        raise ValueError("Metadata timestamps do not span a positive duration")
    if frames is not None and not MIN_FRAMES <= frames <= MAX_FRAMES:
        raise ValueError(f"--frames must be between {MIN_FRAMES} and {MAX_FRAMES}")
    count = frames
    if count is None:
        count = max(MIN_FRAMES, min(MAX_FRAMES, int(round((last - first) * fps))))
    if len(records) < count:
        raise ValueError(f"Recording has only {len(records)} metadata instants; need {count}")

    timestamps = np.asarray([record["playbackTimestampSec"] for record in records])
    targets = np.linspace(first, last, count)
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, len(records) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(
        np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets),
        left,
        right,
    )
    if len(np.unique(indices)) != count:
        raise ValueError("Could not select unique evenly spaced capture instants")
    print(f"Selected {count} stereo capture instants ({first:.3f}s–{last:.3f}s).")
    return [records[int(index)] for index in indices]


def _camera_output(
    calibration: Calibration, width: int, height: int,
    *, undistort: bool, focal_scale: float,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    calibration = calibration.scaled(width, height)
    k = np.array([
        [calibration.fx, 0.0, calibration.cx],
        [0.0, calibration.fy, calibration.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    if not undistort:
        return {
            "model": "OPENCV_FISHEYE", "w": width, "h": height,
            "fx": calibration.fx, "fy": calibration.fy,
            "cx": calibration.cx, "cy": calibration.cy,
            "dist": list(calibration.distortion),
        }, None

    output_k = k.copy()
    output_k[0, 0] *= focal_scale
    output_k[1, 1] *= focal_scale
    distortion = np.asarray(calibration.distortion, dtype=np.float64)
    maps = cv2.fisheye.initUndistortRectifyMap(
        k, distortion, np.eye(3), output_k, (width, height), cv2.CV_16SC2,
    )
    return {
        "model": "PINHOLE", "w": width, "h": height,
        "fx": float(output_k[0, 0]), "fy": float(output_k[1, 1]),
        "cx": float(output_k[0, 2]), "cy": float(output_k[1, 2]),
    }, maps


def _nearest_video_frame(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    target_time: float,
) -> tuple[float, av.VideoFrame, int]:
    """Seek backward to a keyframe, then decode the frame nearest target_time."""
    target_pts = int(target_time / float(stream.time_base))
    container.seek(target_pts, stream=stream, backward=True, any_frame=False)
    previous: tuple[float, av.VideoFrame] | None = None
    decoded = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame.pts is None:
                continue
            decoded += 1
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < target_time:
                previous = timestamp, frame
                continue
            if previous is not None and (
                abs(previous[0] - target_time) <= abs(timestamp - target_time)
            ):
                return previous[0], previous[1], decoded
            return timestamp, frame, decoded
    if previous is not None:
        return previous[0], previous[1], decoded
    raise RuntimeError(
        f"Video stream {stream.index} yielded no frame near {target_time:.6f}s"
    )


def _open_video(
    mp4_path: Path, hardware_device: str | None,
) -> av.container.InputContainer:
    options: dict[str, Any] = {}
    if hardware_device is not None:
        options["hwaccel"] = HWAccel(
            hardware_device, allow_software_fallback=False,
        )
    return av.open(str(mp4_path), **options)


def _probe_hardware_decoder(mp4_path: Path, ncams: int, device: str) -> None:
    for camera_index in range(ncams):
        with _open_video(mp4_path, device) as container:
            stream = container.streams.video[camera_index]
            frame = next(container.decode(stream))
            if not stream.codec_context.is_hwaccel:
                raise RuntimeError(f"video stream {stream.index} used software decoding")
            frame.to_ndarray(format="rgb24")


def _resolve_hardware_decoder(mp4_path: Path, ncams: int, requested: str) -> str | None:
    requested = requested.strip().lower()
    if requested == "none":
        print("Hardware video decoding disabled; using software decoding.", flush=True)
        return None

    available = set(hwdevices_available())
    if requested == "auto":
        preferences = HARDWARE_DECODE_PREFERENCES.get(
            sys.platform, ("cuda", "vaapi", "qsv"),
        )
        candidates = [device for device in preferences if device in available]
    else:
        if requested not in available:
            raise ValueError(
                f"Requested hardware decoder {requested!r} is not exposed by PyAV; "
                f"available devices: {', '.join(sorted(available)) or 'none'}"
            )
        candidates = [requested]

    failures: list[str] = []
    for device in candidates:
        try:
            _probe_hardware_decoder(mp4_path, ncams, device)
        except (OSError, RuntimeError, StopIteration, ValueError) as error:
            failures.append(f"{device}: {error}")
            if requested != "auto":
                raise RuntimeError(
                    f"Hardware decoder {device!r} could not decode both video streams: {error}"
                ) from error
            continue
        print(f"Using PyAV {device} hardware decoding for {ncams} video streams.", flush=True)
        return device

    detail = "; ".join(failures) if failures else "no platform backend exposed by PyAV"
    print(
        f"Warning: hardware video decoding is unavailable ({detail}); "
        "using software decoding with AUTO threading.",
        file=sys.stderr, flush=True,
    )
    return None


def _stream_color_processing(
    stream: av.video.stream.VideoStream,
) -> tuple[bool, bool]:
    codec = stream.codec_context
    tone_map_hlg = int(codec.color_trc or 0) == AVCOL_TRC_ARIB_STD_B67
    convert_bt2020 = int(codec.color_primaries or 0) == AVCOL_PRI_BT2020
    return tone_map_hlg, convert_bt2020


def _tone_map_hlg_rgb_to_sdr_bgr(
    encoded_rgb: np.ndarray, *, convert_bt2020: bool,
) -> np.ndarray:
    """Apply the av1-apps HLG presentation defaults and emit 8-bit sRGB BGR."""
    linear = HLG_TO_LINEAR[encoded_rgb]
    if convert_bt2020:
        linear = cv2.transform(linear, BT2020_TO_BT709)

    scene_luminance = cv2.transform(linear, BT709_LUMINANCE).reshape(linear.shape[:2])
    np.maximum(scene_luminance, 0.0, out=scene_luminance)
    gain = np.zeros_like(scene_luminance)
    np.power(
        scene_luminance,
        HLG_SYSTEM_GAMMA - 1.0,
        out=gain,
        where=scene_luminance > 0.0,
    )
    display_luminance = scene_luminance * gain
    gain *= (
        EXTENDED_REINHARD_WHITE_SCALE + display_luminance
    ) / (
        1.0 + EXTENDED_REINHARD_WHITE_SCALE * display_luminance
    )
    linear *= gain[:, :, np.newaxis]
    np.clip(linear, 0.0, 1.0, out=linear)
    linear_u16 = np.rint(linear * 65535.0).astype(np.uint16)
    return np.ascontiguousarray(LINEAR_TO_SRGB8[linear_u16[:, :, ::-1]])


def extract_selected_frames(
    mp4_path: Path,
    output_dir: Path,
    selected: list[dict[str, Any]],
    calibrations: list[Calibration],
    *,
    ncams: int = 2,
    undistort: bool = True,
    focal_scale: float = 0.8,
    hardware_decode: str = "auto",
) -> list[dict[str, Any]]:
    if focal_scale <= 0:
        raise ValueError("--ph2kb-focal-scale must be positive")
    hardware_device = _resolve_hardware_decoder(mp4_path, ncams, hardware_decode)
    images_dir = output_dir / "images"
    for camera_index in range(ncams):
        (images_dir / f"cam{camera_index}").mkdir(parents=True, exist_ok=True)

    target_times = [record["playbackTimestampSec"] for record in selected]
    saved_images = 0
    output_intrinsics: list[dict[str, Any] | None] = [None] * ncams
    maps: list[tuple[np.ndarray, np.ndarray] | None] = [None] * ncams
    total_images = len(selected) * ncams
    decoded_frames = 0
    last_report_at = time.monotonic()
    last_report_percent = -5

    decoder_description = (
        f"{hardware_device} hardware decoding"
        if hardware_device else "software AUTO threading"
    )
    print(
        f"Extracting {total_images} images from {ncams} video streams "
        f"with keyframe seeking and {decoder_description}.",
        flush=True,
    )

    def report_progress(selected_index: int) -> None:
        nonlocal last_report_at, last_report_percent
        percent = min(100, int((selected_index + 1) * 100 / len(selected)))
        now = time.monotonic()
        if percent < 100 and now - last_report_at < 5 and percent < last_report_percent + 5:
            return
        print(
            f"Extraction progress: decoded {percent}% / "
            f"saved {saved_images} of {total_images}",
            flush=True,
        )
        last_report_at = now
        last_report_percent = percent

    def save(
        camera_index: int,
        selected_index: int,
        frame: av.VideoFrame,
        *,
        tone_map_hlg: bool,
        convert_bt2020: bool,
    ) -> None:
        image = frame.to_ndarray(format="rgb24" if tone_map_hlg else "bgr24")
        height, width = image.shape[:2]
        if output_intrinsics[camera_index] is None:
            output_intrinsics[camera_index], maps[camera_index] = _camera_output(
                calibrations[camera_index], width, height,
                undistort=undistort, focal_scale=focal_scale,
            )
        if maps[camera_index] is not None:
            image = cv2.remap(
                image, maps[camera_index][0], maps[camera_index][1],
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
        if tone_map_hlg:
            image = _tone_map_hlg_rgb_to_sdr_bgr(
                image, convert_bt2020=convert_bt2020,
            )
        timestamp = target_times[selected_index]
        path = images_dir / f"cam{camera_index}" / f"{selected_index:04d}-{timestamp:.6f}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write image: {path}")

    with ExitStack() as stack:
        decoders: list[
            tuple[av.container.InputContainer, av.video.stream.VideoStream, bool, bool]
        ] = []
        for camera_index in range(ncams):
            container = stack.enter_context(_open_video(mp4_path, hardware_device))
            video_streams = list(container.streams.video)
            if len(video_streams) < ncams:
                raise ValueError(f"Expected {ncams} video streams, found {len(video_streams)}")
            stream = video_streams[camera_index]
            if hardware_device is None:
                stream.codec_context.thread_type = "AUTO"
            tone_map_hlg, convert_bt2020 = _stream_color_processing(stream)
            if tone_map_hlg:
                source_primaries = "BT.2020" if convert_bt2020 else "source"
                print(
                    f"Video stream {stream.index}: tone-mapping HLG {source_primaries} "
                    "to 8-bit BT.709 sRGB.",
                    flush=True,
                )
            decoders.append((container, stream, tone_map_hlg, convert_bt2020))

        for selected_index, target_time in enumerate(target_times):
            for camera_index, (
                container, stream, tone_map_hlg, convert_bt2020,
            ) in enumerate(decoders):
                _, frame, decoded = _nearest_video_frame(container, stream, target_time)
                decoded_frames += decoded
                save(
                    camera_index,
                    selected_index,
                    frame,
                    tone_map_hlg=tone_map_hlg,
                    convert_bt2020=convert_bt2020,
                )
                saved_images += 1
            report_progress(selected_index)

    print(
        f"Decoded {decoded_frames} frames from keyframe windows and wrote "
        f"{saved_images} images beneath {images_dir}.",
        flush=True,
    )
    return [intrinsics for intrinsics in output_intrinsics if intrinsics is not None]


def write_scene(
    output_dir: Path,
    selected: list[dict[str, Any]],
    intrinsics: list[dict[str, Any]],
    *,
    ncams: int,
) -> dict[str, np.ndarray]:
    scene: dict[str, Any] = {
        "format": EXTRACTION_FORMAT,
        "cameras": [{"intrinsics": value} for value in intrinsics],
        "frames": [],
    }
    recorded_poses: dict[str, np.ndarray] = {}
    for selected_index, record in enumerate(selected):
        timestamp = record["playbackTimestampSec"]
        avcam0_in_world = np.asarray(record["avcam0InWorld"], dtype=np.float64)
        stereo = record.get("avcam1InAvcam0")
        for camera_index in range(ncams):
            world_from_camera = avcam0_in_world
            if camera_index == 1:
                if stereo is None:
                    raise ValueError("Camera-one pose requested but stereo transform is absent")
                world_from_camera = avcam0_in_world @ np.asarray(stereo, dtype=np.float64)
            name = f"cam{camera_index}/{selected_index:04d}-{timestamp:.6f}.jpg"
            recorded_poses[name] = world_from_camera.copy()
            scene["frames"].append({
                "cam": camera_index,
                "path": f"images/{name}",
                "worldFromCam": world_from_camera.tolist(),
                "playbackTimestampSec": timestamp,
                "middleExposureOfMiddleRowTimestamp": timestamp,
                "linearVelocityMps": record["linearVelocityMps"],
                "angularVelocityRadps": record["angularVelocityRadps"],
            })
    path = output_dir / "main.aiscene"
    path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote scene metadata to {path}.")
    return recorded_poses


def load_extraction_checkpoint(
    output_dir: Path, *, ncams: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    scene_path = output_dir / "main.aiscene"
    if not scene_path.is_file():
        raise ValueError(f"Cannot resume: extraction checkpoint is missing: {scene_path}")
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot resume: invalid extraction checkpoint: {scene_path}") from error
    if not isinstance(scene, dict):
        raise ValueError(f"Cannot resume: invalid extraction checkpoint: {scene_path}")
    if scene.get("format") != EXTRACTION_FORMAT:
        raise ValueError(
            f"Cannot resume: {scene_path} predates the current coordinate/color conversion; "
            "use a new output directory"
        )

    cameras = scene.get("cameras")
    frames = scene.get("frames")
    if not isinstance(cameras, list) or len(cameras) != ncams or not isinstance(frames, list):
        raise ValueError(f"Cannot resume: {scene_path} has an unexpected camera/frame layout")
    if any(not isinstance(camera, dict) for camera in cameras):
        raise ValueError(f"Cannot resume: {scene_path} contains an invalid camera")
    intrinsics = [camera.get("intrinsics") for camera in cameras]
    required = {"model", "w", "h", "fx", "fy", "cx", "cy"}
    if any(not isinstance(value, dict) or not required <= value.keys() for value in intrinsics):
        raise ValueError(f"Cannot resume: {scene_path} has incomplete camera intrinsics")

    recorded_poses: dict[str, np.ndarray] = {}
    camera_counts = [0] * ncams
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("path"), str):
            raise ValueError(f"Cannot resume: {scene_path} contains an invalid frame")
        try:
            camera_index = int(frame["cam"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Cannot resume: {scene_path} contains an invalid camera index") from error
        relative = PurePosixPath(frame["path"])
        if (
            camera_index not in range(ncams)
            or relative.is_absolute()
            or len(relative.parts) != 3
            or relative.parts[:2] != ("images", f"cam{camera_index}")
            or ".." in relative.parts
        ):
            raise ValueError(f"Cannot resume: invalid image path {frame['path']!r}")
        image_path = output_dir.joinpath(*relative.parts)
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"Cannot resume: extracted image is missing or empty: {image_path}")
        image_name = PurePosixPath(*relative.parts[1:]).as_posix()
        if image_name in recorded_poses:
            raise ValueError(f"Cannot resume: duplicate extracted image: {image_name}")
        pose = _pose_matrix(frame.get("worldFromCam"), f"frames[{image_name}].worldFromCam")
        recorded_poses[image_name] = pose
        camera_counts[camera_index] += 1

    if len(set(camera_counts)) != 1 or not MIN_FRAMES <= camera_counts[0] <= MAX_FRAMES:
        raise ValueError(
            f"Cannot resume: expected {MIN_FRAMES}–{MAX_FRAMES} complete stereo pairs, "
            f"found per-camera counts {camera_counts}"
        )
    print(
        f"Resuming from extraction checkpoint with {camera_counts[0]} stereo pairs: "
        f"{scene_path}",
        flush=True,
    )
    return intrinsics, recorded_poses


def _colmap_camera(intrinsics: dict[str, Any], camera_id: int) -> pycolmap.Camera:
    params = [
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"],
    ]
    if intrinsics["model"] == "OPENCV_FISHEYE":
        params.extend(intrinsics["dist"])
    return pycolmap.Camera(
        camera_id=camera_id,
        model=intrinsics["model"],
        width=intrinsics["w"],
        height=intrinsics["h"],
        params=np.asarray(params, dtype=np.float64),
        has_prior_focal_length=True,
    )


def force_database_cameras(
    database_path: Path, intrinsics: list[dict[str, Any]], *, ncams: int,
) -> None:
    database = pycolmap.Database.open(database_path)
    try:
        images = database.read_all_images()
        cameras = database.read_all_cameras()
        if len(cameras) != ncams:
            raise RuntimeError(
                f"COLMAP created {len(cameras)} cameras; expected exactly {ncams}"
            )
        ids_by_folder: dict[str, set[int]] = {}
        for image in images:
            folder = PurePosixPath(image.name.replace("\\", "/")).parts[0]
            ids_by_folder.setdefault(folder, set()).add(image.camera_id)
        for camera_index in range(ncams):
            folder = f"cam{camera_index}"
            camera_ids = ids_by_folder.get(folder, set())
            if len(camera_ids) != 1:
                raise RuntimeError(
                    f"COLMAP image folder {folder} uses camera IDs {sorted(camera_ids)}"
                )
            database.update_camera(
                _colmap_camera(intrinsics[camera_index], next(iter(camera_ids)))
            )
    finally:
        database.close()
    print(f"Forced {ncams} calibrated cameras in {database_path}.")


def _option_from_help(help_text: str, preferred: str, fallback: str) -> str:
    if preferred in help_text:
        return preferred
    if fallback in help_text:
        return fallback
    raise RuntimeError(f"Installed COLMAP supports neither {preferred} nor {fallback}")


def _largest_reconstruction(raw_sparse: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in raw_sparse.iterdir():
        if path.is_dir() and ((path / "cameras.bin").exists() or (path / "cameras.txt").exists()):
            candidates.append((pycolmap.Reconstruction(path).num_reg_images(), path))
    if not candidates:
        raise RuntimeError(f"COLMAP mapper produced no reconstruction in {raw_sparse}")
    registered, path = max(candidates, key=lambda item: item[0])
    print(
        f"Selected mapper component {path.name} with "
        f"{registered} registered images."
    )
    return path


def run_colmap(
    output_dir: Path,
    intrinsics: list[dict[str, Any]],
    *,
    ncams: int,
    colmap: str,
    use_gpu: bool | None,
    num_threads: int | None,
    refine_focal_length: bool,
    guided_matching: bool = True,
) -> pycolmap.Reconstruction:
    images_dir = output_dir / "images"
    work_dir = output_dir / ".avcam-work"
    raw_sparse = work_dir / "sparse-raw"
    database_path = work_dir / "database.db"
    image_list = work_dir / "image-list.txt"
    raw_sparse.mkdir(parents=True)
    names = sorted(path.relative_to(images_dir).as_posix() for path in images_dir.rglob("*.jpg"))
    image_list.write_text("\n".join(names) + "\n", encoding="utf-8")

    result = _run([colmap, "feature_extractor", "-h"], capture=True)
    feature_help = f"{result.stdout}\n{result.stderr}"
    if use_gpu is None:
        use_gpu = sys.platform != "darwin" and not any(
            marker in feature_help.lower()
            for marker in ("without cuda", "without gpu support")
        )
        if use_gpu:
            print("COLMAP GPU feature extraction and matching enabled.", flush=True)
        else:
            print(
                "CUDA COLMAP is unavailable on this platform/build; using CPU feature "
                "extraction and matching.",
                flush=True,
            )
    if not use_gpu and num_threads is None:
        num_threads = DEFAULT_CPU_COLMAP_THREADS
    if num_threads is not None:
        print(f"Limiting COLMAP feature workers to {num_threads}.", flush=True)
    image_list_option = _option_from_help(feature_help, "--image_list_path", "--image_list")
    feature_command: list[object] = [
        colmap, "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        image_list_option, image_list,
        "--ImageReader.camera_model", intrinsics[0]["model"],
        "--ImageReader.single_camera_per_folder", "1",
        _option_from_help(
            feature_help, "--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"
        ), "1" if use_gpu else "0",
    ]
    if num_threads is not None:
        feature_command += [
            _option_from_help(
                feature_help,
                "--FeatureExtraction.num_threads",
                "--SiftExtraction.num_threads",
            ),
            num_threads,
        ]
    _run(feature_command)
    force_database_cameras(database_path, intrinsics, ncams=ncams)

    result = _run([colmap, "exhaustive_matcher", "-h"], capture=True)
    matching_help = f"{result.stdout}\n{result.stderr}"
    matching_command: list[object] = [
        colmap, "exhaustive_matcher", "--database_path", database_path,
        _option_from_help(
            matching_help, "--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"
        ), "1" if use_gpu else "0",
    ]
    if "--FeatureMatching.guided_matching" in matching_help:
        matching_command += ["--FeatureMatching.guided_matching", "1" if guided_matching else "0"]
    elif "--SiftMatching.guided_matching" in matching_help:
        matching_command += ["--SiftMatching.guided_matching", "1" if guided_matching else "0"]
    elif guided_matching:
        print("Warning: installed COLMAP has no guided-matching option.", file=sys.stderr)
    if num_threads is not None:
        matching_command += [
            _option_from_help(
                matching_help,
                "--FeatureMatching.num_threads",
                "--SiftMatching.num_threads",
            ),
            num_threads,
        ]
    _run(matching_command)

    refine_focal = "1" if refine_focal_length else "0"
    _run([
        colmap, "mapper",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--output_path", raw_sparse,
        "--Mapper.ba_refine_focal_length", refine_focal,
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
    ])
    selected_path = _largest_reconstruction(raw_sparse)
    _run([
        colmap, "bundle_adjuster",
        "--input_path", selected_path,
        "--output_path", selected_path,
        "--BundleAdjustment.refine_focal_length", refine_focal,
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extra_params", "0",
    ])
    return pycolmap.Reconstruction(selected_path)


def umeyama_similarity(
    source: np.ndarray, target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Alignment points must be matching N×3 arrays")
    if len(source) < 3:
        raise ValueError(f"Need at least three alignment correspondences, found {len(source)}")
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    source_variance = np.sum(source_centered * source_centered) / len(source)
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("COLMAP camera centers are degenerate")
    if np.linalg.matrix_rank(source_centered, tol=1e-8) < 2:
        raise ValueError("COLMAP camera centers are collinear")
    if np.linalg.matrix_rank(target_centered, tol=1e-8) < 2:
        raise ValueError("Recorded camera centers are collinear")

    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.dot(singular_values, sign) / source_variance)
    translation = target.mean(axis=0) - scale * rotation @ source.mean(axis=0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Alignment produced invalid scale {scale}")
    return scale, rotation, translation


def anchor_similarity_to_pose(
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly correct a center-fit similarity so one full camera pose matches."""
    aligned_rotation = rotation @ source_pose[:3, :3]
    aligned_center = scale * rotation @ source_pose[:3, 3] + translation
    correction_rotation = target_pose[:3, :3] @ aligned_rotation.T
    correction_translation = target_pose[:3, 3] - correction_rotation @ aligned_center
    return (
        correction_rotation @ rotation,
        correction_rotation @ translation + correction_translation,
    )


def align_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    recorded_poses: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    registered: dict[str, pycolmap.Image] = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        registered[image.name.replace("\\", "/")] = image
    matches: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, recorded_pose in recorded_poses.items():
        image = registered.get(name)
        if image is None:
            continue
        source_pose = np.asarray(image.cam_from_world().inverse().matrix(), dtype=np.float64)
        matches.append((name, source_pose, recorded_pose))
    source_array = np.asarray([match[1][:3, 3] for match in matches])
    target_array = np.asarray([match[2][:3, 3] for match in matches])
    scale, rotation, translation = umeyama_similarity(source_array, target_array)
    anchor_name, source_pose, recorded_pose = matches[0]
    rotation, translation = anchor_similarity_to_pose(
        scale, rotation, translation, source_pose, recorded_pose,
    )
    predicted = (scale * (rotation @ source_array.T)).T + translation
    errors = np.linalg.norm(predicted - target_array, axis=1)

    reconstruction.transform(pycolmap.Sim3d(np.c_[scale * rotation, translation]))
    aligned_path = output_dir / "sparse" / "0"
    aligned_path.mkdir(parents=True)
    reconstruction.write(aligned_path)
    alignment = {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "orientationAnchor": anchor_name,
        "correspondenceCount": len(matches),
        "rmsError": float(np.sqrt(np.mean(errors * errors))),
        "medianError": float(np.median(errors)),
    }
    (output_dir / "alignment.json").write_text(
        json.dumps(alignment, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Aligned {len(matches)} cameras using orientation anchor {anchor_name}: "
        f"RMS {alignment['rmsError']:.4f} m, "
        f"median {alignment['medianError']:.4f} m."
    )
    return aligned_path


def prepare_dataset(
    mp4_path: Path,
    output_dir: Path,
    *,
    fps: float = 2.0,
    frames: int | None = None,
    undistort: bool = True,
    ncams: int = 2,
    focal_scale: float = 0.8,
    colmap: str = "colmap",
    colmap_gpu: bool | None = None,
    colmap_threads: int | None = None,
    refine_focal_length: bool = False,
    guided_matching: bool = True,
    hardware_decode: str = "auto",
    resume: bool = False,
) -> Path:
    mp4_path = mp4_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if colmap_threads is not None and colmap_threads < 1:
        raise ValueError("COLMAP thread count must be at least 1")
    if not mp4_path.is_file():
        raise FileNotFoundError(f"Input MP4 not found: {mp4_path}")
    populated = output_dir.exists() and any(output_dir.iterdir())
    if populated and not resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if populated:
        intrinsics, recorded_poses = load_extraction_checkpoint(output_dir, ncams=ncams)
        aligned = output_dir / "sparse" / "0"
        if (output_dir / "alignment.json").is_file() and aligned.is_dir():
            try:
                reconstruction = pycolmap.Reconstruction(aligned)
                if reconstruction.num_reg_images() >= 3:
                    print(
                        f"Reusing aligned COLMAP reconstruction with "
                        f"{reconstruction.num_reg_images()} registered images: {aligned}",
                        flush=True,
                    )
                    return aligned
            except (OSError, RuntimeError, ValueError):
                pass
        for path in (output_dir / ".avcam-work", output_dir / "sparse"):
            if path.exists():
                shutil.rmtree(path)
        print("Restarting COLMAP from the validated extraction checkpoint.", flush=True)
    else:
        records, calibrations = read_av1_metadata(mp4_path, ncams=ncams)
        selected = select_records(records, fps=fps, frames=frames)
        intrinsics = extract_selected_frames(
            mp4_path, output_dir, selected, calibrations,
            ncams=ncams, undistort=undistort, focal_scale=focal_scale,
            hardware_decode=hardware_decode,
        )
        recorded_poses = write_scene(output_dir, selected, intrinsics, ncams=ncams)

    colmap = _resolve_executable(colmap, "COLMAP")
    reconstruction = run_colmap(
        output_dir, intrinsics, ncams=ncams, colmap=colmap,
        use_gpu=colmap_gpu, num_threads=colmap_threads,
        refine_focal_length=refine_focal_length,
        guided_matching=guided_matching,
    )
    aligned = align_reconstruction(reconstruction, recorded_poses, output_dir)
    print(
        f"Retained COLMAP database and raw reconstruction beneath "
        f"{output_dir / '.avcam-work'}.",
        flush=True,
    )
    return aligned


def _add_dataset_arguments(
    parser: argparse.ArgumentParser, *, include_ncams: bool,
) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fps", type=float, default=2.0, help="target pair rate (default: 2)")
    selection.add_argument(
        "--frames", type=int, choices=range(MIN_FRAMES, MAX_FRAMES + 1),
        metavar="N", help="exact number of stereo pairs (100–500)",
    )
    parser.add_argument(
        "--undistort", action=argparse.BooleanOptionalAction, default=True,
        help="apply OpenCV KB4 fisheye undistortion (default: enabled)",
    )
    if include_ncams:
        parser.add_argument("--ncams", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--ph2kb-focal-scale", type=float, default=0.8,
        help="undistorted focal-length multiplier (default: 0.8)",
    )
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable (default: PATH)")
    parser.add_argument(
        "--colmap-gpu", action=argparse.BooleanOptionalAction, default=None,
        help="force GPU or CPU COLMAP features (default: detect CUDA support)",
    )
    parser.add_argument(
        "--colmap-threads", type=int,
        help=f"maximum feature workers (default: {DEFAULT_CPU_COLMAP_THREADS} on CPU)",
    )
    parser.add_argument(
        "--guided-matching", action=argparse.BooleanOptionalAction, default=True,
        help="guided COLMAP rematching (default: enabled); disable to reduce CPU time/memory",
    )
    parser.add_argument(
        "--refine-focal-length", action="store_true",
        help="let COLMAP refine focal length during mapping and final bundle adjustment",
    )
    parser.add_argument(
        "--hardware-decode", default="auto", metavar="DEVICE",
        help="PyAV decoder: auto, none, or a device such as cuda (default: auto)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse a validated extraction or aligned-reconstruction checkpoint",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract an AV Cam stereo MP4 and build an aligned COLMAP dataset."
    )
    parser.add_argument("recording", type=Path, help="stereo Arcturus Vision Camera MP4")
    parser.add_argument("output_directory", type=Path, help="new or empty dataset directory")
    _add_dataset_arguments(parser, include_ncams=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        aligned = prepare_dataset(
            args.recording, args.output_directory,
            fps=args.fps, frames=args.frames, undistort=args.undistort,
            ncams=args.ncams, focal_scale=args.ph2kb_focal_scale,
            colmap=args.colmap, colmap_gpu=args.colmap_gpu,
            colmap_threads=args.colmap_threads,
            refine_focal_length=args.refine_focal_length,
            guided_matching=args.guided_matching,
            hardware_decode=args.hardware_decode, resume=args.resume,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Aligned COLMAP dataset ready: {aligned.parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
