from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    """Store the basic video properties needed throughout the processing pipeline."""

    width: int
    height: int
    fps: float
    frame_count: int | None
    duration: float | None
    has_audio: bool


def require_binary(name: str) -> None:
    """Fail fast if an external executable such as FFmpeg is unavailable."""

    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found on PATH: {name}")


def probe_video(path: Path) -> VideoMetadata:
    """Read video stream metadata from FFprobe and normalize it for the pipeline."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError("Input file does not contain a video stream")

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    if fps <= 0:
        raise RuntimeError("Could not determine a valid video frame rate")

    frame_count = _optional_int(video_stream.get("nb_frames"))
    duration = _optional_float(video_stream.get("duration"))
    if duration is None:
        duration = _optional_float(payload.get("format", {}).get("duration"))
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        has_audio=has_audio,
    )


def mux_frames_to_video(
    frames_pattern: Path,
    source_video: Path,
    output_path: Path,
    fps: float,
    has_audio: bool,
    overwrite: bool,
) -> None:
    """Assemble processed frames into a high-quality MP4 and copy source audio."""

    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        _format_fps(fps),
        "-i",
        str(frames_pattern),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "1:a:0?", "-c:a", "copy"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "1",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True)


def _parse_fps(raw: str | None) -> float:
    """Convert FFprobe FPS strings like ``30000/1001`` into a float value."""

    if not raw or raw == "0/0":
        return 0.0
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        denominator_float = float(denominator)
        if denominator_float == 0:
            return 0.0
        return float(numerator) / denominator_float
    return float(raw)


def _optional_int(raw: object) -> int | None:
    """Return an integer when FFprobe exposes one, otherwise ``None``."""

    if raw in (None, "", "N/A"):
        return None
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_float(raw: object) -> float | None:
    """Return a float when FFprobe exposes one, otherwise ``None``."""

    if raw in (None, "", "N/A"):
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _format_fps(fps: float) -> str:
    """Format FPS for FFmpeg input flags without unnecessary trailing zeros."""

    rounded = round(fps)
    if abs(fps - rounded) < 0.001:
        return str(rounded)
    return f"{fps:.6f}".rstrip("0").rstrip(".")
