from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .mask_providers import MaskProvider, STATIC_RECTANGLE, build_mask_provider
from .roi import Roi
from .video import VideoMetadata, _format_fps, probe_video
from .vlm_analysis import (
    DEFAULT_HEURISTIC_CONTACT_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_HORIZONTAL_OFFSET_PX,
    DEFAULT_HEURISTIC_SHADOW_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX,
    VLM_PROVIDER_NONE,
    VlmAnalysisResult,
    build_affected_mask_provider,
    generate_void_prompt,
    validate_vlm_provider,
)

QUADMASK_PRIMARY = 0
QUADMASK_OVERLAP = 63
QUADMASK_AFFECTED = 127
QUADMASK_BACKGROUND = 255

DEFAULT_VOID_PROMPT = "clean natural background after the selected object is removed"


@dataclass(frozen=True)
class VoidExportConfig:
    """Settings for exporting a VOID-compatible local package."""

    input_path: Path
    output_zip_path: Path
    roi: Roi
    sequence_name: str = "phase5_custom"
    prompt: str = DEFAULT_VOID_PROMPT
    package_dir: Path | None = None
    removal_mode: str = STATIC_RECTANGLE
    prompt_strategy: str | None = None
    prompt_notes: tuple[str, ...] = ()
    reference_frame: int = 0
    mask_padding: int = 0
    start_frame: int = 0
    max_frames: int | None = None
    overwrite: bool = False
    keep_package_dir: bool = True
    progress_callback: Callable[[int, int | None], None] | None = None
    mask_provider: MaskProvider | None = None
    affected_mask_provider: MaskProvider | None = None
    vlm_provider: str = VLM_PROVIDER_NONE
    mask_cache_dir: Path | None = None
    sam2_tracking_backend: str | None = None
    sam2_bidirectional: bool | None = None
    force_mask_cache_rebuild: bool = False
    shadow_dilation_px: int = 0
    heuristic_contact_dilation_px: int = DEFAULT_HEURISTIC_CONTACT_DILATION_PX
    heuristic_shadow_dilation_px: int = DEFAULT_HEURISTIC_SHADOW_DILATION_PX
    heuristic_shadow_vertical_offset_px: int = DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX
    heuristic_shadow_horizontal_offset_px: int = DEFAULT_HEURISTIC_SHADOW_HORIZONTAL_OFFSET_PX
    vlm_analysis: VlmAnalysisResult | None = None
    allow_empty_primary_mask: bool = False


@dataclass(frozen=True)
class VoidExportResult:
    """Paths created by the VOID export pipeline."""

    package_dir: Path
    sequence_dir: Path
    zip_path: Path
    input_video_path: Path
    quadmask_path: Path
    prompt_path: Path
    manifest_path: Path
    frame_count: int
    start_frame: int


def export_void_package(config: VoidExportConfig) -> VoidExportResult:
    """Export input video, basic quadmask, prompt, and manifest as a VOID zip."""

    cv2, np, tqdm = _load_runtime_dependencies()
    metadata = probe_video(config.input_path)
    config.roi.validate_inside(metadata.width, metadata.height)
    if config.start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if config.max_frames is not None and config.max_frames < 1:
        raise ValueError("max_frames must be >= 1 when provided")
    if config.shadow_dilation_px < 0:
        raise ValueError("shadow_dilation_px must be >= 0")
    validate_vlm_provider(config.vlm_provider)

    package_dir = _resolve_package_dir(config)
    sequence_dir = package_dir / config.sequence_name
    _prepare_output_paths(package_dir, config.output_zip_path, config.overwrite)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    mask_provider = config.mask_provider or build_mask_provider(
        removal_mode=config.removal_mode,
        np=np,
        cv2=cv2,
        input_path=config.input_path,
        metadata=metadata,
        roi=config.roi,
        reference_frame=config.reference_frame,
        mask_padding=config.mask_padding,
        sam2_tracking_backend=config.sam2_tracking_backend,
        sam2_bidirectional=config.sam2_bidirectional,
        mask_cache_dir=config.mask_cache_dir,
        force_cache_rebuild=config.force_mask_cache_rebuild,
    )
    mask_provider.prepare()
    affected_mask_provider = config.affected_mask_provider
    if affected_mask_provider is None:
        affected_mask_provider = build_affected_mask_provider(
            vlm_provider=config.vlm_provider,
            np=np,
            cv2=cv2,
            metadata=metadata,
            primary_mask_provider=mask_provider,
            removal_mode=config.removal_mode,
            input_path=config.input_path,
            vlm_analysis=config.vlm_analysis,
            contact_dilation_px=config.heuristic_contact_dilation_px,
            shadow_dilation_px=config.heuristic_shadow_dilation_px,
            shadow_vertical_offset_px=config.heuristic_shadow_vertical_offset_px,
            shadow_horizontal_offset_px=config.heuristic_shadow_horizontal_offset_px,
        )
    if affected_mask_provider is not None:
        affected_mask_provider.prepare()

    prompt_text = config.prompt
    prompt_strategy = config.prompt_strategy or "manual"
    prompt_notes = list(config.prompt_notes)
    if config.prompt_strategy is None and config.vlm_provider != VLM_PROVIDER_NONE:
        prompt_result = generate_void_prompt(
            base_prompt=config.prompt,
            removal_mode=config.removal_mode,
            vlm_provider=config.vlm_provider,
            vlm_analysis=getattr(affected_mask_provider, "analysis", config.vlm_analysis),
        )
        prompt_text = prompt_result.prompt
        prompt_strategy = prompt_result.strategy
        prompt_notes = list(prompt_result.notes)

    input_video_path = sequence_dir / "input_video.mp4"
    if config.start_frame == 0 and config.max_frames is None:
        shutil.copy2(config.input_path, input_video_path)
    else:
        write_input_video_segment(
            input_path=config.input_path,
            output_path=input_video_path,
            metadata=metadata,
            cv2=cv2,
            start_frame=config.start_frame,
            max_frames=config.max_frames,
        )

    quadmask_path = sequence_dir / "quadmask_0.mp4"
    frame_count = write_basic_quadmask_video(
        input_path=config.input_path,
        output_path=quadmask_path,
        metadata=metadata,
        mask_provider=mask_provider,
        cv2=cv2,
        np=np,
        tqdm=tqdm,
        start_frame=config.start_frame,
        max_frames=config.max_frames,
        shadow_dilation_px=config.shadow_dilation_px,
        affected_mask_provider=affected_mask_provider,
        progress_callback=config.progress_callback,
    )
    quadmask_stats = validate_quadmask_video(
        quadmask_path=quadmask_path,
        expected_frame_count=frame_count,
        expected_fps=metadata.fps,
        cv2=cv2,
        np=np,
        allow_empty_primary_mask=config.allow_empty_primary_mask,
    )

    prompt_path = sequence_dir / "prompt.json"
    prompt_path.write_text(json.dumps({"bg": prompt_text}, indent=2))

    manifest_path = sequence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sequence_name": config.sequence_name,
                "input_video": "input_video.mp4",
                "quadmask": "quadmask_0.mp4",
                "prompt": "prompt.json",
                "source_input_path": str(config.input_path),
                "removal_mode": config.removal_mode,
                "prompt_strategy": prompt_strategy,
                "prompt_text": prompt_text,
                "prompt_notes": prompt_notes,
                "vlm_provider": config.vlm_provider,
                "affected_region_provider": getattr(
                    affected_mask_provider,
                    "provider_name",
                    None,
                ),
                "affected_region_config": getattr(affected_mask_provider, "config", None)
                if affected_mask_provider is not None
                else None,
                "vlm_analysis": (
                    config.vlm_analysis.to_manifest()
                    if config.vlm_analysis is not None
                    else None
                ),
                "tracking_backend": getattr(mask_provider, "tracking_backend", None),
                "mask_cache_key": getattr(mask_provider, "cache_key", None),
                "mask_cache_dir": str(getattr(mask_provider, "cache_dir", ""))
                if getattr(mask_provider, "cache_dir", None)
                else None,
                "propagation_mode": getattr(mask_provider, "propagation_mode", None),
                "roi": {
                    "x": config.roi.x,
                    "y": config.roi.y,
                    "width": config.roi.width,
                    "height": config.roi.height,
                },
                "reference_frame": config.reference_frame,
                "mask_padding": config.mask_padding,
                "shadow_dilation_px": config.shadow_dilation_px,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "start_frame": config.start_frame,
                "frame_count": frame_count,
                "source_frame_count": metadata.frame_count,
                "quadmask_values": {
                    "0": "primary object to remove",
                    "63": "primary and affected overlap",
                    "127": "affected interaction region",
                    "255": "background to keep",
                },
                "quadmask_stats": quadmask_stats,
            },
            indent=2,
        )
    )

    _write_package_zip(sequence_dir, config.output_zip_path, config.overwrite)

    if not config.keep_package_dir:
        shutil.rmtree(package_dir, ignore_errors=True)

    return VoidExportResult(
        package_dir=package_dir,
        sequence_dir=sequence_dir,
        zip_path=config.output_zip_path,
        input_video_path=input_video_path,
        quadmask_path=quadmask_path,
        prompt_path=prompt_path,
        manifest_path=manifest_path,
        frame_count=frame_count,
        start_frame=config.start_frame,
    )


def write_basic_quadmask_video(
    input_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    mask_provider,
    cv2,
    np,
    tqdm,
    start_frame: int = 0,
    max_frames: int | None = None,
    shadow_dilation_px: int = 0,
    affected_mask_provider: MaskProvider | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> int:
    """Write a VOID quadmask from primary and optional affected-region masks."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for VOID export: {input_path}")

    temp_avi = output_path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(temp_avi),
        cv2.VideoWriter_fourcc(*"FFV1"),
        metadata.fps,
        (metadata.width, metadata.height),
        isColor=False,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open mask video writer: {temp_avi}")

    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written_count = 0
    total = _target_frame_count(metadata, start_frame=start_frame, max_frames=max_frames)
    progress = tqdm(total=total, unit="frame", desc="Exporting VOID quadmask")
    try:
        while True:
            if max_frames is not None and written_count >= max_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            absolute_one_based_frame = start_frame + written_count + 1
            local_mask = mask_provider.mask_for_frame(absolute_one_based_frame, frame)
            if local_mask.shape[:2] != (metadata.height, metadata.width):
                local_mask = cv2.resize(
                    local_mask,
                    (metadata.width, metadata.height),
                    interpolation=cv2.INTER_NEAREST,
                )
            affected_mask = None
            if affected_mask_provider is not None:
                if hasattr(affected_mask_provider, "mask_from_primary_mask"):
                    affected_mask = affected_mask_provider.mask_from_primary_mask(
                        local_mask,
                        frame_index=absolute_one_based_frame,
                        frame=frame,
                    )
                else:
                    affected_mask = affected_mask_provider.mask_for_frame(
                        absolute_one_based_frame,
                        frame,
                    )
                    if affected_mask.shape[:2] != (metadata.height, metadata.width):
                        affected_mask = cv2.resize(
                            affected_mask,
                            (metadata.width, metadata.height),
                            interpolation=cv2.INTER_NEAREST,
                        )

            quadmask = binary_mask_to_basic_quadmask(
                local_mask,
                np=np,
                cv2=cv2,
                shadow_dilation_px=shadow_dilation_px,
                affected_mask=affected_mask,
            )
            writer.write(quadmask)
            written_count += 1
            progress.update(1)
            if progress_callback:
                progress_callback(written_count, total)
    finally:
        progress.close()
        writer.release()
        capture.release()

    _convert_lossless_mask_video(temp_avi, output_path, metadata.fps)
    temp_avi.unlink(missing_ok=True)
    return written_count


def write_input_video_segment(
    input_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    cv2,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> int:
    """Write a frame-exact input-video segment for one VOID chunk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for VOID segment export: {input_path}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    temp_avi = output_path.with_suffix(".input.avi")
    writer = cv2.VideoWriter(
        str(temp_avi),
        cv2.VideoWriter_fourcc(*"FFV1"),
        metadata.fps,
        (metadata.width, metadata.height),
        isColor=True,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open input segment writer: {temp_avi}")

    written_count = 0
    try:
        while True:
            if max_frames is not None and written_count >= max_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            written_count += 1
    finally:
        writer.release()
        capture.release()

    if written_count == 0:
        temp_avi.unlink(missing_ok=True)
        raise RuntimeError(
            f"No frames were written for segment starting at frame {start_frame}."
        )

    _convert_input_video(temp_avi, output_path, metadata.fps)
    temp_avi.unlink(missing_ok=True)
    return written_count


def binary_mask_to_basic_quadmask(
    mask,
    np,
    cv2=None,
    shadow_dilation_px: int = 0,
    affected_mask=None,
):
    """Convert local primary/affected masks into VOID quadmask values."""

    primary = mask > 0
    affected = np.zeros(mask.shape[:2], dtype=bool)
    quadmask = np.full(mask.shape[:2], QUADMASK_BACKGROUND, dtype=np.uint8)
    if shadow_dilation_px > 0:
        if cv2 is None:
            raise ValueError("cv2 is required when shadow_dilation_px > 0")
        kernel_size = shadow_dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(primary.astype(np.uint8), kernel, iterations=1) > 0
        affected |= dilated & ~primary
    if affected_mask is not None:
        if affected_mask.shape[:2] != mask.shape[:2]:
            if cv2 is None:
                raise ValueError("cv2 is required when affected_mask shape does not match")
            affected_mask = cv2.resize(
                affected_mask,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        affected |= affected_mask > 0

    quadmask[affected & ~primary] = QUADMASK_AFFECTED
    quadmask[primary & ~affected] = QUADMASK_PRIMARY
    quadmask[primary & affected] = QUADMASK_OVERLAP
    return quadmask


def validate_quadmask_video(
    quadmask_path: Path,
    expected_frame_count: int,
    expected_fps: float,
    cv2,
    np,
    fps_tolerance: float = 0.05,
    allow_empty_primary_mask: bool = False,
) -> dict[str, object]:
    """Validate a written VOID quadmask and return compact stats for the manifest."""

    capture = cv2.VideoCapture(str(quadmask_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open written quadmask for validation: {quadmask_path}")

    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    primary_pixels = 0
    affected_pixels = 0
    overlap_pixels = 0
    frames_with_primary_pixels = 0
    observed_values: set[int] = set()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            frame_count += 1
            frame_overlap_pixels = int(np.count_nonzero((gray >= 55) & (gray <= 70)))
            frame_primary_pixels = int(np.count_nonzero(gray <= 8)) + frame_overlap_pixels
            primary_pixels += frame_primary_pixels
            if frame_primary_pixels > 0:
                frames_with_primary_pixels += 1
            affected_pixels += (
                int(np.count_nonzero((gray >= 118) & (gray <= 136)))
                + frame_overlap_pixels
            )
            overlap_pixels += frame_overlap_pixels
            observed_values.update(int(value) for value in np.unique(gray))
    finally:
        capture.release()

    if frame_count != expected_frame_count:
        raise RuntimeError(
            f"Quadmask frame count mismatch: expected {expected_frame_count}, got {frame_count}."
        )
    if reported_frame_count and reported_frame_count != expected_frame_count:
        raise RuntimeError(
            "Quadmask reported frame count mismatch: "
            f"expected {expected_frame_count}, got {reported_frame_count}."
        )
    if expected_fps > 0 and abs(reported_fps - expected_fps) > fps_tolerance:
        raise RuntimeError(
            f"Quadmask FPS mismatch: expected {expected_fps:.3f}, got {reported_fps:.3f}."
        )
    if primary_pixels <= 0 and not allow_empty_primary_mask:
        raise RuntimeError("Quadmask validation failed: no primary object pixels were written.")

    return {
        "frame_count": frame_count,
        "reported_frame_count": reported_frame_count,
        "fps": reported_fps,
        "primary_pixels": primary_pixels,
        "frames_with_primary_pixels": frames_with_primary_pixels,
        "empty_primary_mask": primary_pixels <= 0,
        "affected_pixels": affected_pixels,
        "overlap_pixels": overlap_pixels,
        "observed_values": sorted(observed_values)[:32],
    }


def _load_runtime_dependencies():
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        from tqdm import tqdm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python dependency. Run: python3 -m pip install -r requirements.txt"
        ) from exc
    return cv2, np, tqdm


def _resolve_package_dir(config: VoidExportConfig) -> Path:
    if config.package_dir is not None:
        return config.package_dir
    return config.output_zip_path.with_suffix("")


def _prepare_output_paths(package_dir: Path, output_zip_path: Path, overwrite: bool) -> None:
    if package_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Package directory already exists: {package_dir}")
        shutil.rmtree(package_dir)
    if output_zip_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output zip already exists: {output_zip_path}")
        output_zip_path.unlink()
    package_dir.mkdir(parents=True, exist_ok=True)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)


def _write_package_zip(sequence_dir: Path, output_zip_path: Path, overwrite: bool) -> None:
    if output_zip_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output zip already exists: {output_zip_path}")
        output_zip_path.unlink()

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sequence_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(sequence_dir))


def _convert_lossless_mask_video(temp_avi: Path, output_path: Path, fps: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temp_avi),
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv444p",
        "-r",
        _format_fps(fps),
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _convert_input_video(temp_avi: Path, output_path: Path, fps: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temp_avi),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-r",
        _format_fps(fps),
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _target_frame_count(
    metadata: VideoMetadata,
    start_frame: int,
    max_frames: int | None,
) -> int | None:
    if metadata.frame_count is None:
        return max_frames
    remaining = max(metadata.frame_count - start_frame, 0)
    if max_frames is None:
        return remaining
    return min(max_frames, remaining)
