from __future__ import annotations

import logging
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .inpaint_engines import OPENCV_ENGINE, build_inpaint_engine
from .mask_qa import MaskQaTracker
from .mask_providers import STATIC_RECTANGLE, build_mask_provider
from .roi import Roi
from .video import VideoMetadata, mux_frames_to_video, probe_video


@dataclass(frozen=True)
class InpaintConfig:
    """Collect all runtime settings required to process a single video."""

    input_path: Path
    output_path: Path
    roi: Roi | None
    select_roi: bool
    preview_frame: int
    sample_frame_path: Path | None
    chunk_size: int
    method: str
    radius: float
    mask_padding: int
    overwrite: bool
    keep_temp: bool
    reference_frame: int = 0
    removal_mode: str = STATIC_RECTANGLE
    inpaint_engine: str = OPENCV_ENGINE
    progress_callback: Callable[[int, int | None], None] | None = None
    status_callback: Callable[[str], None] | None = None


def process_video(config: InpaintConfig) -> None:
    """Run the full prototype workflow from metadata read to final export."""

    cv2, np, tqdm = _load_runtime_dependencies()
    if config.status_callback:
        config.status_callback("preparing")

    start = time.perf_counter()
    metadata = probe_video(config.input_path)
    logging.info(
        "Input: %sx%s at %.3f fps%s",
        metadata.width,
        metadata.height,
        metadata.fps,
        f", {metadata.frame_count} frames" if metadata.frame_count else "",
    )

    if config.status_callback:
        config.status_callback("sampling")
    selection_frame_index = config.reference_frame if config.reference_frame else config.preview_frame
    preview_frame = _read_frame(cv2, config.input_path, selection_frame_index)
    if config.sample_frame_path:
        config.sample_frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.sample_frame_path), preview_frame)
        logging.info("Saved preview frame: %s", config.sample_frame_path)

    roi = config.roi
    if roi is None and config.select_roi:
        roi = _select_roi(cv2, preview_frame)
    if roi is None:
        raise RuntimeError("No ROI selected")

    roi.validate_inside(metadata.width, metadata.height)
    logging.info("Using ROI: %s", roi.as_csv())

    temp_root = Path(tempfile.mkdtemp(prefix="video-logo-removal-"))
    frames_dir = temp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Temporary frame directory: %s", frames_dir)

    processed_count = 0
    try:
        mask_provider = build_mask_provider(
            removal_mode=config.removal_mode,
            np=np,
            cv2=cv2,
            input_path=config.input_path,
            metadata=metadata,
            roi=roi,
            reference_frame=selection_frame_index,
            mask_padding=config.mask_padding,
        )
        if config.status_callback:
            config.status_callback(
                "segmenting" if config.removal_mode != STATIC_RECTANGLE else "tracking"
            )
        mask_provider.prepare()

        inpaint_engine = build_inpaint_engine(
            inpaint_engine=config.inpaint_engine,
            cv2=cv2,
            method=config.method,
            radius=config.radius,
        )
        if config.status_callback:
            config.status_callback("inpainting")
        inpaint_engine.prepare()

        processed_count = _process_frames(
            np=np,
            cv2=cv2,
            tqdm=tqdm,
            input_path=config.input_path,
            frames_dir=frames_dir,
            mask_provider=mask_provider,
            inpaint_engine=inpaint_engine,
            chunk_size=config.chunk_size,
            metadata=metadata,
            progress_callback=config.progress_callback,
        )
        if processed_count == 0:
            raise RuntimeError("No frames were processed")

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        if config.status_callback:
            config.status_callback("finalizing")
        mux_frames_to_video(
            frames_pattern=frames_dir / "frame_%08d.png",
            source_video=config.input_path,
            output_path=config.output_path,
            fps=metadata.fps,
            has_audio=metadata.has_audio,
            overwrite=config.overwrite,
        )
    finally:
        if config.keep_temp:
            logging.info("Keeping temporary files: %s", temp_root)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    elapsed = time.perf_counter() - start
    chunks = math.ceil(processed_count / config.chunk_size)
    logging.info("Output: %s", config.output_path)
    logging.info(
        "Completed %s frames in %s chunks in %.2f seconds",
        processed_count,
        chunks,
        elapsed,
    )
    if config.status_callback:
        config.status_callback("completed")


def _load_runtime_dependencies():
    """Import optional runtime libraries only when processing actually begins."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        from tqdm import tqdm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python dependency. Run: python3 -m pip install -r requirements.txt"
        ) from exc
    return cv2, np, tqdm


def _read_frame(cv2, input_path: Path, frame_index: int):
    """Read a single frame from the video for preview, ROI selection, or sampling."""

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    try:
        if frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read preview frame at index {frame_index}")
        return frame
    finally:
        capture.release()


def _select_roi(cv2, frame) -> Roi:
    """Open an interactive OpenCV selector and return the chosen logo rectangle."""

    logging.info("Select the logo region, then press ENTER or SPACE. Press C to cancel.")
    x, y, width, height = cv2.selectROI("Select logo region", frame, showCrosshair=True)
    cv2.destroyWindow("Select logo region")
    roi = Roi(int(x), int(y), int(width), int(height))
    if roi.width <= 0 or roi.height <= 0:
        raise RuntimeError("ROI selection was cancelled or empty")
    return roi


def _process_frames(
    np,
    cv2,
    tqdm,
    input_path: Path,
    frames_dir: Path,
    mask_provider,
    inpaint_engine,
    chunk_size: int,
    metadata: VideoMetadata,
    progress_callback,
) -> int:
    """Stream, inpaint, and write frames without retaining decoded chunks in RAM."""

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    total = metadata.frame_count
    frame_index = 0
    chunk_index = 0
    progress = tqdm(total=total, unit="frame", desc="Inpainting")
    mask_qa = MaskQaTracker(np=np)

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            if frame_index % chunk_size == 0:
                chunk_index += 1
                logging.info(
                    "Processing chunk %s starting at frame %s",
                    chunk_index,
                    frame_index + 1,
                )

            frame_index += 1
            mask = mask_provider.mask_for_frame(frame_index, frame)
            for warning in mask_qa.inspect(mask, frame_index):
                logging.warning("Mask QA warning [%s]: %s", warning.code, warning.message)
            processed = inpaint_engine.inpaint(frame, mask)
            frame_path = frames_dir / f"frame_{frame_index:08d}.png"
            if not cv2.imwrite(str(frame_path), processed):
                raise RuntimeError(f"Could not write processed frame: {frame_path}")
            progress.update(1)
            if progress_callback:
                progress_callback(frame_index, total)
    finally:
        progress.close()
        capture.release()

    return frame_index
