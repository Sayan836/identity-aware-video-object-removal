#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2  # type: ignore[import-not-found]
import numpy as np

from logo_removal.mask_providers import CachedMaskProvider, SAM2_BACKEND_GLOBAL
from logo_removal.roi import Roi
from logo_removal.video import probe_video, require_binary
from logo_removal.void_export import (
    QUADMASK_PRIMARY,
    VoidExportConfig,
    export_void_package,
)


def main() -> int:
    """Smoke-test shared full-video mask cache reads across VOID chunks."""

    args = _parse_args()
    require_binary("ffmpeg")
    require_binary("ffprobe")

    if args.input:
        return _run_external_input_test(args)
    return _run_synthetic_test()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate shared full-video mask cache reads across VOID chunks.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional existing video to test. Defaults to an internally generated clip.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to keep test artifacts for --input runs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=45,
        help="Frames per exported VOID chunk for --input runs. Default matches Colab config.",
    )
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--reference-frame", type=int, default=5)
    parser.add_argument("--x", type=int, default=278)
    parser.add_argument("--y", type=int, default=104)
    parser.add_argument("--width", type=int, default=77)
    parser.add_argument("--height", type=int, default=215)
    parser.add_argument(
        "--move-x-per-frame",
        type=int,
        default=0,
        help="Optional synthetic horizontal mask motion for supplied input videos.",
    )
    return parser.parse_args()


def _run_synthetic_test() -> int:
    with tempfile.TemporaryDirectory(prefix="sam2-motion-cache-void-test-") as tmp:
        root = Path(tmp)
        input_path = root / "moving_box.mp4"
        cache_dir = root / "mask_cache" / "synthetic-cache"
        output_dir = root / "void_chunks"
        output_dir.mkdir(parents=True)

        _write_synthetic_video(input_path)
        metadata = probe_video(input_path)
        _write_synthetic_mask_cache(cache_dir, metadata)

        provider = CachedMaskProvider(
            np=np,
            cv2=cv2,
            metadata=metadata,
            cache_dir=cache_dir,
        )
        provider.prepare()

        chunk_results = []
        for chunk_index, start_frame in enumerate([0, 5]):
            result = export_void_package(
                VoidExportConfig(
                    input_path=input_path,
                    output_zip_path=output_dir / f"synthetic_chunk_{chunk_index:03d}.zip",
                    package_dir=output_dir / f"package_{chunk_index:03d}",
                    sequence_name=f"synthetic_chunk_{chunk_index:03d}",
                    roi=Roi(4, 12, 12, 12),
                    removal_mode="ai_object",
                    reference_frame=4,
                    mask_padding=0,
                    start_frame=start_frame,
                    max_frames=5,
                    overwrite=True,
                    keep_package_dir=True,
                    mask_provider=provider,
                )
            )
            chunk_results.append(result)

        _validate_chunk_zip(chunk_results[0].zip_path)
        _validate_chunk_zip(chunk_results[1].zip_path)
        _validate_manifest(chunk_results[1].manifest_path, cache_dir)
        _validate_chunk_two_first_mask(chunk_results[1].quadmask_path)

        kept_root = PROJECT_ROOT / "eval_outputs" / "sam2_motion_cache_void_smoke"
        if kept_root.exists():
            shutil.rmtree(kept_root)
        shutil.copytree(root, kept_root)
        print(json.dumps({"status": "ok", "artifacts": str(kept_root)}, indent=2))
    return 0


def _run_external_input_test(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.chunks < 1:
        raise ValueError("--chunks must be at least 1")

    output_root = args.output_root
    if output_root is None:
        output_root = PROJECT_ROOT / "eval_outputs" / f"{input_path.stem}_motion_cache_void_test"
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    metadata = probe_video(input_path)
    roi = Roi(args.x, args.y, args.width, args.height)
    roi.validate_inside(metadata.width, metadata.height)
    frame_count = metadata.frame_count or 0
    if frame_count <= 0:
        raise RuntimeError("Input video frame count is required for this tester.")

    cache_dir = output_root / "mask_cache" / "tester-cache"
    output_dir = output_root / "void_chunks"
    output_dir.mkdir(parents=True)
    _write_roi_mask_cache(
        cache_dir=cache_dir,
        metadata=metadata,
        roi=roi,
        reference_frame=args.reference_frame,
        move_x_per_frame=args.move_x_per_frame,
        cache_key="tester-cache",
    )
    provider = CachedMaskProvider(
        np=np,
        cv2=cv2,
        metadata=metadata,
        cache_dir=cache_dir,
    )
    provider.prepare()

    chunk_results = []
    max_chunks = min(args.chunks, (frame_count + args.chunk_size - 1) // args.chunk_size)
    for chunk_index in range(max_chunks):
        start_frame = chunk_index * args.chunk_size
        result = export_void_package(
            VoidExportConfig(
                input_path=input_path,
                output_zip_path=output_dir / f"{input_path.stem}_chunk_{chunk_index:03d}.zip",
                package_dir=output_dir / f"package_{chunk_index:03d}",
                sequence_name=f"{input_path.stem}_chunk_{chunk_index:03d}",
                roi=roi,
                removal_mode="ai_object",
                reference_frame=args.reference_frame,
                mask_padding=0,
                start_frame=start_frame,
                max_frames=min(args.chunk_size, frame_count - start_frame),
                overwrite=True,
                keep_package_dir=True,
                mask_provider=provider,
            )
        )
        chunk_results.append(result)
        _validate_chunk_zip(result.zip_path)
        _validate_manifest(result.manifest_path, cache_dir, expected_cache_key="tester-cache")

    validation_index = 1 if len(chunk_results) > 1 else 0
    _validate_chunk_first_mask(
        path=chunk_results[validation_index].quadmask_path,
        roi=roi,
        absolute_frame=chunk_results[validation_index].start_frame,
        width=metadata.width,
        move_x_per_frame=args.move_x_per_frame,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(input_path),
                "artifacts": str(output_root),
                "chunks": len(chunk_results),
                "chunk_size": args.chunk_size,
            },
            indent=2,
        )
    )
    return 0


def _write_synthetic_video(path: Path) -> None:
    width, height, fps, frame_count = 64, 48, 12.0, 12
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open synthetic video writer: {path}")
    try:
        for frame_idx in range(frame_count):
            frame = np.full((height, width, 3), 235, dtype=np.uint8)
            x = 4 + frame_idx * 3
            cv2.rectangle(frame, (x, 12), (x + 11, 23), (20, 80, 220), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()


def _write_synthetic_mask_cache(cache_dir: Path, metadata) -> None:
    mask_dir = cache_dir / "masks"
    forward_dir = cache_dir / "forward"
    backward_dir = cache_dir / "backward"
    mask_dir.mkdir(parents=True)
    forward_dir.mkdir()
    backward_dir.mkdir()
    frame_count = metadata.frame_count or 12

    for frame_idx in range(frame_count):
        mask = np.zeros((metadata.height, metadata.width), dtype=np.uint8)
        x = 4 + frame_idx * 3
        mask[12:24, x : x + 12] = 255
        for directory in (mask_dir, forward_dir if frame_idx >= 4 else backward_dir):
            if not cv2.imwrite(str(directory / f"{frame_idx:06d}.png"), mask):
                raise RuntimeError(f"Could not write synthetic mask {frame_idx}")

    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache_version": 1,
                "cache_key": "synthetic-cache",
                "source_frame_count": frame_count,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "tracking_backend": SAM2_BACKEND_GLOBAL,
                "propagation_mode": "bidirectional",
                "reference_frame": 4,
            },
            indent=2,
        )
    )


def _write_roi_mask_cache(
    cache_dir: Path,
    metadata,
    roi: Roi,
    reference_frame: int,
    move_x_per_frame: int,
    cache_key: str,
) -> None:
    mask_dir = cache_dir / "masks"
    forward_dir = cache_dir / "forward"
    backward_dir = cache_dir / "backward"
    mask_dir.mkdir(parents=True)
    forward_dir.mkdir()
    backward_dir.mkdir()
    frame_count = metadata.frame_count or 0

    for frame_idx in range(frame_count):
        mask = np.zeros((metadata.height, metadata.width), dtype=np.uint8)
        x = _frame_roi_x(roi, frame_idx, metadata.width, move_x_per_frame)
        mask[roi.y : roi.y + roi.height, x : x + roi.width] = 255
        raw_dir = forward_dir if frame_idx >= reference_frame else backward_dir
        for directory in (mask_dir, raw_dir):
            if not cv2.imwrite(str(directory / f"{frame_idx:06d}.png"), mask):
                raise RuntimeError(f"Could not write test mask {frame_idx}")

    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache_version": 1,
                "cache_key": cache_key,
                "source_frame_count": frame_count,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "tracking_backend": SAM2_BACKEND_GLOBAL,
                "propagation_mode": "bidirectional",
                "reference_frame": reference_frame,
            },
            indent=2,
        )
    )


def _validate_chunk_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
    expected = ["input_video.mp4", "manifest.json", "prompt.json", "quadmask_0.mp4"]
    if names != expected:
        raise RuntimeError(f"Unexpected zip contents for {path}: {names}")


def _validate_manifest(
    path: Path,
    cache_dir: Path,
    expected_cache_key: str = "synthetic-cache",
) -> None:
    manifest = json.loads(path.read_text())
    if manifest.get("mask_cache_key") != expected_cache_key:
        raise RuntimeError(f"Chunk manifest did not preserve mask cache key: {manifest}")
    if Path(manifest.get("mask_cache_dir", "")).resolve() != cache_dir.resolve():
        raise RuntimeError(f"Chunk manifest did not preserve mask cache dir: {manifest}")
    if manifest.get("propagation_mode") != "bidirectional":
        raise RuntimeError(f"Chunk manifest did not preserve propagation mode: {manifest}")


def _validate_chunk_two_first_mask(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read quadmask video: {path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    expected_x = 4 + 5 * 3
    primary_region = gray[12:24, expected_x : expected_x + 12]
    if not np.all(primary_region <= QUADMASK_PRIMARY + 4):
        raise RuntimeError("Chunk 1 did not read the absolute frame-5 mask from cache.")


def _validate_chunk_first_mask(
    path: Path,
    roi: Roi,
    absolute_frame: int,
    width: int,
    move_x_per_frame: int,
) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read quadmask video: {path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    x = _frame_roi_x(roi, absolute_frame, width, move_x_per_frame)
    primary_region = gray[roi.y : roi.y + roi.height, x : x + roi.width]
    if not np.all(primary_region <= QUADMASK_PRIMARY + 4):
        raise RuntimeError(
            f"Chunk did not read the expected absolute frame {absolute_frame} mask."
        )


def _frame_roi_x(roi: Roi, frame_idx: int, width: int, move_x_per_frame: int) -> int:
    return max(0, min(width - roi.width, roi.x + frame_idx * move_x_per_frame))


if __name__ == "__main__":
    raise SystemExit(main())
