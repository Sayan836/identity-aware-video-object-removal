#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logo_removal.mask_providers import (
    STATIC_RECTANGLE,
    VALID_REMOVAL_MODES,
    VALID_SAM2_TRACKING_BACKENDS,
    precompute_mask_cache,
)
from logo_removal.roi import Roi
from logo_removal.video import probe_video, require_binary
from logo_removal.void_export import DEFAULT_VOID_PROMPT, VoidExportConfig, export_void_package


DEFAULT_INPUT = PROJECT_ROOT / "eval_clips" / "bottle-detection.mp4"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "eval_outputs" / "bottle-detection" / "opencv_sam2" / "void_chunks"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export bottle-detection as multiple frame-bounded VOID packages.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-prefix", default="bottle_detection_phase5")
    parser.add_argument("--prompt", default=DEFAULT_VOID_PROMPT)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=45,
        help=(
            "Frames per VOID chunk. Use CogVideoX-safe values where "
            "((chunk_size - 1) // 4 + 1) is even, for example 45 or 85."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument(
        "--removal-mode",
        choices=sorted(VALID_REMOVAL_MODES),
        default=STATIC_RECTANGLE,
        help="Use static_rectangle for the current ROI smoke/research chunks.",
    )
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--mask-padding", type=int, default=2)
    parser.add_argument(
        "--void-shadow-dilation-px",
        type=int,
        default=0,
        help="Optional VOID affected-region shell size. 0 keeps the current binary 0/255 mask.",
    )
    parser.add_argument(
        "--sam2-tracking-backend",
        choices=sorted(VALID_SAM2_TRACKING_BACKENDS),
        default=None,
        help="SAM2-compatible tracking backend for AI modes. Defaults to env/auto.",
    )
    parser.add_argument(
        "--mask-cache-dir",
        type=Path,
        default=None,
        help="Directory used to store one full-video SAM2 mask timeline for all chunks.",
    )
    parser.add_argument(
        "--sam2-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Propagate SAM2 masks forward and backward from the reference frame.",
    )
    parser.add_argument(
        "--force-mask-cache-rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate the SAM2 mask cache before exporting chunks.",
    )
    parser.add_argument("--x", type=int, default=278)
    parser.add_argument("--y", type=int, default=104)
    parser.add_argument("--width", type=int, default=77)
    parser.add_argument("--height", type=int, default=215)
    parser.add_argument("--keep-package-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.chunk_size < 1:
        logging.error("--chunk-size must be at least 1")
        return 1
    latent_frames = (args.chunk_size - 1) // 4 + 1
    if latent_frames % 2 != 0:
        logging.error(
            "--chunk-size=%s produces %s latent frames, which is odd and can fail "
            "inside CogVideoX patch embedding. Use values like 45 or 85.",
            args.chunk_size,
            latent_frames,
        )
        return 1
    if args.start_frame < 0:
        logging.error("--start-frame must be >= 0")
        return 1
    if args.max_chunks is not None and args.max_chunks < 1:
        logging.error("--max-chunks must be >= 1 when provided")
        return 1
    if args.void_shadow_dilation_px < 0:
        logging.error("--void-shadow-dilation-px must be >= 0")
        return 1

    try:
        require_binary("ffmpeg")
        require_binary("ffprobe")
        metadata = probe_video(args.input)
        if metadata.frame_count is None:
            raise RuntimeError("Cannot chunk this input because ffprobe did not report frame count.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        packages_dir = args.output_dir / "packages"
        packages_dir.mkdir(parents=True, exist_ok=True)

        total_remaining = max(metadata.frame_count - args.start_frame, 0)
        chunk_count = math.ceil(total_remaining / args.chunk_size)
        if args.max_chunks is not None:
            chunk_count = min(chunk_count, args.max_chunks)

        logging.info(
            "Exporting %s chunks from %s total frames at %.3f fps.",
            chunk_count,
            metadata.frame_count,
            metadata.fps,
        )
        logging.info(
            "Each %s-frame chunk is about %.2f seconds.",
            args.chunk_size,
            args.chunk_size / metadata.fps,
        )

        shared_mask_provider = None
        if args.removal_mode != STATIC_RECTANGLE:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]

            mask_cache_dir = args.mask_cache_dir or (args.output_dir / "mask_cache")
            logging.info("Precomputing one full-video mask cache at: %s", mask_cache_dir)
            shared_mask_provider = precompute_mask_cache(
                removal_mode=args.removal_mode,
                np=np,
                cv2=cv2,
                input_path=args.input,
                metadata=metadata,
                roi=Roi(args.x, args.y, args.width, args.height),
                reference_frame=args.reference_frame,
                mask_padding=args.mask_padding,
                mask_cache_dir=mask_cache_dir,
                sam2_tracking_backend=args.sam2_tracking_backend,
                sam2_bidirectional=args.sam2_bidirectional,
                force_cache_rebuild=args.force_mask_cache_rebuild,
            )

        for chunk_index in range(chunk_count):
            start_frame = args.start_frame + chunk_index * args.chunk_size
            remaining = metadata.frame_count - start_frame
            frame_count = min(args.chunk_size, remaining)
            sequence_name = f"{args.sequence_prefix}_chunk_{chunk_index:03d}"
            output_zip = args.output_dir / f"{sequence_name}.zip"
            package_dir = packages_dir / f"package_{chunk_index:03d}"

            logging.info(
                "Exporting chunk %03d: frames %s-%s -> %s",
                chunk_index,
                start_frame,
                start_frame + frame_count - 1,
                output_zip,
            )
            result = export_void_package(
                VoidExportConfig(
                    input_path=args.input,
                    output_zip_path=output_zip,
                    package_dir=package_dir,
                    sequence_name=sequence_name,
                    prompt=args.prompt,
                    roi=Roi(args.x, args.y, args.width, args.height),
                    removal_mode=args.removal_mode,
                    reference_frame=args.reference_frame,
                    mask_padding=args.mask_padding,
                    mask_provider=shared_mask_provider,
                    mask_cache_dir=args.mask_cache_dir,
                    sam2_tracking_backend=args.sam2_tracking_backend,
                    sam2_bidirectional=args.sam2_bidirectional,
                    force_mask_cache_rebuild=False,
                    shadow_dilation_px=args.void_shadow_dilation_px,
                    start_frame=start_frame,
                    max_frames=frame_count,
                    overwrite=args.overwrite,
                    keep_package_dir=args.keep_package_dir,
                )
            )
            logging.info("Wrote %s frames: %s", result.frame_count, result.zip_path)
    except KeyboardInterrupt:
        logging.error("Cancelled by user.")
        return 130
    except Exception as exc:
        logging.exception("VOID chunk export failed: %s", exc)
        return 1

    logging.info("Upload/process the chunk zips from: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
