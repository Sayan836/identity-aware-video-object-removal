from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .inpaint_engines import VALID_INPAINT_ENGINES
from .mask_providers import STATIC_RECTANGLE, VALID_REMOVAL_MODES
from .pipeline import InpaintConfig, process_video
from .roi import Roi, parse_roi
from .video import require_binary


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the prototype tool."""

    parser = argparse.ArgumentParser(
        prog="video-logo-remover",
        description="Remove a selected object or region from a video using a pluggable pipeline.",
    )
    parser.add_argument("input", type=Path, help="Input video path.")
    parser.add_argument("output", type=Path, help="Output MP4 path.")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        help="Target rectangle as x,y,width,height. Skips interactive ROI selection.",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Open an OpenCV window to select the target rectangle from a preview frame.",
    )
    parser.add_argument(
        "--preview-frame",
        type=int,
        default=0,
        help="Frame index used for ROI selection and sample export. Default: 0.",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Frame index used as the AI/object selection reference. Default: 0.",
    )
    parser.add_argument(
        "--sample-frame",
        type=Path,
        help="Optional path to save the preview frame used for selection.",
    )
    parser.add_argument(
        "--removal-mode",
        choices=sorted(VALID_REMOVAL_MODES),
        default=STATIC_RECTANGLE,
        help="Mask generation mode. Default: static_rectangle.",
    )
    parser.add_argument(
        "--inpaint-engine",
        choices=sorted(VALID_INPAINT_ENGINES),
        default="opencv",
        help="Inpainting engine. Default: opencv.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Frames processed per chunk. Default: 30.",
    )
    parser.add_argument(
        "--method",
        choices=("telea", "ns"),
        default="telea",
        help="OpenCV inpainting method. Default: telea.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="OpenCV inpainting radius. Default: 3.0.",
    )
    parser.add_argument(
        "--mask-padding",
        type=int,
        default=0,
        help="Pixels to expand the selected mask on all sides. Default: 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary processed frame PNGs for debugging.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity. Default: INFO.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before any video processing starts."""

    if not args.input.exists():
        raise FileNotFoundError(f"Input video does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite: {args.output}")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.radius <= 0:
        raise ValueError("--radius must be greater than 0")
    if args.mask_padding < 0:
        raise ValueError("--mask-padding cannot be negative")
    if args.preview_frame < 0:
        raise ValueError("--preview-frame cannot be negative")
    if args.reference_frame < 0:
        raise ValueError("--reference-frame cannot be negative")
    if args.roi is None and not args.select_roi:
        raise ValueError("Provide --roi x,y,w,h or use --select-roi for interactive selection.")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, prepare the runtime config, and execute the pipeline."""

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        validate_args(args)
        require_binary("ffmpeg")
        require_binary("ffprobe")
        config = InpaintConfig(
            input_path=args.input,
            output_path=args.output,
            roi=args.roi if isinstance(args.roi, Roi) else None,
            select_roi=args.select_roi,
            preview_frame=args.preview_frame,
            reference_frame=args.reference_frame,
            sample_frame_path=args.sample_frame,
            chunk_size=args.chunk_size,
            removal_mode=args.removal_mode,
            inpaint_engine=args.inpaint_engine,
            method=args.method,
            radius=args.radius,
            mask_padding=args.mask_padding,
            overwrite=args.overwrite,
            keep_temp=args.keep_temp,
        )
        process_video(config)
        return 0
    except KeyboardInterrupt:
        logging.error("Cancelled by user.")
        return 130
    except Exception as exc:
        logging.error("%s", exc)
        return 1
