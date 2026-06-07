#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logo_removal.mask_providers import STATIC_RECTANGLE, VALID_REMOVAL_MODES
from logo_removal.roi import Roi
from logo_removal.video import require_binary
from logo_removal.void_export import DEFAULT_VOID_PROMPT, VoidExportConfig, export_void_package


DEFAULT_INPUT = PROJECT_ROOT / "eval_clips" / "bottle-detection.mp4"
DEFAULT_PACKAGE_DIR = (
    PROJECT_ROOT / "eval_outputs" / "bottle-detection" / "opencv_sam2" / "void_phase5_package"
)
DEFAULT_OUTPUT_ZIP = (
    PROJECT_ROOT / "eval_outputs" / "bottle-detection" / "opencv_sam2" / "void_phase5_input.zip"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the bottle-detection eval clip as a VOID Phase 5 package.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-zip", type=Path, default=DEFAULT_OUTPUT_ZIP)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--sequence-name", default="bottle_detection_phase5")
    parser.add_argument("--prompt", default=DEFAULT_VOID_PROMPT)
    parser.add_argument(
        "--removal-mode",
        choices=sorted(VALID_REMOVAL_MODES),
        default=STATIC_RECTANGLE,
        help=(
            "Use ai_object for SAM2 propagated masks, or static_rectangle for a quick "
            "ROI-only smoke test."
        ),
    )
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--mask-padding", type=int, default=2)
    parser.add_argument(
        "--void-shadow-dilation-px",
        type=int,
        default=0,
        help="Optional VOID affected-region shell size. 0 keeps the current binary 0/255 mask.",
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

    try:
        if args.void_shadow_dilation_px < 0:
            raise ValueError("--void-shadow-dilation-px must be >= 0")
        require_binary("ffmpeg")
        require_binary("ffprobe")
        config = VoidExportConfig(
            input_path=args.input,
            output_zip_path=args.output_zip,
            package_dir=args.package_dir,
            sequence_name=args.sequence_name,
            prompt=args.prompt,
            roi=Roi(args.x, args.y, args.width, args.height),
            removal_mode=args.removal_mode,
            reference_frame=args.reference_frame,
            mask_padding=args.mask_padding,
            shadow_dilation_px=args.void_shadow_dilation_px,
            overwrite=args.overwrite,
            keep_package_dir=args.keep_package_dir,
        )
        result = export_void_package(config)
    except KeyboardInterrupt:
        logging.error("Cancelled by user.")
        return 130
    except Exception as exc:
        logging.exception("VOID package export failed: %s", exc)
        return 1

    logging.info("VOID package directory: %s", result.sequence_dir)
    logging.info("VOID package zip: %s", result.zip_path)
    logging.info("Frames exported: %s", result.frame_count)
    logging.info("Upload this zip to Colab: %s", result.zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
