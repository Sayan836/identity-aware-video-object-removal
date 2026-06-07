#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logo_removal.video import require_binary


DEFAULT_CHUNKS_DIR = (
    PROJECT_ROOT / "eval_outputs" / "bottle-detection" / "opencv_sam2" / "void_colab_outputs"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "eval_outputs" / "bottle-detection" / "opencv_sam2" / "void_stitched.mp4"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stitch downloaded VOID chunk outputs.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", default="*.mp4")
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
        require_binary("ffmpeg")
        chunk_paths = [
            path
            for path in sorted(args.chunks_dir.glob(args.pattern))
            if path.is_file() and not path.name.endswith("_tuple.mp4")
        ]
        if not chunk_paths:
            raise RuntimeError(f"No chunk videos found in {args.chunks_dir} matching {args.pattern}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
            list_path = Path(list_file.name)
            for path in chunk_paths:
                escaped = str(path.resolve()).replace("'", "'\\''")
                list_file.write(f"file '{escaped}'\n")

        command = [
            "ffmpeg",
            "-y" if args.overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(args.output),
        ]
        logging.info("Stitching %s chunks into %s", len(chunk_paths), args.output)
        subprocess.run(command, check=True)
        list_path.unlink(missing_ok=True)
    except KeyboardInterrupt:
        logging.error("Cancelled by user.")
        return 130
    except Exception as exc:
        logging.exception("VOID chunk stitching failed: %s", exc)
        return 1

    logging.info("Stitched output: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
