#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logo_removal.mask_providers import VALID_REMOVAL_MODES, VALID_SAM2_TRACKING_BACKENDS
from logo_removal.roi import Roi
from logo_removal.void_export import DEFAULT_VOID_PROMPT
from logo_removal.void_pipeline import (
    PreparedVoidPipelineConfig,
    VALID_QUALITY_RESTORATION_MODES,
    VoidRuntimeConfig,
    resolve_void_resource_profile,
    run_full_video_void_pipeline,
)
from logo_removal.vlm_analysis import (
    DEFAULT_HEURISTIC_CONTACT_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX,
    VLM_PROVIDER_HEURISTIC,
    VALID_VLM_PROVIDERS,
)


DEFAULT_INPUT = PROJECT_ROOT / "bottle-detection.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_outputs" / "full_void_pipeline"


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        profile = resolve_void_resource_profile(args.resource_profile)
        logging.info(
            "Using VOID profile %s: %s, %s frames, %s steps",
            profile.name,
            profile.sample_size,
            profile.max_video_length,
            profile.num_inference_steps,
        )
        full_result = run_full_video_void_pipeline(
            prepare_config=PreparedVoidPipelineConfig(
                input_path=args.input,
                output_dir=args.output_dir,
                roi=Roi(args.x, args.y, args.width, args.height),
                sequence_prefix=args.sequence_prefix,
                prompt=args.prompt,
                removal_mode=args.removal_mode,
                tracking_backend=args.sam2_tracking_backend,
                reference_frame=args.reference_frame,
                mask_padding=args.mask_padding,
                start_frame=args.start_frame,
                max_chunks=args.max_chunks,
                force_mask_cache_rebuild=args.force_mask_cache_rebuild,
                void_shadow_dilation_px=args.void_shadow_dilation_px,
                vlm_provider=args.vlm_provider,
                heuristic_contact_dilation_px=args.heuristic_contact_dilation_px,
                heuristic_shadow_dilation_px=args.heuristic_shadow_dilation_px,
                heuristic_shadow_vertical_offset_px=args.heuristic_shadow_vertical_offset_px,
                sam2_bidirectional=args.sam2_bidirectional,
                resource_profile=args.resource_profile,
                keep_package_dir=args.keep_package_dir,
                overwrite=args.overwrite,
            ),
            runtime_config=VoidRuntimeConfig(
                void_repo=args.void_repo,
                data_root=args.colab_data_root,
                output_dir=args.colab_output_dir,
                upload_dir=args.colab_upload_dir,
                chunk_outputs_dir=args.colab_chunk_outputs_dir,
                merged_output_path=args.colab_merged_output_path,
                resource_profile=args.resource_profile,
                quality_restoration=args.quality_restoration,
            ),
            run_void=args.run_void,
        )
        prepared = full_result.prepared
        logging.info("Prepared %s chunk zip(s): %s", len(prepared.chunk_zips), prepared.chunk_zip_dir)
        logging.info("Preparation manifest: %s", prepared.manifest_path)

        run_result = full_result.void_run
        if run_result:
            logging.info("VOID run manifest: %s", run_result.run_manifest_path)
            logging.info("VOID merged output: %s", run_result.merged_output_path)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "prepared_manifest": str(prepared.manifest_path),
                    "chunk_zip_dir": str(prepared.chunk_zip_dir),
                    "chunk_zips": [str(path) for path in prepared.chunk_zips],
                    "mask_cache_dir": str(prepared.mask_cache_dir)
                    if prepared.mask_cache_dir
                    else None,
                    "void_run_manifest": str(run_result.run_manifest_path)
                    if run_result
                    else None,
                    "void_merged_output": str(run_result.merged_output_path)
                    if run_result and run_result.merged_output_path
                    else None,
                },
                indent=2,
            )
        )
    except KeyboardInterrupt:
        logging.error("Cancelled by user.")
        return 130
    except Exception as exc:
        logging.exception("Full VOID pipeline failed: %s", exc)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare full-video motion masks, Colab-safe VOID chunks, and optionally run VOID Pass 1.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-prefix", default="motion_void")
    parser.add_argument("--prompt", default=DEFAULT_VOID_PROMPT)
    parser.add_argument("--resource-profile", default="l4_pro_balanced")
    parser.add_argument(
        "--quality-restoration",
        choices=sorted(VALID_QUALITY_RESTORATION_MODES),
        default="ffmpeg_bicubic",
        help=(
            "Post-VOID restoration mode. VEnhancer/Real-ESRGAN require command "
            "templates via VENHANCER_COMMAND_TEMPLATE or REALESRGAN_COMMAND_TEMPLATE."
        ),
    )
    parser.add_argument(
        "--removal-mode",
        choices=sorted(VALID_REMOVAL_MODES),
        default="ai_object",
    )
    parser.add_argument(
        "--sam2-tracking-backend",
        choices=sorted(VALID_SAM2_TRACKING_BACKENDS),
        default="samurai",
    )
    parser.add_argument("--reference-frame", type=int, default=5)
    parser.add_argument("--mask-padding", type=int, default=2)
    parser.add_argument("--void-shadow-dilation-px", type=int, default=0)
    parser.add_argument(
        "--vlm-provider",
        choices=sorted(VALID_VLM_PROVIDERS),
        default=VLM_PROVIDER_HEURISTIC,
        help=(
            "Affected-region conditioning provider. Ollama uses a local vision "
            "model and falls back to heuristic masks if unavailable."
        ),
    )
    parser.add_argument(
        "--heuristic-contact-dilation-px",
        type=int,
        default=DEFAULT_HEURISTIC_CONTACT_DILATION_PX,
    )
    parser.add_argument(
        "--heuristic-shadow-dilation-px",
        type=int,
        default=DEFAULT_HEURISTIC_SHADOW_DILATION_PX,
    )
    parser.add_argument(
        "--heuristic-shadow-vertical-offset-px",
        type=int,
        default=DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX,
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--force-mask-cache-rebuild", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam2-bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x", type=int, default=278)
    parser.add_argument("--y", type=int, default=104)
    parser.add_argument("--width", type=int, default=77)
    parser.add_argument("--height", type=int, default=215)
    parser.add_argument("--keep-package-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-void", action="store_true")
    parser.add_argument("--void-repo", type=Path, default=Path("/content/void-model"))
    parser.add_argument("--colab-data-root", type=Path, default=Path("/content/void_phase5_data"))
    parser.add_argument("--colab-output-dir", type=Path, default=Path("/content/void_phase5_outputs"))
    parser.add_argument("--colab-upload-dir", type=Path, default=Path("/content/void_phase5_upload"))
    parser.add_argument(
        "--colab-chunk-outputs-dir",
        type=Path,
        default=Path("/content/void_phase5_chunk_outputs"),
    )
    parser.add_argument(
        "--colab-merged-output-path",
        type=Path,
        default=Path("/content/void_phase5_merged.mp4"),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
