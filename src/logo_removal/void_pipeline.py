from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .mask_providers import STATIC_RECTANGLE, precompute_mask_cache
from .roi import Roi
from .video import VideoMetadata, _format_fps, probe_video, require_binary
from .void_export import DEFAULT_VOID_PROMPT, VoidExportConfig, export_void_package
from .vlm_analysis import (
    DEFAULT_HEURISTIC_CONTACT_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX,
    VLM_PROVIDER_HEURISTIC,
    VLM_PROVIDER_OLLAMA,
    VALID_VLM_PROVIDERS,
    PromptGenerationResult,
    VlmAnalysisResult,
    analyze_ollama_scene,
    generate_void_prompt,
    validate_vlm_provider,
)


DEFAULT_COLAB_DATA_ROOT = Path("/content/void_phase5_data")
DEFAULT_COLAB_OUTPUT_DIR = Path("/content/void_phase5_outputs")
DEFAULT_COLAB_UPLOAD_DIR = Path("/content/void_phase5_upload")
DEFAULT_COLAB_CHUNK_OUTPUTS_DIR = Path("/content/void_phase5_chunk_outputs")
DEFAULT_COLAB_MERGED_OUTPUT_PATH = Path("/content/void_phase5_merged.mp4")
DEFAULT_COLAB_VOID_REPO = Path("/content/void-model")
QUALITY_RESTORATION_NONE = "none"
QUALITY_RESTORATION_FFMPEG_BICUBIC = "ffmpeg_bicubic"
QUALITY_RESTORATION_VENHANCER = "venhancer"
QUALITY_RESTORATION_REALESRGAN = "realesrgan"
VALID_QUALITY_RESTORATION_MODES = {
    QUALITY_RESTORATION_NONE,
    QUALITY_RESTORATION_FFMPEG_BICUBIC,
    QUALITY_RESTORATION_VENHANCER,
    QUALITY_RESTORATION_REALESRGAN,
}


@dataclass(frozen=True)
class VoidResourceProfile:
    """VOID/CogVideoX runtime settings mirrored from in_colab.ipynb."""

    name: str
    sample_size: str
    max_video_length: int
    temporal_window_size: int
    gpu_memory_mode: str
    num_inference_steps: int
    temporal_multidiffusion_stride: int

    @property
    def latent_temporal_frames(self) -> int:
        return validate_temporal_window_size(self.temporal_window_size)


@dataclass(frozen=True)
class PreparedVoidPipelineConfig:
    """Local package-preparation settings for a full-video VOID run."""

    input_path: Path
    output_dir: Path
    roi: Roi
    sequence_prefix: str = "motion_void"
    prompt: str = DEFAULT_VOID_PROMPT
    removal_mode: str = "ai_object"
    tracking_backend: str = "samurai"
    reference_frame: int = 5
    mask_padding: int = 2
    start_frame: int = 0
    max_chunks: int | None = None
    force_mask_cache_rebuild: bool = True
    void_shadow_dilation_px: int = 0
    vlm_provider: str = VLM_PROVIDER_HEURISTIC
    heuristic_contact_dilation_px: int = DEFAULT_HEURISTIC_CONTACT_DILATION_PX
    heuristic_shadow_dilation_px: int = DEFAULT_HEURISTIC_SHADOW_DILATION_PX
    heuristic_shadow_vertical_offset_px: int = DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX
    sam2_bidirectional: bool = True
    resource_profile: str = "l4_pro_balanced"
    keep_package_dir: bool = True
    overwrite: bool = True
    progress_callback: Callable[[int, int | None], None] | None = None
    status_callback: Callable[[str], None] | None = None
    log_callback: Callable[[str], None] | None = None


@dataclass(frozen=True)
class VoidRuntimeConfig:
    """VOID Pass 1 execution settings mirrored from in_colab.ipynb."""

    void_repo: Path = DEFAULT_COLAB_VOID_REPO
    data_root: Path = DEFAULT_COLAB_DATA_ROOT
    output_dir: Path = DEFAULT_COLAB_OUTPUT_DIR
    upload_dir: Path = DEFAULT_COLAB_UPLOAD_DIR
    chunk_outputs_dir: Path = DEFAULT_COLAB_CHUNK_OUTPUTS_DIR
    merged_output_path: Path = DEFAULT_COLAB_MERGED_OUTPUT_PATH
    resource_profile: str = "l4_pro_balanced"
    fps_fallback: int = 12
    guidance_scale: float = 1.0
    seed: int = 42
    cuda_visible_devices: str = "0"
    force_cuda_device: str = "cuda"
    pytorch_cuda_alloc_conf: str = "expandable_segments:True"
    quality_restoration: str = QUALITY_RESTORATION_FFMPEG_BICUBIC
    duration_tolerance_seconds: float = 0.1


@dataclass(frozen=True)
class PreparedVoidPipelineResult:
    """Outputs from local full-video package preparation."""

    output_dir: Path
    chunk_zip_dir: Path
    package_dir: Path
    mask_cache_dir: Path | None
    chunk_zips: list[Path]
    manifest_path: Path
    resource_profile: VoidResourceProfile


@dataclass(frozen=True)
class VoidRunResult:
    """Outputs from a VOID Pass 1 chunk run."""

    chunk_outputs: list[Path]
    merged_output_path: Path | None
    run_manifest_path: Path


@dataclass(frozen=True)
class FullVoidPipelineResult:
    """Combined result for package preparation and optional VOID execution."""

    prepared: PreparedVoidPipelineResult
    void_run: VoidRunResult | None


def run_full_video_void_pipeline(
    prepare_config: PreparedVoidPipelineConfig,
    runtime_config: VoidRuntimeConfig | None = None,
    run_void: bool = False,
) -> FullVoidPipelineResult:
    """Run the whole video flow: masks, VOID chunks, and optionally VOID Pass 1."""

    if run_void:
        if runtime_config is None:
            runtime_config = VoidRuntimeConfig(
                resource_profile=prepare_config.resource_profile,
            )
        elif runtime_config.resource_profile != prepare_config.resource_profile:
            runtime_config = VoidRuntimeConfig(
                void_repo=runtime_config.void_repo,
                data_root=runtime_config.data_root,
                output_dir=runtime_config.output_dir,
                upload_dir=runtime_config.upload_dir,
                chunk_outputs_dir=runtime_config.chunk_outputs_dir,
                merged_output_path=runtime_config.merged_output_path,
                resource_profile=prepare_config.resource_profile,
                fps_fallback=runtime_config.fps_fallback,
                guidance_scale=runtime_config.guidance_scale,
                seed=runtime_config.seed,
                cuda_visible_devices=runtime_config.cuda_visible_devices,
                force_cuda_device=runtime_config.force_cuda_device,
                pytorch_cuda_alloc_conf=runtime_config.pytorch_cuda_alloc_conf,
                quality_restoration=runtime_config.quality_restoration,
                duration_tolerance_seconds=runtime_config.duration_tolerance_seconds,
            )
        validate_void_runtime_config(runtime_config)

    prepared = prepare_full_video_void_pipeline(prepare_config)
    void_run = None
    if run_void:
        void_run = run_void_pass1_for_chunks(
            chunk_zips=prepared.chunk_zips,
            runtime_config=runtime_config,
            progress_callback=prepare_config.progress_callback,
            status_callback=prepare_config.status_callback,
            log_callback=prepare_config.log_callback,
        )
    elif prepare_config.log_callback:
        prepare_config.log_callback("VOID Pass 1 was skipped; prepared chunk zips are ready.")
    return FullVoidPipelineResult(prepared=prepared, void_run=void_run)


def validate_void_runtime_config(runtime_config: VoidRuntimeConfig) -> None:
    """Fail early when the VOID Pass 1 repo/model files are not ready."""

    if runtime_config.quality_restoration not in VALID_QUALITY_RESTORATION_MODES:
        raise ValueError(
            "quality_restoration must be one of: "
            f"{', '.join(sorted(VALID_QUALITY_RESTORATION_MODES))}"
        )
    if runtime_config.duration_tolerance_seconds <= 0:
        raise ValueError("duration_tolerance_seconds must be greater than 0")

    void_repo = runtime_config.void_repo.expanduser()
    required_paths = [
        ("VOID repo directory", void_repo),
        ("VOID predict script", void_repo / "inference" / "cogvideox_fun" / "predict_v2v.py"),
        ("VOID config", void_repo / "config" / "quadmask_cogvideox.py"),
        ("CogVideoX-Fun model directory", void_repo / "CogVideoX-Fun-V1.5-5b-InP"),
        ("VOID Pass 1 checkpoint", void_repo / "void_pass1.safetensors"),
    ]
    missing = [f"{label}: {path}" for label, path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "VOID runtime is not ready, so Run VOID pass cannot start. Missing:\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\nRun the VOID setup/download cells from in_colab.ipynb first, or set "
            "VOID_REPO_DIR/VOID_REPO to the directory containing the VOID repo, "
            "CogVideoX-Fun-V1.5-5b-InP, and void_pass1.safetensors."
        )


def validate_temporal_window_size(frame_count: int) -> int:
    """Validate CogVideoX latent temporal frame parity, matching the notebook."""

    latent_frames = (frame_count - 1) // 4 + 1
    if latent_frames % 2 != 0:
        raise ValueError(
            f"Invalid temporal window {frame_count}: latent frame count is {latent_frames}, "
            "but CogVideoX patch embedding expects an even latent count. "
            "Use values like 13, 21, 29, 37, 45, 53, 61, 69, 77, or 85."
        )
    return latent_frames


def resolve_void_resource_profile(name: str) -> VoidResourceProfile:
    """Return the exact profile constants used by in_colab.ipynb."""

    if name in {"t4_highram_safe", "free_colab_emergency"}:
        return VoidResourceProfile(name, "192x320", 45, 45, "sequential_cpu_offload", 8, 16)
    if name in {"t4_highram_qfloat8", "free_colab_smoke"}:
        return VoidResourceProfile(
            name, "192x320", 45, 45, "model_cpu_offload_and_qfloat8", 8, 16
        )
    if name == "t4_highram_long_lowres":
        return VoidResourceProfile(name, "192x320", 85, 85, "sequential_cpu_offload", 8, 16)
    if name in {"l4_pro_balanced", "free_colab_balanced"}:
        return VoidResourceProfile(
            name, "256x448", 45, 45, "model_cpu_offload_and_qfloat8", 20, 16
        )
    if name == "l4_pro_long_lowres":
        return VoidResourceProfile(
            name, "192x320", 85, 85, "model_cpu_offload_and_qfloat8", 12, 16
        )
    if name == "a100_quality":
        return VoidResourceProfile(
            name, "384x672", 85, 85, "model_cpu_offload_and_qfloat8", 30, 16
        )
    raise ValueError(f"Unknown RESOURCE_PROFILE: {name}")


def prepare_full_video_void_pipeline(
    config: PreparedVoidPipelineConfig,
) -> PreparedVoidPipelineResult:
    """Prepare full-video SAM2/SAMURAI masks and Colab-safe VOID chunk zips."""

    require_binary("ffmpeg")
    require_binary("ffprobe")
    cv2, np = _load_cv_runtime()
    profile = resolve_void_resource_profile(config.resource_profile)
    chunk_size = profile.max_video_length
    metadata = probe_video(config.input_path)
    if metadata.frame_count is None:
        raise RuntimeError("Input video frame count is required for VOID chunk export.")
    _validate_prepare_config(config, metadata, chunk_size)
    _emit_status(config.status_callback, "object_detection")
    _emit_log(
        config.log_callback,
        "Object detection/selection: using marked box "
        f"{config.roi.x},{config.roi.y},{config.roi.width},{config.roi.height} "
        f"on reference frame {config.reference_frame}.",
    )
    _emit_log(
        config.log_callback,
        "VOID profile: "
        f"{profile.name}, sample_size={profile.sample_size}, "
        f"max_video_length={profile.max_video_length}, "
        f"temporal_window_size={profile.temporal_window_size}, "
        f"gpu_memory_mode={profile.gpu_memory_mode}, "
        f"steps={profile.num_inference_steps}, "
        f"stride={profile.temporal_multidiffusion_stride}.",
    )

    output_dir = config.output_dir.expanduser().resolve()
    chunk_zip_dir = output_dir / "void_chunks"
    packages_dir = chunk_zip_dir / "packages"
    mask_cache_root = chunk_zip_dir / "mask_cache"
    if config.overwrite and chunk_zip_dir.exists():
        shutil.rmtree(chunk_zip_dir)
    chunk_zip_dir.mkdir(parents=True, exist_ok=True)
    packages_dir.mkdir(parents=True, exist_ok=True)

    shared_mask_provider = None
    if config.removal_mode != STATIC_RECTANGLE:
        _emit_status(config.status_callback, "masking")
        _emit_log(
            config.log_callback,
            "Masking: precomputing full-video motion-aware masks "
            f"with backend={config.tracking_backend}, bidirectional={config.sam2_bidirectional}.",
        )
        shared_mask_provider = precompute_mask_cache(
            removal_mode=config.removal_mode,
            np=np,
            cv2=cv2,
            input_path=config.input_path,
            metadata=metadata,
            roi=config.roi,
            reference_frame=config.reference_frame,
            mask_padding=config.mask_padding,
            mask_cache_dir=mask_cache_root,
            sam2_tracking_backend=config.tracking_backend,
            sam2_bidirectional=config.sam2_bidirectional,
            force_cache_rebuild=config.force_mask_cache_rebuild,
        )
        _emit_log(
            config.log_callback,
            f"Masking: reusable mask cache ready at {shared_mask_provider.cache_dir}.",
        )
    else:
        _emit_status(config.status_callback, "masking")
        _emit_log(config.log_callback, "Masking: using static rectangle masks from the marked box.")

    vlm_analysis = _run_scene_analysis_if_needed(
        config=config,
        metadata=metadata,
        mask_provider=shared_mask_provider,
        np=np,
        cv2=cv2,
    )
    prompt_result = generate_void_prompt(
        base_prompt=config.prompt,
        removal_mode=config.removal_mode,
        vlm_provider=config.vlm_provider,
        vlm_analysis=vlm_analysis,
    )
    _emit_log(
        config.log_callback,
        "Scene conditioning: "
        f"vlm_provider={config.vlm_provider}, prompt_strategy={prompt_result.strategy}.",
    )
    if config.vlm_provider != "none":
        provider_label = (
            "Ollama-guided affected regions with heuristic fallback"
            if config.vlm_provider == VLM_PROVIDER_OLLAMA
            else "heuristic contact/shadow settings"
        )
        _emit_log(
            config.log_callback,
            f"Affected regions: {provider_label} "
            f"contact={config.heuristic_contact_dilation_px}px, "
            f"shadow={config.heuristic_shadow_dilation_px}px, "
            f"vertical_offset={config.heuristic_shadow_vertical_offset_px}px.",
        )

    chunk_zips: list[Path] = []
    total_remaining = max(metadata.frame_count - config.start_frame, 0)
    chunk_count = math.ceil(total_remaining / chunk_size)
    if config.max_chunks is not None:
        chunk_count = min(chunk_count, config.max_chunks)
    total_export_frames = min(total_remaining, chunk_count * chunk_size)
    _emit_status(config.status_callback, "chunking")
    _emit_log(
        config.log_callback,
        f"Chunking: exporting {chunk_count} VOID package chunk(s), {chunk_size} frames per chunk.",
    )

    for chunk_index in range(chunk_count):
        start_frame = config.start_frame + chunk_index * chunk_size
        remaining = metadata.frame_count - start_frame
        frame_count = min(chunk_size, remaining)
        sequence_name = f"{config.sequence_prefix}_chunk_{chunk_index:03d}"
        output_zip = chunk_zip_dir / f"{sequence_name}.zip"
        _emit_log(
            config.log_callback,
            f"Chunking: writing chunk {chunk_index + 1}/{chunk_count} "
            f"from frame {start_frame} with {frame_count} frame(s).",
        )
        result = export_void_package(
            VoidExportConfig(
                input_path=config.input_path,
                output_zip_path=output_zip,
                package_dir=packages_dir / f"package_{chunk_index:03d}",
                sequence_name=sequence_name,
                prompt=prompt_result.prompt,
                prompt_strategy=prompt_result.strategy,
                prompt_notes=prompt_result.notes,
                roi=config.roi,
                removal_mode=config.removal_mode,
                reference_frame=config.reference_frame,
                mask_padding=config.mask_padding,
                mask_provider=shared_mask_provider,
                mask_cache_dir=mask_cache_root,
                sam2_tracking_backend=config.tracking_backend,
                sam2_bidirectional=config.sam2_bidirectional,
                force_mask_cache_rebuild=False,
                shadow_dilation_px=config.void_shadow_dilation_px,
                vlm_provider=config.vlm_provider,
                heuristic_contact_dilation_px=config.heuristic_contact_dilation_px,
                heuristic_shadow_dilation_px=config.heuristic_shadow_dilation_px,
                heuristic_shadow_vertical_offset_px=config.heuristic_shadow_vertical_offset_px,
                vlm_analysis=vlm_analysis,
                start_frame=start_frame,
                max_frames=frame_count,
                overwrite=config.overwrite,
                keep_package_dir=config.keep_package_dir,
                progress_callback=_offset_progress_callback(
                    config.progress_callback,
                    offset=start_frame - config.start_frame,
                    total=total_export_frames,
                ),
                allow_empty_primary_mask=True,
            )
        )
        chunk_zips.append(result.zip_path)
        chunk_manifest = json.loads(result.manifest_path.read_text())
        if _manifest_has_empty_primary_mask(chunk_manifest):
            _emit_log(
                config.log_callback,
                f"Chunking: chunk {chunk_index + 1}/{chunk_count} has no primary "
                "object pixels; it will be passed through unchanged during VOID execution.",
            )
        _emit_log(config.log_callback, f"Chunking: created {result.zip_path}.")

    manifest_path = output_dir / "void_full_pipeline_manifest.json"
    manifest = _build_prepare_manifest(
        config=config,
        metadata=metadata,
        profile=profile,
        chunk_zips=chunk_zips,
        mask_cache_root=mask_cache_root if shared_mask_provider else None,
        prompt_result=prompt_result,
        vlm_analysis=vlm_analysis,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _emit_log(config.log_callback, f"Chunking: preparation manifest written to {manifest_path}.")

    return PreparedVoidPipelineResult(
        output_dir=output_dir,
        chunk_zip_dir=chunk_zip_dir,
        package_dir=packages_dir,
        mask_cache_dir=mask_cache_root if shared_mask_provider else None,
        chunk_zips=chunk_zips,
        manifest_path=manifest_path,
        resource_profile=profile,
    )


def build_void_pass1_command(
    save_path: Path,
    data_root: Path,
    sequence_name: str,
    runtime_config: VoidRuntimeConfig,
    package_fps: float,
) -> list[str]:
    """Build the same VOID Pass 1 command used in the Colab notebook."""

    profile = resolve_void_resource_profile(runtime_config.resource_profile)
    run_fps = round(package_fps) if package_fps > 0 else runtime_config.fps_fallback
    return [
        sys.executable,
        "-u",
        "inference/cogvideox_fun/predict_v2v.py",
        "--config",
        "config/quadmask_cogvideox.py",
        f"--config.data.data_rootdir={data_root}",
        f"--config.experiment.run_seqs={sequence_name}",
        f"--config.experiment.save_path={save_path}",
        "--config.experiment.skip_if_exists=False",
        f"--config.video_model.model_name={runtime_config.void_repo / 'CogVideoX-Fun-V1.5-5b-InP'}",
        f"--config.video_model.transformer_path={runtime_config.void_repo / 'void_pass1.safetensors'}",
        f"--config.data.sample_size={profile.sample_size}",
        f"--config.data.max_video_length={profile.max_video_length}",
        f"--config.data.fps={run_fps}",
        f"--config.video_model.temporal_window_size={profile.temporal_window_size}",
        f"--config.video_model.temproal_multidiffusion_stride={profile.temporal_multidiffusion_stride}",
        f"--config.video_model.num_inference_steps={profile.num_inference_steps}",
        f"--config.video_model.guidance_scale={runtime_config.guidance_scale}",
        f"--config.system.gpu_memory_mode={profile.gpu_memory_mode}",
        f"--config.system.device={runtime_config.force_cuda_device}",
        "--config.system.ulysses_degree=1",
        "--config.system.ring_degree=1",
        f"--config.system.seed={runtime_config.seed}",
    ]


def run_void_pass1_for_chunks(
    chunk_zips: Iterable[Path],
    runtime_config: VoidRuntimeConfig,
    progress_callback: Callable[[int, int | None], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> VoidRunResult:
    """Run VOID Pass 1 over prepared chunk zips using the notebook configuration."""

    require_binary("ffmpeg")
    validate_void_runtime_config(runtime_config)
    _emit_status(status_callback, "void_pass")
    _emit_log(log_callback, "VOID Pass 1: starting chunk execution with notebook-matched config.")
    os.environ["CUDA_VISIBLE_DEVICES"] = runtime_config.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = runtime_config.pytorch_cuda_alloc_conf
    runtime_config.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config.chunk_outputs_dir.mkdir(parents=True, exist_ok=True)

    sorted_chunk_zips = sorted(Path(path) for path in chunk_zips)
    chunk_outputs: list[Path] = []
    chunk_frame_counts: list[int] = []
    output_fps: float | None = None
    output_width: int | None = None
    output_height: int | None = None
    for chunk_index, chunk_zip in enumerate(sorted_chunk_zips):
        _emit_log(
            log_callback,
            f"VOID Pass 1: preparing chunk {chunk_index + 1}/{len(sorted_chunk_zips)} from {chunk_zip}.",
        )
        data_root, sequence_name, manifest = prepare_void_zip_for_runtime(
            zip_path=chunk_zip,
            data_root=runtime_config.data_root,
            upload_dir=runtime_config.upload_dir,
            sequence_name="phase5_custom",
        )
        save_path = runtime_config.chunk_outputs_dir / f"chunk_{chunk_index:03d}"
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        manifest_frame_count = int(manifest.get("frame_count") or 0)
        manifest_fps = float(manifest.get("fps") or runtime_config.fps_fallback)
        manifest_width = int(manifest.get("width") or 0)
        manifest_height = int(manifest.get("height") or 0)
        if output_fps is None and manifest_fps > 0:
            output_fps = manifest_fps
        if output_width is None and manifest_width > 0:
            output_width = manifest_width
        if output_height is None and manifest_height > 0:
            output_height = manifest_height
        if _manifest_has_empty_primary_mask(manifest):
            stable_output = runtime_config.chunk_outputs_dir / f"void_chunk_{chunk_index:03d}.mp4"
            original_segment = data_root / sequence_name / "input_video.mp4"
            save_stable_chunk_output(
                source_path=original_segment,
                target_path=stable_output,
                frame_count=manifest_frame_count,
                fps=manifest_fps,
                width=manifest_width,
                height=manifest_height,
                duration_tolerance_seconds=runtime_config.duration_tolerance_seconds,
            )
            chunk_outputs.append(stable_output)
            chunk_frame_counts.append(manifest_frame_count)
            _emit_progress(progress_callback, chunk_index + 1, len(sorted_chunk_zips))
            _emit_log(
                log_callback,
                "VOID Pass 1: chunk has no primary object pixels, "
                f"copied original segment to {stable_output}.",
            )
            continue
        command = build_void_pass1_command(
            save_path=save_path,
            data_root=data_root,
            sequence_name=sequence_name,
            runtime_config=runtime_config,
            package_fps=float(manifest.get("fps") or runtime_config.fps_fallback),
        )
        _emit_log(log_callback, "VOID Pass 1 command: " + " ".join(command))
        _run_logged_subprocess(command, cwd=runtime_config.void_repo, log_callback=log_callback)
        main_output = find_main_void_output(save_path)
        stable_output = runtime_config.chunk_outputs_dir / f"void_chunk_{chunk_index:03d}.mp4"
        save_stable_chunk_output(
            source_path=main_output,
            target_path=stable_output,
            frame_count=manifest_frame_count,
            fps=manifest_fps,
            width=manifest_width,
            height=manifest_height,
            duration_tolerance_seconds=runtime_config.duration_tolerance_seconds,
        )
        chunk_outputs.append(stable_output)
        chunk_frame_counts.append(manifest_frame_count)
        _emit_progress(progress_callback, chunk_index + 1, len(sorted_chunk_zips))
        _emit_log(log_callback, f"VOID Pass 1: stable chunk output ready at {stable_output}.")

    merged_output = None
    if len(chunk_outputs) > 1:
        _emit_status(status_callback, "merging")
        _emit_log(log_callback, f"Merging: stitching {len(chunk_outputs)} VOID chunk outputs.")
        merged_output = stitch_void_chunk_outputs(
            chunk_outputs=chunk_outputs,
            output_path=runtime_config.merged_output_path,
            fps=output_fps or runtime_config.fps_fallback,
            width=output_width or 0,
            height=output_height or 0,
            frame_count=sum(count for count in chunk_frame_counts if count > 0),
            duration_tolerance_seconds=runtime_config.duration_tolerance_seconds,
        )
    elif chunk_outputs:
        merged_output = chunk_outputs[0]
        validate_timed_video_output(
            merged_output,
            fps=output_fps or runtime_config.fps_fallback,
            width=output_width or 0,
            height=output_height or 0,
            frame_count=sum(count for count in chunk_frame_counts if count > 0),
            duration_tolerance_seconds=runtime_config.duration_tolerance_seconds,
            label="single VOID chunk output",
        )
    raw_merged_output = merged_output
    if merged_output:
        merged_output = post_enhance_void_output(
            input_path=merged_output,
            quality_restoration=runtime_config.quality_restoration,
            fps=output_fps or runtime_config.fps_fallback,
            width=output_width or 0,
            height=output_height or 0,
            frame_count=sum(count for count in chunk_frame_counts if count > 0),
            duration_tolerance_seconds=runtime_config.duration_tolerance_seconds,
            log_callback=log_callback,
        )
    if merged_output:
        _emit_log(log_callback, f"Final output ready at {merged_output}.")

    run_manifest_path = runtime_config.chunk_outputs_dir / "void_run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "resource_profile": _profile_to_dict(
                    resolve_void_resource_profile(runtime_config.resource_profile)
                ),
                "chunk_outputs": [str(path) for path in chunk_outputs],
                "raw_merged_output_path": str(raw_merged_output) if raw_merged_output else None,
                "merged_output_path": str(merged_output) if merged_output else None,
                "output_fps": output_fps,
                "output_width": output_width,
                "output_height": output_height,
                "output_frame_count": sum(count for count in chunk_frame_counts if count > 0),
                "quality_restoration": runtime_config.quality_restoration,
                "duration_tolerance_seconds": runtime_config.duration_tolerance_seconds,
            },
            indent=2,
        )
    )
    return VoidRunResult(
        chunk_outputs=chunk_outputs,
        merged_output_path=merged_output,
        run_manifest_path=run_manifest_path,
    )


def _run_logged_subprocess(
    command: list[str],
    cwd: Path,
    log_callback: Callable[[str], None] | None = None,
) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    _emit_log(log_callback, f"VOID: launching subprocess in {cwd}.")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _emit_log(log_callback, f"VOID: subprocess pid {process.pid}.")
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            _emit_log(log_callback, f"VOID: {line}")
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def prepare_void_zip_for_runtime(
    zip_path: Path,
    data_root: Path,
    upload_dir: Path,
    sequence_name: str,
) -> tuple[Path, str, dict[str, object]]:
    """Extract one prepared VOID zip into the sequence layout expected by VOID."""

    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    if data_root.exists():
        shutil.rmtree(data_root)
    upload_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(upload_dir)

    source_video = _find_first(upload_dir, ["input_video.mp4", "input.mp4"])
    source_mask = _find_first(upload_dir, ["quadmask_0.mp4", "trimask_quadmask.mp4"])
    source_prompt = _find_first(upload_dir, ["prompt.json"])
    source_manifest = _find_first(upload_dir, ["manifest.json"])
    if source_video is None or source_mask is None:
        raise FileNotFoundError("VOID zip must contain input_video.mp4 and quadmask_0.mp4")

    sequence_dir = data_root / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_video, sequence_dir / "input_video.mp4")
    shutil.copy2(source_mask, sequence_dir / "quadmask_0.mp4")
    normalize_void_prompt(source_prompt, sequence_dir / "prompt.json")
    manifest = {}
    if source_manifest:
        manifest = json.loads(source_manifest.read_text())
        shutil.copy2(source_manifest, sequence_dir / "manifest.json")
    return data_root, sequence_name, manifest


def normalize_void_prompt(source_path: Path | None, target_path: Path) -> None:
    if source_path and source_path.exists():
        data = json.loads(source_path.read_text())
        if isinstance(data, str):
            data = {"bg": data}
        elif "bg" not in data:
            data = {
                "bg": data.get(
                    "prompt", "clean natural background after the selected object is removed"
                )
            }
    else:
        data = {"bg": "clean natural background after the selected object is removed"}
    target_path.write_text(json.dumps(data, indent=2))


def find_main_void_output(output_dir: Path) -> Path:
    outputs = sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    main_outputs = [path for path in outputs if not path.name.endswith("_tuple.mp4")]
    if not main_outputs:
        raise FileNotFoundError(f"No main MP4 output found in {output_dir}")
    return main_outputs[-1]


def save_stable_chunk_output(
    source_path: Path,
    target_path: Path,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    duration_tolerance_seconds: float = 0.1,
) -> Path:
    """Rewrite a VOID chunk with source-video timing and dimensions.

    VOID can emit correct frame pixels with a slow/default playback timestamp.  The
    chunk merger must not inherit those timestamps, otherwise a normal-length input
    turns into a much longer output with a frozen tail.
    """

    if fps <= 0 or width <= 0 or height <= 0:
        metadata = probe_video(source_path)
        if fps <= 0:
            fps = metadata.fps
        if width <= 0:
            width = metadata.width
        if height <= 0:
            height = metadata.height

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()

    fps_arg = _format_fps(fps)
    with tempfile.TemporaryDirectory(prefix="void_chunk_frames_") as temp_dir:
        frame_pattern = Path(temp_dir) / "frame_%06d.png"
        extract_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-an",
        ]
        if frame_count > 0:
            extract_command.extend(["-frames:v", str(frame_count)])
        extract_command.extend(["-vsync", "0", str(frame_pattern)])
        subprocess.run(extract_command, check=True)

        assemble_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            fps_arg,
            "-i",
            str(frame_pattern),
            "-vf",
            f"setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos",
        ]
        if frame_count > 0:
            assemble_command.extend(["-frames:v", str(frame_count)])
        assemble_command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target_path),
            ]
        )
        subprocess.run(assemble_command, check=True)
    validate_timed_video_output(
        target_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_tolerance_seconds=duration_tolerance_seconds,
        label="stable VOID chunk output",
    )
    return target_path


def stitch_void_chunk_outputs(
    chunk_outputs: list[Path],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    duration_tolerance_seconds: float = 0.1,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_suffix(".concat.txt")
    list_path.write_text("".join(f"file {str(path.resolve())!r}\n" for path in chunk_outputs))
    if output_path.exists():
        output_path.unlink()

    command = [
        "ffmpeg",
        "-y",
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
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(
        command,
        check=True,
    )
    list_path.unlink(missing_ok=True)
    validate_timed_video_output(
        output_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_tolerance_seconds=duration_tolerance_seconds,
        label="stitched VOID output",
    )
    return output_path


def post_enhance_void_output(
    input_path: Path,
    quality_restoration: str,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    duration_tolerance_seconds: float = 0.1,
    log_callback: Callable[[str], None] | None = None,
) -> Path:
    """Run optional Phase 1 quality restoration after VOID chunk stitching."""

    if quality_restoration == QUALITY_RESTORATION_NONE:
        _emit_log(log_callback, "Quality restoration: disabled.")
        return input_path
    if quality_restoration not in VALID_QUALITY_RESTORATION_MODES:
        raise ValueError(
            "quality_restoration must be one of: "
            f"{', '.join(sorted(VALID_QUALITY_RESTORATION_MODES))}"
        )

    if quality_restoration == QUALITY_RESTORATION_VENHANCER:
        enhanced = _try_template_enhancer(
            input_path=input_path,
            mode=quality_restoration,
            env_key="VENHANCER_COMMAND_TEMPLATE",
            suffix="venhancer",
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_tolerance_seconds=duration_tolerance_seconds,
            log_callback=log_callback,
        )
        if enhanced is not None:
            return enhanced
        raise RuntimeError(
            "Quality restoration was set to VEnhancer, but VENHANCER_COMMAND_TEMPLATE "
            "is not configured."
        )

    if quality_restoration == QUALITY_RESTORATION_REALESRGAN:
        enhanced = _try_template_enhancer(
            input_path=input_path,
            mode=quality_restoration,
            env_key="REALESRGAN_COMMAND_TEMPLATE",
            suffix="realesrgan",
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_tolerance_seconds=duration_tolerance_seconds,
            log_callback=log_callback,
        )
        if enhanced is not None:
            return enhanced
        raise RuntimeError(
            "Quality restoration was set to Real-ESRGAN, but REALESRGAN_COMMAND_TEMPLATE "
            "is not configured."
        )

    return _ffmpeg_bicubic_restore(
        input_path=input_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_tolerance_seconds=duration_tolerance_seconds,
        log_callback=log_callback,
    )


def _try_template_enhancer(
    input_path: Path,
    mode: str,
    env_key: str,
    suffix: str,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    duration_tolerance_seconds: float,
    log_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """Run an optional external enhancer command template when configured."""

    template = os.environ.get(env_key, "").strip()
    if not template:
        return None

    output_path = input_path.with_name(f"{input_path.stem}_{suffix}.mp4")
    if output_path.exists():
        output_path.unlink()
    command = template.format(
        input=str(input_path),
        output=str(output_path),
        fps=_format_fps(fps),
        width=width,
        height=height,
        frame_count=frame_count,
    )
    _emit_log(log_callback, f"Quality restoration: running {mode} command template.")
    returncode, output_tail = _run_streaming_shell_command(
        command,
        log_callback=log_callback,
        label=f"Quality restoration {mode}",
    )
    if returncode != 0:
        raise RuntimeError(
            f"Quality restoration command failed for {mode} with exit code "
            f"{returncode}. Last output:\n{output_tail}"
        )
    validate_timed_video_output(
        output_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_tolerance_seconds=duration_tolerance_seconds,
        label=f"{mode} restored output",
    )
    return output_path


def _run_streaming_shell_command(
    command: str,
    log_callback: Callable[[str], None] | None,
    label: str,
) -> tuple[int, str]:
    """Run a shell command while forwarding stdout/stderr into job logs."""

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        return process.wait(), ""

    buffer: list[str] = []
    tail_lines: list[str] = []
    last_logged = ""

    def flush_buffer() -> None:
        nonlocal last_logged
        line = "".join(buffer).strip()
        buffer.clear()
        if not line or line == last_logged:
            return
        last_logged = line
        tail_lines.append(line)
        while len("\n".join(tail_lines)) > 4000 and len(tail_lines) > 1:
            tail_lines.pop(0)
        _emit_log(log_callback, f"{label}: {line}")

    try:
        while True:
            chunk = process.stdout.read(1)
            if chunk == "":
                break
            if chunk in {"\n", "\r"}:
                flush_buffer()
            else:
                buffer.append(chunk)
        flush_buffer()
        returncode = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise

    return returncode, "\n".join(tail_lines)


def _tail_for_log(text: str, limit: int = 4000) -> str:
    """Keep external enhancer output useful without flooding Celery logs."""

    if len(text) <= limit:
        return text.rstrip()
    return "... output truncated ...\n" + text[-limit:].rstrip()


def _ffmpeg_bicubic_restore(
    input_path: Path,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    duration_tolerance_seconds: float,
    log_callback: Callable[[str], None] | None = None,
) -> Path:
    """Normalize final timing and apply the built-in FFmpeg restoration fallback."""

    metadata = probe_video(input_path)
    if fps <= 0:
        fps = metadata.fps
    if width <= 0:
        width = metadata.width
    if height <= 0:
        height = metadata.height

    output_path = input_path.with_name(f"{input_path.stem}_restored.mp4")
    if output_path.resolve() == input_path.resolve():
        output_path = input_path.with_name(f"{input_path.stem}_restored_tmp.mp4")
    if output_path.exists():
        output_path.unlink()

    fps_arg = _format_fps(fps)
    filters = f"setpts=PTS-STARTPTS,scale={width}:{height}:flags=bicubic,fps={fps_arg}"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filters,
    ]
    if frame_count > 0:
        command.extend(["-frames:v", str(frame_count)])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    _emit_log(log_callback, "Quality restoration: normalizing final video with FFmpeg bicubic.")
    subprocess.run(command, check=True)
    validate_timed_video_output(
        output_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_tolerance_seconds=duration_tolerance_seconds,
        label="FFmpeg bicubic restored output",
    )
    return output_path


def validate_timed_video_output(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    duration_tolerance_seconds: float = 0.1,
    label: str = "video output",
) -> None:
    """Verify that a generated video kept the expected timing and dimensions."""

    metadata = probe_video(output_path)
    if width > 0 and metadata.width != width:
        raise RuntimeError(
            f"{label} width mismatch: expected {width}, got {metadata.width}."
        )
    if height > 0 and metadata.height != height:
        raise RuntimeError(
            f"{label} height mismatch: expected {height}, got {metadata.height}."
        )
    if fps > 0 and abs(metadata.fps - fps) > 0.05:
        raise RuntimeError(
            f"{label} FPS mismatch: expected {fps:.3f}, got {metadata.fps:.3f}."
        )
    if frame_count > 0 and metadata.frame_count is not None:
        # Some MP4 muxers report a one-frame difference after stream copy; anything
        # larger points to a real chunk timing/duplication problem.
        if abs(metadata.frame_count - frame_count) > 1:
            raise RuntimeError(
                f"{label} frame count mismatch: "
                f"expected {frame_count}, got {metadata.frame_count}."
            )
    if frame_count > 0 and fps > 0 and metadata.duration is not None:
        expected_duration = frame_count / fps
        allowed = max(duration_tolerance_seconds, 2 / fps)
        if abs(metadata.duration - expected_duration) > allowed:
            raise RuntimeError(
                f"{label} duration mismatch: expected {expected_duration:.3f}s, "
                f"got {metadata.duration:.3f}s."
            )


def _validate_prepare_config(
    config: PreparedVoidPipelineConfig,
    metadata: VideoMetadata,
    chunk_size: int,
) -> None:
    if config.start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if config.max_chunks is not None and config.max_chunks < 1:
        raise ValueError("max_chunks must be >= 1 when provided")
    if config.void_shadow_dilation_px < 0:
        raise ValueError("void_shadow_dilation_px must be >= 0")
    validate_vlm_provider(config.vlm_provider)
    if config.heuristic_contact_dilation_px < 0:
        raise ValueError("heuristic_contact_dilation_px must be >= 0")
    if config.heuristic_shadow_dilation_px < 0:
        raise ValueError("heuristic_shadow_dilation_px must be >= 0")
    if config.heuristic_shadow_vertical_offset_px < 0:
        raise ValueError("heuristic_shadow_vertical_offset_px must be >= 0")
    if config.reference_frame < 0:
        raise ValueError("reference_frame must be >= 0")
    if metadata.frame_count is not None and config.reference_frame >= metadata.frame_count:
        raise ValueError("reference_frame is outside the input video")
    validate_temporal_window_size(chunk_size)
    config.roi.validate_inside(metadata.width, metadata.height)


def _run_scene_analysis_if_needed(
    *,
    config: PreparedVoidPipelineConfig,
    metadata: VideoMetadata,
    mask_provider,
    np,
    cv2,
) -> VlmAnalysisResult | None:
    if config.vlm_provider != VLM_PROVIDER_OLLAMA:
        return None
    _emit_status(config.status_callback, "scene_analysis")
    _emit_log(
        config.log_callback,
        "Scene conditioning: running local Ollama VLM analysis for affected regions.",
    )
    analysis = analyze_ollama_scene(
        np=np,
        cv2=cv2,
        metadata=metadata,
        primary_mask_provider=mask_provider,
        removal_mode=config.removal_mode,
        input_path=config.input_path,
    )
    if analysis.error or not analysis.usable:
        reason = analysis.error or f"confidence={analysis.confidence:.2f}"
        _emit_log(
            config.log_callback,
            f"Scene conditioning: Ollama unavailable or low confidence; "
            f"heuristic fallback remains active ({reason}).",
        )
    else:
        _emit_log(
            config.log_callback,
            "Scene conditioning: Ollama guidance ready "
            f"(confidence={analysis.confidence:.2f}, model={analysis.model}).",
        )
    return analysis


def _build_prepare_manifest(
    config: PreparedVoidPipelineConfig,
    metadata: VideoMetadata,
    profile: VoidResourceProfile,
    chunk_zips: list[Path],
    mask_cache_root: Path | None,
    prompt_result: PromptGenerationResult,
    vlm_analysis: VlmAnalysisResult | None,
) -> dict[str, object]:
    return {
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "chunk_zip_count": len(chunk_zips),
        "chunk_zips": [str(path) for path in chunk_zips],
        "mask_cache_dir": str(mask_cache_root) if mask_cache_root else None,
        "resource_profile": _profile_to_dict(profile),
        "removal_mode": config.removal_mode,
        "tracking_backend": config.tracking_backend,
        "reference_frame": config.reference_frame,
        "mask_padding": config.mask_padding,
        "void_shadow_dilation_px": config.void_shadow_dilation_px,
        "vlm_provider": config.vlm_provider,
        "valid_vlm_providers": sorted(VALID_VLM_PROVIDERS),
        "prompt_strategy": prompt_result.strategy,
        "prompt_text": prompt_result.prompt,
        "prompt_notes": list(prompt_result.notes),
        "vlm_analysis": vlm_analysis.to_manifest() if vlm_analysis is not None else None,
        "heuristic_affected_regions": {
            "contact_dilation_px": config.heuristic_contact_dilation_px,
            "shadow_dilation_px": config.heuristic_shadow_dilation_px,
            "shadow_vertical_offset_px": config.heuristic_shadow_vertical_offset_px,
        },
        "roi": {
            "x": config.roi.x,
            "y": config.roi.y,
            "width": config.roi.width,
            "height": config.roi.height,
        },
        "source": {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
        },
        "colab_upload": {
            "chunk_zip_dir": "/content/void_phase5_chunks",
            "google_drive_chunk_zip_dir": "/content/drive/MyDrive/void_phase5_chunks",
            "resource_profile": profile.name,
        },
    }


def _profile_to_dict(profile: VoidResourceProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "sample_size": profile.sample_size,
        "max_video_length": profile.max_video_length,
        "temporal_window_size": profile.temporal_window_size,
        "latent_temporal_frames": profile.latent_temporal_frames,
        "gpu_memory_mode": profile.gpu_memory_mode,
        "num_inference_steps": profile.num_inference_steps,
        "temporal_multidiffusion_stride": profile.temporal_multidiffusion_stride,
    }


def _find_first(root: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _manifest_has_empty_primary_mask(manifest: dict[str, object]) -> bool:
    stats = manifest.get("quadmask_stats")
    if not isinstance(stats, dict):
        return False
    if stats.get("empty_primary_mask") is True:
        return True
    try:
        return int(stats.get("primary_pixels") or 0) <= 0
    except (TypeError, ValueError):
        return False


def _emit_status(callback: Callable[[str], None] | None, phase: str) -> None:
    if callback:
        callback(phase)


def _emit_progress(
    callback: Callable[[int, int | None], None] | None,
    current: int,
    total: int | None,
) -> None:
    if callback:
        callback(current, total)


def _emit_log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)


def _offset_progress_callback(
    callback: Callable[[int, int | None], None] | None,
    offset: int,
    total: int | None,
) -> Callable[[int, int | None], None] | None:
    if callback is None:
        return None

    def _callback(current: int, _: int | None) -> None:
        callback(offset + current, total)

    return _callback


def _load_cv_runtime():
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Missing Python dependency. Run: python3 -m pip install -r requirements.txt") from exc
    return cv2, np
