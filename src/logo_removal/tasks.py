from __future__ import annotations

import logging
import os
import shutil
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from celery import Celery

from .job_store import DEFAULT_REDIS_URL, RedisJobStore
from .pipeline import InpaintConfig, process_video
from .roi import Roi
from .void_pipeline import (
    PreparedVoidPipelineConfig,
    VoidRuntimeConfig,
    run_full_video_void_pipeline,
)


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)


celery_app = Celery(
    "video_object_removal",
    broker=os.environ.get("CELERY_BROKER_URL", _redis_url()),
    backend=os.environ.get("CELERY_RESULT_BACKEND", _redis_url()),
)
celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_acks_on_failure_or_timeout=True,
    task_reject_on_worker_lost=False,
)


@celery_app.task(name="video_object_removal.process_video_job")
def process_video_job(payload: dict[str, Any]) -> dict[str, str]:
    """Run one heavy video-processing job in a Celery worker process."""

    job_id = str(payload["job_id"])
    store = RedisJobStore()
    store.update(job_id, state="running", phase="preparing", error=None)
    _job_log(store, job_id, "Job accepted by worker.")

    try:
        with _job_logging(store, job_id):
            pipeline_mode = str(payload.get("pipeline_mode") or "legacy")
            if pipeline_mode == "void":
                _process_void_video_job(payload, store, job_id)
            else:
                _process_legacy_video_job(payload, store, job_id)
    except Exception as exc:
        logging.exception("Video object removal job failed: %s", job_id)
        _job_log(store, job_id, f"Job failed: {exc}", phase="failed")
        store.update(job_id, state="failed", phase="failed", error=str(exc))
        raise

    download_url = f"/api/jobs/{job_id}/download"
    store.update(
        job_id,
        state="completed",
        phase="completed",
        error=None,
        download_url=download_url,
    )
    return {"job_id": job_id, "download_url": download_url}


def _process_legacy_video_job(
    payload: dict[str, Any],
    store: RedisJobStore,
    job_id: str,
) -> None:
    roi = _payload_roi(payload)
    _job_log(store, job_id, "Object detection/selection: using the marked ROI.")
    config = InpaintConfig(
        input_path=Path(payload["input_path"]),
        output_path=Path(payload["output_path"]),
        roi=roi,
        select_roi=False,
        preview_frame=int(payload.get("preview_frame", 0)),
        sample_frame_path=None,
        chunk_size=int(payload["chunk_size"]),
        reference_frame=int(payload["reference_frame"]),
        removal_mode=str(payload["removal_mode"]),
        inpaint_engine=str(payload["inpaint_engine"]),
        method=str(payload["method"]),
        radius=float(payload["radius"]),
        mask_padding=int(payload["mask_padding"]),
        overwrite=True,
        keep_temp=False,
        progress_callback=lambda current, total: store.update_progress(job_id, current, total),
        status_callback=lambda phase: _update_phase(store, job_id, phase),
    )
    _job_log(store, job_id, "Masking/inpainting: starting existing frame pipeline.")
    process_video(config)
    _job_log(store, job_id, f"Final output written to {payload['output_path']}.")


def _process_void_video_job(
    payload: dict[str, Any],
    store: RedisJobStore,
    job_id: str,
) -> None:
    roi = _payload_roi(payload)
    output_path = Path(payload["output_path"])
    output_dir = output_path.with_suffix("")
    run_void = _as_bool(payload.get("run_void", False))
    _job_log(
        store,
        job_id,
        "Starting full SAM2 motion-aware VOID pipeline "
        f"(run_void={run_void}, backend={payload.get('sam2_tracking_backend', 'samurai')}).",
    )
    full_result = run_full_video_void_pipeline(
        prepare_config=PreparedVoidPipelineConfig(
            input_path=Path(payload["input_path"]),
            output_dir=output_dir,
            roi=roi,
            sequence_prefix=f"{job_id}_void",
            prompt=str(payload.get("prompt") or "clean natural background after the selected object is removed"),
            removal_mode=str(payload["removal_mode"]),
            tracking_backend=str(payload.get("sam2_tracking_backend") or "samurai"),
            reference_frame=int(payload["reference_frame"]),
            mask_padding=int(payload["mask_padding"]),
            start_frame=0,
            max_chunks=_optional_int(payload.get("max_chunks")),
            force_mask_cache_rebuild=_as_bool(payload.get("force_mask_cache_rebuild", True)),
            void_shadow_dilation_px=int(payload.get("void_shadow_dilation_px") or 0),
            vlm_provider=str(
                payload.get("vlm_provider")
                or os.environ.get("VLM_PROVIDER")
                or "heuristic"
            ),
            heuristic_contact_dilation_px=_payload_int(
                payload,
                "heuristic_contact_dilation_px",
                10,
            ),
            heuristic_shadow_dilation_px=_payload_int(
                payload,
                "heuristic_shadow_dilation_px",
                30,
            ),
            heuristic_shadow_vertical_offset_px=_payload_int(
                payload,
                "heuristic_shadow_vertical_offset_px",
                16,
            ),
            sam2_bidirectional=_as_bool(payload.get("sam2_bidirectional", True)),
            resource_profile=str(payload.get("resource_profile") or "l4_pro_balanced"),
            keep_package_dir=True,
            overwrite=True,
            progress_callback=lambda current, total: store.update_progress(job_id, current, total),
            status_callback=lambda phase: _update_phase(store, job_id, phase),
            log_callback=lambda message: _job_log(store, job_id, message),
        ),
        runtime_config=VoidRuntimeConfig(
            void_repo=_runtime_path(payload, "void_repo", "VOID_REPO_DIR", "VOID_REPO", "/content/void-model"),
            data_root=Path(payload.get("colab_data_root") or "/content/void_phase5_data"),
            output_dir=Path(payload.get("colab_output_dir") or "/content/void_phase5_outputs"),
            upload_dir=Path(payload.get("colab_upload_dir") or "/content/void_phase5_upload"),
            chunk_outputs_dir=Path(
                payload.get("colab_chunk_outputs_dir") or "/content/void_phase5_chunk_outputs"
            ),
            merged_output_path=Path(
                payload.get("colab_merged_output_path") or "/content/void_phase5_merged.mp4"
            ),
            resource_profile=str(payload.get("resource_profile") or "l4_pro_balanced"),
            quality_restoration=str(
                payload.get("quality_restoration")
                or os.environ.get("QUALITY_RESTORATION")
                or "ffmpeg_bicubic"
            ),
        ),
        run_void=run_void,
    )
    if full_result.void_run and full_result.void_run.merged_output_path:
        source_output = full_result.void_run.merged_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if source_output.resolve() != output_path.resolve():
            shutil.copy2(source_output, output_path)
        store.update(job_id, output_path=str(output_path))
        _job_log(store, job_id, f"Final VOID video copied to {output_path}.")
        return

    artifact_path = _bundle_prepared_void_chunks(full_result.prepared.chunk_zips, output_path)
    store.update(job_id, output_path=str(artifact_path))
    _job_log(store, job_id, f"Prepared VOID chunk artifact ready at {artifact_path}.")


def _payload_roi(payload: dict[str, Any]) -> Roi:
    roi_payload = payload["roi"]
    return Roi(
        x=int(roi_payload["x"]),
        y=int(roi_payload["y"]),
        width=int(roi_payload["width"]),
        height=int(roi_payload["height"]),
    )


def _bundle_prepared_void_chunks(chunk_zips: list[Path], output_path: Path) -> Path:
    if len(chunk_zips) == 1:
        artifact_path = output_path.with_suffix(".zip")
        shutil.copy2(chunk_zips[0], artifact_path)
        return artifact_path

    artifact_path = output_path.with_suffix(".void_chunks.zip")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for chunk_zip in chunk_zips:
            archive.write(chunk_zip, arcname=chunk_zip.name)
    return artifact_path


def _update_phase(store: RedisJobStore, job_id: str, phase: str) -> None:
    store.update_phase(job_id, phase)
    phase_labels = {
        "preparing": "Preparing job inputs.",
        "sampling": "Sampling reference frame.",
        "object_detection": "Object detection/selection step started.",
        "segmenting": "Segmenting target object.",
        "tracking": "Tracking target object across frames.",
        "masking": "Masking step started.",
        "chunking": "Chunking step started.",
        "void_pass": "VOID Pass 1 started.",
        "merging": "Merging chunk outputs.",
        "inpainting": "Inpainting frames.",
        "finalizing": "Finalizing output.",
        "completed": "Completed.",
    }
    if phase in phase_labels:
        _job_log(store, job_id, phase_labels[phase], phase=phase)


def _job_log(store: RedisJobStore, job_id: str, message: str, phase: str | None = None) -> None:
    store.append_log(job_id, message, phase=phase)
    logging.getLogger(__name__).info(
        "Job %s: %s",
        job_id,
        message,
        extra={"_job_log_mirrored": True},
    )


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key)
    if value in {None, ""}:
        return default
    return int(value)


def _runtime_path(
    payload: dict[str, Any],
    payload_key: str,
    primary_env_key: str,
    secondary_env_key: str,
    default: str,
) -> Path:
    return Path(
        payload.get(payload_key)
        or os.environ.get(primary_env_key)
        or os.environ.get(secondary_env_key)
        or default
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


class _RedisJobLogHandler(logging.Handler):
    def __init__(self, store: RedisJobStore, job_id: str) -> None:
        super().__init__(level=logging.INFO)
        self.store = store
        self.job_id = job_id
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "_job_log_mirrored", False):
            return
        try:
            self.store.append_log(self.job_id, self.format(record))
        except Exception:
            self.handleError(record)


@contextmanager
def _job_logging(store: RedisJobStore, job_id: str) -> Iterator[None]:
    handler = _RedisJobLogHandler(store, job_id)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
