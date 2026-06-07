from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.logo_removal.inpaint_engines import VALID_INPAINT_ENGINES
from src.logo_removal.job_store import JobRecord, RedisJobStore
from src.logo_removal.mask_providers import AI_OBJECT, VALID_REMOVAL_MODES, VALID_SAM2_TRACKING_BACKENDS
from src.logo_removal.roi import Roi
from src.logo_removal.video import probe_video, require_binary
from src.logo_removal.void_pipeline import (
    VALID_QUALITY_RESTORATION_MODES,
    resolve_void_resource_profile,
)
from src.logo_removal.vlm_analysis import (
    DEFAULT_HEURISTIC_CONTACT_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_DILATION_PX,
    DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX,
    VLM_PROVIDER_HEURISTIC,
    VALID_VLM_PROVIDERS,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "web"
RUNTIME_DIR = Path(tempfile.gettempdir()) / "video-logo-removal-web"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
DEFAULT_VLM_PROVIDER = os.environ.get("VLM_PROVIDER", VLM_PROVIDER_HEURISTIC)
DEFAULT_QUALITY_RESTORATION = os.environ.get("QUALITY_RESTORATION", "ffmpeg_bicubic")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare runtime folders and verify external video tools during startup."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_binary("ffmpeg")
    require_binary("ffprobe")
    try:
        RedisJobStore().ping()
    except Exception as exc:
        logging.warning("Redis is not reachable yet: %s", exc)
    yield


app = FastAPI(title="Video Object Removal Prototype", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
job_store = RedisJobStore()


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page prototype UI."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Expose a minimal health check for the web server."""

    return {"status": "ok"}


@app.post("/api/video-frames")
def extract_video_frames(
    video: UploadFile = File(...),
    thumbnail_width: int = Form(180),
) -> dict[str, object]:
    """Extract true video frames as scrollable JPEG thumbnails for ROI selection."""

    if thumbnail_width < 80 or thumbnail_width > 360:
        raise HTTPException(status_code=400, detail="Thumbnail width must be between 80 and 360.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    temp_path = UPLOAD_DIR / f"frames_{uuid.uuid4().hex}{suffix}"
    try:
        with temp_path.open("wb") as destination:
            shutil.copyfileobj(video.file, destination)

        metadata = probe_video(temp_path)
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="OpenCV is required to extract frames.") from exc

        capture = cv2.VideoCapture(str(temp_path))
        try:
            if not capture.isOpened():
                raise HTTPException(status_code=400, detail="Could not open uploaded video.")

            frames: list[dict[str, object]] = []
            frame_index = 0
            thumbnail_height = max(1, round(thumbnail_width * metadata.height / metadata.width))
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 74]
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                thumbnail = cv2.resize(
                    frame,
                    (thumbnail_width, thumbnail_height),
                    interpolation=cv2.INTER_AREA,
                )
                encoded_ok, encoded = cv2.imencode(".jpg", thumbnail, encode_params)
                if not encoded_ok:
                    raise HTTPException(status_code=500, detail="Could not encode frame thumbnail.")
                frames.append(
                    {
                        "index": frame_index,
                        "time": frame_index / metadata.fps if metadata.fps > 0 else None,
                        "src": "data:image/jpeg;base64,"
                        + base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                )
                frame_index += 1
        finally:
            capture.release()

        if not frames:
            raise HTTPException(status_code=400, detail="No video frames could be extracted.")
        return {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frame_count": len(frames),
            "frames": frames,
        }
    finally:
        temp_path.unlink(missing_ok=True)


def validate_job_options(
    removal_mode: str,
    inpaint_engine: str,
    method: str,
    reference_frame: int,
    chunk_size: int,
    radius: float,
    mask_padding: int,
    pipeline_mode: str = "legacy",
    sam2_tracking_backend: str = "samurai",
    resource_profile: str = "l4_pro_balanced",
    quality_restoration: str = DEFAULT_QUALITY_RESTORATION,
    void_shadow_dilation_px: int = 0,
    vlm_provider: str = DEFAULT_VLM_PROVIDER,
    heuristic_contact_dilation_px: int = DEFAULT_HEURISTIC_CONTACT_DILATION_PX,
    heuristic_shadow_dilation_px: int = DEFAULT_HEURISTIC_SHADOW_DILATION_PX,
    heuristic_shadow_vertical_offset_px: int = DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX,
    max_chunks: int | None = None,
) -> None:
    """Validate job options shared by the API and tests."""

    if pipeline_mode not in {"legacy", "void"}:
        raise HTTPException(status_code=400, detail="Pipeline mode must be 'legacy' or 'void'.")
    if removal_mode not in VALID_REMOVAL_MODES:
        raise HTTPException(status_code=400, detail="Unsupported removal mode.")
    if inpaint_engine not in VALID_INPAINT_ENGINES:
        raise HTTPException(status_code=400, detail="Unsupported inpaint engine.")
    if method not in {"telea", "ns"}:
        raise HTTPException(status_code=400, detail="Method must be 'telea' or 'ns'.")
    if reference_frame < 0:
        raise HTTPException(status_code=400, detail="Reference frame cannot be negative.")
    if chunk_size < 1:
        raise HTTPException(status_code=400, detail="Chunk size must be at least 1.")
    if radius <= 0:
        raise HTTPException(status_code=400, detail="Radius must be greater than 0.")
    if mask_padding < 0:
        raise HTTPException(status_code=400, detail="Mask padding cannot be negative.")
    if sam2_tracking_backend not in VALID_SAM2_TRACKING_BACKENDS:
        raise HTTPException(status_code=400, detail="Unsupported SAM2 tracking backend.")
    if void_shadow_dilation_px < 0:
        raise HTTPException(status_code=400, detail="VOID shadow dilation cannot be negative.")
    if vlm_provider not in VALID_VLM_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported VLM/affected-region provider.")
    if heuristic_contact_dilation_px < 0:
        raise HTTPException(status_code=400, detail="Heuristic contact dilation cannot be negative.")
    if heuristic_shadow_dilation_px < 0:
        raise HTTPException(status_code=400, detail="Heuristic shadow dilation cannot be negative.")
    if heuristic_shadow_vertical_offset_px < 0:
        raise HTTPException(status_code=400, detail="Heuristic shadow offset cannot be negative.")
    if max_chunks is not None and max_chunks < 1:
        raise HTTPException(status_code=400, detail="Max chunks must be at least 1.")
    try:
        resolve_void_resource_profile(resource_profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if quality_restoration not in VALID_QUALITY_RESTORATION_MODES:
        raise HTTPException(status_code=400, detail="Unsupported quality restoration mode.")
    required_restoration_template = {
        "venhancer": "VENHANCER_COMMAND_TEMPLATE",
        "realesrgan": "REALESRGAN_COMMAND_TEMPLATE",
    }.get(quality_restoration)
    if required_restoration_template and not os.environ.get(required_restoration_template):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{quality_restoration} requires {required_restoration_template} "
                "to be configured before starting the job."
            ),
        )


@app.post("/api/jobs")
def create_job(
    video: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    reference_frame: int = Form(0),
    removal_mode: str = Form(AI_OBJECT),
    inpaint_engine: str = Form("opencv"),
    chunk_size: int = Form(30),
    method: str = Form("telea"),
    radius: float = Form(3.0),
    mask_padding: int = Form(0),
    pipeline_mode: str = Form("legacy"),
    sam2_tracking_backend: str = Form("samurai"),
    resource_profile: str = Form("l4_pro_balanced"),
    quality_restoration: str = Form(DEFAULT_QUALITY_RESTORATION),
    sam2_bidirectional: bool = Form(True),
    run_void: bool = Form(False),
    void_shadow_dilation_px: int = Form(0),
    vlm_provider: str = Form(DEFAULT_VLM_PROVIDER),
    heuristic_contact_dilation_px: int = Form(DEFAULT_HEURISTIC_CONTACT_DILATION_PX),
    heuristic_shadow_dilation_px: int = Form(DEFAULT_HEURISTIC_SHADOW_DILATION_PX),
    heuristic_shadow_vertical_offset_px: int = Form(
        DEFAULT_HEURISTIC_SHADOW_VERTICAL_OFFSET_PX
    ),
    max_chunks: int | None = Form(None),
    force_mask_cache_rebuild: bool = Form(True),
) -> dict[str, str]:
    """Accept an uploaded video and queue a logo removal job."""

    validate_job_options(
        removal_mode=removal_mode,
        inpaint_engine=inpaint_engine,
        method=method,
        reference_frame=reference_frame,
        chunk_size=chunk_size,
        radius=radius,
        mask_padding=mask_padding,
        pipeline_mode=pipeline_mode,
        sam2_tracking_backend=sam2_tracking_backend,
        resource_profile=resource_profile,
        quality_restoration=quality_restoration,
        void_shadow_dilation_px=void_shadow_dilation_px,
        vlm_provider=vlm_provider,
        heuristic_contact_dilation_px=heuristic_contact_dilation_px,
        heuristic_shadow_dilation_px=heuristic_shadow_dilation_px,
        heuristic_shadow_vertical_offset_px=heuristic_shadow_vertical_offset_px,
        max_chunks=max_chunks,
    )

    roi = Roi(x=x, y=y, width=width, height=height)
    if roi.width <= 0 or roi.height <= 0:
        raise HTTPException(status_code=400, detail="ROI width and height must be positive.")

    job_id = uuid.uuid4().hex
    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    with input_path.open("wb") as destination:
        shutil.copyfileobj(video.file, destination)

    record = JobRecord(id=job_id, input_path=str(input_path), output_path=str(output_path))
    try:
        job_store.create(record)
    except Exception as exc:
        input_path.unlink(missing_ok=True)
        logging.exception("Could not create Redis job record: %s", job_id)
        raise HTTPException(status_code=503, detail=f"Redis is not available: {exc}") from exc

    payload = {
        "job_id": job_id,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "roi": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
        "preview_frame": 0,
        "reference_frame": reference_frame,
        "removal_mode": removal_mode,
        "inpaint_engine": inpaint_engine,
        "chunk_size": chunk_size,
        "method": method,
        "radius": radius,
        "mask_padding": mask_padding,
        "pipeline_mode": pipeline_mode,
        "sam2_tracking_backend": sam2_tracking_backend,
        "resource_profile": resource_profile,
        "quality_restoration": quality_restoration,
        "sam2_bidirectional": sam2_bidirectional,
        "run_void": run_void,
        "void_shadow_dilation_px": void_shadow_dilation_px,
        "vlm_provider": vlm_provider,
        "heuristic_contact_dilation_px": heuristic_contact_dilation_px,
        "heuristic_shadow_dilation_px": heuristic_shadow_dilation_px,
        "heuristic_shadow_vertical_offset_px": heuristic_shadow_vertical_offset_px,
        "max_chunks": max_chunks,
        "force_mask_cache_rebuild": force_mask_cache_rebuild,
    }
    try:
        from src.logo_removal.tasks import process_video_job

        process_video_job.apply_async(args=[payload], task_id=job_id)
    except Exception as exc:
        job_store.update(job_id, state="failed", phase="failed", error=str(exc))
        logging.exception("Could not enqueue Celery job: %s", job_id)
        raise HTTPException(status_code=503, detail=f"Celery/Redis queue is not available: {exc}") from exc

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    """Return the latest status for a queued or completed job."""

    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = _refresh_failed_celery_state(job)
    return job.snapshot()


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    """Download the processed MP4 for a completed job."""

    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    output_path = Path(job.output_path)
    if job.state != "completed" or not output_path.exists():
        raise HTTPException(status_code=409, detail="Output is not ready yet.")
    if output_path.suffix == ".zip":
        return FileResponse(output_path, media_type="application/zip", filename=output_path.name)
    return FileResponse(output_path, media_type="video/mp4", filename="logo_removed.mp4")


def _refresh_failed_celery_state(job: JobRecord) -> JobRecord:
    """Reflect worker-lost failures from Celery into the Redis job state."""

    if job.state not in {"queued", "running"}:
        return job
    try:
        from src.logo_removal.tasks import process_video_job

        result = process_video_job.AsyncResult(job.id)
        if result.failed():
            error = str(result.result) if result.result else "Celery worker failed."
            job_store.update(job.id, state="failed", phase="failed", error=error)
            refreshed = job_store.get(job.id)
            return refreshed or job
    except Exception:
        logging.debug("Could not refresh Celery state for job %s", job.id, exc_info=True)
    return job


def main() -> None:
    """Start the FastAPI development server for the prototype UI."""

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
