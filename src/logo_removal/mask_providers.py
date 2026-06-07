from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import subprocess
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .mask_qa import MaskQaTracker
from .mask_refiners import MorphologicalMaskRefiner
from .roi import Roi
from .video import VideoMetadata


STATIC_RECTANGLE = "static_rectangle"
AI_OBJECT = "ai_object"
AI_TEXT_OR_LOGO = "ai_text_or_logo"
AI_HUMAN_CROWDED = "ai_human_crowded"
VALID_REMOVAL_MODES = {STATIC_RECTANGLE, AI_OBJECT, AI_TEXT_OR_LOGO, AI_HUMAN_CROWDED}

SAM2_BACKEND_AUTO = "auto"
SAM2_BACKEND_GLOBAL = "global"
SAM2_BACKEND_SAMURAI = "samurai"
SAM2_BACKEND_DAM4SAM = "dam4sam"
SAM2_BACKEND_SAM2LONG_RESEARCH = "sam2long_research"
VALID_SAM2_TRACKING_BACKENDS = {
    SAM2_BACKEND_AUTO,
    SAM2_BACKEND_GLOBAL,
    SAM2_BACKEND_SAMURAI,
    SAM2_BACKEND_DAM4SAM,
    SAM2_BACKEND_SAM2LONG_RESEARCH,
}
DEFAULT_SAM2_CACHE_VERSION = 1
CROWDED_HUMAN_TRACKING_BACKEND = "botsort_reid"
DEFAULT_CROWDED_HUMAN_DETECTOR_MODEL = "yolo11m.pt"
DEFAULT_CROWDED_HUMAN_REID_MODEL = "osnet_x0_25_msmt17.pt"
DEFAULT_CROWDED_HUMAN_DETECTOR_IMGSZ = 960
DEFAULT_CROWDED_HUMAN_PERSON_CONF = 0.45
DEFAULT_CROWDED_HUMAN_PERSON_IOU = 0.70
DEFAULT_CROWDED_HUMAN_TARGET_ROI_MIN_IOU = 0.25
DEFAULT_CROWDED_HUMAN_SAM2_IMAGE_SIZE = 1024
DEFAULT_CROWDED_HUMAN_SAM2_REFINE_EVERY_N_FRAMES = 1
DEFAULT_CROWDED_HUMAN_MISSING_TARGET_WARNING_FRAMES = 3
DEFAULT_CROWDED_HUMAN_NEIGHBOR_MASK_EROSION_PX = 3


class MaskProvider(Protocol):
    """Produces one binary removal mask for each processed video frame."""

    def prepare(self) -> None:
        """Load resources or precompute state before frame processing begins."""

    def mask_for_frame(self, frame_index: int, frame):
        """Return a uint8 single-channel mask for the given frame."""


@dataclass
class CachedMaskProvider:
    """Read a precomputed full-video mask timeline by absolute frame index."""

    np: object
    cv2: object
    metadata: VideoMetadata
    cache_dir: Path

    def prepare(self) -> None:
        """Load cache metadata and validate that every expected mask exists."""

        self.cache_dir = self.cache_dir.expanduser().resolve()
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Mask cache metadata was not found: {metadata_path}")
        try:
            self.cache_metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Mask cache metadata is invalid JSON: {metadata_path}") from exc

        frame_count = int(self.cache_metadata.get("source_frame_count") or 0)
        if frame_count <= 0:
            frame_count = self.metadata.frame_count or 0
        if frame_count <= 0:
            raise RuntimeError("Mask cache does not declare a usable frame count.")

        self._mask_dir = self.cache_dir / "masks"
        self._mask_paths: dict[int, Path] = {}
        for frame_idx in range(frame_count):
            mask_path = self._mask_dir / f"{frame_idx:06d}.png"
            if not mask_path.exists():
                raise RuntimeError(f"Mask cache is missing frame {frame_idx}: {mask_path}")
            self._mask_paths[frame_idx] = mask_path

        self.tracking_backend = self.cache_metadata.get("tracking_backend")
        self.propagation_mode = self.cache_metadata.get("propagation_mode")
        self.cache_key = self.cache_metadata.get("cache_key") or self.cache_dir.name

    def mask_for_frame(self, frame_index: int, frame):
        """Return the cached mask for the one-based pipeline frame index."""

        zero_based_frame_index = frame_index - 1
        mask_path = self._mask_paths.get(zero_based_frame_index)
        if mask_path is None or not mask_path.exists():
            raise RuntimeError(f"Mask cache does not contain frame {zero_based_frame_index}.")

        mask = self.cv2.imread(str(mask_path), self.cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read cached mask: {mask_path}")
        if frame is not None and mask.shape[:2] != frame.shape[:2]:
            mask = self.cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=self.cv2.INTER_NEAREST,
            )
        return mask


@dataclass
class StaticRectangleMaskProvider:
    """Reuses the selected rectangular ROI as the mask for every frame."""

    np: object
    cv2: object
    metadata: VideoMetadata
    roi: Roi
    mask_padding: int

    def prepare(self) -> None:
        """Validate and build the static mask once."""

        padded_roi = self.roi.padded(self.mask_padding, self.metadata.width, self.metadata.height)
        padded_roi.validate_inside(self.metadata.width, self.metadata.height)
        self._mask = self._create_mask(padded_roi)

    def mask_for_frame(self, frame_index: int, frame):
        """Return the same static mask for every frame."""

        return self._mask

    def _create_mask(self, roi: Roi):
        """Create an OpenCV-compatible mask for the selected rectangle."""

        mask = self.np.zeros((self.metadata.height, self.metadata.width), dtype=self.np.uint8)
        mask[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = 255
        return mask


@dataclass
class Sam2VideoMaskProvider:
    """Propagate a user-selected target mask through the video with SAM 2."""

    np: object
    cv2: object
    input_path: Path
    metadata: VideoMetadata
    roi: Roi
    reference_frame: int
    mask_padding: int
    checkpoint_path: Path | None = None
    model_cfg: str | None = None
    device: str | None = None
    sam2_image_size: int | None = None
    mask_close_kernel_px: int = 3
    mask_blur_px: int = 0
    mask_min_component_area: int = 0
    sam2_tracking_backend: str | None = None
    bidirectional: bool | None = None
    mask_cache_dir: Path | None = None
    force_cache_rebuild: bool = False

    def prepare(self) -> None:
        """Load SAM 2, extract video frames, and precompute one mask per frame."""

        self.tracking_backend = self._resolve_tracking_backend()
        self.bidirectional = self._resolve_bidirectional()
        self.propagation_mode = "bidirectional" if self.bidirectional else "forward_only"
        cache_root = self._resolve_cache_root()
        cache_key = self._build_cache_key(self.tracking_backend, self.propagation_mode)
        self.cache_key = cache_key
        if cache_root is not None:
            cached_provider = self._try_load_cache(cache_root / cache_key)
            if cached_provider is not None:
                self.cache_dir = cached_provider.cache_dir
                self.cache_metadata = cached_provider.cache_metadata
                self._mask_dir = cached_provider._mask_dir
                self._mask_paths = cached_provider._mask_paths
                logging.info("Using cached SAM 2 mask timeline: %s", self.cache_dir)
                return

        if self.tracking_backend in {
            SAM2_BACKEND_SAMURAI,
            SAM2_BACKEND_DAM4SAM,
            SAM2_BACKEND_SAM2LONG_RESEARCH,
        }:
            self._prepare_with_subprocess_backend(cache_root, cache_key)
            return

        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "AI object/text tracking needs SAM 2, but it is not installed. "
                "Use 'Static Area' mode for now, or install Meta SAM 2 and configure "
                "its checkpoint before selecting Moving Object or Text/Logo mode."
            ) from exc

        build_sam2_video_predictor = self._import_sam2_video_predictor()
        checkpoint_path = self._resolve_checkpoint_path()
        model_cfg = self._resolve_model_cfg(checkpoint_path, self.tracking_backend)
        device = self.device or os.environ.get("SAM2_DEVICE") or self._select_torch_device(torch)
        sam2_image_size = self.sam2_image_size or self._resolve_sam2_image_size()
        logging.info(
            "Preparing SAM 2 %s mask propagation with %s on %s at %sx%s inference size",
            self.tracking_backend,
            checkpoint_path.name,
            device,
            sam2_image_size,
            sam2_image_size,
        )

        self._frame_temp = tempfile.TemporaryDirectory(prefix="sam2-frames-")
        self._frame_dir = Path(self._frame_temp.name)
        if cache_root is None:
            self._cache_temp = tempfile.TemporaryDirectory(prefix="sam2-mask-cache-")
            self.cache_dir = Path(self._cache_temp.name)
        else:
            self.cache_dir = cache_root / cache_key
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mask_dir = self.cache_dir / "masks"
        self._forward_mask_dir = self.cache_dir / "forward"
        self._backward_mask_dir = self.cache_dir / "backward"
        self._mask_dir.mkdir(parents=True, exist_ok=True)
        self._forward_mask_dir.mkdir(parents=True, exist_ok=True)
        self._backward_mask_dir.mkdir(parents=True, exist_ok=True)
        self._mask_paths: dict[int, Path] = {}
        self._warnings: list[str] = []
        self._mask_refiner = MorphologicalMaskRefiner(
            np=self.np,
            cv2=self.cv2,
            dilation_px=self.mask_padding,
            close_kernel_px=self.mask_close_kernel_px,
            blur_px=self.mask_blur_px,
            min_component_area=self.mask_min_component_area,
        )
        self._mask_refiner.prepare()

        frame_count = self._extract_video_frames()
        if self.reference_frame >= frame_count:
            raise RuntimeError(
                f"Reference frame {self.reference_frame} is outside the video "
                f"range. Extracted {frame_count} frames."
            )

        padded_roi = self.roi.padded(self.mask_padding, self.metadata.width, self.metadata.height)
        padded_roi.validate_inside(self.metadata.width, self.metadata.height)
        box = self.np.array(
            [
                padded_roi.x,
                padded_roi.y,
                padded_roi.x + padded_roi.width,
                padded_roi.y + padded_roi.height,
            ],
            dtype=self.np.float32,
        )

        try:
            predictor = build_sam2_video_predictor(
                model_cfg,
                str(checkpoint_path),
                device=device,
                hydra_overrides_extra=[f"model.image_size={sam2_image_size}"],
                apply_postprocessing=False,
            )
            with torch.inference_mode():
                forward_masks = self._run_sam2_pass(
                    predictor=predictor,
                    box=box,
                    reverse=False,
                    output_dir=self._forward_mask_dir,
                )
                backward_masks: dict[int, Path] = {}
                if self.bidirectional and self.reference_frame > 0:
                    try:
                        backward_masks = self._run_sam2_pass(
                            predictor=predictor,
                            box=box,
                            reverse=True,
                            output_dir=self._backward_mask_dir,
                        )
                    except Exception as exc:  # pragma: no cover - exercised by integration runs
                        warning = f"Reverse SAM 2 propagation failed; using forward masks only: {exc}"
                        logging.warning(warning)
                        self._warnings.append(warning)
                        self.propagation_mode = "forward_only_reverse_failed"
                self._merge_propagation_masks(
                    frame_count=frame_count,
                    forward_masks=forward_masks,
                    backward_masks=backward_masks,
                )
        except Exception as exc:
            raise RuntimeError(f"SAM 2 mask propagation failed: {exc}") from exc

        if not self._mask_paths:
            raise RuntimeError("SAM 2 did not produce any propagated masks.")
        self._qa_summary = self._write_cache_qa(frame_count)
        self._write_cache_metadata(
            frame_count=frame_count,
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            device=device,
            sam2_image_size=sam2_image_size,
        )
        logging.info("SAM 2 produced %s propagated frame masks.", len(self._mask_paths))

    def mask_for_frame(self, frame_index: int, frame):
        """Read the precomputed SAM 2 mask for the one-based pipeline frame index."""

        zero_based_frame_index = frame_index - 1
        mask_path = self._mask_paths.get(zero_based_frame_index)
        if mask_path is None or not mask_path.exists():
            raise RuntimeError(f"SAM 2 did not produce a mask for frame {zero_based_frame_index}.")

        mask = self.cv2.imread(str(mask_path), self.cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read SAM 2 mask: {mask_path}")
        if mask.shape[:2] != frame.shape[:2]:
            mask = self.cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=self.cv2.INTER_NEAREST,
            )
        return mask

    def _resolve_tracking_backend(self) -> str:
        """Resolve the configured SAM2-compatible tracking backend."""

        requested = (
            self.sam2_tracking_backend
            or os.environ.get("SAM2_TRACKING_BACKEND")
            or SAM2_BACKEND_AUTO
        ).strip().lower()
        if requested not in VALID_SAM2_TRACKING_BACKENDS:
            raise RuntimeError(
                "SAM2_TRACKING_BACKEND must be one of: "
                f"{', '.join(sorted(VALID_SAM2_TRACKING_BACKENDS))}"
            )
        if requested == SAM2_BACKEND_AUTO:
            samurai_repo = self._default_samurai_repo_dir()
            if samurai_repo.exists():
                return SAM2_BACKEND_SAMURAI
            logging.info("SAMURAI_REPO_DIR is not configured; using global SAM 2 backend.")
            return SAM2_BACKEND_GLOBAL
        return requested

    def _default_samurai_repo_dir(self) -> Path:
        raw = os.environ.get("SAMURAI_REPO_DIR")
        if raw:
            return Path(raw).expanduser().resolve()
        return Path(__file__).resolve().parents[2] / "models" / "samurai_repo"

    def _prepare_with_subprocess_backend(self, cache_root: Path | None, cache_key: str) -> None:
        """Generate a mask cache through a forked SAM2 backend in a fresh process."""

        checkpoint_path = self._resolve_checkpoint_path()
        model_cfg = self._resolve_model_cfg(checkpoint_path, self.tracking_backend)
        device = self.device or os.environ.get("SAM2_DEVICE") or "cuda"
        sam2_image_size = self.sam2_image_size or self._resolve_sam2_image_size()
        warnings: list[str] = []
        if self.tracking_backend == SAM2_BACKEND_SAM2LONG_RESEARCH and sam2_image_size != 1024:
            warnings.append(
                "SAM2Long uses the upstream 1024 image size because its memory-tree "
                "implementation assumes the standard SAM2 feature grid."
            )
            sam2_image_size = 1024
        if self.tracking_backend == SAM2_BACKEND_DAM4SAM:
            try:
                import torch  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("DAM4SAM requires torch in the active Python environment.") from exc
            if not device.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError(
                    "DAM4SAM's public tracker wrapper currently requires CUDA. "
                    "Run this backend in a CUDA runtime, or use samurai/global/sam2long_research."
                )

        self._frame_temp = tempfile.TemporaryDirectory(prefix=f"{self.tracking_backend}-frames-")
        self._frame_dir = Path(self._frame_temp.name)
        if cache_root is None:
            self._cache_temp = tempfile.TemporaryDirectory(prefix=f"{self.tracking_backend}-cache-")
            self.cache_dir = Path(self._cache_temp.name)
        else:
            self.cache_dir = cache_root / cache_key
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._mask_dir = self.cache_dir / "masks"
        self._forward_mask_dir = self.cache_dir / "forward"
        self._backward_mask_dir = self.cache_dir / "backward"
        self._warnings = warnings
        frame_count = self._extract_video_frames()
        if self.reference_frame >= frame_count:
            raise RuntimeError(
                f"Reference frame {self.reference_frame} is outside the video "
                f"range. Extracted {frame_count} frames."
            )

        padded_roi = self.roi.padded(self.mask_padding, self.metadata.width, self.metadata.height)
        padded_roi.validate_inside(self.metadata.width, self.metadata.height)
        repo_dir = self._resolve_fork_backend_repo_dir(self.tracking_backend)
        runner = Path(__file__).resolve().parents[2] / "scripts" / "generate_fork_backend_cache.py"
        command = [
            sys.executable,
            str(runner),
            "--backend",
            self.tracking_backend,
            "--repo-dir",
            str(repo_dir),
            "--frame-dir",
            str(self._frame_dir),
            "--output-cache-dir",
            str(self.cache_dir),
            "--x",
            str(padded_roi.x),
            "--y",
            str(padded_roi.y),
            "--width",
            str(padded_roi.width),
            "--height",
            str(padded_roi.height),
            "--reference-frame",
            str(self.reference_frame),
            "--checkpoint",
            str(checkpoint_path),
            "--model-cfg",
            model_cfg,
            "--device",
            device,
            "--sam2-image-size",
            str(sam2_image_size),
        ]
        if self.bidirectional:
            command.append("--bidirectional")
        logging.info("Generating %s mask cache via subprocess.", self.tracking_backend)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{self.tracking_backend} mask cache generation failed. "
                "Check backend setup, CUDA availability, and checkpoint compatibility."
            ) from exc

        self._mask_paths = {}
        for frame_idx in range(frame_count):
            mask_path = self._mask_dir / f"{frame_idx:06d}.png"
            if mask_path.exists():
                self._mask_paths[frame_idx] = mask_path
        if len(self._mask_paths) != frame_count:
            raise RuntimeError(
                f"{self.tracking_backend} produced {len(self._mask_paths)} masks for "
                f"{frame_count} frames."
            )

        backend_report_path = self.cache_dir / "backend_report.json"
        if backend_report_path.exists():
            try:
                backend_report = json.loads(backend_report_path.read_text())
                self._warnings.extend(str(warning) for warning in backend_report.get("warnings", []))
            except json.JSONDecodeError:
                self._warnings.append("Backend report was not valid JSON.")
        self._qa_summary = self._write_cache_qa(frame_count)
        self._write_cache_metadata(
            frame_count=frame_count,
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            device=device,
            sam2_image_size=sam2_image_size,
        )
        logging.info("%s produced %s frame masks.", self.tracking_backend, len(self._mask_paths))

    def _resolve_fork_backend_repo_dir(self, tracking_backend: str) -> Path:
        project_root = Path(__file__).resolve().parents[2]
        if tracking_backend == SAM2_BACKEND_DAM4SAM:
            raw = os.environ.get("DAM4SAM_REPO_DIR") or str(project_root / "models" / "dam4sam_repo")
            repo_dir = Path(raw).expanduser().resolve()
            if not repo_dir.exists():
                raise RuntimeError(
                    "SAM2_TRACKING_BACKEND=dam4sam requires DAM4SAM_REPO_DIR or "
                    f"a local checkout at {repo_dir}."
                )
            return repo_dir
        if tracking_backend == SAM2_BACKEND_SAMURAI:
            repo_dir = self._default_samurai_repo_dir()
            if not repo_dir.exists():
                raise RuntimeError(
                    "SAM2_TRACKING_BACKEND=samurai requires SAMURAI_REPO_DIR or "
                    f"a local checkout at {repo_dir}."
                )
            return repo_dir
        if tracking_backend == SAM2_BACKEND_SAM2LONG_RESEARCH:
            raw = os.environ.get("SAM2LONG_REPO_DIR") or str(
                project_root / "models" / "sam2long_repo"
            )
            repo_dir = Path(raw).expanduser().resolve()
            if not repo_dir.exists():
                raise RuntimeError(
                    "SAM2_TRACKING_BACKEND=sam2long_research requires SAM2LONG_REPO_DIR "
                    f"or a local checkout at {repo_dir}."
                )
            return repo_dir
        raise RuntimeError(f"Unsupported fork backend: {tracking_backend}")

    def _resolve_bidirectional(self) -> bool:
        if self.bidirectional is not None:
            return bool(self.bidirectional)
        raw = os.environ.get("SAM2_BIDIRECTIONAL", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _resolve_cache_root(self) -> Path | None:
        cache_dir = self.mask_cache_dir or (
            Path(os.environ["SAM2_MASK_CACHE_DIR"])
            if os.environ.get("SAM2_MASK_CACHE_DIR")
            else None
        )
        if cache_dir is None:
            return None
        return cache_dir.expanduser().resolve()

    def _import_sam2_video_predictor(self):
        """Import the selected backend without silently mixing SAM2 forks."""

        if self.tracking_backend == SAM2_BACKEND_GLOBAL:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-not-found]

            return build_sam2_video_predictor

        if self.tracking_backend == SAM2_BACKEND_SAMURAI:
            repo_dir = self._default_samurai_repo_dir()
            if not repo_dir.exists():
                raise RuntimeError(
                    "SAM2_TRACKING_BACKEND=samurai requires SAMURAI_REPO_DIR or "
                    f"a local checkout at {repo_dir}."
                )
            import_root = repo_dir / "sam2"
            if not (import_root / "sam2").exists():
                import_root = repo_dir

            loaded_sam2 = sys.modules.get("sam2")
            if loaded_sam2 is not None:
                loaded_file = Path(getattr(loaded_sam2, "__file__", "")).resolve()
                if import_root not in loaded_file.parents:
                    raise RuntimeError(
                        "A different sam2 package is already imported in this process. "
                        "Run SAMURAI in a fresh process or use SAM2_TRACKING_BACKEND=global."
                    )
            sys.path.insert(0, str(import_root))
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-not-found]

            return build_sam2_video_predictor

        raise RuntimeError(f"Unsupported SAM 2 tracking backend: {self.tracking_backend}")

    def _resolve_checkpoint_path(self) -> Path:
        """Find the configured SAM 2 checkpoint path, defaulting to the local tiny model."""

        if self.checkpoint_path is not None:
            checkpoint_path = self.checkpoint_path
        elif os.environ.get("SAM2_CHECKPOINT_PATH"):
            checkpoint_path = Path(os.environ["SAM2_CHECKPOINT_PATH"])
        else:
            checkpoint_path = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "sam2_repo"
                / "checkpoints"
                / "sam2.1_hiera_tiny.pt"
            )
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise RuntimeError(
                "SAM 2 checkpoint was not found. Expected a local checkpoint at "
                f"{checkpoint_path}. Download Meta SAM 2 checkpoints or set "
                "SAM2_CHECKPOINT_PATH to the desired .pt file."
            )
        return checkpoint_path

    def _resolve_model_cfg(self, checkpoint_path: Path, tracking_backend: str) -> str:
        """Choose the matching SAM 2 Hydra config for the selected checkpoint."""

        if self.model_cfg is not None:
            return self.model_cfg
        if os.environ.get("SAM2_MODEL_CFG"):
            return os.environ["SAM2_MODEL_CFG"]

        checkpoint_name = checkpoint_path.name
        if tracking_backend == SAM2_BACKEND_SAMURAI:
            config_root = "configs/samurai"
        elif tracking_backend == SAM2_BACKEND_DAM4SAM:
            config_root = "sam21pp"
        else:
            config_root = "configs/sam2.1"
        if "hiera_small" in checkpoint_name:
            if tracking_backend == SAM2_BACKEND_DAM4SAM:
                return "sam21pp_hiera_s.yaml"
            return f"{config_root}/sam2.1_hiera_s.yaml"
        if "hiera_base_plus" in checkpoint_name:
            if tracking_backend == SAM2_BACKEND_DAM4SAM:
                return "sam21pp_hiera_b+.yaml"
            return f"{config_root}/sam2.1_hiera_b+.yaml"
        if "hiera_large" in checkpoint_name:
            if tracking_backend == SAM2_BACKEND_DAM4SAM:
                return "sam21pp_hiera_l.yaml"
            return f"{config_root}/sam2.1_hiera_l.yaml"
        if tracking_backend == SAM2_BACKEND_DAM4SAM:
            return "sam21pp_hiera_t.yaml"
        return f"{config_root}/sam2.1_hiera_t.yaml"

    def _select_torch_device(self, torch) -> str:
        """Select a stable Torch device without probing Apple MPS in worker processes."""

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_sam2_image_size(self) -> int:
        """Resolve SAM 2's square inference size, using a memory-safe prototype default."""

        raw_image_size = os.environ.get("SAM2_IMAGE_SIZE", "512")
        try:
            image_size = int(raw_image_size)
        except ValueError as exc:
            raise RuntimeError("SAM2_IMAGE_SIZE must be an integer.") from exc
        if image_size < 256:
            raise RuntimeError("SAM2_IMAGE_SIZE must be at least 256.")
        if image_size % 16 != 0:
            raise RuntimeError("SAM2_IMAGE_SIZE must be divisible by 16.")
        return image_size

    def _extract_video_frames(self) -> int:
        """Save the input video as numbered JPEGs for SAM 2's video predictor."""

        capture = self.cv2.VideoCapture(str(self.input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for SAM 2 frame extraction: {self.input_path}")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_path = self._frame_dir / f"{frame_index:05d}.jpg"
                written = self.cv2.imwrite(
                    str(frame_path),
                    frame,
                    [int(self.cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if not written:
                    raise RuntimeError(f"Could not write SAM 2 frame: {frame_path}")
                frame_index += 1
        finally:
            capture.release()

        if frame_index == 0:
            raise RuntimeError("Could not extract any frames for SAM 2.")
        return frame_index

    def _run_sam2_pass(self, predictor, box, reverse: bool, output_dir: Path) -> dict[int, Path]:
        """Run one SAM2 propagation direction and write raw masks."""

        output_dir.mkdir(parents=True, exist_ok=True)
        inference_state = predictor.init_state(
            video_path=str(self._frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=True,
        )
        prompt_frame_idx, object_ids, mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=self.reference_frame,
            obj_id=1,
            box=box,
        )
        mask_paths: dict[int, Path] = {}
        self._write_mask(prompt_frame_idx, object_ids, mask_logits, output_dir, mask_paths)
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=self.reference_frame,
            reverse=reverse,
        ):
            self._write_mask(frame_idx, object_ids, mask_logits, output_dir, mask_paths)
        return mask_paths

    def _merge_propagation_masks(
        self,
        frame_count: int,
        forward_masks: dict[int, Path],
        backward_masks: dict[int, Path],
    ) -> None:
        """Merge forward/backward raw masks into the absolute full-video timeline."""

        for frame_idx in range(frame_count):
            source_path = None
            if frame_idx < self.reference_frame and backward_masks:
                source_path = backward_masks.get(frame_idx)
            elif frame_idx == self.reference_frame:
                source_path = forward_masks.get(frame_idx) or backward_masks.get(frame_idx)
            else:
                source_path = forward_masks.get(frame_idx)

            if source_path is None:
                continue
            merged_path = self._mask_dir / f"{frame_idx:06d}.png"
            shutil.copy2(source_path, merged_path)
            self._mask_paths[frame_idx] = merged_path

    def _write_mask(
        self,
        frame_idx: int,
        object_ids,
        mask_logits,
        output_dir: Path,
        mask_paths: dict[int, Path],
    ) -> None:
        """Convert SAM 2 logits for object id 1 into a binary PNG mask."""

        object_id_list = [int(object_id) for object_id in object_ids]
        mask_index = object_id_list.index(1) if 1 in object_id_list else 0
        mask = (mask_logits[mask_index] > 0.0).detach().cpu().numpy()
        mask = self.np.squeeze(mask).astype(self.np.uint8) * 255
        if mask.shape != (self.metadata.height, self.metadata.width):
            mask = self.cv2.resize(
                mask,
                (self.metadata.width, self.metadata.height),
                interpolation=self.cv2.INTER_NEAREST,
            )
        mask = self._postprocess_mask(mask)

        mask_path = output_dir / f"{frame_idx:06d}.png"
        if not self.cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Could not write SAM 2 mask: {mask_path}")
        mask_paths[int(frame_idx)] = mask_path

    def _postprocess_mask(self, mask):
        """Clean and slightly expand propagated masks before inpainting."""

        return self._mask_refiner.refine(mask, frame=None, frame_index=0)

    def _build_cache_key(self, tracking_backend: str, propagation_mode: str) -> str:
        payload = {
            "cache_version": DEFAULT_SAM2_CACHE_VERSION,
            "source_sha256": _sha256_file(self.input_path),
            "roi": {
                "x": self.roi.x,
                "y": self.roi.y,
                "width": self.roi.width,
                "height": self.roi.height,
            },
            "reference_frame": self.reference_frame,
            "mask_padding": self.mask_padding,
            "mask_close_kernel_px": self.mask_close_kernel_px,
            "mask_blur_px": self.mask_blur_px,
            "mask_min_component_area": self.mask_min_component_area,
            "tracking_backend": tracking_backend,
            "propagation_mode": propagation_mode,
            "checkpoint_path": str(
                self.checkpoint_path or os.environ.get("SAM2_CHECKPOINT_PATH", "")
            ),
            "model_cfg": self.model_cfg or os.environ.get("SAM2_MODEL_CFG", ""),
            "sam2_image_size": self.sam2_image_size or os.environ.get("SAM2_IMAGE_SIZE", "512"),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:24]

    def _try_load_cache(self, cache_dir: Path) -> CachedMaskProvider | None:
        if self.force_cache_rebuild or not cache_dir.exists():
            return None
        provider = CachedMaskProvider(
            np=self.np,
            cv2=self.cv2,
            metadata=self.metadata,
            cache_dir=cache_dir,
        )
        try:
            provider.prepare()
        except RuntimeError as exc:
            logging.info("Ignoring invalid SAM 2 mask cache %s: %s", cache_dir, exc)
            return None
        return provider

    def _write_cache_metadata(
        self,
        frame_count: int,
        checkpoint_path: Path,
        model_cfg: str,
        device: str,
        sam2_image_size: int,
    ) -> None:
        metadata = {
            "cache_version": DEFAULT_SAM2_CACHE_VERSION,
            "cache_key": self.cache_key,
            "source_input_path": str(self.input_path),
            "source_sha256": _sha256_file(self.input_path),
            "source_frame_count": frame_count,
            "width": self.metadata.width,
            "height": self.metadata.height,
            "fps": self.metadata.fps,
            "roi": {
                "x": self.roi.x,
                "y": self.roi.y,
                "width": self.roi.width,
                "height": self.roi.height,
            },
            "reference_frame": self.reference_frame,
            "mask_padding": self.mask_padding,
            "tracking_backend": self.tracking_backend,
            "propagation_mode": self.propagation_mode,
            "checkpoint_path": str(checkpoint_path),
            "model_cfg": model_cfg,
            "device": device,
            "sam2_image_size": sam2_image_size,
            "warnings": self._warnings,
            "qa": getattr(self, "_qa_summary", None),
        }
        self.cache_metadata = metadata
        (self.cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def _write_cache_qa(self, frame_count: int) -> dict[str, object]:
        """Inspect merged masks and write a compact QA report for the cache."""

        tracker = MaskQaTracker(np=self.np)
        frames: list[dict[str, object]] = []
        warning_payloads: list[dict[str, object]] = []
        for zero_based_frame in range(frame_count):
            mask_path = self._mask_paths.get(zero_based_frame)
            if mask_path is None:
                continue
            mask = self.cv2.imread(str(mask_path), self.cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            one_based_frame = zero_based_frame + 1
            active = mask > 0
            area = int(self.np.count_nonzero(active))
            height, width = mask.shape[:2]
            frame_warnings = tracker.inspect(mask, frame_index=one_based_frame)
            frames.append(
                {
                    "frame_index": zero_based_frame,
                    "area": area,
                    "area_ratio": area / max(1, height * width),
                    "warning_codes": [warning.code for warning in frame_warnings],
                }
            )
            for warning in frame_warnings:
                warning_payloads.append(
                    {
                        "frame_index": warning.frame_index,
                        "code": warning.code,
                        "message": warning.message,
                    }
                )

        summary = {
            "frame_count": len(frames),
            "warning_count": len(warning_payloads),
            "warnings": warning_payloads,
            "frames": frames,
        }
        (self.cache_dir / "qa_scores.json").write_text(json.dumps(summary, indent=2))
        return {
            "frame_count": summary["frame_count"],
            "warning_count": summary["warning_count"],
            "qa_scores": "qa_scores.json",
        }


@dataclass
class CrowdedHumanMaskProvider(Sam2VideoMaskProvider):
    """Track one selected person with YOLO + BoT-SORT/ReID, then refine with SAM2."""

    detector_model: str | Path = DEFAULT_CROWDED_HUMAN_DETECTOR_MODEL
    reid_model: str | Path = DEFAULT_CROWDED_HUMAN_REID_MODEL
    detector_imgsz: int = DEFAULT_CROWDED_HUMAN_DETECTOR_IMGSZ
    person_conf: float = DEFAULT_CROWDED_HUMAN_PERSON_CONF
    person_iou: float = DEFAULT_CROWDED_HUMAN_PERSON_IOU
    target_roi_min_iou: float = DEFAULT_CROWDED_HUMAN_TARGET_ROI_MIN_IOU
    sam2_refine_every_n_frames: int = DEFAULT_CROWDED_HUMAN_SAM2_REFINE_EVERY_N_FRAMES
    missing_target_warning_frames: int = DEFAULT_CROWDED_HUMAN_MISSING_TARGET_WARNING_FRAMES
    neighbor_exclusion: bool = True
    neighbor_mask_erosion_px: int = DEFAULT_CROWDED_HUMAN_NEIGHBOR_MASK_EROSION_PX

    def prepare(self) -> None:
        """Precompute an identity-locked crowded-human mask timeline."""

        self._resolve_l4_hyperparameters_from_env()
        self._validate_l4_hyperparameters()
        self.tracking_backend = CROWDED_HUMAN_TRACKING_BACKEND
        self.propagation_mode = "yolo_botsort_reid_sam2_image"
        cache_root = self._resolve_cache_root()
        self.cache_key = self._build_crowded_cache_key()
        if cache_root is not None:
            cached_provider = self._try_load_cache(cache_root / self.cache_key)
            if cached_provider is not None:
                self.cache_dir = cached_provider.cache_dir
                self.cache_metadata = cached_provider.cache_metadata
                self._mask_dir = cached_provider._mask_dir
                self._mask_paths = cached_provider._mask_paths
                logging.info("Using cached crowded-human mask timeline: %s", self.cache_dir)
                return

        detector_cls, boxmot_module, torch, sam2_image_predictor_cls, build_sam2 = (
            self._import_crowded_human_dependencies()
        )
        checkpoint_path = self._resolve_checkpoint_path()
        model_cfg = self._resolve_model_cfg(checkpoint_path, SAM2_BACKEND_GLOBAL)
        device = self.device or os.environ.get("SAM2_DEVICE") or self._select_torch_device(torch)
        sam2_image_size = self._resolve_crowded_human_sam2_image_size()

        self._frame_temp = tempfile.TemporaryDirectory(prefix="crowded-human-frames-")
        self._frame_dir = Path(self._frame_temp.name)
        if cache_root is None:
            self._cache_temp = tempfile.TemporaryDirectory(prefix="crowded-human-cache-")
            self.cache_dir = Path(self._cache_temp.name)
        else:
            self.cache_dir = cache_root / self.cache_key
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mask_dir = self.cache_dir / "masks"
        self._mask_dir.mkdir(parents=True, exist_ok=True)
        self._mask_paths: dict[int, Path] = {}
        self._warnings: list[str] = []
        self._tracking_frames: list[dict[str, object]] = []
        self._mask_refiner = MorphologicalMaskRefiner(
            np=self.np,
            cv2=self.cv2,
            dilation_px=self.mask_padding,
            close_kernel_px=self.mask_close_kernel_px,
            blur_px=self.mask_blur_px,
            min_component_area=self.mask_min_component_area,
        )
        self._mask_refiner.prepare()

        frame_count = self._extract_video_frames()
        if self.reference_frame >= frame_count:
            raise RuntimeError(
                f"Reference frame {self.reference_frame} is outside the video "
                f"range. Extracted {frame_count} frames."
            )

        logging.info(
            "Preparing crowded-human masks with detector=%s, tracker=BoT-SORT/ReID, "
            "reid=%s, imgsz=%s, conf=%.2f, iou=%.2f on %s.",
            self.detector_model,
            self.reid_model,
            self.detector_imgsz,
            self.person_conf,
            self.person_iou,
            device,
        )
        detector = detector_cls(str(self.detector_model))
        tracker = self._build_botsort_tracker(
            boxmot_module=boxmot_module,
            reid_weights=self.reid_model,
            device=device,
        )
        tracks_by_frame = self._track_people(detector, tracker, frame_count)
        target_track_id = self._select_target_track_id(tracks_by_frame)
        if target_track_id is None:
            warning = (
                "No BoT-SORT person track overlapped the selected ROI; falling back to "
                "SAMURAI tracking for this job."
            )
            logging.warning(warning)
            self._warnings.append(warning)
            self._prepare_samurai_fallback(cache_root=cache_root)
            return

        sam2_model = build_sam2(
            model_cfg,
            str(checkpoint_path),
            device=device,
            hydra_overrides_extra=[],
            apply_postprocessing=False,
        )
        predictor = sam2_image_predictor_cls(sam2_model)
        self._write_crowded_human_masks(
            predictor=predictor,
            tracks_by_frame=tracks_by_frame,
            target_track_id=target_track_id,
            frame_count=frame_count,
        )

        if not self._mask_paths:
            raise RuntimeError("Crowded-human tracking did not produce any target masks.")
        self._qa_summary = self._write_cache_qa(frame_count)
        self._write_crowded_cache_metadata(
            frame_count=frame_count,
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            device=device,
            sam2_image_size=sam2_image_size,
            target_track_id=target_track_id,
        )
        logging.info("Crowded-human provider wrote %s frame masks.", len(self._mask_paths))

    def _resolve_l4_hyperparameters_from_env(self) -> None:
        self.detector_model = os.environ.get(
            "CROWDED_HUMAN_DETECTOR_MODEL", str(self.detector_model)
        )
        self.reid_model = os.environ.get("CROWDED_HUMAN_REID_MODEL", str(self.reid_model))
        self.detector_imgsz = _env_int(
            "CROWDED_HUMAN_DETECTOR_IMGSZ", self.detector_imgsz
        )
        self.person_conf = _env_float("CROWDED_HUMAN_PERSON_CONF", self.person_conf)
        self.person_iou = _env_float("CROWDED_HUMAN_PERSON_IOU", self.person_iou)
        self.target_roi_min_iou = _env_float(
            "CROWDED_HUMAN_TARGET_ROI_MIN_IOU", self.target_roi_min_iou
        )
        self.sam2_refine_every_n_frames = _env_int(
            "CROWDED_HUMAN_SAM2_REFINE_EVERY_N_FRAMES",
            self.sam2_refine_every_n_frames,
        )
        self.missing_target_warning_frames = _env_int(
            "CROWDED_HUMAN_MISSING_TARGET_WARNING_FRAMES",
            self.missing_target_warning_frames,
        )
        self.neighbor_exclusion = _env_bool(
            "CROWDED_HUMAN_NEIGHBOR_EXCLUSION", self.neighbor_exclusion
        )
        self.neighbor_mask_erosion_px = _env_int(
            "CROWDED_HUMAN_NEIGHBOR_MASK_EROSION_PX",
            self.neighbor_mask_erosion_px,
        )

    def _resolve_crowded_human_sam2_image_size(self) -> int:
        if self.sam2_image_size is not None:
            raw_image_size = str(self.sam2_image_size)
        else:
            raw_image_size = os.environ.get(
                "CROWDED_HUMAN_SAM2_IMAGE_SIZE",
                str(DEFAULT_CROWDED_HUMAN_SAM2_IMAGE_SIZE),
            )
        try:
            image_size = int(raw_image_size)
        except ValueError as exc:
            raise RuntimeError("CROWDED_HUMAN_SAM2_IMAGE_SIZE must be an integer.") from exc
        if image_size != 1024:
            raise RuntimeError(
                "Crowded-human SAM2 image refinement requires "
                "CROWDED_HUMAN_SAM2_IMAGE_SIZE=1024. Smaller SAM2 image sizes can "
                "produce mismatched image and prompt embedding grids in SAM2ImagePredictor."
            )
        return image_size

    def _validate_l4_hyperparameters(self) -> None:
        if self.detector_imgsz < 320:
            raise RuntimeError("CROWDED_HUMAN_DETECTOR_IMGSZ must be at least 320.")
        if not 0 < self.person_conf < 1:
            raise RuntimeError("CROWDED_HUMAN_PERSON_CONF must be between 0 and 1.")
        if not 0 < self.person_iou <= 1:
            raise RuntimeError("CROWDED_HUMAN_PERSON_IOU must be between 0 and 1.")
        if not 0 <= self.target_roi_min_iou <= 1:
            raise RuntimeError("CROWDED_HUMAN_TARGET_ROI_MIN_IOU must be between 0 and 1.")
        if self.sam2_refine_every_n_frames < 1:
            raise RuntimeError("CROWDED_HUMAN_SAM2_REFINE_EVERY_N_FRAMES must be >= 1.")
        if self.missing_target_warning_frames < 1:
            raise RuntimeError("CROWDED_HUMAN_MISSING_TARGET_WARNING_FRAMES must be >= 1.")
        if self.neighbor_mask_erosion_px < 0:
            raise RuntimeError("CROWDED_HUMAN_NEIGHBOR_MASK_EROSION_PX cannot be negative.")

    def _import_crowded_human_dependencies(self):
        try:
            import torch  # type: ignore[import-not-found]
            from ultralytics import YOLO  # type: ignore[import-not-found]
            import boxmot  # type: ignore[import-not-found]
            from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
            import sam2.sam2_image_predictor as sam2_image_predictor_module  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Crowded human tracking requires optional dependencies: "
                "ultralytics, boxmot, torch, and SAM2. In Colab L4 install them with "
                "`python -m pip install ultralytics boxmot`, then ensure SAM2 is installed "
                "and SAM2_CHECKPOINT_PATH points to a SAM2 checkpoint. The intended L4 "
                "models are detector=yolo11m.pt and reid=osnet_x0_25_msmt17.pt."
            ) from exc
        sam2_image_predictor_module = self._patch_sam2_image_predictor_module(
            sam2_image_predictor_module
        )
        SAM2ImagePredictor = sam2_image_predictor_module.SAM2ImagePredictor
        return YOLO, boxmot, torch, SAM2ImagePredictor, build_sam2

    def _patch_sam2_image_predictor_module(self, module):
        """Patch upstream SAM2 for PyTorch stacks where permute(...).view(...) fails."""
        module_file = Path(str(getattr(module, "__file__", "")))
        if not module_file.exists():
            return module

        old = "feat.permute(1, 2, 0).view(1, -1, *feat_size)"
        new = "feat.permute(1, 2, 0).reshape(1, -1, *feat_size)"
        try:
            source = module_file.read_text()
        except OSError as exc:
            logging.warning("Could not read SAM2 image predictor for compatibility patch: %s", exc)
            return module

        if old not in source:
            return module

        try:
            module_file.write_text(source.replace(old, new))
        except OSError as exc:
            logging.warning("Could not patch SAM2 image predictor for Colab compatibility: %s", exc)
            return module

        logging.info(
            "Patched SAM2 image predictor for Colab/PyTorch compatibility: %s",
            module_file,
        )
        importlib.invalidate_caches()
        return importlib.reload(module)

    def _build_botsort_tracker(self, boxmot_module, reid_weights: str | Path, device: str):
        tracker_classes = self._resolve_boxmot_botsort_classes(boxmot_module)
        create_tracker_candidates = self._resolve_boxmot_create_tracker_functions(boxmot_module)
        half = str(device).startswith("cuda")
        device_candidates = [device]
        if device == "cuda":
            device_candidates.append("cuda:0")
        reid_candidates = [Path(str(reid_weights)), str(reid_weights)]
        last_error: Exception | None = None

        tracker_config = self._resolve_boxmot_botsort_config(boxmot_module)
        for create_tracker in create_tracker_candidates:
            for resolved_device in device_candidates:
                for resolved_reid in reid_candidates:
                    for kwargs in (
                        {
                            "tracker_type": "botsort",
                            "tracker_config": tracker_config,
                            "reid_weights": resolved_reid,
                            "device": resolved_device,
                            "half": half,
                            "per_class": False,
                        },
                        {
                            "tracker_type": "botsort",
                            "tracker_config": tracker_config,
                            "reid_weights": resolved_reid,
                            "device": resolved_device,
                            "half": half,
                        },
                        {
                            "tracker_name": "botsort",
                            "tracker_config": tracker_config,
                            "reid_weights": resolved_reid,
                            "device": resolved_device,
                            "half": half,
                        },
                    ):
                        try:
                            return self._validate_boxmot_tracker(create_tracker(**kwargs))
                        except Exception as exc:  # pragma: no cover - depends on boxmot versions
                            last_error = exc
                    try:
                        return self._validate_boxmot_tracker(
                            create_tracker(
                                "botsort",
                                tracker_config,
                                resolved_reid,
                                resolved_device,
                                half,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - depends on boxmot versions
                        last_error = exc
                    try:
                        return self._validate_boxmot_tracker(
                            create_tracker(
                                "botsort",
                                tracker_config=tracker_config,
                                reid_weights=resolved_reid,
                                device=resolved_device,
                                half=half,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - depends on boxmot versions
                        last_error = exc

        for tracker_cls in tracker_classes:
            for resolved_device in device_candidates:
                for resolved_reid in reid_candidates:
                    reid_model = None
                    try:
                        reid_model = self._build_boxmot_reid_model(
                            resolved_reid,
                            resolved_device,
                            half,
                        )
                    except Exception as exc:  # pragma: no cover - depends on boxmot versions
                        last_error = exc

                    for kwargs in (
                        {
                            "reid_model": reid_model,
                            "device": resolved_device,
                            "half": half,
                            "per_class": False,
                        },
                        {
                            "reid_model": reid_model,
                            "per_class": False,
                        },
                        {
                            "reid_weights": resolved_reid,
                            "device": resolved_device,
                            "half": half,
                            "per_class": False,
                        },
                        {
                            "reid_weights": resolved_reid,
                            "device": resolved_device,
                            "half": half,
                        },
                        {
                            "model_weights": resolved_reid,
                            "device": resolved_device,
                            "half": half,
                        },
                    ):
                        if kwargs.get("reid_model") is None and "reid_model" in kwargs:
                            continue
                        try:
                            tracker = tracker_cls(**kwargs)
                            return self._validate_boxmot_tracker(tracker)
                        except Exception as exc:  # pragma: no cover - depends on boxmot versions
                            last_error = exc
                            continue

        detail = f" Last BoxMOT error: {last_error}" if last_error else ""
        if not detail:
            detail = (
                f" Resolved tracker factories={len(create_tracker_candidates)} "
                f"and BoT-SORT classes={len(tracker_classes)}."
            )
        raise RuntimeError(
            "Could not initialize BoxMOT BoT-SORT. Confirm that `boxmot` is installed "
            "and that the ReID weight is available. Recommended L4 ReID model: "
            f"{DEFAULT_CROWDED_HUMAN_REID_MODEL}.{detail}"
        ) from last_error

    def _validate_boxmot_tracker(self, tracker):
        with_reid = bool(getattr(tracker, "with_reid", False))
        reid_model = getattr(tracker, "model", None) or getattr(tracker, "reid_model", None)
        if with_reid and reid_model is None:
            raise RuntimeError(
                "BoxMOT BoT-SORT initialized without a ReID model. This usually means "
                "the raw BotSort class ignored `reid_weights`; use "
                "`boxmot.trackers.tracker_zoo.create_tracker` or pass `reid_model`."
            )
        if reid_model is not None and hasattr(reid_model, "warmup"):
            reid_model.warmup()
        return tracker

    def _build_boxmot_reid_model(self, reid_weights: str | Path, device: str, half: bool):
        reid_cls = None
        for module_name in ("boxmot.reid.core", "boxmot.reid"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            reid_cls = getattr(module, "ReID", None)
            if reid_cls is not None:
                break
        if reid_cls is None:
            raise RuntimeError("Could not resolve BoxMOT ReID class.")
        return reid_cls(weights=reid_weights, device=device, half=half).model

    def _resolve_boxmot_botsort_classes(self, boxmot_module) -> list[object]:
        classes: list[object] = []
        for name in ("BotSort", "BoTSORT", "BOTSORT"):
            candidate = getattr(boxmot_module, name, None)
            if candidate is not None:
                classes.append(candidate)

        for module_name in (
            "boxmot.trackers.bbox.botsort.botsort",
            "boxmot.trackers.bbox.botsort",
            "boxmot.trackers.botsort.bot_sort",
            "boxmot.trackers.botsort.botsort",
            "boxmot.trackers.botsort",
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for name in ("BotSort", "BoTSORT", "BOTSORT"):
                candidate = getattr(module, name, None)
                if candidate is not None and candidate not in classes:
                    classes.append(candidate)
        return classes

    def _resolve_boxmot_create_tracker_functions(self, boxmot_module) -> list[object]:
        functions: list[object] = []
        candidate = getattr(boxmot_module, "create_tracker", None)
        if candidate is not None:
            functions.append(candidate)
        for module_name in (
            "boxmot.trackers.tracker_zoo",
            "boxmot.tracker_zoo",
            "boxmot.tracker_zoo.tracker_zoo",
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            candidate = getattr(module, "create_tracker", None)
            if candidate is not None and candidate not in functions:
                functions.append(candidate)
        return functions

    def _resolve_boxmot_botsort_config(self, boxmot_module) -> Path | None:
        module_file = getattr(boxmot_module, "__file__", None)
        if not module_file:
            return None
        boxmot_root = Path(module_file).resolve().parent
        for candidate in (
            boxmot_root / "configs" / "botsort.yaml",
            boxmot_root / "configs" / "trackers" / "botsort.yaml",
            boxmot_root / "trackers" / "botsort" / "configs" / "botsort.yaml",
            boxmot_root / "trackers" / "botsort" / "botsort.yaml",
            boxmot_root / "trackers" / "botsort" / "config.yaml",
        ):
            if candidate.exists():
                return candidate
        return None

    def _track_people(self, detector, tracker, frame_count: int) -> list[dict[int, dict[str, object]]]:
        tracks_by_frame: list[dict[int, dict[str, object]]] = []
        for frame_idx in range(frame_count):
            frame = self._read_extracted_frame(frame_idx)
            detections = self._detect_people(detector, frame)
            tracks = tracker.update(detections, frame)
            tracks_by_frame.append(self._parse_tracks(tracks))
        return tracks_by_frame

    def _detect_people(self, detector, frame):
        results = detector.predict(
            frame,
            classes=[0],
            conf=self.person_conf,
            iou=self.person_iou,
            imgsz=self.detector_imgsz,
            verbose=False,
        )
        if not results:
            return self.np.empty((0, 6), dtype=self.np.float32)
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return self.np.empty((0, 6), dtype=self.np.float32)
        xyxy = self._to_numpy(boxes.xyxy)
        if xyxy.size == 0:
            return self.np.empty((0, 6), dtype=self.np.float32)
        conf = self._to_numpy(boxes.conf).reshape(-1, 1)
        cls = self._to_numpy(boxes.cls).reshape(-1, 1)
        return self.np.concatenate([xyxy, conf, cls], axis=1).astype(self.np.float32)

    def _parse_tracks(self, tracks) -> dict[int, dict[str, object]]:
        array = self._to_numpy(tracks)
        if array.size == 0:
            return {}
        if array.ndim == 1:
            array = array.reshape(1, -1)
        parsed: dict[int, dict[str, object]] = {}
        for row in array:
            if len(row) < 5:
                continue
            track_id = int(row[4])
            confidence = float(row[5]) if len(row) > 5 else 0.0
            class_id = int(row[6]) if len(row) > 6 else 0
            if class_id != 0:
                continue
            box = [float(value) for value in row[:4]]
            parsed[track_id] = {
                "box": box,
                "confidence": confidence,
                "class_id": class_id,
            }
        return parsed

    def _select_target_track_id(
        self,
        tracks_by_frame: list[dict[int, dict[str, object]]],
    ) -> int | None:
        if self.reference_frame >= len(tracks_by_frame):
            return None
        roi_box = [
            float(self.roi.x),
            float(self.roi.y),
            float(self.roi.x + self.roi.width),
            float(self.roi.y + self.roi.height),
        ]
        best_track_id = None
        best_iou = 0.0
        for track_id, track in tracks_by_frame[self.reference_frame].items():
            iou = _box_iou(roi_box, track["box"])
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        if best_iou < self.target_roi_min_iou:
            return None
        return best_track_id

    def _write_crowded_human_masks(
        self,
        predictor,
        tracks_by_frame: list[dict[int, dict[str, object]]],
        target_track_id: int,
        frame_count: int,
    ) -> None:
        last_valid_box: list[float] | None = None
        last_valid_mask = None
        missing_streak = 0
        for frame_idx in range(frame_count):
            frame = self._read_extracted_frame(frame_idx)
            tracks = tracks_by_frame[frame_idx]
            target_track = tracks.get(target_track_id)
            low_confidence = target_track is None
            if target_track is not None:
                target_box = target_track["box"]
                last_valid_box = list(target_box)
                missing_streak = 0
            elif last_valid_box is not None:
                target_box = last_valid_box
                missing_streak += 1
            else:
                target_box = None
                missing_streak += 1

            warning_codes: list[str] = []
            if low_confidence:
                warning_codes.append("target_track_missing")
            if missing_streak >= self.missing_target_warning_frames:
                warning_codes.append("target_missing_streak")

            if target_box is None:
                mask = self.np.zeros((self.metadata.height, self.metadata.width), dtype=self.np.uint8)
            elif (
                last_valid_mask is not None
                and self.sam2_refine_every_n_frames > 1
                and frame_idx % self.sam2_refine_every_n_frames != 0
                and not low_confidence
            ):
                mask = last_valid_mask.copy()
            else:
                mask = self._segment_box_with_sam2(predictor, frame, target_box)
                last_valid_mask = mask.copy()

            original_area = int(self.np.count_nonzero(mask > 0))
            excluded_neighbors = 0
            if self.neighbor_exclusion and target_box is not None:
                neighbor_mask, excluded_neighbors = self._build_neighbor_mask(
                    tracks=tracks,
                    target_track_id=target_track_id,
                    shape=mask.shape,
                )
                if excluded_neighbors:
                    mask = self.np.where(neighbor_mask > 0, 0, mask).astype(self.np.uint8)
                    reduced_area = int(self.np.count_nonzero(mask > 0))
                    if original_area > 0 and reduced_area < original_area * 0.6:
                        warning_codes.append("high_neighbor_overlap")

            mask = self._postprocess_mask(mask)
            mask_path = self._mask_dir / f"{frame_idx:06d}.png"
            if not self.cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"Could not write crowded-human mask: {mask_path}")
            self._mask_paths[frame_idx] = mask_path
            self._tracking_frames.append(
                {
                    "frame_index": frame_idx,
                    "target_track_id": target_track_id,
                    "tracking_mode": "botsort_high_conf"
                    if not low_confidence
                    else "botsort_last_known_box",
                    "low_confidence": bool(low_confidence),
                    "missing_streak": missing_streak,
                    "n_detected_persons": len(tracks),
                    "n_excluded_neighbors": excluded_neighbors,
                    "target_box": target_box,
                    "warning_codes": warning_codes,
                }
            )

    def _segment_box_with_sam2(self, predictor, frame, box: list[float]):
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        box_array = self.np.asarray(box, dtype=self.np.float32)
        masks, _, _ = predictor.predict(box=box_array[None, :], multimask_output=False)
        mask = self.np.squeeze(masks[0]).astype(self.np.uint8) * 255
        if mask.shape != (self.metadata.height, self.metadata.width):
            mask = self.cv2.resize(
                mask,
                (self.metadata.width, self.metadata.height),
                interpolation=self.cv2.INTER_NEAREST,
            )
        return mask

    def _build_neighbor_mask(
        self,
        tracks: dict[int, dict[str, object]],
        target_track_id: int,
        shape: tuple[int, int],
    ) -> tuple[object, int]:
        neighbor_mask = self.np.zeros(shape, dtype=self.np.uint8)
        excluded = 0
        for track_id, track in tracks.items():
            if track_id == target_track_id:
                continue
            x1, y1, x2, y2 = _clip_box(track["box"], self.metadata.width, self.metadata.height)
            if x2 <= x1 or y2 <= y1:
                continue
            neighbor_mask[y1:y2, x1:x2] = 255
            excluded += 1
        if excluded and self.neighbor_mask_erosion_px > 0:
            kernel_size = self.neighbor_mask_erosion_px * 2 + 1
            kernel = self.cv2.getStructuringElement(
                self.cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            neighbor_mask = self.cv2.erode(neighbor_mask, kernel, iterations=1)
        return neighbor_mask, excluded

    def _read_extracted_frame(self, frame_idx: int):
        frame_path = self._frame_dir / f"{frame_idx:05d}.jpg"
        frame = self.cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read extracted crowded-human frame: {frame_path}")
        return frame

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return self.np.asarray(value)

    def _prepare_samurai_fallback(self, cache_root: Path | None) -> None:
        fallback = Sam2VideoMaskProvider(
            np=self.np,
            cv2=self.cv2,
            input_path=self.input_path,
            metadata=self.metadata,
            roi=self.roi,
            reference_frame=self.reference_frame,
            mask_padding=self.mask_padding,
            checkpoint_path=self.checkpoint_path,
            model_cfg=self.model_cfg,
            device=self.device,
            sam2_image_size=self.sam2_image_size,
            mask_close_kernel_px=self.mask_close_kernel_px,
            mask_blur_px=self.mask_blur_px,
            mask_min_component_area=self.mask_min_component_area,
            sam2_tracking_backend=SAM2_BACKEND_SAMURAI,
            bidirectional=self.bidirectional,
            mask_cache_dir=cache_root,
            force_cache_rebuild=self.force_cache_rebuild,
        )
        fallback.prepare()
        self.cache_dir = fallback.cache_dir
        self.cache_key = fallback.cache_key
        self.cache_metadata = fallback.cache_metadata
        self._mask_dir = fallback._mask_dir
        self._mask_paths = fallback._mask_paths
        self.tracking_backend = f"{CROWDED_HUMAN_TRACKING_BACKEND}_fallback_samurai"
        self.propagation_mode = fallback.propagation_mode

    def _build_crowded_cache_key(self) -> str:
        payload = {
            "cache_version": DEFAULT_SAM2_CACHE_VERSION,
            "source_sha256": _sha256_file(self.input_path),
            "roi": {
                "x": self.roi.x,
                "y": self.roi.y,
                "width": self.roi.width,
                "height": self.roi.height,
            },
            "reference_frame": self.reference_frame,
            "mask_padding": self.mask_padding,
            "tracking_backend": CROWDED_HUMAN_TRACKING_BACKEND,
            "detector_model": str(self.detector_model),
            "reid_model": str(self.reid_model),
            "detector_imgsz": self.detector_imgsz,
            "person_conf": self.person_conf,
            "person_iou": self.person_iou,
            "target_roi_min_iou": self.target_roi_min_iou,
            "crowded_human_sam2_image_size": self._resolve_crowded_human_sam2_image_size(),
            "sam2_refine_every_n_frames": self.sam2_refine_every_n_frames,
            "neighbor_exclusion": self.neighbor_exclusion,
            "neighbor_mask_erosion_px": self.neighbor_mask_erosion_px,
            "checkpoint_path": str(
                self.checkpoint_path or os.environ.get("SAM2_CHECKPOINT_PATH", "")
            ),
            "model_cfg": self.model_cfg or os.environ.get("SAM2_MODEL_CFG", ""),
            "sam2_image_size": self._resolve_crowded_human_sam2_image_size(),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:24]

    def _write_crowded_cache_metadata(
        self,
        frame_count: int,
        checkpoint_path: Path,
        model_cfg: str,
        device: str,
        sam2_image_size: int,
        target_track_id: int,
    ) -> None:
        metadata = {
            "cache_version": DEFAULT_SAM2_CACHE_VERSION,
            "cache_key": self.cache_key,
            "source_input_path": str(self.input_path),
            "source_sha256": _sha256_file(self.input_path),
            "source_frame_count": frame_count,
            "width": self.metadata.width,
            "height": self.metadata.height,
            "fps": self.metadata.fps,
            "roi": {
                "x": self.roi.x,
                "y": self.roi.y,
                "width": self.roi.width,
                "height": self.roi.height,
            },
            "reference_frame": self.reference_frame,
            "mask_padding": self.mask_padding,
            "tracking_backend": self.tracking_backend,
            "propagation_mode": self.propagation_mode,
            "detector_model": str(self.detector_model),
            "reid_model": str(self.reid_model),
            "detector_imgsz": self.detector_imgsz,
            "person_conf": self.person_conf,
            "person_iou": self.person_iou,
            "target_roi_min_iou": self.target_roi_min_iou,
            "target_track_id": target_track_id,
            "sam2_refine_every_n_frames": self.sam2_refine_every_n_frames,
            "missing_target_warning_frames": self.missing_target_warning_frames,
            "neighbor_exclusion": self.neighbor_exclusion,
            "neighbor_mask_erosion_px": self.neighbor_mask_erosion_px,
            "checkpoint_path": str(checkpoint_path),
            "model_cfg": model_cfg,
            "device": device,
            "sam2_image_size": sam2_image_size,
            "warnings": self._warnings,
            "tracking_frames": self._tracking_frames,
            "qa": getattr(self, "_qa_summary", None),
        }
        self.cache_metadata = metadata
        (self.cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def build_mask_provider(
    removal_mode: str,
    np,
    cv2,
    input_path: Path,
    metadata: VideoMetadata,
    roi: Roi,
    reference_frame: int,
    mask_padding: int,
    sam2_tracking_backend: str | None = None,
    sam2_bidirectional: bool | None = None,
    mask_cache_dir: Path | None = None,
    force_cache_rebuild: bool = False,
) -> MaskProvider:
    """Create the mask provider requested by the job configuration."""

    if removal_mode == STATIC_RECTANGLE:
        return StaticRectangleMaskProvider(
            np=np,
            cv2=cv2,
            metadata=metadata,
            roi=roi,
            mask_padding=mask_padding,
        )
    if removal_mode == AI_HUMAN_CROWDED:
        return CrowdedHumanMaskProvider(
            np=np,
            cv2=cv2,
            input_path=input_path,
            metadata=metadata,
            roi=roi,
            reference_frame=reference_frame,
            mask_padding=mask_padding,
            sam2_tracking_backend=sam2_tracking_backend,
            bidirectional=sam2_bidirectional,
            mask_cache_dir=mask_cache_dir,
            force_cache_rebuild=force_cache_rebuild,
        )
    if removal_mode in {AI_OBJECT, AI_TEXT_OR_LOGO}:
        return Sam2VideoMaskProvider(
            np=np,
            cv2=cv2,
            input_path=input_path,
            metadata=metadata,
            roi=roi,
            reference_frame=reference_frame,
            mask_padding=mask_padding,
            sam2_tracking_backend=sam2_tracking_backend,
            bidirectional=sam2_bidirectional,
            mask_cache_dir=mask_cache_dir,
            force_cache_rebuild=force_cache_rebuild,
        )
    raise ValueError(f"Unsupported removal_mode: {removal_mode}")


def precompute_mask_cache(
    *,
    removal_mode: str,
    np,
    cv2,
    input_path: Path,
    metadata: VideoMetadata,
    roi: Roi,
    reference_frame: int,
    mask_padding: int,
    mask_cache_dir: Path,
    sam2_tracking_backend: str | None = None,
    sam2_bidirectional: bool | None = None,
    force_cache_rebuild: bool = False,
) -> CachedMaskProvider:
    """Precompute and return a reusable full-video mask cache provider."""

    provider = build_mask_provider(
        removal_mode=removal_mode,
        np=np,
        cv2=cv2,
        input_path=input_path,
        metadata=metadata,
        roi=roi,
        reference_frame=reference_frame,
        mask_padding=mask_padding,
        sam2_tracking_backend=sam2_tracking_backend,
        sam2_bidirectional=sam2_bidirectional,
        mask_cache_dir=mask_cache_dir,
        force_cache_rebuild=force_cache_rebuild,
    )
    if not isinstance(provider, Sam2VideoMaskProvider):
        raise RuntimeError("Mask cache precompute is only supported for SAM 2 removal modes.")
    provider.prepare()
    return CachedMaskProvider(
        np=np,
        cv2=cv2,
        metadata=metadata,
        cache_dir=provider.cache_dir,
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number.") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _box_iou(box_a: list[float], box_b: object) -> float:
    b = [float(value) for value in box_b]  # type: ignore[operator]
    x1 = max(float(box_a[0]), b[0])
    y1 = max(float(box_a[1]), b[1])
    x2 = min(float(box_a[2]), b[2])
    y2 = min(float(box_a[3]), b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _clip_box(box: object, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]  # type: ignore[operator]
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
