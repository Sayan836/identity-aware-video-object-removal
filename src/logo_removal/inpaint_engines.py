from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


OPENCV_ENGINE = "opencv"
LAMA_ENGINE = "lama"
VALID_INPAINT_ENGINES = {OPENCV_ENGINE, LAMA_ENGINE}


class InpaintEngine:
    """Base interface for replacing masked frame regions."""

    def prepare(self) -> None:
        """Load any model or runtime resources needed before processing frames."""

    def inpaint(self, frame, mask):
        """Return a processed frame with the masked region removed."""

        raise NotImplementedError


@dataclass
class OpenCvInpaintEngine(InpaintEngine):
    """Fast OpenCV inpainting engine using TELEA or Navier-Stokes."""

    cv2: object
    method: str
    radius: float

    def prepare(self) -> None:
        """Validate the configured OpenCV inpainting method."""

        if self.method not in {"telea", "ns"}:
            raise ValueError("OpenCV method must be 'telea' or 'ns'.")

    def inpaint(self, frame, mask):
        """Inpaint one frame using OpenCV's classical algorithms."""

        inpaint_flag = self.cv2.INPAINT_TELEA if self.method == "telea" else self.cv2.INPAINT_NS
        return self.cv2.inpaint(frame, mask, self.radius, inpaint_flag)


class LamaInpaintEngine(InpaintEngine):
    """Optional LaMa-based frame inpainting engine."""

    def prepare(self) -> None:
        """Load the Simple LaMa model for frame-by-frame image inpainting."""

        try:
            import torch  # type: ignore[import-not-found]
            self._prepare_model_cache()
            from simple_lama_inpainting import SimpleLama  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "AI LaMa inpainting is not installed. Use 'Fast OpenCV' for now, "
                "or install/configure simple-lama-inpainting before selecting AI LaMa."
            ) from exc

        device = self._select_torch_device(torch)
        self._simple_lama = SimpleLama(device=device)

    def inpaint(self, frame, mask):
        """Inpaint one BGR OpenCV frame with Simple LaMa and return BGR output."""

        if not hasattr(self, "_simple_lama"):
            raise RuntimeError("LaMa inpainting was used before prepare() completed.")

        import numpy as np  # type: ignore[import-not-found]

        rgb_frame = frame[:, :, ::-1]
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        result = self._simple_lama(rgb_frame, binary_mask)
        result_array = np.asarray(result)
        if result_array.ndim == 2:
            result_array = np.stack([result_array] * 3, axis=-1)
        return result_array[:, :, :3][:, :, ::-1].copy()

    def _select_torch_device(self, torch):
        """Select a Torch device for LaMa, using CPU when accelerators are unavailable."""

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _prepare_model_cache(self) -> None:
        """Point torch downloads at the project's local model cache by default."""

        cache_root = Path(
            os.environ.get(
                "LAMA_MODEL_CACHE_DIR",
                Path(__file__).resolve().parents[2] / "models" / "lama_cache",
            )
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_root))


def build_inpaint_engine(inpaint_engine: str, cv2, method: str, radius: float) -> InpaintEngine:
    """Create the requested inpainting engine."""

    if inpaint_engine == OPENCV_ENGINE:
        return OpenCvInpaintEngine(cv2=cv2, method=method, radius=radius)
    if inpaint_engine == LAMA_ENGINE:
        return LamaInpaintEngine()
    raise ValueError(f"Unsupported inpaint_engine: {inpaint_engine}")
