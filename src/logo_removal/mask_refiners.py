from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class MaskRefiner(Protocol):
    """Refines binary removal masks before inpainting."""

    def prepare(self) -> None:
        """Load any optional model resources required by the refiner."""

    def refine(self, mask, frame, frame_index: int):
        """Return a refined mask for the given frame."""


@dataclass
class MorphologicalMaskRefiner:
    """Clean and expand masks using deterministic OpenCV morphology."""

    np: object
    cv2: object
    dilation_px: int = 0
    close_kernel_px: int = 3
    blur_px: int = 0
    min_component_area: int = 0

    def prepare(self) -> None:
        """Validate refinement settings."""

        if self.dilation_px < 0:
            raise ValueError("dilation_px cannot be negative.")
        if self.close_kernel_px < 0:
            raise ValueError("close_kernel_px cannot be negative.")
        if self.blur_px < 0:
            raise ValueError("blur_px cannot be negative.")
        if self.min_component_area < 0:
            raise ValueError("min_component_area cannot be negative.")

    def refine(self, mask, frame, frame_index: int):
        """Apply thresholding, cleanup, dilation, and optional edge softening."""

        refined = self.np.where(mask > 0, 255, 0).astype(self.np.uint8)
        refined = self._remove_small_components(refined)
        refined = self._close(refined)
        refined = self._dilate(refined)
        refined = self._blur(refined)
        return refined

    def _remove_small_components(self, mask):
        if self.min_component_area <= 0:
            return mask

        component_count, labels, stats, _ = self.cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        cleaned = self.np.zeros_like(mask)
        area_column = self.cv2.CC_STAT_AREA
        for component_index in range(1, component_count):
            if stats[component_index, area_column] >= self.min_component_area:
                cleaned[labels == component_index] = 255
        return cleaned

    def _close(self, mask):
        kernel_size = self._odd_kernel_size(self.close_kernel_px)
        if kernel_size <= 1:
            return mask
        kernel = self.cv2.getStructuringElement(
            self.cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        return self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)

    def _dilate(self, mask):
        if self.dilation_px <= 0:
            return mask
        kernel_size = (self.dilation_px * 2) + 1
        kernel = self.cv2.getStructuringElement(
            self.cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        return self.cv2.dilate(mask, kernel, iterations=1)

    def _blur(self, mask):
        kernel_size = self._odd_kernel_size(self.blur_px)
        if kernel_size <= 1:
            return mask
        return self.cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    def _odd_kernel_size(self, value: int) -> int:
        if value <= 1:
            return value
        return value if value % 2 == 1 else value + 1


@dataclass
class BiRefNetMaskRefiner:
    """Optional learned mask boundary refiner placeholder."""

    model_path: Path | None = None
    device: str | None = None

    def prepare(self) -> None:
        """Fail clearly until BiRefNet dependencies and weights are configured."""

        raise RuntimeError(
            "BiRefNet mask refinement is not configured yet. Use the morphology "
            "refiner for now, or install BiRefNet and provide a model path before "
            "selecting this refiner."
        )

    def refine(self, mask, frame, frame_index: int):
        """Return a refined mask."""

        raise RuntimeError("BiRefNet mask refinement was used before prepare() completed.")
