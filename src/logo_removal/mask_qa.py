from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskQaConfig:
    """Thresholds for detecting likely mask propagation failures."""

    min_area_ratio: float = 0.0001
    max_area_jump_ratio: float = 0.5
    max_missing_frames: int = 3
    max_centroid_jump_ratio: float = 0.25


@dataclass(frozen=True)
class MaskQaWarning:
    """One warning produced while inspecting a mask sequence."""

    frame_index: int
    code: str
    message: str


class MaskQaTracker:
    """Track mask stability across a video and flag suspicious frames."""

    def __init__(self, np, config: MaskQaConfig | None = None) -> None:
        self.np = np
        self.config = config or MaskQaConfig()
        self.warnings: list[MaskQaWarning] = []
        self._previous_area: int | None = None
        self._previous_centroid: tuple[float, float] | None = None
        self._missing_streak = 0

    def inspect(self, mask, frame_index: int) -> list[MaskQaWarning]:
        """Inspect one frame mask and return warnings emitted for this frame."""

        frame_warnings: list[MaskQaWarning] = []
        height, width = mask.shape[:2]
        frame_area = max(1, height * width)
        active_pixels = mask > 0
        area = int(self.np.count_nonzero(active_pixels))
        area_ratio = area / frame_area
        centroid = self._centroid(active_pixels)

        if area == 0:
            self._missing_streak += 1
            frame_warnings.append(
                self._warning(
                    frame_index,
                    "empty_mask",
                    f"Mask is empty at frame {frame_index}.",
                )
            )
        elif area_ratio < self.config.min_area_ratio:
            self._missing_streak += 1
            frame_warnings.append(
                self._warning(
                    frame_index,
                    "tiny_mask",
                    f"Mask area ratio {area_ratio:.6f} is below threshold "
                    f"{self.config.min_area_ratio:.6f}.",
                )
            )
        else:
            self._missing_streak = 0

        if self._missing_streak > self.config.max_missing_frames:
            frame_warnings.append(
                self._warning(
                    frame_index,
                    "missing_mask_streak",
                    f"Mask has been missing or tiny for {self._missing_streak} frames.",
                )
            )

        if self._previous_area and area > 0:
            area_jump = abs(area - self._previous_area) / self._previous_area
            if area_jump > self.config.max_area_jump_ratio:
                frame_warnings.append(
                    self._warning(
                        frame_index,
                        "area_jump",
                        f"Mask area changed by {area_jump:.2f}, exceeding "
                        f"{self.config.max_area_jump_ratio:.2f}.",
                    )
                )

        if self._previous_centroid is not None and centroid is not None:
            max_dimension = max(1, width, height)
            centroid_jump = self._distance(self._previous_centroid, centroid) / max_dimension
            if centroid_jump > self.config.max_centroid_jump_ratio:
                frame_warnings.append(
                    self._warning(
                        frame_index,
                        "centroid_jump",
                        f"Mask centroid moved by {centroid_jump:.2f} of frame max dimension, "
                        f"exceeding {self.config.max_centroid_jump_ratio:.2f}.",
                    )
                )

        self._previous_area = area
        self._previous_centroid = centroid
        self.warnings.extend(frame_warnings)
        return frame_warnings

    def _centroid(self, active_pixels) -> tuple[float, float] | None:
        ys, xs = self.np.nonzero(active_pixels)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    def _distance(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        dx = first[0] - second[0]
        dy = first[1] - second[1]
        return float((dx * dx + dy * dy) ** 0.5)

    def _warning(self, frame_index: int, code: str, message: str) -> MaskQaWarning:
        return MaskQaWarning(frame_index=frame_index, code=code, message=message)
