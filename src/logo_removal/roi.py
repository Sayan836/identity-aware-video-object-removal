from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Roi:
    """Represent a rectangular region of interest inside a video frame."""

    x: int
    y: int
    width: int
    height: int

    def padded(self, padding: int, frame_width: int, frame_height: int) -> "Roi":
        """Return a new ROI expanded by the given padding and clamped to the frame."""

        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(frame_width, self.x + self.width + padding)
        y2 = min(frame_height, self.y + self.height + padding)
        return Roi(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def validate_inside(self, frame_width: int, frame_height: int) -> None:
        """Ensure this ROI is positive-sized and fully contained within the frame."""

        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be greater than zero")
        if self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ValueError(
                f"ROI {self.as_csv()} exceeds frame bounds {frame_width}x{frame_height}"
            )

    def as_csv(self) -> str:
        """Serialize the ROI to the x,y,width,height format used by the CLI."""

        return f"{self.x},{self.y},{self.width},{self.height}"


def parse_roi(raw: str) -> Roi:
    """Parse an ROI string from the CLI into a validated ``Roi`` instance."""

    parts = raw.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must use x,y,width,height format")
    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    roi = Roi(x=x, y=y, width=width, height=height)
    if roi.width <= 0 or roi.height <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be greater than zero")
    return roi
