"""Create a vertical before/after comparison video.

Example:
    python video_edit.py \
        --before input.mp4 \
        --after output.mp4 \
        --out comparison.mp4 \
        --circle 520,340,36 \
        --slow-factor 2.4 \
        --canvas 1080x1350 \
        --mark-every 1.5 \
        --mark-duration 0.45

For moving objects, use a CSV marks file with a header:
    time,x,y,r,label
    0.0,520,340,36,Removed object
    1.5,610,360,34,Removed object

or for box marks:
    time,x,y,w,h,label
    0.0,480,300,90,80,Removed object
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MARK_COLOR = "#ffcc00"
DEFAULT_TEXT_COLOR = "#ffffff"

cv2 = None
np = None


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class Marker:
    time_seconds: float | None = None
    frame_index: int | None = None
    x: float = 0.0
    y: float = 0.0
    radius: float | None = None
    width: float | None = None
    height: float | None = None
    label: str = "Removed object"


class FrameSampler:
    """Read frames by timeline position while avoiding unnecessary seeks."""

    def __init__(self, path: Path, meta: VideoMeta):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.meta = meta
        self.next_frame_index = 0

    def read_at(self, time_seconds: float):
        target_index = int(round(time_seconds * self.meta.fps))
        if self.meta.frame_count > 0:
            target_index = min(max(target_index, 0), self.meta.frame_count - 1)

        if target_index != self.next_frame_index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            self.next_frame_index = target_index

        ok, frame = self.capture.read()
        if not ok:
            return None
        self.next_frame_index += 1
        return frame

    def release(self) -> None:
        self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stack a before video above an after video and mark the removed "
            "object on selected before frames."
        )
    )
    parser.add_argument(
        "--before",
        "--input-video",
        dest="before_path",
        required=True,
        help="Original/input video path.",
    )
    parser.add_argument(
        "--after",
        "--removed-video",
        "--output-video",
        dest="after_path",
        required=True,
        help="Processed/output video path after object removal.",
    )
    parser.add_argument(
        "--out",
        "--comparison-output",
        dest="output_path",
        required=True,
        help="Path for the generated comparison video.",
    )
    parser.add_argument(
        "--circle",
        help="Static circle marker in original before-video coordinates: x,y,r.",
    )
    parser.add_argument(
        "--box",
        help="Static box marker in original before-video coordinates: x,y,w,h.",
    )
    parser.add_argument(
        "--marks-file",
        type=Path,
        help=(
            "Optional CSV for moving objects. Use columns time,x,y,r or "
            "frame,x,y,r. For boxes use time,x,y,w,h or frame,x,y,w,h."
        ),
    )
    parser.add_argument(
        "--mark-every",
        type=float,
        default=1.5,
        help="Seconds between static marker flashes. Use 0 to show continuously.",
    )
    parser.add_argument(
        "--mark-duration",
        type=float,
        default=0.45,
        help="Seconds each marker flash stays visible.",
    )
    parser.add_argument(
        "--marker-label",
        default="Removed object",
        help="Text shown next to the marker.",
    )
    parser.add_argument(
        "--before-label",
        default="BEFORE: target object marked",
        help="Label drawn on the top video.",
    )
    parser.add_argument(
        "--after-label",
        default="AFTER: object removed",
        help="Label drawn on the bottom video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help=(
            "Output canvas width in pixels when --canvas is not used. "
            "Heights are scaled proportionally."
        ),
    )
    parser.add_argument(
        "--canvas",
        help=(
            "Fixed output canvas, for example 1080x1350. This keeps the "
            "stacked video from becoming too tall by fitting each panel into "
            "half the canvas."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Output FPS. Defaults to the before video's FPS.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Optional maximum source duration in seconds before slow motion.",
    )
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=1.0,
        help="Slow motion multiplier. For 5 seconds to 12 seconds, use 2.4.",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        help=(
            "Desired final output duration in seconds. Overrides --slow-factor "
            "after applying --max-duration."
        ),
    )
    parser.add_argument(
        "--mark-color",
        default=DEFAULT_MARK_COLOR,
        help="Marker color as hex RGB, for example #ffcc00.",
    )
    parser.add_argument(
        "--text-color",
        default=DEFAULT_TEXT_COLOR,
        help="Text color as hex RGB, for example #ffffff.",
    )
    parser.add_argument(
        "--separator",
        type=int,
        default=10,
        help="Separator height between videos in pixels.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw BEFORE/AFTER labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    before_path = Path(args.before_path).expanduser()
    after_path = Path(args.after_path).expanduser()
    output_path = Path(args.output_path).expanduser()

    if not before_path.exists():
        raise FileNotFoundError(f"Before video not found: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"After video not found: {after_path}")

    before_meta = read_video_meta(before_path)
    after_meta = read_video_meta(after_path)

    output_fps = args.fps or before_meta.fps or after_meta.fps or 30.0
    source_duration = min(before_meta.duration_seconds, after_meta.duration_seconds)
    if args.max_duration is not None:
        source_duration = min(source_duration, max(args.max_duration, 0.0))
    if source_duration <= 0:
        raise RuntimeError("Could not determine a positive comparison duration.")

    slow_factor = args.slow_factor
    if args.target_duration is not None:
        slow_factor = args.target_duration / source_duration
    if slow_factor <= 0:
        raise ValueError("--slow-factor or --target-duration must be greater than 0.")
    output_duration = source_duration * slow_factor

    separator_height = make_even(max(0, args.separator))
    if args.canvas:
        output_size = parse_canvas(args.canvas)
        panel_height = make_even(max(2, (output_size[1] - separator_height) // 2))
        top_canvas_size = (output_size[0], panel_height)
        bottom_canvas_size = (output_size[0], output_size[1] - separator_height - panel_height)
        top_size = None
        bottom_size = None
    else:
        width = make_even(max(2, args.width))
        top_size = scaled_size(before_meta.width, before_meta.height, width)
        bottom_size = scaled_size(after_meta.width, after_meta.height, width)
        output_size = (width, top_size[1] + separator_height + bottom_size[1])
        top_canvas_size = None
        bottom_canvas_size = None

    static_markers = build_static_markers(args)
    file_markers = read_markers_file(args.marks_file) if args.marks_file else []
    mark_color = parse_hex_color(args.mark_color)
    text_color = parse_hex_color(args.text_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    before_sampler = FrameSampler(before_path, before_meta)
    after_sampler = FrameSampler(after_path, after_meta)
    total_frames = max(1, int(round(output_duration * output_fps)))

    try:
        for output_index in range(total_frames):
            output_time_seconds = output_index / output_fps
            source_time_seconds = min(output_time_seconds / slow_factor, source_duration)
            before_frame = before_sampler.read_at(source_time_seconds)
            after_frame = after_sampler.read_at(source_time_seconds)
            if before_frame is None or after_frame is None:
                break

            if top_canvas_size and bottom_canvas_size:
                marked_before = draw_markers(
                    frame=before_frame.copy(),
                    original_size=(before_meta.width, before_meta.height),
                    static_markers=static_markers,
                    file_markers=file_markers,
                    output_index=output_index,
                    output_time_seconds=output_time_seconds,
                    source_time_seconds=source_time_seconds,
                    output_fps=output_fps,
                    mark_every_seconds=args.mark_every,
                    mark_duration_seconds=args.mark_duration,
                    mark_color=mark_color,
                    text_color=text_color,
                )
                top = fit_frame_to_canvas(marked_before, top_canvas_size)
                bottom = fit_frame_to_canvas(after_frame, bottom_canvas_size)
            else:
                top = cv2.resize(before_frame, top_size, interpolation=cv2.INTER_AREA)
                bottom = cv2.resize(after_frame, bottom_size, interpolation=cv2.INTER_AREA)
                top = draw_markers(
                    frame=top,
                    original_size=(before_meta.width, before_meta.height),
                    static_markers=static_markers,
                    file_markers=file_markers,
                    output_index=output_index,
                    output_time_seconds=output_time_seconds,
                    source_time_seconds=source_time_seconds,
                    output_fps=output_fps,
                    mark_every_seconds=args.mark_every,
                    mark_duration_seconds=args.mark_duration,
                    mark_color=mark_color,
                    text_color=text_color,
                )

            if not args.no_labels:
                draw_panel_label(top, args.before_label, text_color=text_color)
                draw_panel_label(bottom, args.after_label, text_color=text_color)

            if separator_height:
                separator = np.full((separator_height, width, 3), (18, 18, 18), dtype=np.uint8)
                combined = np.vstack([top, separator, bottom])
            else:
                combined = np.vstack([top, bottom])

            writer.write(combined)
            if output_index and output_index % 100 == 0:
                print(f"Processed {output_index}/{total_frames} frames")
    finally:
        before_sampler.release()
        after_sampler.release()
        writer.release()

    print(f"Saved comparison video: {output_path}")


def load_runtime_dependencies() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    try:
        import cv2 as cv2_module  # type: ignore[import-not-found]
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install project requirements first: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    cv2 = cv2_module
    np = np_module


def read_video_meta(path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read video dimensions: {path}")
    if fps <= 0:
        fps = 30.0
    duration_seconds = frame_count / fps if frame_count > 0 else 0.0
    return VideoMeta(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )


def scaled_size(source_width: int, source_height: int, target_width: int) -> tuple[int, int]:
    target_height = int(round(source_height * (target_width / source_width)))
    return make_even(target_width), make_even(max(2, target_height))


def parse_canvas(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if "x" not in normalized:
        raise ValueError("--canvas expects WIDTHxHEIGHT, for example 1080x1350")
    width_text, height_text = normalized.split("x", 1)
    width = make_even(max(2, int(width_text)))
    height = make_even(max(2, int(height_text)))
    return width, height


def fit_frame_to_canvas(frame, canvas_size: tuple[int, int]):
    canvas_width, canvas_height = canvas_size
    frame_height, frame_width = frame.shape[:2]
    scale = min(canvas_width / frame_width, canvas_height / frame_height)
    resized_width = make_even(max(2, int(round(frame_width * scale))))
    resized_height = make_even(max(2, int(round(frame_height * scale))))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((canvas_height, canvas_width, 3), (12, 12, 12), dtype=np.uint8)
    x = max(0, (canvas_width - resized_width) // 2)
    y = max(0, (canvas_height - resized_height) // 2)
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def make_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def build_static_markers(args: argparse.Namespace) -> list[Marker]:
    markers: list[Marker] = []
    if args.circle:
        x, y, radius = parse_float_tuple(args.circle, 3, "--circle")
        markers.append(Marker(x=x, y=y, radius=radius, label=args.marker_label))
    if args.box:
        x, y, width, height = parse_float_tuple(args.box, 4, "--box")
        markers.append(
            Marker(x=x, y=y, width=width, height=height, label=args.marker_label)
        )
    return markers


def parse_float_tuple(value: str, expected_count: int, option_name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != expected_count:
        raise ValueError(f"{option_name} expects {expected_count} comma-separated numbers.")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{option_name} contains a non-numeric value: {value}") from exc


def read_markers_file(path: Path) -> list[Marker]:
    if not path.exists():
        raise FileNotFoundError(f"Marks CSV not found: {path}")

    markers: list[Marker] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            normalized = {key.strip().lower(): (value or "").strip() for key, value in row.items()}
            try:
                markers.append(marker_from_row(normalized))
            except ValueError as exc:
                raise ValueError(f"Invalid marker CSV row {row_number}: {exc}") from exc
    return markers


def marker_from_row(row: dict[str, str]) -> Marker:
    time_seconds = parse_optional_float(row.get("time") or row.get("time_seconds"))
    frame_index = parse_optional_int(row.get("frame") or row.get("frame_index"))
    if time_seconds is None and frame_index is None:
        raise ValueError("expected either time or frame column")

    x = parse_required_float(row, "x")
    y = parse_required_float(row, "y")
    radius = parse_optional_float(row.get("r") or row.get("radius"))
    width = parse_optional_float(row.get("w") or row.get("width"))
    height = parse_optional_float(row.get("h") or row.get("height"))
    label = row.get("label") or "Removed object"

    if radius is None and (width is None or height is None):
        raise ValueError("expected circle columns x,y,r or box columns x,y,w,h")

    return Marker(
        time_seconds=time_seconds,
        frame_index=frame_index,
        x=x,
        y=y,
        radius=radius,
        width=width,
        height=height,
        label=label,
    )


def parse_required_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    parsed = parse_optional_float(value)
    if parsed is None:
        raise ValueError(f"missing numeric {key} column")
    return parsed


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def draw_markers(
    *,
    frame: np.ndarray,
    original_size: tuple[int, int],
    static_markers: Iterable[Marker],
    file_markers: Iterable[Marker],
    output_index: int,
    output_time_seconds: float,
    source_time_seconds: float,
    output_fps: float,
    mark_every_seconds: float,
    mark_duration_seconds: float,
    mark_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> np.ndarray:
    original_width, original_height = original_size
    scale_x = frame.shape[1] / original_width
    scale_y = frame.shape[0] / original_height

    if should_show_static_marker(output_time_seconds, mark_every_seconds, mark_duration_seconds):
        for marker in static_markers:
            draw_marker(frame, marker, scale_x, scale_y, mark_color, text_color)

    active_file_markers = active_markers(
        file_markers,
        output_index=output_index,
        source_time_seconds=source_time_seconds,
        output_fps=output_fps,
        hold_seconds=mark_duration_seconds,
    )
    for marker in active_file_markers:
        draw_marker(frame, marker, scale_x, scale_y, mark_color, text_color)

    return frame


def should_show_static_marker(
    time_seconds: float,
    mark_every_seconds: float,
    mark_duration_seconds: float,
) -> bool:
    if mark_every_seconds <= 0:
        return True
    if mark_duration_seconds <= 0:
        return False
    return (time_seconds % mark_every_seconds) < mark_duration_seconds


def active_markers(
    markers: Iterable[Marker],
    *,
    output_index: int,
    source_time_seconds: float,
    output_fps: float,
    hold_seconds: float,
) -> list[Marker]:
    frame_hold = max(1, int(round(hold_seconds * output_fps / 2)))
    time_hold = max(1.0 / output_fps, hold_seconds / 2)
    active: list[Marker] = []
    for marker in markers:
        if marker.time_seconds is not None:
            if abs(source_time_seconds - marker.time_seconds) <= time_hold:
                active.append(marker)
        elif marker.frame_index is not None:
            if abs(output_index - marker.frame_index) <= frame_hold:
                active.append(marker)
    return active


def draw_marker(
    frame: np.ndarray,
    marker: Marker,
    scale_x: float,
    scale_y: float,
    mark_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    thickness = max(3, int(round(frame.shape[1] / 360)))
    x = int(round(marker.x * scale_x))
    y = int(round(marker.y * scale_y))

    if marker.radius is not None:
        radius = max(4, int(round(marker.radius * (scale_x + scale_y) / 2)))
        cv2.circle(frame, (x, y), radius, mark_color, thickness)
        cv2.circle(frame, (x, y), 3, mark_color, -1)
        label_anchor = (x + radius + 12, max(24, y - radius))
    else:
        width = int(round((marker.width or 0) * scale_x))
        height = int(round((marker.height or 0) * scale_y))
        cv2.rectangle(frame, (x, y), (x + width, y + height), mark_color, thickness)
        label_anchor = (x + width + 12, max(24, y))

    draw_text_badge(frame, marker.label, label_anchor, mark_color, text_color)


def draw_panel_label(
    frame: np.ndarray,
    label: str,
    *,
    text_color: tuple[int, int, int],
) -> None:
    draw_text_badge(frame, label, (18, 22), (25, 25, 25), text_color)


def draw_text_badge(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    background_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.58, frame.shape[1] / 1500)
    thickness = max(1, int(round(frame.shape[1] / 900)))
    padding_x = 10
    padding_y = 7
    x, y = anchor
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size
    x = min(max(6, x), max(6, frame.shape[1] - text_width - 2 * padding_x - 6))
    y = min(max(text_height + padding_y + 6, y), max(text_height + padding_y + 6, frame.shape[0] - 6))

    top_left = (x, y - text_height - padding_y)
    bottom_right = (x + text_width + 2 * padding_x, y + baseline + padding_y)

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, background_color, -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, dst=frame)
    cv2.putText(
        frame,
        text,
        (x + padding_x, y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def parse_hex_color(value: str) -> tuple[int, int, int]:
    color = value.strip().lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {value}")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return blue, green, red


if __name__ == "__main__":
    main()
