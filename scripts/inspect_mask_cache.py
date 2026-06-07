#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2  # type: ignore[import-not-found]
import numpy as np

from logo_removal.mask_qa import MaskQaTracker
from logo_removal.video import probe_video


def main() -> int:
    args = build_parser().parse_args()
    video_path = args.video.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = probe_video(video_path)
    qa_report = inspect_cache(
        video_path=video_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        frame_count=metadata.frame_count or 0,
        contact_sheet_frames=args.contact_sheet_frames,
    )
    report_path = output_dir / "mask_cache_inspection.json"
    report_path.write_text(json.dumps(qa_report, indent=2))
    print(json.dumps({"status": "ok", "report": str(report_path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a full-video SAM2 mask cache.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_outputs" / "mask_cache_inspection",
    )
    parser.add_argument("--contact-sheet-frames", type=int, default=12)
    return parser


def inspect_cache(
    video_path: Path,
    cache_dir: Path,
    output_dir: Path,
    frame_count: int,
    contact_sheet_frames: int,
) -> dict[str, object]:
    if frame_count <= 0:
        raise RuntimeError("Video frame count is required for mask cache inspection.")

    mask_dir = cache_dir / "masks"
    if not mask_dir.exists():
        raise RuntimeError(f"Mask cache does not contain a masks directory: {mask_dir}")

    tracker = MaskQaTracker(np=np)
    frames: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    sheet_frames = _sample_frame_indices(frame_count, contact_sheet_frames)
    overlays = []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        for frame_idx in range(frame_count):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            mask_path = mask_dir / f"{frame_idx:06d}.png"
            if not mask_path.exists():
                raise RuntimeError(f"Missing cached mask for frame {frame_idx}: {mask_path}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not read cached mask: {mask_path}")
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            active = mask > 0
            area = int(np.count_nonzero(active))
            frame_warnings = tracker.inspect(mask, frame_index=frame_idx + 1)
            frames.append(
                {
                    "frame_index": frame_idx,
                    "area": area,
                    "area_ratio": area / max(1, mask.shape[0] * mask.shape[1]),
                    "warning_codes": [warning.code for warning in frame_warnings],
                }
            )
            for warning in frame_warnings:
                warnings.append(
                    {
                        "frame_index": warning.frame_index,
                        "code": warning.code,
                        "message": warning.message,
                    }
                )
            if frame_idx in sheet_frames:
                overlays.append(_overlay_mask(frame, mask, label=f"{frame_idx:06d}"))
    finally:
        capture.release()

    contact_sheet_path = output_dir / "mask_contact_sheet.jpg"
    if overlays:
        contact_sheet = _make_contact_sheet(overlays)
        cv2.imwrite(str(contact_sheet_path), contact_sheet)

    qa = {
        "video": str(video_path),
        "cache_dir": str(cache_dir),
        "frame_count": len(frames),
        "warning_count": len(warnings),
        "warnings": warnings,
        "frames": frames,
        "contact_sheet": str(contact_sheet_path) if overlays else None,
    }
    (cache_dir / "qa_scores.json").write_text(json.dumps(qa, indent=2))
    return qa


def _sample_frame_indices(frame_count: int, count: int) -> set[int]:
    if count <= 0:
        return set()
    if frame_count <= count:
        return set(range(frame_count))
    return {
        int(round(index * (frame_count - 1) / (count - 1)))
        for index in range(count)
    }


def _overlay_mask(frame, mask, label: str):
    output = frame.copy()
    red = np.zeros_like(frame)
    red[:, :, 2] = 255
    active = mask > 0
    output[active] = cv2.addWeighted(output[active], 0.45, red[active], 0.55, 0)
    cv2.putText(
        output,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _make_contact_sheet(images):
    thumb_width = 240
    resized = []
    for image in images:
        height, width = image.shape[:2]
        scale = thumb_width / max(1, width)
        thumb = cv2.resize(
            image,
            (thumb_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        resized.append(thumb)

    columns = min(4, len(resized))
    rows = (len(resized) + columns - 1) // columns
    cell_height = max(image.shape[0] for image in resized)
    sheet = np.zeros((rows * cell_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, image in enumerate(resized):
        row = index // columns
        column = index % columns
        y = row * cell_height
        x = column * thumb_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    return sheet


if __name__ == "__main__":
    raise SystemExit(main())
