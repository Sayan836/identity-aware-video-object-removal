#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
from PIL import Image


BACKEND_DAM4SAM = "dam4sam"
BACKEND_SAM2LONG = "sam2long_research"
BACKEND_SAMURAI = "samurai"


def main() -> int:
    args = build_parser().parse_args()
    repo_dir = args.repo_dir.expanduser().resolve()
    frame_dir = args.frame_dir.expanduser().resolve()
    output_cache_dir = args.output_cache_dir.expanduser().resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Backend repository does not exist: {repo_dir}")
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")

    import_root = repo_dir / "sam2" if (repo_dir / "sam2" / "sam2").exists() else repo_dir
    sys.path.insert(0, str(import_root))
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    for child in ("masks", "forward", "backward"):
        child_path = output_cache_dir / child
        if child_path.exists():
            shutil.rmtree(child_path)
        child_path.mkdir(parents=True)

    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No JPG frames found in {frame_dir}")
    if args.reference_frame < 0 or args.reference_frame >= len(frame_paths):
        raise RuntimeError(f"Reference frame {args.reference_frame} is outside frame range.")

    if args.backend == BACKEND_SAMURAI:
        report = run_samurai(args, frame_paths, output_cache_dir)
    elif args.backend == BACKEND_DAM4SAM:
        report = run_dam4sam(args, repo_dir, frame_paths, output_cache_dir)
    elif args.backend == BACKEND_SAM2LONG:
        report = run_sam2long(args, frame_dir, frame_paths, output_cache_dir)
    else:
        raise RuntimeError(f"Unsupported backend: {args.backend}")

    (output_cache_dir / "backend_report.json").write_text(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate masks with a forked SAM2 backend.")
    parser.add_argument(
        "--backend",
        choices=[BACKEND_SAMURAI, BACKEND_DAM4SAM, BACKEND_SAM2LONG],
        required=True,
    )
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-image-size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--num-pathway", type=int, default=3)
    parser.add_argument("--iou-thre", type=float, default=0.1)
    parser.add_argument("--uncertainty", type=float, default=2.0)
    return parser


def run_dam4sam(args, repo_dir: Path, frame_paths: list[Path], output_cache_dir: Path) -> dict:
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("DAM4SAM currently requires CUDA in its public tracker wrapper.")
    _ensure_dam4sam_checkpoint(repo_dir, args.checkpoint)

    from dam4sam_tracker import DAM4SAMTracker

    tracker_name = _dam4sam_tracker_name(args.checkpoint)
    bbox = [args.x, args.y, args.width, args.height]
    forward = _run_dam4sam_sequence(
        tracker=DAM4SAMTracker(tracker_name),
        frame_paths=frame_paths[args.reference_frame :],
        bbox=bbox,
    )
    for offset, mask in enumerate(forward):
        frame_idx = args.reference_frame + offset
        _write_binary_mask(output_cache_dir / "forward" / f"{frame_idx:06d}.png", mask)

    backward = []
    if args.bidirectional and args.reference_frame > 0:
        reverse_paths = list(reversed(frame_paths[: args.reference_frame + 1]))
        backward = _run_dam4sam_sequence(
            tracker=DAM4SAMTracker(tracker_name),
            frame_paths=reverse_paths,
            bbox=bbox,
        )
        for offset, mask in enumerate(backward):
            frame_idx = args.reference_frame - offset
            _write_binary_mask(output_cache_dir / "backward" / f"{frame_idx:06d}.png", mask)

    _merge_directional_masks(
        output_cache_dir=output_cache_dir,
        frame_count=len(frame_paths),
        reference_frame=args.reference_frame,
        use_backward=bool(backward),
    )
    return {
        "backend": BACKEND_DAM4SAM,
        "tracker_name": tracker_name,
        "forward_masks": len(forward),
        "backward_masks": len(backward),
        "warnings": [],
    }


def run_samurai(args, frame_paths: list[Path], output_cache_dir: Path) -> dict:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(args.checkpoint.expanduser().resolve()),
        device=device,
        hydra_overrides_extra=[f"model.image_size={args.sam2_image_size}"],
        apply_postprocessing=False,
    )
    box = np.array(
        [args.x, args.y, args.x + args.width, args.y + args.height],
        dtype=np.float32,
    )
    forward_frame_dir = _make_ordered_frame_dir(
        output_cache_dir / "_samurai_forward_frames",
        frame_paths[args.reference_frame :],
    )
    forward = _run_standard_sam2_sequence(
        predictor=predictor,
        frame_dir=forward_frame_dir,
        output_offset=args.reference_frame,
        output_step=1,
        box=box,
        output_dir=output_cache_dir / "forward",
    )
    backward = {}
    if args.bidirectional and args.reference_frame > 0:
        backward_frame_dir = _make_ordered_frame_dir(
            output_cache_dir / "_samurai_backward_frames",
            list(reversed(frame_paths[: args.reference_frame + 1])),
        )
        backward = _run_standard_sam2_sequence(
            predictor=predictor,
            frame_dir=backward_frame_dir,
            output_offset=args.reference_frame,
            output_step=-1,
            box=box,
            output_dir=output_cache_dir / "backward",
        )

    _merge_directional_masks(
        output_cache_dir=output_cache_dir,
        frame_count=len(frame_paths),
        reference_frame=args.reference_frame,
        use_backward=bool(backward),
    )
    return {
        "backend": BACKEND_SAMURAI,
        "device": device,
        "forward_masks": len(forward),
        "backward_masks": len(backward),
        "warnings": [],
    }


def _run_standard_sam2_sequence(
    predictor,
    frame_dir: Path,
    output_offset: int,
    output_step: int,
    box: np.ndarray,
    output_dir: Path,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_state = predictor.init_state(
        video_path=str(frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )
    prompt_frame_idx, object_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=box,
    )
    paths = {
        output_offset: _write_sam_logits_mask(
            output_dir / f"{output_offset:06d}.png",
            object_ids,
            mask_logits,
        )
    }
    for sequence_idx, object_ids, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
    ):
        frame_idx = output_offset + sequence_idx * output_step
        paths[frame_idx] = _write_sam_logits_mask(
            output_dir / f"{frame_idx:06d}.png",
            object_ids,
            mask_logits,
        )
    return paths


def _run_dam4sam_sequence(tracker, frame_paths: list[Path], bbox: list[int]) -> list[np.ndarray]:
    masks = []
    for index, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path).convert("RGB")
        if index == 0:
            output = tracker.initialize(image, None, bbox=bbox)
        else:
            output = tracker.track(image)
        masks.append(output["pred_mask"].astype(np.uint8) * 255)
    return masks


def run_sam2long(args, frame_dir: Path, frame_paths: list[Path], output_cache_dir: Path) -> dict:
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(args.checkpoint.expanduser().resolve()),
        device=device,
        hydra_overrides_extra=[f"model.image_size={args.sam2_image_size}"],
        apply_postprocessing=False,
    )
    box = np.array(
        [args.x, args.y, args.x + args.width, args.y + args.height],
        dtype=np.float32,
    )
    forward_frame_dir = _make_ordered_frame_dir(
        output_cache_dir / "_sam2long_forward_frames",
        frame_paths[args.reference_frame :],
    )
    forward = _run_sam2long_sequence(
        predictor=predictor,
        frame_dir=forward_frame_dir,
        output_offset=args.reference_frame,
        output_step=1,
        box=box,
        output_dir=output_cache_dir / "forward",
        num_pathway=args.num_pathway,
        iou_thre=args.iou_thre,
        uncertainty=args.uncertainty,
    )
    backward = {}
    if args.bidirectional and args.reference_frame > 0:
        backward_frame_dir = _make_ordered_frame_dir(
            output_cache_dir / "_sam2long_backward_frames",
            list(reversed(frame_paths[: args.reference_frame + 1])),
        )
        backward = _run_sam2long_sequence(
            predictor=predictor,
            frame_dir=backward_frame_dir,
            output_offset=args.reference_frame,
            output_step=-1,
            box=box,
            output_dir=output_cache_dir / "backward",
            num_pathway=args.num_pathway,
            iou_thre=args.iou_thre,
            uncertainty=args.uncertainty,
        )

    _merge_directional_masks(
        output_cache_dir=output_cache_dir,
        frame_count=len(frame_paths),
        reference_frame=args.reference_frame,
        use_backward=bool(backward),
    )
    return {
        "backend": BACKEND_SAM2LONG,
        "device": device,
        "num_pathway": args.num_pathway,
        "iou_thre": args.iou_thre,
        "uncertainty": args.uncertainty,
        "forward_masks": len(forward),
        "backward_masks": len(backward),
        "warnings": [],
    }


def _run_sam2long_sequence(
    predictor,
    frame_dir: Path,
    output_offset: int,
    output_step: int,
    box: np.ndarray,
    output_dir: Path,
    num_pathway: int,
    iou_thre: float,
    uncertainty: float,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_state = predictor.init_state(
        video_path=str(frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )
    inference_state["num_pathway"] = num_pathway
    inference_state["iou_thre"] = iou_thre
    inference_state["uncertainty"] = uncertainty
    prompt_frame_idx, object_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=box,
    )
    paths = {}
    out_obj_ids, mask_sequence = predictor.propagate_in_video(inference_state, start_frame_idx=0)
    for sequence_idx, mask_logits in enumerate(mask_sequence):
        frame_idx = output_offset + sequence_idx * output_step
        paths[frame_idx] = _write_sam_logits_mask(
            output_dir / f"{frame_idx:06d}.png",
            out_obj_ids,
            mask_logits,
        )
    return paths


def _make_ordered_frame_dir(output_dir: Path, frame_paths: list[Path]) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for index, frame_path in enumerate(frame_paths):
        shutil.copy2(frame_path, output_dir / f"{index:05d}.jpg")
    return output_dir


def _write_sam_logits_mask(path: Path, object_ids, mask_logits) -> Path:
    object_id_list = [int(object_id) for object_id in object_ids]
    mask_index = object_id_list.index(1) if 1 in object_id_list else 0
    mask = (mask_logits[mask_index] > 0.0).detach().cpu().numpy()
    mask = np.squeeze(mask).astype(np.uint8) * 255
    _write_binary_mask(path, mask)
    return path


def _merge_directional_masks(
    output_cache_dir: Path,
    frame_count: int,
    reference_frame: int,
    use_backward: bool,
) -> None:
    merged_dir = output_cache_dir / "masks"
    forward_dir = output_cache_dir / "forward"
    backward_dir = output_cache_dir / "backward"
    for frame_idx in range(frame_count):
        if frame_idx < reference_frame and use_backward:
            source = backward_dir / f"{frame_idx:06d}.png"
        elif frame_idx == reference_frame:
            source = forward_dir / f"{frame_idx:06d}.png"
            if not source.exists():
                source = backward_dir / f"{frame_idx:06d}.png"
        else:
            source = forward_dir / f"{frame_idx:06d}.png"
        if not source.exists():
            raise RuntimeError(f"Backend did not produce frame {frame_idx}: {source}")
        shutil.copy2(source, merged_dir / f"{frame_idx:06d}.png")


def _write_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max(initial=0) <= 1:
        mask = mask * 255
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write mask: {path}")


def _dam4sam_tracker_name(checkpoint: Path) -> str:
    name = checkpoint.name
    if "base_plus" in name:
        return "sam21pp-B"
    if "small" in name:
        return "sam21pp-S"
    if "large" in name:
        return "sam21pp-L"
    return "sam21pp-T"


def _ensure_dam4sam_checkpoint(repo_dir: Path, checkpoint: Path) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    name = checkpoint.name
    targets = [
        repo_dir / "checkpoints" / name,
        repo_dir / name,
    ]
    for target in targets:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(checkpoint)
        except FileExistsError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
