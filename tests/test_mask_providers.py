import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import cv2

from logo_removal.mask_providers import (
    AI_HUMAN_CROWDED,
    CROWDED_HUMAN_TRACKING_BACKEND,
    DEFAULT_CROWDED_HUMAN_DETECTOR_IMGSZ,
    DEFAULT_CROWDED_HUMAN_DETECTOR_MODEL,
    DEFAULT_CROWDED_HUMAN_REID_MODEL,
    DEFAULT_CROWDED_HUMAN_SAM2_IMAGE_SIZE,
    SAM2_BACKEND_GLOBAL,
    SAM2_BACKEND_SAMURAI,
    STATIC_RECTANGLE,
    CachedMaskProvider,
    CrowdedHumanMaskProvider,
    Sam2VideoMaskProvider,
    StaticRectangleMaskProvider,
    build_mask_provider,
    precompute_mask_cache,
)
from logo_removal.roi import Roi
from logo_removal.video import VideoMetadata


class MaskProviderTests(unittest.TestCase):
    def setUp(self):
        self.metadata = VideoMetadata(
            width=12,
            height=8,
            fps=30.0,
            frame_count=10,
            duration=0.33,
            has_audio=False,
        )

    def test_static_rectangle_mask_shape_and_values(self):
        provider = StaticRectangleMaskProvider(
            np=np,
            cv2=None,
            metadata=self.metadata,
            roi=Roi(2, 1, 3, 2),
            mask_padding=0,
        )
        provider.prepare()

        mask = provider.mask_for_frame(1, None)

        self.assertEqual(mask.shape, (8, 12))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask[1:3, 2:5].min(), 255)
        self.assertEqual(mask[0, 0], 0)

    def test_static_rectangle_padding_clamps_to_frame(self):
        provider = StaticRectangleMaskProvider(
            np=np,
            cv2=None,
            metadata=self.metadata,
            roi=Roi(0, 0, 2, 2),
            mask_padding=3,
        )
        provider.prepare()

        mask = provider.mask_for_frame(1, None)

        self.assertEqual(mask[:5, :5].min(), 255)
        self.assertEqual(mask[5:, 5:].max(), 0)

    def test_build_static_provider(self):
        provider = build_mask_provider(
            removal_mode=STATIC_RECTANGLE,
            np=np,
            cv2=None,
            input_path=None,
            metadata=self.metadata,
            roi=Roi(1, 1, 2, 2),
            reference_frame=0,
            mask_padding=0,
        )

        self.assertIsInstance(provider, StaticRectangleMaskProvider)

    def test_build_crowded_human_provider_uses_l4_defaults(self):
        provider = build_mask_provider(
            removal_mode=AI_HUMAN_CROWDED,
            np=np,
            cv2=cv2,
            input_path=Path("input.mp4"),
            metadata=self.metadata,
            roi=Roi(1, 1, 4, 5),
            reference_frame=0,
            mask_padding=2,
        )

        self.assertIsInstance(provider, CrowdedHumanMaskProvider)
        self.assertEqual(provider.detector_model, DEFAULT_CROWDED_HUMAN_DETECTOR_MODEL)
        self.assertEqual(provider.reid_model, DEFAULT_CROWDED_HUMAN_REID_MODEL)
        self.assertEqual(provider.detector_imgsz, DEFAULT_CROWDED_HUMAN_DETECTOR_IMGSZ)
        self.assertEqual(provider.person_conf, 0.45)
        self.assertEqual(provider.person_iou, 0.70)

    def test_crowded_human_sam2_image_size_ignores_generic_512_default(self):
        provider = CrowdedHumanMaskProvider(
            np=np,
            cv2=cv2,
            input_path=Path("input.mp4"),
            metadata=self.metadata,
            roi=Roi(1, 1, 4, 5),
            reference_frame=0,
            mask_padding=2,
        )

        with mock.patch.dict(os.environ, {"SAM2_IMAGE_SIZE": "512"}, clear=True):
            self.assertEqual(
                provider._resolve_crowded_human_sam2_image_size(),
                DEFAULT_CROWDED_HUMAN_SAM2_IMAGE_SIZE,
            )

    def test_crowded_human_rejects_unsafe_sam2_image_size(self):
        provider = CrowdedHumanMaskProvider(
            np=np,
            cv2=cv2,
            input_path=Path("input.mp4"),
            metadata=self.metadata,
            roi=Roi(1, 1, 4, 5),
            reference_frame=0,
            mask_padding=2,
        )

        with mock.patch.dict(os.environ, {"CROWDED_HUMAN_SAM2_IMAGE_SIZE": "512"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires.*1024"):
                provider._resolve_crowded_human_sam2_image_size()

    def test_crowded_human_cache_key_includes_tracker_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.mp4"
            input_path.write_bytes(b"video")
            provider = CrowdedHumanMaskProvider(
                np=np,
                cv2=cv2,
                input_path=input_path,
                metadata=self.metadata,
                roi=Roi(1, 1, 4, 5),
                reference_frame=0,
                mask_padding=2,
            )

            key = provider._build_crowded_cache_key()
            provider.detector_model = "yolov8l.pt"
            changed_key = provider._build_crowded_cache_key()

        self.assertEqual(len(key), 24)
        self.assertNotEqual(key, changed_key)
        self.assertEqual(CROWDED_HUMAN_TRACKING_BACKEND, "botsort_reid")

    def test_ai_object_provider_fails_clearly_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.mp4"
            input_path.write_bytes(b"not a real video")
            provider = Sam2VideoMaskProvider(
                np=np,
                cv2=None,
                input_path=input_path,
                metadata=self.metadata,
                roi=Roi(1, 1, 2, 2),
                reference_frame=0,
                mask_padding=0,
                checkpoint_path=Path("missing-sam2-checkpoint.pt"),
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                provider.prepare()

    def test_cached_mask_provider_uses_one_based_frame_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            mask_dir = cache_dir / "masks"
            mask_dir.mkdir()
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "source_frame_count": 3,
                        "tracking_backend": SAM2_BACKEND_GLOBAL,
                        "propagation_mode": "bidirectional",
                        "cache_key": "unit-cache",
                    }
                )
            )
            for index, value in enumerate([40, 80, 120]):
                mask = np.full((8, 12), value, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(mask_dir / f"{index:06d}.png"), mask))

            provider = CachedMaskProvider(
                np=np,
                cv2=cv2,
                metadata=self.metadata,
                cache_dir=cache_dir,
            )
            provider.prepare()

            frame = np.zeros((8, 12, 3), dtype=np.uint8)
            mask = provider.mask_for_frame(2, frame)

            self.assertEqual(mask[0, 0], 80)
            self.assertEqual(provider.cache_key, "unit-cache")
            self.assertEqual(provider.propagation_mode, "bidirectional")

    def test_cached_mask_provider_reports_missing_absolute_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            mask_dir = cache_dir / "masks"
            mask_dir.mkdir()
            (cache_dir / "metadata.json").write_text(json.dumps({"source_frame_count": 2}))
            mask = np.ones((8, 12), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(mask_dir / "000000.png"), mask))

            provider = CachedMaskProvider(
                np=np,
                cv2=cv2,
                metadata=self.metadata,
                cache_dir=cache_dir,
            )

            with self.assertRaisesRegex(RuntimeError, "missing frame 1"):
                provider.prepare()

    def test_samurai_backend_resolves_samurai_config_names(self):
        provider = Sam2VideoMaskProvider(
            np=np,
            cv2=cv2,
            input_path=Path("input.mp4"),
            metadata=self.metadata,
            roi=Roi(1, 1, 2, 2),
            reference_frame=0,
            mask_padding=0,
        )

        config = provider._resolve_model_cfg(
            Path("sam2.1_hiera_tiny.pt"),
            SAM2_BACKEND_SAMURAI,
        )

        self.assertEqual(config, "configs/samurai/sam2.1_hiera_t.yaml")

    def test_samurai_backend_missing_repo_fails_clearly(self):
        provider = Sam2VideoMaskProvider(
            np=np,
            cv2=cv2,
            input_path=Path("input.mp4"),
            metadata=self.metadata,
            roi=Roi(1, 1, 2, 2),
            reference_frame=0,
            mask_padding=0,
        )
        provider.tracking_backend = SAM2_BACKEND_SAMURAI

        with mock.patch.dict(os.environ, {"SAMURAI_REPO_DIR": "/definitely/missing/samurai"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "samurai"):
                provider._import_sam2_video_predictor()

    def test_precompute_mask_cache_rejects_static_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "only supported for SAM 2"):
                precompute_mask_cache(
                    removal_mode=STATIC_RECTANGLE,
                    np=np,
                    cv2=cv2,
                    input_path=Path("input.mp4"),
                    metadata=self.metadata,
                    roi=Roi(1, 1, 2, 2),
                    reference_frame=0,
                    mask_padding=0,
                    mask_cache_dir=Path(tmp),
                )


if __name__ == "__main__":
    unittest.main()
