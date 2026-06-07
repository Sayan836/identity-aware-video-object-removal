import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import cv2

from logo_removal.void_export import (
    QUADMASK_AFFECTED,
    QUADMASK_BACKGROUND,
    QUADMASK_OVERLAP,
    QUADMASK_PRIMARY,
    _write_package_zip,
    binary_mask_to_basic_quadmask,
    validate_quadmask_video,
)


class VoidExportTests(unittest.TestCase):
    def test_binary_mask_to_basic_quadmask_inverts_local_mask_convention(self):
        local_mask = np.array(
            [
                [0, 1, 255],
                [0, 128, 0],
            ],
            dtype=np.uint8,
        )

        quadmask = binary_mask_to_basic_quadmask(local_mask, np=np)

        expected = np.array(
            [
                [QUADMASK_BACKGROUND, QUADMASK_PRIMARY, QUADMASK_PRIMARY],
                [QUADMASK_BACKGROUND, QUADMASK_PRIMARY, QUADMASK_BACKGROUND],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(quadmask, expected)

    def test_binary_mask_to_basic_quadmask_can_write_affected_shell(self):
        local_mask = np.zeros((7, 7), dtype=np.uint8)
        local_mask[3, 3] = 255

        quadmask = binary_mask_to_basic_quadmask(
            local_mask,
            np=np,
            cv2=cv2,
            shadow_dilation_px=1,
        )

        self.assertEqual(quadmask[3, 3], QUADMASK_PRIMARY)
        self.assertEqual(int(np.count_nonzero(quadmask == QUADMASK_AFFECTED)), 4)
        self.assertEqual(quadmask[0, 0], QUADMASK_BACKGROUND)

    def test_binary_mask_to_basic_quadmask_accepts_explicit_affected_mask(self):
        local_mask = np.zeros((5, 5), dtype=np.uint8)
        local_mask[2, 2] = 255
        affected_mask = np.zeros((5, 5), dtype=np.uint8)
        affected_mask[2, 2] = 255
        affected_mask[2, 3] = 255

        quadmask = binary_mask_to_basic_quadmask(
            local_mask,
            np=np,
            affected_mask=affected_mask,
        )

        self.assertEqual(quadmask[2, 2], QUADMASK_OVERLAP)
        self.assertEqual(quadmask[2, 3], QUADMASK_AFFECTED)
        self.assertEqual(quadmask[0, 0], QUADMASK_BACKGROUND)

    def test_write_package_zip_places_void_files_at_zip_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            sequence_dir = Path(tmp) / "sequence"
            sequence_dir.mkdir()
            (sequence_dir / "input_video.mp4").write_bytes(b"video")
            (sequence_dir / "quadmask_0.mp4").write_bytes(b"mask")
            (sequence_dir / "prompt.json").write_text(json.dumps({"bg": "clean background"}))
            (sequence_dir / "manifest.json").write_text("{}")
            output_zip = Path(tmp) / "void_phase5_input.zip"

            _write_package_zip(sequence_dir, output_zip, overwrite=False)

            with zipfile.ZipFile(output_zip) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    [
                        "input_video.mp4",
                        "manifest.json",
                        "prompt.json",
                        "quadmask_0.mp4",
                    ],
                )

    def test_validate_quadmask_can_allow_empty_primary_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            quadmask_path = Path(tmp) / "empty_quadmask.mp4"
            self._write_constant_mask_video(quadmask_path, value=QUADMASK_BACKGROUND)

            stats = validate_quadmask_video(
                quadmask_path=quadmask_path,
                expected_frame_count=3,
                expected_fps=12.0,
                cv2=cv2,
                np=np,
                allow_empty_primary_mask=True,
            )

            self.assertTrue(stats["empty_primary_mask"])
            self.assertEqual(stats["primary_pixels"], 0)
            self.assertEqual(stats["frames_with_primary_pixels"], 0)

    def test_validate_quadmask_still_rejects_empty_primary_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            quadmask_path = Path(tmp) / "empty_quadmask.mp4"
            self._write_constant_mask_video(quadmask_path, value=QUADMASK_BACKGROUND)

            with self.assertRaisesRegex(RuntimeError, "no primary object pixels"):
                validate_quadmask_video(
                    quadmask_path=quadmask_path,
                    expected_frame_count=3,
                    expected_fps=12.0,
                    cv2=cv2,
                    np=np,
                )

    def _write_constant_mask_video(self, path: Path, value: int) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            12.0,
            (8, 8),
            isColor=False,
        )
        self.assertTrue(writer.isOpened())
        try:
            frame = np.full((8, 8), value, dtype=np.uint8)
            for _ in range(3):
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
