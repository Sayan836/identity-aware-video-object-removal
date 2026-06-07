import unittest

import cv2
import numpy as np

from logo_removal.mask_refiners import BiRefNetMaskRefiner, MorphologicalMaskRefiner


class MorphologicalMaskRefinerTests(unittest.TestCase):
    def test_dilation_expands_binary_mask(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 255
        refiner = MorphologicalMaskRefiner(np=np, cv2=cv2, dilation_px=1)
        refiner.prepare()

        refined = refiner.refine(mask, frame=None, frame_index=0)

        self.assertGreater(np.count_nonzero(refined), np.count_nonzero(mask))
        self.assertEqual(refined.dtype, np.uint8)
        self.assertEqual(set(np.unique(refined)), {0, 255})

    def test_close_fills_small_hole(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        mask[3, 3] = 0
        refiner = MorphologicalMaskRefiner(np=np, cv2=cv2, close_kernel_px=3)
        refiner.prepare()

        refined = refiner.refine(mask, frame=None, frame_index=0)

        self.assertEqual(refined[3, 3], 255)

    def test_component_cleanup_removes_tiny_islands(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1, 1] = 255
        mask[4:7, 4:7] = 255
        refiner = MorphologicalMaskRefiner(
            np=np,
            cv2=cv2,
            close_kernel_px=0,
            min_component_area=4,
        )
        refiner.prepare()

        refined = refiner.refine(mask, frame=None, frame_index=0)

        self.assertEqual(refined[1, 1], 0)
        self.assertEqual(refined[5, 5], 255)

    def test_invalid_settings_fail_clearly(self):
        refiner = MorphologicalMaskRefiner(np=np, cv2=cv2, dilation_px=-1)

        with self.assertRaisesRegex(ValueError, "dilation_px"):
            refiner.prepare()


class BiRefNetMaskRefinerTests(unittest.TestCase):
    def test_birefnet_placeholder_fails_clearly(self):
        refiner = BiRefNetMaskRefiner()

        with self.assertRaisesRegex(RuntimeError, "BiRefNet"):
            refiner.prepare()


if __name__ == "__main__":
    unittest.main()
