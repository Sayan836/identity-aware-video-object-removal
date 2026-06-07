import unittest

import numpy as np

from logo_removal.mask_qa import MaskQaConfig, MaskQaTracker


class MaskQaTrackerTests(unittest.TestCase):
    def test_empty_mask_warns(self):
        tracker = MaskQaTracker(np=np)
        mask = np.zeros((10, 10), dtype=np.uint8)

        warnings = tracker.inspect(mask, frame_index=1)

        self.assertEqual(warnings[0].code, "empty_mask")

    def test_area_jump_warns(self):
        tracker = MaskQaTracker(
            np=np,
            config=MaskQaConfig(min_area_ratio=0.0, max_area_jump_ratio=0.5),
        )
        small = np.zeros((20, 20), dtype=np.uint8)
        small[1:3, 1:3] = 255
        large = np.zeros((20, 20), dtype=np.uint8)
        large[1:10, 1:10] = 255

        tracker.inspect(small, frame_index=1)
        warnings = tracker.inspect(large, frame_index=2)

        self.assertIn("area_jump", {warning.code for warning in warnings})

    def test_missing_mask_streak_warns(self):
        tracker = MaskQaTracker(
            np=np,
            config=MaskQaConfig(max_missing_frames=1),
        )
        mask = np.zeros((10, 10), dtype=np.uint8)

        tracker.inspect(mask, frame_index=1)
        warnings = tracker.inspect(mask, frame_index=2)

        self.assertIn("missing_mask_streak", {warning.code for warning in warnings})

    def test_centroid_jump_warns(self):
        tracker = MaskQaTracker(
            np=np,
            config=MaskQaConfig(min_area_ratio=0.0, max_centroid_jump_ratio=0.2),
        )
        first = np.zeros((20, 20), dtype=np.uint8)
        first[1:4, 1:4] = 255
        second = np.zeros((20, 20), dtype=np.uint8)
        second[15:18, 15:18] = 255

        tracker.inspect(first, frame_index=1)
        warnings = tracker.inspect(second, frame_index=2)

        self.assertIn("centroid_jump", {warning.code for warning in warnings})

    def test_stable_mask_sequence_has_no_warnings(self):
        tracker = MaskQaTracker(np=np)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 255

        first = tracker.inspect(mask, frame_index=1)
        second = tracker.inspect(mask.copy(), frame_index=2)

        self.assertEqual(first, [])
        self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
