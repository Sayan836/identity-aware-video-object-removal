import argparse
import unittest


from logo_removal.roi import Roi, parse_roi


class RoiTests(unittest.TestCase):
    def test_parse_roi_accepts_csv_values(self):
        self.assertEqual(parse_roi("10,20,30,40"), Roi(10, 20, 30, 40))

    def test_parse_roi_strips_spaces(self):
        self.assertEqual(parse_roi(" 1, 2, 3, 4 "), Roi(1, 2, 3, 4))

    def test_parse_roi_rejects_invalid_shape(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_roi("1,2,3")

    def test_roi_padding_clamps_to_frame_bounds(self):
        roi = Roi(2, 3, 10, 20)
        self.assertEqual(roi.padded(5, frame_width=20, frame_height=25), Roi(0, 0, 17, 25))

    def test_roi_validate_inside_rejects_out_of_bounds(self):
        with self.assertRaises(ValueError):
            Roi(90, 10, 20, 20).validate_inside(frame_width=100, frame_height=100)


if __name__ == "__main__":
    unittest.main()
