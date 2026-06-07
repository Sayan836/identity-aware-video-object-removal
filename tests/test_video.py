import unittest

from logo_removal.video import _format_fps, _parse_fps


class VideoHelperTests(unittest.TestCase):
    def test_parse_fractional_fps(self):
        self.assertEqual(round(_parse_fps("30000/1001"), 3), 29.97)

    def test_parse_invalid_fps(self):
        self.assertEqual(_parse_fps("0/0"), 0.0)

    def test_format_whole_fps(self):
        self.assertEqual(_format_fps(30.0), "30")

    def test_format_fractional_fps(self):
        self.assertEqual(_format_fps(29.97002997), "29.97003")


if __name__ == "__main__":
    unittest.main()
