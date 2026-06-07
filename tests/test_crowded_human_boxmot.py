import types
import sys
import unittest

from logo_removal.mask_providers import CrowdedHumanMaskProvider


class CrowdedHumanBoxmotTests(unittest.TestCase):
    def test_botsort_builder_accepts_top_level_boxmot_class(self):
        class FakeBotSort:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        provider = object.__new__(CrowdedHumanMaskProvider)
        boxmot_module = types.SimpleNamespace(BotSort=FakeBotSort, __file__=__file__)

        tracker = provider._build_botsort_tracker(
            boxmot_module=boxmot_module,
            reid_weights="osnet_x0_25_msmt17.pt",
            device="cuda",
        )

        self.assertIsInstance(tracker, FakeBotSort)
        self.assertEqual(str(tracker.kwargs["reid_weights"]), "osnet_x0_25_msmt17.pt")
        self.assertTrue(tracker.kwargs["half"])

    def test_botsort_builder_uses_v19_tracker_zoo_factory(self):
        class FakeTracker:
            pass

        captured = {}

        def create_tracker(**kwargs):
            captured.update(kwargs)
            return FakeTracker()

        tracker_zoo = types.ModuleType("boxmot.trackers.tracker_zoo")
        tracker_zoo.create_tracker = create_tracker
        previous = sys.modules.get("boxmot.trackers.tracker_zoo")
        sys.modules["boxmot.trackers.tracker_zoo"] = tracker_zoo
        try:
            provider = object.__new__(CrowdedHumanMaskProvider)
            boxmot_module = types.SimpleNamespace(__file__=__file__)

            tracker = provider._build_botsort_tracker(
                boxmot_module=boxmot_module,
                reid_weights="osnet_x0_25_msmt17.pt",
                device="cuda",
            )
        finally:
            if previous is None:
                sys.modules.pop("boxmot.trackers.tracker_zoo", None)
            else:
                sys.modules["boxmot.trackers.tracker_zoo"] = previous

        self.assertIsInstance(tracker, FakeTracker)
        self.assertEqual(captured["tracker_type"], "botsort")
        self.assertEqual(str(captured["reid_weights"]), "osnet_x0_25_msmt17.pt")
        self.assertEqual(captured["device"], "cuda")
        self.assertTrue(captured["half"])

    def test_botsort_builder_rejects_tracker_without_reid_model(self):
        class BadTracker:
            with_reid = True
            model = None

        def create_tracker(*args, **kwargs):
            return BadTracker()

        tracker_zoo = types.ModuleType("boxmot.trackers.tracker_zoo")
        tracker_zoo.create_tracker = create_tracker
        previous = sys.modules.get("boxmot.trackers.tracker_zoo")
        sys.modules["boxmot.trackers.tracker_zoo"] = tracker_zoo
        try:
            provider = object.__new__(CrowdedHumanMaskProvider)
            boxmot_module = types.SimpleNamespace(__file__=__file__)

            with self.assertRaisesRegex(RuntimeError, "without a ReID model"):
                provider._build_botsort_tracker(
                    boxmot_module=boxmot_module,
                    reid_weights="osnet_x0_25_msmt17.pt",
                    device="cuda",
                )
        finally:
            if previous is None:
                sys.modules.pop("boxmot.trackers.tracker_zoo", None)
            else:
                sys.modules["boxmot.trackers.tracker_zoo"] = previous


if __name__ == "__main__":
    unittest.main()
