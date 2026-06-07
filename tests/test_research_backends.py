import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from logo_removal.research_backends import (
    DAM4SAM_BACKEND,
    SAM2LONG_BACKEND,
    check_dam4sam_setup,
    check_research_backend,
    check_sam2long_setup,
)


class ResearchBackendTests(unittest.TestCase):
    def test_dam4sam_missing_repo_reports_clear_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            report = check_dam4sam_setup(Path("/definitely/missing/dam4sam"))

        self.assertFalse(report.ready)
        self.assertEqual(report.status, "missing_repo")
        self.assertEqual(report.expected_env_var, "DAM4SAM_REPO_DIR")

    def test_sam2long_noncommercial_license_is_research_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "README.md").write_text("SAM2Long")
            (repo / "LICENSE").write_text("Creative Commons CC-BY-NC 4.0 NonCommercial")

            report = check_sam2long_setup(repo)

        self.assertTrue(report.ready)
        self.assertEqual(report.status, "ready_for_research_eval")
        self.assertEqual(report.license_status, "noncommercial")

    def test_check_research_backend_dispatches(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            dam4sam = check_research_backend(
                DAM4SAM_BACKEND,
                Path("/definitely/missing/dam4sam"),
            )
            sam2long = check_research_backend(
                SAM2LONG_BACKEND,
                Path("/definitely/missing/sam2long"),
            )

        self.assertEqual(dam4sam.backend, DAM4SAM_BACKEND)
        self.assertEqual(sam2long.backend, SAM2LONG_BACKEND)


if __name__ == "__main__":
    unittest.main()
