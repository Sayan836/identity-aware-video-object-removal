from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DAM4SAM_BACKEND = "dam4sam"
SAM2LONG_BACKEND = "sam2long_research"
VALID_RESEARCH_BACKENDS = {DAM4SAM_BACKEND, SAM2LONG_BACKEND}


@dataclass(frozen=True)
class ResearchBackendReport:
    """Setup status for a non-default research tracking backend."""

    backend: str
    repo_dir: str | None
    ready: bool
    status: str
    license_status: str
    expected_env_var: str
    notes: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "repo_dir": self.repo_dir,
            "ready": self.ready,
            "status": self.status,
            "license_status": self.license_status,
            "expected_env_var": self.expected_env_var,
            "notes": self.notes,
        }


def check_research_backend(backend: str, repo_dir: Path | None = None) -> ResearchBackendReport:
    """Check whether an optional research backend is locally evaluable."""

    if backend == DAM4SAM_BACKEND:
        return check_dam4sam_setup(repo_dir)
    if backend == SAM2LONG_BACKEND:
        return check_sam2long_setup(repo_dir)
    raise ValueError(f"Unsupported research backend: {backend}")


def check_dam4sam_setup(repo_dir: Path | None = None) -> ResearchBackendReport:
    """Validate local DAM4SAM setup without importing its tracker."""

    resolved = _resolve_repo_dir(repo_dir, "DAM4SAM_REPO_DIR", "models/dam4sam_repo")
    notes = [
        "DAM4SAM is kept as an explicit evaluation backend for distractor-heavy clips.",
        "The public wrapper is frame-by-frame and is not the same API as the current SAM2 cache path.",
    ]
    if resolved is None:
        return ResearchBackendReport(
            backend=DAM4SAM_BACKEND,
            repo_dir=None,
            ready=False,
            status="missing_repo",
            license_status="unknown",
            expected_env_var="DAM4SAM_REPO_DIR",
            notes=notes + ["Set DAM4SAM_REPO_DIR to a local DAM4SAM checkout."],
        )

    license_status = _detect_license_status(resolved)
    expected_files = ["README.md"]
    missing = [name for name in expected_files if not (resolved / name).exists()]
    ready = not missing
    status = "ready_for_manual_eval" if ready else "incomplete_setup"
    if missing:
        notes.append(f"Missing expected file(s): {', '.join(missing)}")
    if license_status in {"missing", "unknown"}:
        notes.append("Review DAM4SAM licensing before production use.")

    return ResearchBackendReport(
        backend=DAM4SAM_BACKEND,
        repo_dir=str(resolved),
        ready=ready,
        status=status,
        license_status=license_status,
        expected_env_var="DAM4SAM_REPO_DIR",
        notes=notes,
    )


def check_sam2long_setup(repo_dir: Path | None = None) -> ResearchBackendReport:
    """Validate local SAM2Long setup and mark it research-only."""

    resolved = _resolve_repo_dir(repo_dir, "SAM2LONG_REPO_DIR", "models/sam2long_repo")
    notes = [
        "SAM2Long is research-only in this project unless commercial constraints are cleared.",
        "Expected upstream license is CC-BY-NC 4.0.",
    ]
    if resolved is None:
        return ResearchBackendReport(
            backend=SAM2LONG_BACKEND,
            repo_dir=None,
            ready=False,
            status="missing_repo",
            license_status="cc-by-nc-expected",
            expected_env_var="SAM2LONG_REPO_DIR",
            notes=notes + ["Set SAM2LONG_REPO_DIR to a local SAM2Long checkout."],
        )

    license_status = _detect_license_status(resolved)
    expected_files = ["README.md"]
    missing = [name for name in expected_files if not (resolved / name).exists()]
    ready = not missing and "noncommercial" in license_status
    status = "ready_for_research_eval" if ready else "incomplete_or_license_unconfirmed"
    if missing:
        notes.append(f"Missing expected file(s): {', '.join(missing)}")
    if "noncommercial" not in license_status:
        notes.append("License did not clearly identify non-commercial terms; review manually.")

    return ResearchBackendReport(
        backend=SAM2LONG_BACKEND,
        repo_dir=str(resolved),
        ready=ready,
        status=status,
        license_status=license_status,
        expected_env_var="SAM2LONG_REPO_DIR",
        notes=notes,
    )


def write_research_backend_report(report: ResearchBackendReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.as_dict(), indent=2))


def _resolve_repo_dir(repo_dir: Path | None, env_var: str, default_relative: str) -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    raw = repo_dir or (
        Path(os.environ[env_var])
        if os.environ.get(env_var)
        else project_root / default_relative
    )
    if raw is None:
        return None
    resolved = raw.expanduser().resolve()
    if not resolved.exists():
        return None
    return resolved


def _detect_license_status(repo_dir: Path) -> str:
    license_files = [
        path
        for path in repo_dir.iterdir()
        if path.is_file() and path.name.lower().startswith("license")
    ]
    if not license_files:
        return "missing"

    text = "\n".join(path.read_text(errors="ignore").lower() for path in license_files)
    if "cc-by-nc" in text or "noncommercial" in text or "non-commercial" in text:
        return "noncommercial"
    if "apache license" in text or "apache-2.0" in text:
        return "apache-2.0"
    if "mit license" in text:
        return "mit"
    return "unknown"
