#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from logo_removal.research_backends import (
    VALID_RESEARCH_BACKENDS,
    check_research_backend,
    write_research_backend_report,
)


def main() -> int:
    args = build_parser().parse_args()
    report = check_research_backend(args.backend, args.repo_dir)
    output_path = args.output or (
        PROJECT_ROOT / "eval_outputs" / "research_backends" / f"{args.backend}_setup.json"
    )
    write_research_backend_report(report, output_path)
    payload = report.as_dict()
    payload["report_path"] = str(output_path)
    print(json.dumps(payload, indent=2))
    if args.require_ready and not report.ready:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check optional DAM4SAM/SAM2Long research backend setup.",
    )
    parser.add_argument("--backend", choices=sorted(VALID_RESEARCH_BACKENDS), required=True)
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero when the backend is not ready for evaluation.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
