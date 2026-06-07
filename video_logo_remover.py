#!/usr/bin/env python3
"""Convenience entrypoint for the Phase 1 video logo removal prototype."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from logo_removal.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
