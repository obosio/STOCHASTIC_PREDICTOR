#!/usr/bin/env python3
"""Generate test execution JSON report from a provided exit code."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from framework.discovery import get_project_root

FRAMEWORK_VERSION = "2.1.0"


def _collect_files(root: Path) -> list[str]:
    files: list[str] = []
    if not root.exists():
        return files
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        files.append(str(path.relative_to(get_project_root())))
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tests execution report JSON")
    parser.add_argument(
        "--exit-code",
        type=int,
        required=True,
        help="Pytest exit code",
    )
    args = parser.parse_args()

    root = get_project_root()
    results_dir = root / "Test" / "results"
    results_dir.mkdir(exist_ok=True)

    status = "PASS" if args.exit_code == 0 else "FAIL"

    payload = {
        "metadata": {
            "report_id": "tests_generation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "source": "Test/scripts/tests_generation.py",
            "framework_version": FRAMEWORK_VERSION,
            "notes": "Pytest execution summary",
        },
        "summary": {
            "title": "Execution Summary",
            "metrics": [
                {"label": "Status", "value": status},
                {"label": "Exit Code", "value": args.exit_code},
            ],
        },
        "scope": {
            "targets": {
                "folders": ["Test/tests/unit"],
                "files": _collect_files(root / "Test" / "tests" / "unit"),
                "modules": [],
                "functions": [],
                "classes": [],
            }
        },
        "details": {
            "type": "table",
            "columns": ["Metric", "Value"],
            "rows": [
                ["Exit Code", str(args.exit_code)],
                ["Framework Version", FRAMEWORK_VERSION],
            ],
        },
        "issues": {
            "type": "list",
            "items": ["No findings"] if args.exit_code == 0 else ["Tests failed"],
        },
        "extras": [],
    }

    json_file = results_dir / "tests_generation_last.json"
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Test report JSON: {json_file.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
