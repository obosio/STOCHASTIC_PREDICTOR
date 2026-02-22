#!/usr/bin/env python3
"""Generate Markdown reports from JSON results.

Reads JSON reports from Test/results and renders Markdown files into Test/reports.
All report content is derived from JSON sources using a fixed schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Test" / "results"
REPORTS_DIR = ROOT / "Test" / "reports"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_markdown(filename: str, content: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / filename).write_text(content.rstrip() + "\n", encoding="utf-8")


def _format_table(rows: Iterable[Tuple[str, str]], header: Tuple[str, str]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return lines


def _truncate(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _title_from_id(report_id: str) -> str:
    words = report_id.replace("_", " ").split()
    return " ".join(word.capitalize() for word in words) + " Report"


def _render_block(block: Dict[str, Any]) -> List[str]:
    block_type = block.get("type", "text")
    lines: List[str] = []

    if block_type == "table":
        columns = block.get("columns", [])
        rows = block.get("rows", [])
        if columns:
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(item) for item in row) + " |")
        return lines

    if block_type == "list":
        items = block.get("items", [])
        if not items:
            lines.append("- No findings")
        else:
            for item in items:
                lines.append(f"- {item}")
        return lines

    if block_type == "code":
        language = block.get("language", "text")
        content = block.get("content", "")
        lines.append(f"```{language}")
        lines.append(_truncate(str(content), 6000))
        lines.append("```")
        return lines

    if block_type == "breakdown":
        # Special handling for breakdown type (e.g., tools list + classification)
        tools = block.get("tools", [])
        if tools:
            lines.append("### Tools")
            lines.append("")
            lines.append(", ".join(tools))
        classification = block.get("classification", "")
        if classification:
            lines.append("")
            lines.append("### Classification")
            lines.append("")
            lines.append(classification)
        return lines

    content = block.get("content", "")
    if content:
        lines.append(str(content))
    return lines


def render_report(data: Dict[str, Any]) -> str:
    metadata = data.get("metadata", {})
    report_id = metadata.get("report_id", "report")
    title = _title_from_id(str(report_id))

    lines = [f"# {title}", ""]

    lines.append("## Metadata")
    lines.append("")
    metadata_rows = [
        ("Report ID", str(metadata.get("report_id", ""))),
        ("Timestamp (UTC)", str(metadata.get("timestamp_utc", ""))),
        ("Status", str(metadata.get("status", ""))),
        ("Source", str(metadata.get("source", ""))),
        ("Framework Version", str(metadata.get("framework_version", ""))),
        ("Notes", str(metadata.get("notes", ""))),
    ]
    lines.extend(_format_table(metadata_rows, ("Field", "Value")))
    lines.append("")

    summary = data.get("summary", {})
    lines.append(f"## {summary.get('title', 'Execution Summary')}")
    lines.append("")
    metrics = summary.get("metrics", [])
    metric_rows = [(m.get("label", ""), str(m.get("value", ""))) for m in metrics]
    lines.extend(_format_table(metric_rows, ("Metric", "Value")))
    lines.append("")

    scope = data.get("scope", {})
    targets = scope.get("targets", {})
    lines.append("## Scope")
    lines.append("")
    for key in ["folders", "files", "modules", "functions", "classes"]:
        values = targets.get(key, [])
        lines.append(f"### {key.capitalize()}")
        lines.append("")
        if values:
            for item in values:
                lines.append(f"- {item}")
        else:
            lines.append("- None")
        lines.append("")

    details = data.get("details", {})
    lines.append("## Details")
    lines.append("")
    lines.extend(_render_block(details))
    lines.append("")

    # ISSUES SECTION (MANDATORY - 3-level classification)
    issues_blocking = data.get("issues_blocking", {})
    issues_errors = data.get("issues_errors", {})
    issues_warnings = data.get("issues_warnings", {})

    # Check if we have 3-level classification fields
    has_3level = any([issues_blocking.get("type"), issues_errors.get("type"), issues_warnings.get("type")])

    if has_3level:
        # Use 3-level classification structure (STANDARD)
        lines.append("## Issues & Warnings")
        lines.append("")

        if issues_blocking.get("type"):
            lines.append("### Blocking Issues")
            lines.append("")
            lines.extend(_render_block(issues_blocking))
            lines.append("")

        if issues_errors.get("type"):
            lines.append("### Error Issues")
            lines.append("")
            lines.extend(_render_block(issues_errors))
            lines.append("")

        if issues_warnings.get("type"):
            lines.append("### Warning Issues")
            lines.append("")
            lines.extend(_render_block(issues_warnings))
            lines.append("")
    else:
        # Fallback: no 3-level structure found
        lines.append("## Issues & Warnings")
        lines.append("")
        lines.append("### Blocking Issues")
        lines.append("- No blocking issues")
        lines.append("")
        lines.append("### Error Issues")
        lines.append("- No error issues")
        lines.append("")
        lines.append("### Warning Issues")
        lines.append("- No warnings")
        lines.append("")

    extras = data.get("extras", [])
    if extras:
        lines.append("## Extras")
        lines.append("")
        for extra in extras:
            title = extra.get("title") or extra.get("id", "Extra")
            lines.append(f"### {title}")
            lines.append("")
            lines.extend(_render_block(extra))
            lines.append("")

    return "\n".join(lines)


def generate_all_reports() -> int:
    report_specs = {
        "code_fixtures_last.json": ("CODE_FIXTURES_LAST.md", render_report),
        "dependency_check_last.json": ("DEPENDENCY_CHECK_LAST.md", render_report),
        "code_lint_last.json": ("CODE_LINT_LAST.md", render_report),
        "code_alignment_last.json": ("CODE_ALIGNMENT_LAST.md", render_report),
        "code_structure_last.json": ("CODE_STRUCTURE_LAST.md", render_report),
        "tests_generation_last.json": ("TESTS_GENERATION_LAST.md", render_report),
        "summary_last.json": ("SUMMARY_LAST.md", render_report),
    }

    generated = 0
    for json_name, (md_name, renderer) in report_specs.items():
        data = _load_json(RESULTS_DIR / json_name)
        if not data:
            continue
        _write_markdown(md_name, renderer(data))
        generated += 1

    return generated


def main() -> int:
    generated = generate_all_reports()
    print(f"Generated {generated} Markdown report(s) in Test/reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
