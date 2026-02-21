"""HTML telemetry dashboard generation.

Generates a static HTML dashboard from telemetry records without
external dependencies. Intended for offline inspection and audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from Python.io.telemetry import TelemetryBuffer, materialize_telemetry_batch


@dataclass(frozen=True)
class DashboardSeries:
    label: str
    values: list[float]


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_series(records: Iterable[dict], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        raw = record.get(key)
        scalar = _coerce_float(raw)
        if scalar is None:
            continue
        values.append(scalar)
    return values


def _svg_line_chart(series: DashboardSeries, width: int, height: int, spread_epsilon: float) -> str:
    if not series.values:
        return '<div class="chart-empty">No data</div>'

    min_val = min(series.values)
    max_val = max(series.values)
    spread = max(max_val - min_val, spread_epsilon)

    points = []
    count = len(series.values)
    for idx, val in enumerate(series.values):
        x = int(idx * (width - 2) / max(count - 1, 1)) + 1
        normalized = (val - min_val) / spread
        y = int((1.0 - normalized) * (height - 2)) + 1
        points.append(f"{x},{y}")

    polyline = " ".join(points)
    return (
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"<polyline fill='none' stroke='currentColor' stroke-width='2' points='{polyline}'/>"
        f"</svg>"
    )


def build_dashboard_html(
    records: list[dict],
    output_path: str | Path,
    config: Any,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = config.telemetry_dashboard_title
    chart_width = config.telemetry_dashboard_width
    chart_height = config.telemetry_dashboard_height
    spread_epsilon = config.telemetry_dashboard_spread_epsilon
    recent_rows = config.telemetry_dashboard_recent_rows

    series = [
        DashboardSeries("Free Energy", _extract_series(records, "free_energy")),
        DashboardSeries("Holder Exponent", _extract_series(records, "holder_exponent")),
        DashboardSeries("DGM Entropy", _extract_series(records, "dgm_entropy")),
        DashboardSeries("Adaptive Threshold", _extract_series(records, "adaptive_threshold")),
    ]

    rows = []
    for record in records[-recent_rows:]:
        step = record.get("step", "-")
        prediction = record.get("prediction", "-")
        mode = record.get("mode", "-")
        free_energy = record.get("free_energy", "-")
        rows.append(
            "<tr>" f"<td>{step}</td>" f"<td>{prediction}</td>" f"<td>{mode}</td>" f"<td>{free_energy}</td>" "</tr>"
        )

    charts_html = "\n".join(
        "<div class='card'>"
        f"<h3>{item.label}</h3>"
        f"{_svg_line_chart(item, chart_width, chart_height, spread_epsilon)}"
        "</div>"
        for item in series
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --ink: #1a1a1a;
      --muted: #5a5a5a;
      --card: #ffffff;
      --accent: #003049;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 32px 40px 10px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 32px;
      color: var(--accent);
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      padding: 20px 40px;
    }}
    .card {{
      background: var(--card);
      padding: 16px;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }}
    .chart {{
      width: 100%;
      height: 180px;
      color: var(--accent);
    }}
    .chart-empty {{
      padding: 40px 0;
      text-align: center;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid #e3dccf;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    footer {{
      padding: 0 40px 30px;
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="sub">Generated: {timestamp} · Records: {len(records)}</div>
  </header>
  <section class="grid">
    {charts_html}
  </section>
  <section class="grid">
    <div class="card" style="grid-column: 1 / -1;">
      <h3>Latest Telemetry</h3>
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Prediction</th>
            <th>Mode</th>
            <th>Free Energy</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </div>
  </section>
  <footer>
    USP dashboard generated from telemetry buffer drain.
  </footer>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def export_dashboard_snapshot(
    telemetry_buffer: TelemetryBuffer,
    output_path: str | Path,
    config: Any,
) -> int:
    """Drain telemetry buffer and write an HTML dashboard.

    Returns the number of records exported.
    """
    records = telemetry_buffer.drain()
    payloads = materialize_telemetry_batch(records, config)
    build_dashboard_html(payloads, output_path, config)
    return len(payloads)


__all__ = [
    "DashboardSeries",
    "build_dashboard_html",
    "export_dashboard_snapshot",
]
