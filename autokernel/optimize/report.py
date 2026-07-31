"""Morning report generation for optimize campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .state import write_text_atomic


def write_morning_report(
    path: Path,
    *,
    receipt: Mapping[str, Any],
    stage_summaries: Mapping[str, Any] | None = None,
) -> Path:
    """Write a human-readable morning report distinguishing isolated vs e2e."""
    terminal = receipt.get("terminal") or receipt.get("status") or "unknown"
    lines = [
        "# MotionKernel Optimize Campaign — Morning Report",
        "",
        f"**Terminal status:** `{terminal}`",
        f"**Model:** `{receipt.get('model')}`",
        f"**Workload:** `{receipt.get('workload')}`",
        f"**Baseline mode:** `{receipt.get('baseline_mode')}`",
        f"**Min e2e speedup for promotion:** `{receipt.get('min_e2e_speedup')}`",
        f"**Budget (hours):** `{receipt.get('budget_hours')}`",
        f"**Started:** {receipt.get('started_at')}",
        f"**Finished:** {receipt.get('finished_at')}",
        "",
        "## Policy",
        "",
        "- Isolated operator speedup alone **never** promotes a kernel.",
        "- Promotion requires end-to-end generation improvement at or above "
        "the configured threshold with acceptable parity.",
        "- A neutral or low-impact outcome is reported as "
        "`no_worthwhile_candidate`, not success.",
        "",
        "## Completed stages",
        "",
    ]
    for stage in receipt.get("completed_stages") or []:
        lines.append(f"- `{stage}`")
    failed = receipt.get("failed_stages") or {}
    if failed:
        lines.extend(["", "## Failed stages", ""])
        for name, reason in failed.items():
            lines.append(f"- `{name}`: {reason}")

    records = receipt.get("stage_records") or {}
    if records:
        lines.extend(["", "## Stage metrics", ""])
        for name, record in records.items():
            metrics = record.get("metrics") or {}
            msg = record.get("message") or ""
            lines.append(
                f"- **{name}** status=`{record.get('status')}` "
                f"metrics=`{metrics}` {msg}"
            )

    if stage_summaries:
        lines.extend(["", "## Stage summaries", ""])
        for name, summary in stage_summaries.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"```json\n{summary}\n```")
            lines.append("")

    candidates = receipt.get("candidates") or []
    lines.extend(["", "## Candidates", ""])
    if not candidates:
        lines.append("_No candidates retained._")
    else:
        for item in candidates:
            lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"Run directory: `{receipt.get('output')}`",
            f"Receipt: `receipt.json`",
            f"State: `state.json`",
            "",
            f"## Conclusion",
            "",
            f"{receipt.get('message') or terminal}",
            "",
        ]
    )
    text = "\n".join(lines)
    write_text_atomic(path, text)
    return path
