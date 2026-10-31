# -*- coding: utf-8 -*-
"""报告渲染：文本（人读）与 JSON（机读，CI 用）。"""

import json

from accel_preflight import db

_ORDER = {"blocked": 0, "variant": 1, "partial": 2, "unknown": 3, "ok": 4}


def render_text(findings, platform):
    """文本报告。空 findings 也必须明确说出 0 blocker，而不是沉默。"""
    lines = [f"accel-preflight · platform={platform}"]
    ordered = sorted(findings, key=lambda f: _ORDER.get(f["status"], 9))
    blockers = [f for f in ordered if f["status"] == "blocked"]
    others = [f for f in ordered if f["status"] != "blocked"]

    if blockers:
        lines.append("")
        lines.append(f"BLOCKER ({len(blockers)}):")
        for f in blockers:
            lines.append(
                f"  [{f['id']}] {f['match']}  →  {f['status']}"
                f"  (证据 {f['evidence']})"
            )
            lines.append(f"      {f['message']}")

    if others:
        lines.append("")
        lines.append(f"NOTICE ({len(others)}):")
        for f in others:
            lines.append(
                f"  [{f['id']}] {f['match']}  →  {f['status'].upper()}"
                f"  (证据 {f['evidence']})"
            )
            lines.append(f"      {f['message']}")

    lines.append("")
    lines.append(f"summary: {len(blockers)} blocker, {len(others)} notice")
    return "\n".join(lines)


def render_json(findings, platform):
    blocker_count = sum(1 for f in findings if f["status"] == "blocked")
    return json.dumps(
        {
            "platform": platform,
            "findings": findings,
            "blocker_count": blocker_count,
        },
        ensure_ascii=False,
        indent=2,
    )


def exit_code(findings):
    """约定：有 blocked → 1（CI 应失败）；否则 0。"""
    has_blocker = any(f["status"] == "blocked" for f in findings)
    return 1 if has_blocker else 0
