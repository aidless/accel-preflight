# -*- coding: utf-8 -*-
"""报告渲染：文本（人读）、JSON（CI）、Markdown（贴进迁移指南），以及证据解释。"""

import json

from accel_preflight import db

_ORDER = {"blocked": 0, "variant": 1, "partial": 2, "unknown": 3, "ok": 4}


def _fix_line(finding, indent):
    fix = finding.get("fix") or ""
    if not fix.strip():
        return None
    return f"{indent}建议：{fix}"


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
            fix_line = _fix_line(f, "      ")
            if fix_line:
                lines.append(fix_line)

    if others:
        lines.append("")
        lines.append(f"NOTICE ({len(others)}):")
        for f in others:
            lines.append(
                f"  [{f['id']}] {f['match']}  →  {f['status'].upper()}"
                f"  (证据 {f['evidence']})"
            )
            lines.append(f"      {f['message']}")
            fix_line = _fix_line(f, "      ")
            if fix_line:
                lines.append(fix_line)

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


def _md_cell(text):
    """表格单元格转义：| 必须转义，换行压成空格，否则表格会散架。"""
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(findings, platform):
    """Markdown 报告：一段能直接贴进迁移指南/工单的表格。"""
    ordered = sorted(findings, key=lambda f: _ORDER.get(f["status"], 9))
    blockers = [f for f in ordered if f["status"] == "blocked"]

    lines = [
        f"# accel-preflight 迁移预检报告",
        "",
        f"- 目标平台：`{platform}`",
        f"- blocker：**{len(blockers)}**",
        f"- 条目合计：{len(ordered)}",
        "",
    ]

    if not ordered:
        lines.append(f"未发现需要处理的问题（0 blocker）。")
        return "\n".join(lines)

    lines.append("| 规则 | 状态 | 命中 | 证据 | 说明 | 迁移建议 |")
    lines.append("|---|---|---|---|---|---|")
    for f in ordered:
        lines.append(
            "| `{id}` | {status} | `{match}` | {ev} | {msg} | {fix} |".format(
                id=_md_cell(f["id"]),
                status=_md_cell(f["status"]),
                match=_md_cell(f["match"]),
                ev=_md_cell(f["evidence"]),
                msg=_md_cell(f["message"]),
                fix=_md_cell(f.get("fix") or "—"),
            )
        )
    return "\n".join(lines)


def render_explain(evidence_id):
    """打印一条证据的完整出处，让「这个判断凭什么」经得起追问。"""
    ev = db.EVIDENCE.get(evidence_id)
    if ev is None:
        raise KeyError(evidence_id)
    lines = [
        f"证据 {evidence_id}",
        f"  类型：{ev['kind']}",
        f"  来源：{ev['source']}",
    ]
    if ev.get("as_of"):
        lines.append(f"  记录日期：{ev['as_of']}")
    lines.append(f"  说明：{ev['note']}")

    used_by = [r["id"] for r in db.ALL_RULES if r["evidence"] == evidence_id]
    lines.append("  引用规则：" + (", ".join(used_by) if used_by else "（无，兜底/指路用）"))
    return "\n".join(lines)


def exit_code(findings):
    """约定：有 blocked → 1（CI 应失败）；否则 0。"""
    has_blocker = any(f["status"] == "blocked" for f in findings)
    return 1 if has_blocker else 0
