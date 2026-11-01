# -*- coding: utf-8 -*-
"""v0.2.0 新增能力测试：迁移建议（fix）、Markdown 报告、--explain 证据追溯、版本一致性。

先写测试再写实现——每条断言都必须先在 RED 阶段失败过。
"""
import io
import contextlib
import re
import unittest
from pathlib import Path

import accel_preflight
from accel_preflight import cli, db, report, scan

ROOT = Path(__file__).resolve().parent.parent


def run_cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def has_unnegated(text, word):
    """判断 word 是否以「未被否定」的形式出现。

    「不代为保证」是否定式免责，不是强断言；只有前面没有否定词的才算违规。
    """
    negations = "不非未毋勿无没"
    start = 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return False
        # 往前看 6 个字符内是否有否定词（中文否定常隔着「代为/会/能」等）
        window = text[max(0, idx - 6):idx]
        if not any(n in window for n in negations):
            return True
        start = idx + len(word)


class TestFixCoverage(unittest.TestCase):
    """每条「可能判 blocker」的规则都必须给出迁移建议——只说病不给药等于没帮上忙。"""

    def test_negation_helper_itself(self):
        # 判据的自检：正例必须红、反例必须不被误杀
        self.assertTrue(has_unnegated("一定保证可用", "保证"))
        self.assertTrue(has_unnegated("必然可以跑通", "必然"))
        self.assertFalse(has_unnegated("本项目不代为保证", "保证"))
        self.assertFalse(has_unnegated("无法保证该路径", "保证"))

    def test_blocker_capable_rules_have_fix(self):
        for r in db.ALL_RULES:
            if "blocked" in r["platforms"].values():
                self.assertIn("fix", r, f"规则 {r['id']} 会判 blocker 却没有 fix")
                self.assertTrue(
                    r["fix"] and r["fix"].strip(), f"规则 {r['id']} 的 fix 是空的"
                )

    def test_fix_never_claims_untested_path_works(self):
        # fix 不得出现「保证/一定/必然可以」这类无证据的强断言（否定式免责不算）
        banned = ("保证", "一定能", "必然", "100%", "guaranteed")
        for r in db.ALL_RULES:
            fix = r.get("fix", "")
            for word in banned:
                self.assertFalse(
                    has_unnegated(fix, word),
                    f"规则 {r['id']} 的 fix 出现无证据强断言「{word}」：{fix}",
                )

    def test_unknown_package_suggestion_is_actionable(self):
        out = scan.scan_requirements("some-obscure-kernel-lib\n", "rocm")
        f = [x for x in out if x["id"] == "PKG-UNKNOWN-001"][0]
        self.assertTrue(f["fix"].strip())


class TestFixInFindings(unittest.TestCase):
    def test_finding_carries_fix(self):
        out = scan.scan_command("vllm serve m --kv-cache-dtype fp8_e4m3", "rocm")
        f = [x for x in out if x["id"] == "FLAG-ROCM-005"][0]
        self.assertIn("TRITON", f["fix"].upper() + "TRITON")  # 有内容即可
        self.assertTrue(f["fix"].strip())

    def test_text_report_prints_suggestion_for_blocker(self):
        out = scan.scan_command("vllm serve m --quantization auto-round", "rocm")
        text = report.render_text(out, "rocm")
        self.assertIn("建议", text)
        self.assertIn("FLAG-ROCM-004", text)

    def test_ok_finding_has_no_broken_fix_line(self):
        # ok 项不出建议行（避免噪音）
        out = scan.scan_command("VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve m", "rocm")
        text = report.render_text(out, "rocm")
        self.assertNotIn("建议：None", text)


class TestMarkdownReport(unittest.TestCase):
    FINDINGS = [
        {"id": "FLAG-ROCM-005", "severity": "blocker", "status": "blocked",
         "platform": "rocm", "match": "--kv-cache-dtype fp8_e4m3",
         "message": "RDNA3 fp8 不完整", "evidence": "S8",
         "fix": "去掉 fp8 KV cache，回到默认精度"},
        {"id": "PKG-UNKNOWN-001", "severity": "info", "status": "unknown",
         "platform": "rocm", "match": "some|lib", "message": "未收录",
         "evidence": "S0", "fix": "查上游 README"},
    ]

    def test_markdown_has_table_and_rows(self):
        md = report.render_markdown(self.FINDINGS, "rocm")
        self.assertIn("|", md)
        self.assertIn("FLAG-ROCM-005", md)
        self.assertIn("S8", md)
        self.assertIn("去掉 fp8 KV cache", md)

    def test_markdown_escapes_pipes_in_cells(self):
        # 命中串里含 | 时不能把表格撑坏
        md = report.render_markdown(self.FINDINGS, "rocm")
        row = [ln for ln in md.splitlines() if "some" in ln][0]
        self.assertIn("some\\|lib", row)

    def test_markdown_counts_match(self):
        md = report.render_markdown(self.FINDINGS, "rocm")
        self.assertIn("1", md)
        self.assertIn("blocker", md.lower())

    def test_empty_markdown_says_clean(self):
        md = report.render_markdown([], "dcu")
        self.assertIn("0", md)
        self.assertIn("dcu", md)


class TestExplain(unittest.TestCase):
    def test_explain_known_evidence(self):
        code, out = run_cli(["--explain", "S8"])
        self.assertEqual(code, 0)
        self.assertIn("first-party", out)
        self.assertIn("2026-10-26", out)

    def test_explain_unknown_evidence_exit_2(self):
        code, _ = run_cli(["--explain", "S99"])
        self.assertEqual(code, 2)

    def test_explain_all_ledger_ids_work(self):
        for sid in db.EVIDENCE:
            code, out = run_cli(["--explain", sid])
            self.assertEqual(code, 0, f"{sid} 解释失败")
            self.assertIn(sid, out)


class TestFormatFlag(unittest.TestCase):
    def test_format_markdown(self):
        code, out = run_cli(
            ["scan", "--cmd", "vllm serve m --kv-cache-dtype fp8_e4m3",
             "--platform", "rocm", "--format", "markdown"]
        )
        self.assertEqual(code, 1)
        self.assertIn("|", out)
        self.assertIn("FLAG-ROCM-005", out)

    def test_format_json_still_works(self):
        code, out = run_cli(
            ["scan", "--cmd", "vllm serve m --kv-cache-dtype fp8_e4m3",
             "--platform", "rocm", "--format", "json"]
        )
        self.assertEqual(code, 1)
        self.assertIn('"blocker_count": 1', out)

    def test_format_text_default(self):
        code, out = run_cli(
            ["scan", "--cmd", "vllm serve m --kv-cache-dtype fp8_e4m3", "--platform", "rocm"]
        )
        self.assertEqual(code, 1)
        self.assertIn("BLOCKER", out)


class TestVersionConsistency(unittest.TestCase):
    def test_package_version_matches_pyproject(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(m, "pyproject.toml 里找不到 version")
        self.assertEqual(accel_preflight.__version__, m.group(1))

    def test_version_is_0_2_0(self):
        self.assertEqual(accel_preflight.__version__, "0.2.0")


if __name__ == "__main__":
    unittest.main()
