# -*- coding: utf-8 -*-
"""报告与 CLI 测试：文本/JSON 渲染、退出码、--prove-red 自检必须能红能绿。"""
import io
import contextlib
import json
import unittest

from accel_preflight import cli, report


def run_cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


class TestReport(unittest.TestCase):
    FINDINGS = [
        {"id": "FLAG-ROCM-005", "severity": "blocker", "status": "blocked",
         "platform": "rocm", "match": "--kv-cache-dtype fp8_e4m3",
         "message": "RDNA3 上 fp8 KV cache 不完整", "evidence": "S8"},
        {"id": "PKG-UNKNOWN-001", "severity": "info", "status": "unknown",
         "platform": "rocm", "match": "some-lib",
         "message": "无来源，不猜测", "evidence": "S0"},
    ]

    def test_text_report_has_sections(self):
        text = report.render_text(self.FINDINGS, "rocm")
        self.assertIn("BLOCKER", text)
        self.assertIn("UNKNOWN", text)
        self.assertIn("FLAG-ROCM-005", text)

    def test_json_report_roundtrip(self):
        data = json.loads(report.render_json(self.FINDINGS, "rocm"))
        self.assertEqual(data["platform"], "rocm")
        self.assertEqual(len(data["findings"]), 2)
        self.assertEqual(data["blocker_count"], 1)

    def test_empty_findings_renders_clean(self):
        text = report.render_text([], "rocm")
        self.assertIn("0 blocker", text.lower())


class TestCli(unittest.TestCase):
    def test_scan_bad_command_exit_1(self):
        code, out = run_cli(
            ["scan", "--cmd", "vllm serve m --kv-cache-dtype fp8_e4m3", "--platform", "rocm"]
        )
        self.assertEqual(code, 1)
        self.assertIn("FLAG-ROCM-005", out)

    def test_scan_clean_command_exit_0(self):
        code, out = run_cli(
            ["scan", "--cmd", "vllm serve m --port 8000", "--platform", "rocm"]
        )
        self.assertEqual(code, 0)

    def test_scan_requirements_file(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("torch\nflashinfer\n")
            path = f.name
        try:
            code, out = run_cli(
                ["scan", "--requirements", path, "--platform", "rocm"]
            )
            self.assertEqual(code, 1)
            self.assertIn("PKG-FLASHINFER-001", out)
        finally:
            os.unlink(path)

    def test_json_output_flag(self):
        code, out = run_cli(
            ["scan", "--cmd", "vllm serve m --kv-cache-dtype fp8_e4m3",
             "--platform", "rocm", "--json"]
        )
        self.assertEqual(code, 1)
        data = json.loads(out)
        self.assertEqual(data["blocker_count"], 1)

    def test_usage_error_exit_2(self):
        code, _ = run_cli(["scan"])
        self.assertEqual(code, 2)

    def test_prove_red_passes(self):
        code, out = run_cli(["--prove-red"])
        self.assertEqual(code, 0, out)
        self.assertIn("prove-red", out.lower())


if __name__ == "__main__":
    unittest.main()
