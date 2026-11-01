# -*- coding: utf-8 -*-
"""v0.3.0：vLLM 量化后端矩阵 + DCU 不再是 unknown。

两条都是「先有上游原文，再写规则」：
- S10 vLLM 官方量化支持矩阵（AMD GPU 列：AWQ/GPTQ/Marlin/INT8/bnb/DeepSpeedFP 全 ❌，
  FP8 W8A8 与 GGUF ✅）
- S11 sglang 官方 README（Hygon HCU/DCU 仍列在「Integrations in progress」）
- S12 HYGON-AI inference-cookbook-das（按型号×模型逐项 ✅ 的厂商验证矩阵）
- S13 HYGON-AI vllm-plugin-das（DTK 在 /opt/dtk，用 ROCM_PATH 指路；CI 按 DTK
  26.04/26.10 release line 匹配）

本文件先写断言再写实现（RED 先红）。
"""
import unittest

from accel_preflight import db, report, scan

ROCM, NVIDIA, DCU = "rocm", "nvidia", "dcu"


def ids(findings):
    return [f["id"] for f in findings]


def by_id(findings, rule_id):
    hit = [f for f in findings if f["id"] == rule_id]
    return hit[0] if hit else None


class TestNewEvidence(unittest.TestCase):
    """新证据必须落台账、字段齐全、kind 合法。"""

    def test_s10_to_s13_registered(self):
        for ev_id in ("S10", "S11", "S12", "S13"):
            self.assertIn(ev_id, db.EVIDENCE, f"{ev_id} 未登记")
            ev = db.EVIDENCE[ev_id]
            self.assertTrue(ev.get("source"), f"{ev_id} 缺 source")
            self.assertTrue(ev.get("note"), f"{ev_id} 缺 note")
            self.assertIn(ev.get("kind"), ("upstream-doc", "first-party", "meta"))

    def test_vendor_evidence_is_upstream_doc_not_first_party(self):
        """厂商 README 是上游文档，不是我们的一手实测——不能混类。"""
        for ev_id in ("S10", "S11", "S12", "S13"):
            self.assertEqual(db.EVIDENCE[ev_id]["kind"], "upstream-doc", ev_id)


class TestQuantizationRules(unittest.TestCase):
    """S10 的 AMD GPU 列决定 rocm 判定。"""

    def test_cuda_only_quant_formats_blocked_on_rocm(self):
        rule = [r for r in db.ALL_RULES if r["id"] == "FLAG-QUANT-001"]
        self.assertTrue(rule, "FLAG-QUANT-001 未定义")
        rule = rule[0]
        self.assertEqual(rule["platforms"].get(ROCM), "blocked")
        self.assertEqual(rule["platforms"].get(NVIDIA), "ok")
        self.assertEqual(rule["evidence"], "S10")
        self.assertTrue((rule.get("fix") or "").strip(), "blocker 必须带迁移建议")
        for fmt in ("awq", "gptq", "gptq_marlin", "marlin", "bitsandbytes", "deepspeedfp"):
            self.assertIn(fmt, rule["values"], f"{fmt} 应列入 CUDA-only 量化格式")

    def test_gguf_ok_on_rocm(self):
        rule = [r for r in db.ALL_RULES if r["id"] == "FLAG-QUANT-002"]
        self.assertTrue(rule, "FLAG-QUANT-002 未定义")
        rule = rule[0]
        self.assertEqual(rule["platforms"].get(ROCM), "ok")
        self.assertEqual(rule["evidence"], "S10")
        self.assertIn("gguf", rule["values"])

    def test_scan_gptq_marlin_blocked_on_rocm(self):
        out = scan.scan_command("vllm serve m --quantization gptq_marlin", ROCM)
        self.assertIn("FLAG-QUANT-001", ids(out))
        self.assertEqual(by_id(out, "FLAG-QUANT-001")["status"], "blocked")
        self.assertEqual(report.exit_code(out), 1)

    def test_scan_bitsandbytes_blocked_on_rocm(self):
        out = scan.scan_command("vllm serve m --quantization=bitsandbytes", ROCM)
        self.assertEqual(by_id(out, "FLAG-QUANT-001")["status"], "blocked")

    def test_scan_awq_ok_on_nvidia(self):
        out = scan.scan_command("vllm serve m --quantization awq", NVIDIA)
        self.assertEqual(by_id(out, "FLAG-QUANT-001")["status"], "ok")
        self.assertEqual(report.exit_code(out), 0)

    def test_scan_gguf_not_a_blocker_on_rocm(self):
        out = scan.scan_command("vllm serve m --quantization gguf", ROCM)
        hit = by_id(out, "FLAG-QUANT-002")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["status"], "ok")
        self.assertEqual(report.exit_code(out), 0)


class TestDcuWiring(unittest.TestCase):
    """DCU 从 unknown 升级为「有据」：厂商插件 + 厂商验证矩阵。"""

    def test_das_plugins_ok_on_dcu(self):
        for rule_id, pkg in (
            ("PKG-VLLM-DAS-001", "vllm-plugin-das"),
            ("PKG-SGLANG-DAS-001", "sglang-das"),
        ):
            rule = [r for r in db.ALL_RULES if r["id"] == rule_id]
            self.assertTrue(rule, f"{rule_id} 未定义")
            self.assertEqual(rule[0]["platforms"].get(DCU), "ok", rule_id)
            self.assertEqual(rule[0]["kind"], "package")
            self.assertIn(pkg, rule[0]["package"])

    def test_das_plugin_scan_on_dcu(self):
        out = scan.scan_requirements("vllm-plugin-das==0.28.1\n", DCU)
        self.assertIn("PKG-VLLM-DAS-001", ids(out))
        self.assertEqual(by_id(out, "PKG-VLLM-DAS-001")["status"], "ok")

    def test_sglang_das_on_dcu_ok_on_rocm_variant(self):
        dcu_out = scan.scan_requirements("sglang-das\n", DCU)
        self.assertEqual(by_id(dcu_out, "PKG-SGLANG-DAS-001")["status"], "ok")
        rocm_out = scan.scan_requirements("sglang-das\n", ROCM)
        self.assertEqual(by_id(rocm_out, "PKG-SGLANG-DAS-001")["status"], "variant")

    def test_dcu_is_no_longer_all_unknown(self):
        """核心断言：DCU 上跑常见依赖，至少要有一条非 unknown 的判定。"""
        out = scan.scan_requirements("vllm\nsglang\nvllm-plugin-das\n", DCU)
        statuses = [f["status"] for f in out]
        self.assertNotIn("unknown", statuses, f"DCU 仍有 unknown: {out}")
        self.assertEqual(len({f["id"] for f in out}), 3)

    def test_dcu_findings_point_at_vendor_sources(self):
        """DCU 的判定必须能追到厂商侧来源（S7/S12/S13），不能只挂上游通用证据。"""
        out = scan.scan_requirements("vllm\nsglang\nvllm-plugin-das\n", DCU)
        dcu_sources = {"S7", "S12", "S13"}
        for f in out:
            traceable = f["evidence"] in dcu_sources or any(
                s in (f.get("message") or "") for s in dcu_sources
            )
            self.assertTrue(traceable, f"{f['id']} 无法追到厂商来源: {f}")


class TestExplainCoversNewEvidence(unittest.TestCase):
    def test_explain_s12_lists_citing_rules(self):
        text = report.render_explain("S12")
        self.assertIn("S12", text)
        self.assertIn("PKG-SGLANG-DAS-001", text)

    def test_explain_s13_lists_das_plugin(self):
        text = report.render_explain("S13")
        self.assertIn("PKG-VLLM-DAS-001", text)


if __name__ == "__main__":
    unittest.main()
