# -*- coding: utf-8 -*-
"""扫描器测试：命令行 / 环境变量 / requirements / pyproject 四种输入，注入违规必须被抓住（RED 反证）。"""
import unittest

from accel_preflight import scan

ROCM = "rocm"


def ids(findings):
    return [f["id"] for f in findings]


class TestScanCommand(unittest.TestCase):
    def test_flashinfer_mamba_backend_flagged_on_rocm(self):
        out = scan.scan_command(
            "vllm serve Qwen/Qwen3-8B --mamba-backend flashinfer --port 8000", ROCM
        )
        self.assertIn("FLAG-ROCM-001", ids(out))
        blocker = [f for f in out if f["id"] == "FLAG-ROCM-001"]
        self.assertEqual(blocker[0]["status"], "blocked")

    def test_flag_equals_form_also_caught(self):
        out = scan.scan_command(
            "vllm serve m --linear-attn-prefill-backend=flashinfer", ROCM
        )
        self.assertIn("FLAG-ROCM-002", ids(out))

    def test_nextn_speculative_flagged_on_rocm(self):
        out = scan.scan_command(
            "python -m sglang.launch_server m --speculative-algorithm NEXTN", ROCM
        )
        self.assertIn("FLAG-ROCM-003", ids(out))

    def test_auto_round_quantization_flagged_on_rocm(self):
        out = scan.scan_command("vllm serve m --quantization auto-round", ROCM)
        self.assertIn("FLAG-ROCM-004", ids(out))

    def test_fp8_kv_cache_flagged_on_rocm(self):
        out = scan.scan_command("vllm serve m --kv-cache-dtype fp8_e4m3", ROCM)
        self.assertIn("FLAG-ROCM-005", ids(out))

    def test_nvfp4_quantization_flagged(self):
        out = scan.scan_command("vllm serve m --quantization modelopt_fp4", "nvidia")
        self.assertIn("FLAG-NVFP4-001", ids(out))

    def test_clean_command_has_no_blockers(self):
        out = scan.scan_command(
            "vllm serve Qwen/Qwen3-8B --max-model-len 16384 "
            "--gpu-memory-utilization 0.8 --port 8000",
            ROCM,
        )
        self.assertEqual([f for f in out if f["status"] == "blocked"], [])

    def test_quoted_values_do_not_confuse_parser(self):
        out = scan.scan_command(
            'vllm serve m --served-model-name "my --quantization auto-round model"',
            ROCM,
        )
        # 模型名里的字符串不是真的 flag，不应命中
        self.assertNotIn("FLAG-ROCM-004", ids(out))

    def test_env_backend_rules(self):
        out = scan.scan_command(
            "VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve m", ROCM
        )
        self.assertIn("ENV-ROCM-001", ids(out))
        out2 = scan.scan_command(
            "VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve m", ROCM
        )
        self.assertIn("ENV-ROCM-002", ids(out2))
        self.assertEqual(
            [f for f in out2 if f["id"] == "ENV-ROCM-002"][0]["status"], "ok"
        )


class TestScanRequirements(unittest.TestCase):
    def test_flashinfer_dependency_flagged_on_rocm(self):
        out = scan.scan_requirements(
            "torch\nflashinfer\nvllm\n", ROCM
        )
        self.assertIn("PKG-FLASHINFER-001", ids(out))
        self.assertEqual(
            [f for f in out if f["id"] == "PKG-FLASHINFER-001"][0]["status"], "blocked"
        )

    def test_flash_attn_is_variant_not_blocker_on_rocm(self):
        out = scan.scan_requirements("flash-attn\n", ROCM)
        self.assertIn("PKG-FLASHATTN-001", ids(out))
        self.assertEqual(
            [f for f in out if f["id"] == "PKG-FLASHATTN-001"][0]["status"], "variant"
        )

    def test_vllm_ok_on_rocm(self):
        out = scan.scan_requirements("vllm==0.11.0\n", ROCM)
        self.assertIn("PKG-VLLM-001", ids(out))
        self.assertEqual(
            [f for f in out if f["id"] == "PKG-VLLM-001"][0]["status"], "ok"
        )

    def test_vllm_on_dcu_points_to_vendor_plugin(self):
        out = scan.scan_requirements("vllm\n", "dcu")
        self.assertIn("PKG-VLLM-001", ids(out))
        self.assertEqual(
            [f for f in out if f["id"] == "PKG-VLLM-001"][0]["status"], "partial"
        )

    def test_unknown_package_reports_unknown_not_guess(self):
        out = scan.scan_requirements("some-obscure-kernel-lib\n", ROCM)
        self.assertIn("PKG-UNKNOWN-001", ids(out))
        f = [f for f in out if f["id"] == "PKG-UNKNOWN-001"][0]
        self.assertEqual(f["status"], "unknown")

    def test_comments_and_blank_lines_ignored(self):
        out = scan.scan_requirements("# flashinfer  # 注释行不算依赖\n\n", ROCM)
        self.assertEqual(ids(out), [])

    def test_name_normalization_ignores_extras_and_case(self):
        out = scan.scan_requirements("FlashInfer[all]==0.2.2\n", ROCM)
        self.assertIn("PKG-FLASHINFER-001", ids(out))


class TestScanPyproject(unittest.TestCase):
    def test_pyproject_dependencies_scanned(self):
        text = """
[project]
name = "demo"
dependencies = [
  "torch>=2.9",
  "flashinfer",
]
"""
        out = scan.scan_pyproject(text, ROCM)
        self.assertIn("PKG-FLASHINFER-001", ids(out))


class TestScanTorchWheel(unittest.TestCase):
    def test_cu_wheel_blocked_on_rocm(self):
        out = scan.scan_requirements("torch==2.9.1+cu128\n", ROCM)
        self.assertIn("PKG-TORCHWHEEL-001", ids(out))
        self.assertEqual(
            [f for f in out if f["id"] == "PKG-TORCHWHEEL-001"][0]["status"], "blocked"
        )

    def test_rocm_wheel_ok_on_rocm(self):
        out = scan.scan_requirements("torch==2.9.1+rocm7.2\n", ROCM)
        self.assertIn("PKG-TORCHWHEEL-001", ids(out))
        self.assertEqual(
            [f for f in out if f["id"] == "PKG-TORCHWHEEL-001"][0]["status"], "ok"
        )

    def test_plain_pin_is_unknown_on_rocm(self):
        out = scan.scan_requirements("torch==2.9.1\n", ROCM)
        self.assertEqual(
            [f for f in out if f["id"] == "PKG-TORCHWHEEL-001"][0]["status"], "unknown"
        )


if __name__ == "__main__":
    unittest.main()
