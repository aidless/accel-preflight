# -*- coding: utf-8 -*-
"""Honesty drift guards.

These tests pin the false greens that let CUDA-only inputs exit 0, the DCU
claim boundary (no first-party basis), and the numbers CI actually runs.
They do not contact a GPU or the network.
"""
import ast
import importlib.util
import re
import unittest
from pathlib import Path

from accel_preflight import db, report, scan

ROOT = Path(__file__).resolve().parent.parent


def ids(findings):
    return [f["id"] for f in findings]


def one(findings, rule_id):
    hits = [f for f in findings if f["id"] == rule_id]
    return hits[0] if hits else None


class TestFalseGreensClosed(unittest.TestCase):
    def test_pep503_flash_attn_underscore_and_dot(self):
        for line in ("flash_attn\n", "flash.attn\n"):
            out = scan.scan_requirements(line, "rocm")
            hit = one(out, "PKG-FLASHATTN-001")
            self.assertIsNotNone(hit, line)
            self.assertEqual(hit["status"], "variant")
            self.assertNotEqual(hit["status"], "unknown")

    def test_flashinfer_distribution_names_blocked(self):
        for name in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
            for platform in ("rocm", "dcu"):
                out = scan.scan_requirements(name + "\n", platform)
                hit = one(out, "PKG-FLASHINFER-001")
                self.assertIsNotNone(hit, (name, platform))
                self.assertEqual(hit["status"], "blocked")
                self.assertEqual(hit["basis"], "upstream-doc")
                self.assertEqual(hit["evidence"], "S1")
                self.assertNotIn("scope", hit)

    def test_cuda_index_url_blocked_on_non_nvidia(self):
        text = (
            "--extra-index-url https://download.pytorch.org/whl/cu118\n"
            "torch==2.9.1\n"
        )
        for platform in ("rocm", "dcu"):
            out = scan.scan_requirements(text, platform)
            hit = one(out, "PKG-INDEXURL-001")
            self.assertIsNotNone(hit, platform)
            self.assertEqual(hit["status"], "blocked")
            self.assertEqual(hit["basis"], "upstream-doc")
            self.assertEqual(report.exit_code(out), 1)
        nvidia = scan.scan_requirements(text, "nvidia")
        hit = one(nvidia, "PKG-INDEXURL-001")
        self.assertEqual(hit["status"], "ok")
        self.assertEqual(hit["basis"], "upstream-doc")
        self.assertEqual(report.exit_code(nvidia), 0)

    def test_torch_direct_url_cuda_channel_blocked(self):
        line = (
            "torch @ https://download.pytorch.org/whl/cu128/"
            "torch-2.5.0%2Bcu128-cp310-cp310-manylinux.whl\n"
        )
        out = scan.scan_requirements(line, "rocm")
        hit = one(out, "PKG-TORCHWHEEL-001")
        self.assertEqual(hit["status"], "blocked")
        self.assertEqual(hit["basis"], "upstream-doc")

    def test_non_torch_cu_tag_blocked_on_rocm_and_dcu(self):
        line = "torchvision==0.20.1+cu128\n"
        for platform in ("rocm", "dcu"):
            out = scan.scan_requirements(line, platform)
            hit = one(out, "PKG-WHEELTAG-001")
            self.assertIsNotNone(hit, platform)
            self.assertEqual(hit["status"], "blocked")
            self.assertEqual(hit["basis"], "upstream-doc")
            self.assertNotIn("PKG-UNKNOWN-001", ids(out))
        nvidia = scan.scan_requirements(line, "nvidia")
        self.assertIsNone(one(nvidia, "PKG-WHEELTAG-001"))
        self.assertEqual(one(nvidia, "PKG-UNKNOWN-001")["status"], "unknown")

    def test_cpu_local_version_is_not_a_cuda_ok(self):
        out = scan.scan_requirements("torch==2.9.1+cpu\n", "nvidia")
        hit = one(out, "PKG-TORCHWHEEL-001")
        self.assertEqual(hit["status"], "unknown")
        self.assertEqual(hit["basis"], "unknown")
        self.assertNotEqual(hit["status"], "ok")

    def test_rocm_tag_on_dcu_stays_unknown(self):
        out = scan.scan_requirements("torch==2.9.1+rocm7.2\n", "dcu")
        hit = one(out, "PKG-TORCHWHEEL-001")
        self.assertEqual(hit["status"], "unknown")
        self.assertEqual(hit["basis"], "unknown")

    def test_plain_torch_pin_stays_unknown(self):
        out = scan.scan_requirements("torch==2.9.1\n", "rocm")
        self.assertEqual(one(out, "PKG-TORCHWHEEL-001")["status"], "unknown")
        self.assertEqual(report.exit_code(out), 0)

    def test_prefix_ok_is_downgraded_and_blocked_prefix_stays_blocked(self):
        inherited = scan.scan_requirements("vllm-custom-build\n", "rocm")
        hit = one(inherited, "PKG-VLLM-001")
        self.assertEqual(hit["status"], "partial")
        self.assertNotEqual(hit["status"], "ok")
        blocked = scan.scan_requirements("flashinfer-nightly\n", "rocm")
        hit = one(blocked, "PKG-FLASHINFER-001")
        self.assertEqual(hit["status"], "blocked")

    def test_optional_and_group_dependencies_are_scanned(self):
        text = """
[project]
dependencies = ["numpy"]

[project.optional-dependencies]
cuda = ["flashinfer-python"]

[dependency-groups]
dev = ["flashinfer-cubin"]

[tool.other]
note = "flashinfer"
"""
        out = scan.scan_pyproject(text, "rocm")
        flash = [f for f in out if f["id"] == "PKG-FLASHINFER-001"]
        self.assertEqual(len(flash), 2)
        self.assertTrue(all(f["status"] == "blocked" for f in flash))
        # A quoted name outside the dependency sections must not become a hit.
        self.assertEqual(len(flash), 2)

    def test_includes_are_not_opened(self):
        out = scan.scan_requirements("-r cuda-reqs.txt\nnumpy\n", "rocm")
        self.assertIn("PKG-INCLUDE-001", ids(out))
        self.assertEqual(one(out, "PKG-INCLUDE-001")["status"], "unknown")
        self.assertNotIn("PKG-FLASHINFER-001", ids(out))
        self.assertEqual(report.exit_code(out), 0)

    def test_auto_round_underscore_is_a_known_hole(self):
        out = scan.scan_command("vllm serve m --quantization auto_round", "rocm")
        self.assertNotIn("FLAG-ROCM-004", ids(out))
        self.assertEqual(report.exit_code(out), 0)


class TestDcuClaimBoundary(unittest.TestCase):
    def test_nvfp4_on_nvidia_is_unknown(self):
        out = scan.scan_command("vllm serve m --quantization modelopt_fp4", "nvidia")
        hit = one(out, "FLAG-NVFP4-001")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["status"], "unknown")
        self.assertEqual(hit["basis"], "unknown")
        self.assertNotIn("scope", hit)

    def test_dcu_plugin_ok_is_upstream_doc(self):
        out = scan.scan_requirements("vllm-plugin-das==0.28.1\n", "dcu")
        hit = one(out, "PKG-VLLM-DAS-001")
        self.assertEqual(hit["status"], "ok")
        self.assertEqual(hit["basis"], "upstream-doc")
        self.assertNotEqual(hit["basis"], "observed")
        text = report.render_text(out, "dcu")
        self.assertIn("upstream-doc", text)
        self.assertNotIn("basis=observed", text)

    def test_w7900_flags_do_not_become_observed_on_dcu(self):
        cmd = (
            "vllm serve m --kv-cache-dtype fp8_e4m3 "
            "--quantization auto-round --speculative-algorithm NEXTN"
        )
        out = scan.scan_command(cmd, "dcu")
        self.assertTrue(out)
        for finding in out:
            self.assertNotEqual(finding["basis"], "observed", finding["id"])
            self.assertEqual(finding["status"], "unknown", finding["id"])
            self.assertNotEqual(finding["evidence"], "S8", finding["id"])

    def test_observed_scope_is_only_on_observed_rows(self):
        rocm = scan.scan_command("vllm serve m --kv-cache-dtype fp8_e4m3", "rocm")
        hit = one(rocm, "FLAG-ROCM-005")
        self.assertEqual(hit["basis"], "observed")
        self.assertIn("gfx1100", hit["scope"])
        text = report.render_text(rocm, "rocm")
        self.assertIn("basis=observed", text)
        self.assertIn("gfx1100", text)


class TestOfflineAndDrift(unittest.TestCase):
    def test_package_sources_do_not_import_network_clients(self):
        banned_roots = {"urllib", "requests", "httpx", "socket", "subprocess", "pip"}
        for path in (ROOT / "accel_preflight").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                self.assertFalse(
                    banned_roots.intersection(names),
                    f"{path.name} imports {names}",
                )

    def test_scan_text_takes_a_string(self):
        out = scan.scan_text("vllm serve m --quantization auto-round\n", "rocm")
        self.assertIn("FLAG-ROCM-004", ids(out))

    def test_readme_problem_statement_is_at_most_three_lines(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(
            r"<!-- problem:start -->\n(.*?)\n<!-- problem:end -->",
            readme,
            re.S,
        )
        self.assertIsNotNone(match, "README is missing the problem-statement markers")
        lines = [ln for ln in match.group(1).splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 1)
        self.assertLessEqual(len(lines), 3)
        for word in ("OBSERVED", "UNKNOWN", "Offline", "WIP"):
            self.assertIn(word, readme)

    def test_readme_test_count_matches_discovery(self):
        # Same files CI loads (`test_*.py`), counted without re-entering discover
        # from inside a running test (that recurses).
        loader = unittest.TestLoader()
        discovered = 0
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            spec = importlib.util.spec_from_file_location(f"honesty_count_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            discovered += loader.loadTestsFromModule(module).countTestCases()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"Discovered tests: (\d+)\.", readme)
        self.assertIsNotNone(match, "README must state 'Discovered tests: N.'")
        self.assertEqual(
            int(match.group(1)),
            discovered,
            f"README says {match.group(1)} but test_*.py contains {discovered}",
        )

    def test_readme_ci_python_matches_workflow(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        listed = re.search(r"python-version:\s*\[([^\]]+)\]", workflow)
        self.assertIsNotNone(listed)
        versions = re.findall(r'"([^"]+)"', listed.group(1))
        self.assertEqual(versions, ["3.10", "3.11", "3.12", "3.13"])
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("CI Python: " + ", ".join(versions), readme)
        self.assertNotIn("tested on 3.8", readme.lower())
        self.assertIn("python -m unittest discover -s tests -v", workflow)
        self.assertIn("python -m accel_preflight --prove-red", workflow)
        self.assertNotIn("pip install", workflow)

    def test_no_pypi_and_no_author_paths(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        blob = readme.lower() + "\n" + pyproject.lower()
        self.assertNotIn("pypi.org", blob)
        self.assertNotIn("shields.io/pypi", blob)
        self.assertIn("actions/workflows/ci.yml", readme)
        self.assertTrue((ROOT / ".github" / "workflows" / "ci.yml").is_file())
        allowed_suffixes = {".py", ".md", ".yml", ".toml", ".txt"}
        # Split so this source file does not itself contain the banned strings.
        banned = ("/Users" + "/", "/home" + "/", "C:" + "\\Users\\")
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix not in allowed_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in banned:
                self.assertNotIn(needle, text, str(path))

    def test_evidence_ids_match_sources_md(self):
        text = (ROOT / "docs" / "sources.md").read_text(encoding="utf-8")
        ids_in_doc = set(re.findall(r"\|\s*(S\d+)\s*\|", text))
        self.assertEqual(ids_in_doc, set(db.EVIDENCE))

    def test_status_and_failure_modes_are_present(self):
        status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
        modes = (ROOT / "FAILURE_MODES.md").read_text(encoding="utf-8")
        self.assertIn("WIP", status)
        self.assertIn("DCU", modes)
        self.assertIn("CUDA", modes)
        self.assertIn("observed", modes)


if __name__ == "__main__":
    unittest.main()
