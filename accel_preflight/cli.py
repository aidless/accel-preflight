# -*- coding: utf-8 -*-
"""CLI 入口。

用法：
    accel-preflight scan --cmd "vllm serve ..." --platform rocm
    accel-preflight scan --requirements requirements.txt --platform dcu --format markdown
    accel-preflight scan --pyproject pyproject.toml --platform dcu
    accel-preflight --explain S8
    accel-preflight --prove-red

退出码：0 = 无 blocker；1 = 有 blocker（CI 应失败）；2 = 用法错误。
"""

import argparse
import sys

from accel_preflight import db, report, scan

_USAGE = """usage:
  accel-preflight scan --cmd "<命令行>" [--platform {nvidia,rocm,dcu}] [--format {text,json,markdown}]
  accel-preflight scan --requirements <file> [--platform ...] [--format ...]
  accel-preflight scan --pyproject <file> [--platform ...] [--format ...]
  accel-preflight --explain <证据ID>
  accel-preflight --prove-red
"""


def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _render(findings, platform, fmt):
    if fmt == "json":
        return report.render_json(findings, platform)
    if fmt == "markdown":
        return report.render_markdown(findings, platform)
    return report.render_text(findings, platform)


def _do_scan(args):
    findings = []
    if args.cmd is not None:
        findings += scan.scan_command(args.cmd, args.platform)
    if args.requirements is not None:
        findings += scan.scan_requirements(_read_file(args.requirements), args.platform)
    if args.pyproject is not None:
        findings += scan.scan_pyproject(_read_file(args.pyproject), args.platform)

    print(_render(findings, args.platform, args.format))
    return report.exit_code(findings)


def _do_explain(evidence_id):
    try:
        print(report.render_explain(evidence_id))
    except KeyError:
        print(
            f"未知证据 ID: {evidence_id}（台账里只有 {', '.join(sorted(db.EVIDENCE))}）",
            file=sys.stderr,
        )
        return 2
    return 0


def _prove_red():
    """自检：注入违规必须被抓（红），干净输入必须放行（绿）。任何一条不成立 → 退出 1。"""
    bad = scan.scan_command(
        "vllm serve m --kv-cache-dtype fp8_e4m3 --quantization auto-round", "rocm"
    )
    clean = scan.scan_command("vllm serve m --port 8000", "rocm")
    bad_req = scan.scan_requirements(
        "torch==2.9.1+cu128\nflashinfer\nsome-obscure-kernel-lib\n", "rocm"
    )
    cuda_index = scan.scan_requirements(
        "--extra-index-url https://download.pytorch.org/whl/cu118\ntorch==2.9.1\n",
        "rocm",
    )
    dcu_fp8 = scan.scan_command("vllm serve m --kv-cache-dtype fp8_e4m3", "dcu")
    dcu_plugin = scan.scan_requirements("vllm-plugin-das==0.28.1\n", "dcu")

    checks = [
        ("fp8 KV cache 被判 blocked", any(
            f["id"] == "FLAG-ROCM-005" and f["status"] == "blocked" for f in bad)),
        ("fp8 KV cache 的 basis 是 observed", any(
            f["id"] == "FLAG-ROCM-005" and f["basis"] == "observed" for f in bad)),
        ("auto-round 被判 blocked", any(
            f["id"] == "FLAG-ROCM-004" and f["status"] == "blocked" for f in bad)),
        ("干净命令 0 blocker", not any(f["status"] == "blocked" for f in clean)),
        ("CUDA wheel 装到 rocm 被判 blocked", any(
            f["id"] == "PKG-TORCHWHEEL-001" and f["status"] == "blocked" for f in bad_req)),
        ("flashinfer 被判 blocked", any(
            f["id"] == "PKG-FLASHINFER-001" and f["status"] == "blocked" for f in bad_req)),
        ("未知包写 unknown 而不是猜", any(
            f["id"] == "PKG-UNKNOWN-001" and f["status"] == "unknown" for f in bad_req)),
        ("whl/cu118 索引在 rocm 上被判 blocked", any(
            f["id"] == "PKG-INDEXURL-001" and f["status"] == "blocked" for f in cuda_index)),
        ("DCU 不继承 W7900 的 fp8 结论", all(
            f["basis"] != "observed" for f in dcu_fp8) and any(
            f["status"] == "unknown" for f in dcu_fp8)),
        ("DCU 插件 ok 的 basis 不是 observed", any(
            f["id"] == "PKG-VLLM-DAS-001" and f["status"] == "ok"
            and f["basis"] == "upstream-doc" for f in dcu_plugin)),
        ("每条 blocker 都带迁移建议", all(
            f.get("fix", "").strip() for f in bad if f["status"] == "blocked")),
        ("markdown 报告含表格结构", "|" in report.render_markdown(bad, "rocm")),
    ]
    print("accel-preflight prove-red（自检：门必须能红，也要能绿）")
    failed = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
    if failed:
        print(f"prove-red: {failed} 项自检失败", file=sys.stderr)
        return 1
    print("prove-red: all checks passed")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="accel-preflight",
        description="异构算力可移植性预检：在按下回车前，先告诉你哪些 CUDA-only 路径过不去。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_USAGE,
    )
    parser.add_argument("--prove-red", action="store_true", help="运行自检（RED→GREEN 反证）")
    parser.add_argument("--explain", metavar="证据ID", help="解释一条证据的出处（如 S8）")
    sub = parser.add_subparsers(dest="sub")

    p_scan = sub.add_parser("scan", help="扫描命令行 / 依赖文件")
    p_scan.add_argument("--cmd", help="要扫描的启动命令")
    p_scan.add_argument("--requirements", help="requirements.txt 路径")
    p_scan.add_argument("--pyproject", help="pyproject.toml 路径")
    p_scan.add_argument("--platform", choices=db.PLATFORMS, default="rocm")
    p_scan.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text",
        help="报告格式（默认 text）",
    )
    p_scan.add_argument("--json", action="store_true", help="等价于 --format json")

    args = parser.parse_args(argv)

    if args.explain is not None:
        return _do_explain(args.explain)
    if args.prove_red:
        return _prove_red()

    if args.sub != "scan":
        parser.print_usage(sys.stderr)
        return 2
    if args.cmd is None and args.requirements is None and args.pyproject is None:
        parser.print_usage(sys.stderr)
        return 2
    if args.json:
        args.format = "json"

    return _do_scan(args)
