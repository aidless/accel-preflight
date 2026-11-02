#!/usr/bin/env python3
"""accel-preflight 扫描延迟基准（真实测量，非估算）。

度量「一条 vllm/sglang 启动命令被 scan_command 扫一遍」的延迟，
以及「一份 requirements.txt 被 scan_requirements 扫一遍」的延迟。

为什么这是基准而不是玩具：accel-preflight 是一个 CI 预检 CLI，
其价值在于**毫秒级完成、零网络、纯标准库**——如果扫描要几百毫秒，
把它接进「按下回车前」的预检就失去意义。所以这里量的是：
  (a) 冷启动 import 成本（每次 CLI 进程都要付）；
  (b) 典型命令的扫描延迟（一条 vllm serve 启动串，含若干 CUDA-only 参数）；
  (c) 大 requirements.txt 的扫描延迟（200 行依赖，模拟真实迁移清单）。

结果写入 bench/results.json，数字全部来自本机实测（time.perf_counter），
不是任何估算或外部来源。

用法：
    python3 bench/scan_latency.py            # 写 bench/results.json 并打印
    python3 bench/scan_latency.py --repeat 20   # 每项重复 20 次取中位
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from accel_preflight import db, scan  # noqa: E402


# 典型 vllm serve 启动串：混入几个真实 CUDA-only 参数（来自 db.py 规则值）
TYPICAL_CMD = (
    "vllm serve Qwen/Qwen3-8B "
    "--tensor-parallel-size 4 "
    "--kv-cache-dtype fp8_e4m3 "
    "--mamba-backend flashinfer "
    "--speculative-algorithm NEXTN "
    "--quantization auto-round "
    "--max-model-len 32768 "
    "--gpu-memory-utilization 0.85"
)

# 200 行真实样式 requirements.txt（依赖名 + 平台后缀轮子，模拟迁移清单）
BIG_REQS = "\n".join(
    f"some-pkg{i}==1.2.{i % 10}\n"
    f"torch==2.5.1+cu128\n" if i % 10 == 0 else f"pkg-{i}>=0.{i % 5}.0"
    for i in range(200)
)


def bench_one(fn, repeat):
    """跑 fn repeat 次，返回 (中位秒, 最小秒, 最大秒)。"""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def main():
    repeat = 20
    if "--repeat" in sys.argv:
        try:
            repeat = int(sys.argv[sys.argv.index("--repeat") + 1])
        except (ValueError, IndexError):
            repeat = 20

    # 冷启动 import（每次 CLI 进程都付的固定成本）
    t0 = time.perf_counter()
    import accel_preflight.cli  # noqa: F401
    cold_import_s = time.perf_counter() - t0

    # 典型命令扫描（rocm 平台，命中若干 blocker）
    med_cmd, mn_cmd, mx_cmd = bench_one(
        lambda: scan.scan_command(TYPICAL_CMD, "rocm"), repeat
    )
    n_cmd_findings = len(scan.scan_command(TYPICAL_CMD, "rocm"))

    # 大 requirements.txt 扫描（dcu 平台）
    med_req, mn_req, mx_req = bench_one(
        lambda: scan.scan_requirements(BIG_REQS, "dcu"), repeat
    )
    n_req_findings = len(scan.scan_requirements(BIG_REQS, "dcu"))

    # 规则库规模（证据口径）
    n_flag = len(db.flag_rules())
    n_env = len(db.env_rules())
    n_pkg = len(db.package_rules())

    results = {
        "tool": "accel-preflight",
        "version": "0.3.0",
        "method": "time.perf_counter, median of %d reps" % repeat,
        "cold_import_s": round(cold_import_s, 6),
        "scan_command": {
            "cmd_sample": TYPICAL_CMD,
            "platform": "rocm",
            "findings": n_cmd_findings,
            "median_s": round(med_cmd, 6),
            "min_s": round(mn_cmd, 6),
            "max_s": round(mx_cmd, 6),
        },
        "scan_requirements_200": {
            "platform": "dcu",
            "lines": len(BIG_REQS.splitlines()),
            "findings": n_req_findings,
            "median_s": round(med_req, 6),
            "min_s": round(mn_req, 6),
            "max_s": round(mx_req, 6),
        },
        "rule_counts": {
            "flag_rules": n_flag,
            "env_rules": n_env,
            "package_rules": n_pkg,
        },
    }

    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"accel-preflight scan latency (median of {repeat}):")
    print(f"  cold import        : {cold_import_s:.6f} s")
    print(f"  cmd scan (rocm)    : {med_cmd:.6f} s  [{n_cmd_findings} findings]")
    print(f"  reqs 200 (dcu)     : {med_req:.6f} s  [{n_req_findings} findings]")
    print(f"  rules              : {n_flag} flag / {n_env} env / {n_pkg} pkg")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
