# PREREG — accel-preflight 预注册与评测规格

> 本文件是 accel-preflight 的**预注册规格**（registered-report 风格）：声明本仓库**实际执行、可从落盘证据复算**的异构可移植性预检假设、评测方案、成功标准与明确边界。
> 所有数字来自仓库内文件（`accel_preflight/db.py`、`bench/results.json`、`tests/`）与本机实测，**不是虚构实验**。

---

## 0. 一句话假设（先读这行）

> accel-preflight 能在**毫秒级**、**零网络、纯标准库**下，把一条 CUDA-only 的 vLLM/SGLang 启动命令或一份 `requirements.txt` 扫成带证据背书的 `blocked`/`partial`/`unknown` 判定，**且判定是静态规则匹配、不声称运行时验证**。

---

## 1. 主假设 H1（可移植性预检可判定）

- **假设**：给定 `--cmd "vllm serve ... --kv-cache-dtype fp8_e4m3" --platform rocm`，扫描器产生 `BLOCKER` 判定（证据 S8），退出码 1（CI 应失败）。
- **可证伪方向**：若同一条命令扫出 `blocked` 却无对应证据背书，或退出码不为 1，则 H1 不成立。
- **评测方案**：`scan_command` 逐 flag 匹配 `db.flag_rules()`，逐 env 匹配 `db.env_rules()`；每条 finding 带 `{id, severity, status, platform, match, evidence, note}`；`blocked` 才计 blocker。
- **成功标准**：README 顶部示例输出（1 blocker）可复现，`exit=1`。

## 2. 假设 H2（证据不外推 / first-party 口径）

- **假设**：一手实测证据（`kind=first-party`）只给 ROCm 平台背书；涉及 DCU 的状态一律来自上游公开文档（`partial`/`ok`/`unknown`），**绝不外推**。
- **可证伪方向**：若某条 DCU 判定用了 first-party 证据，则 H2 不成立。
- **评测方案**：`db.py` 证据台账 S0–S13，`docs/sources.md` 记录来源与日期；`--explain` 可追问单条证据出处。
- **成功标准**：DCU 相关状态口径为「厂商插件/适配仓→`ok`，仅上游包→`partial`，都没有→`unknown`」，逐型号×逐模型（S12）。

## 3. 假设 H3（性能：扫描延迟在毫秒级）

- **假设**：单条命令扫描延迟 < 1 ms，200 行 requirements 扫描延迟 < 5 ms，冷启动 import < 10 ms（**bench/results.json 实测**）。
- **可证伪方向**：若实测中位延迟超过上述量级，则「CI 预检」定位不成立。
- **评测方案**：`bench/scan_latency.py` 用 `time.perf_counter`，每项重复 20 次取中位。
- **成功标准**：冷启动 2.289 ms；命令扫描 0.090 ms（4 findings）；200 行 reqs 0.715 ms（220 findings）（2026-11-01 实测）。

## 4. 假设 H4（门必须能红也能绿 / RED→GREEN）

- **假设**：`--prove-red` 注入明确违规 → 断言真拦（红）；注入干净 → 断言不误杀（绿）。
- **可证伪方向**：若任一注入违规未拦，或干净输入被误杀，则门是装饰。
- **评测方案**：`accel_preflight/cli.py::_prove_red` + `tests/test_fixes.py`。
- **成功标准**：`python3 -m accel_preflight --prove-red` exit≠0 且断言全部触发。

---

## 5. 明确边界（不证明什么）

| 维度 | 不证明 | 原因 |
|---|---|---|
| 目标机跑通 | 「一定能跑通」 | 静态规则匹配，非运行时验证；`blocked` 清掉后仍须真实冒烟 |
| DCU 可用性 | DCU/HCU 上的可用性 | 不同硬件/驱动栈/厂商插件；一手证据从未在 DCU 产生过 |
| 规则完备性 | 覆盖全部包/参数 | 只覆盖台账收录项；未收录报 `PKG-UNKNOWN`/`FLAG-UNKNOWN` 并指向出处 |
| 上游矩阵稳定性 | vLLM/SGLang 官方支持矩阵稳定 | S10 等上游来源自述 "subject to change"，需按 `as_of` 日期重查 |

---

## 6. 结论纪律

- 所有判定数字与规则在仓库内重算，零外部服务（见 README「复算」节）。
- `blocked` 被清掉 ≠ 已证明能跑通——必须真实冒烟。
- 本 PREREG 是规格声明，不新增实验，不改变已冻结规则库（`db.py`）与测试基线（`tests/`）。
