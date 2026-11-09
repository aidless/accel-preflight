# Changelog

## 0.4.0 — 2026-11-09

诚实边界，不是新的硬件结果。状态仍是 WIP：没有 DCU 捕获，S8 的原始日志不在仓库里。

### Changed
- 报告增加 `basis`：`observed` / `upstream-doc` / `unknown`。只有 S8 且状态不是 unknown 才是 `observed`，并带上 gfx1100 的 `scope`。
- FlashInfer 系 flag 在 DCU 上改为 `blocked`，证据是 S1（CUDA-only 发行物），不引用 S8。W7900 上的 fp8 / NEXTN / auto-round 在 DCU 上仍是 `unknown`。
- NVFP4 不再给 NVIDIA 写 `partial`。没有单独的非 Blackwell 机器记录。
- 包名按 PEP 503 折叠。`flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache` 列入 S1（2026-11-09 重读 README）。
- `whl/cuNNN` 索引、非 torch 的 `+cu` 标签、`torch @ …/whl/cu…` URL 在 rocm/dcu 上拒绝。`+cpu` 不再被 `startswith("cu")` 误判成 CUDA 并在 nvidia 上给 `ok`。
- `ok` 规则的前缀命中降为 `partial`。optional-dependencies 与 dependency-groups 会扫。`-r` 不打开，并显式标 unknown。
- S9 记了 2026-11-09 对 pytorch.org 安装页的重读：示例是 `whl/cu118`，ROCm 6.3 带删除线。`+cu`/`+rocm` 本地版本仍是 2026-10-31 的台账约定，这次页面正文没有再印出。

### Docs
- README 的问题陈述压到三行，并分开 OBSERVED 与 UNKNOWN。
- 新增 `STATUS.md`（WIP）和 `FAILURE_MODES.md`（假绿与 DCU 假结论）。
- 测试数、CI 的 Python 版本、禁止 PyPI 徽章和作者绝对路径，由 `tests/test_honesty.py` 对着仓库现况核对。

## 0.3.0 — 2026-10-31

规则库按「先有上游原文，再写规则」扩了一轮，并把 DCU 从「一片 unknown」变成「有据可查」。

### Added
- **量化格式判定（S10）**：vLLM 官方 Supported Hardware 矩阵——AWQ / GPTQ / Marlin /
  llm-compressor INT8 / bitsandbytes / DeepSpeedFP 在 AMD GPU 列全为 ❌ → `FLAG-QUANT-001`
  在 rocm 判 blocked（这是换内核问题，不是调参问题）；FP8 W8A8 与 GGUF 为 ✅ →
  `FLAG-QUANT-002`。
- **DCU 厂商件规则**：`vllm-plugin-das`（S13）、`sglang-das`（S12）在 dcu 判 ok，
  在其他平台判 variant。
- **证据 S10–S13**（全部 `upstream-doc`，不是一手实测）：S10 量化矩阵；S11 上游 sglang
  把 Hygon 列在「Integrations in progress」；S12 厂商 cookbook 的型号×模型 ✅ 矩阵；
  S13 厂商插件 README——`export ROCM_PATH=/opt/dtk`、CI 按 DTK 26.04/26.10 release line
  匹配。

### Changed
- `vllm` / `sglang` 的 dcu 说明从「需插件」细化为「逐型号逐模型，指到 S11/S12/S13」。
- **包名匹配改成精确优先于前缀**：`vllm-plugin-das` 原本会被 `vllm` 的前缀匹配抢先命中，
  判成上游 vllm 的 `partial`，把「厂商插件可用」这个更要紧的结论抹掉。
- 版本一致性测试不再钉死 `"0.2.0"`，改为 semver 格式 + 与 pyproject 相等。

### Tests
- 54 → **69** 个测试，全绿；新增 `tests/test_quant_and_dcu.py`（15 条），先红后绿。
- 新断言：DCU 上跑 `vllm / sglang / vllm-plugin-das` 不得出现任何 `unknown`，
  且每条判定必须能追到厂商侧来源（S7/S12/S13）——「有据」不是「有个状态就算」。

## 0.2.0 — 2026-10-31

从「指出病情」到「给出处方」。

### Added
- **迁移建议（`fix`）**：每条会判 blocker 的规则都带一条修复建议（15 条规则全覆盖）。
  措辞守证据：能指到实测恢复路径的就指（flashinfer 系 → `VLLM_ATTENTION_BACKEND=TRITON_ATTN`，S8），
  只能"关掉这个选项"的就直说关掉，不暗示未验证的替代方案。
- **Markdown 迁移报告**：`--format markdown` 输出一张可直接贴进迁移指南/工单的表格
  （规则 / 状态 / 命中 / 证据 / 说明 / 迁移建议），单元格内的 `|` 会自动转义。
- **证据追溯 `--explain <证据ID>`**：打印证据的类型、来源、记录日期、说明，以及引用它
  的全部规则；未知 ID 退出 2。
- 文本报告的 blocker/notice 条目下新增「建议」行。

### Changed
- `--json` 保留为 `--format json` 的别名（旧脚本不受影响）。
- `--prove-red` 自检从 6 项扩到 8 项（新增「每条 blocker 都带迁移建议」与
  「markdown 报告含表格结构」）。
- 版本号 0.1.0 → 0.2.0。

### Tests
- 35 → **54** 个单元测试，全绿。
- 新增反例断言：`fix` 不得出现「保证 / 一定能 / 必然 / 100%」这类无证据强断言。
  该判据本身配了正反自检（`has_unnegated`）——否定式免责（"本项目不代为保证"）
  不算违规，真实强断言必须红。
- 新增版本一致性断言：`accel_preflight.__version__` 必须等于 `pyproject.toml` 的 version。

## 0.1.0 — 2026-10-31

首版。

### Added
- 证据台账驱动的规则库（S0–S9）：上游 README 引用（flashinfer / flash-attention /
  rocm-flash-attention / sglang / triton / vllm / HYGON-AI org / pytorch.org wheel 渠道）
  ＋ 一手实测记录（S8，AMD Radeon PRO W7900、ROCm 7.2.1）。
- 扫描命令行参数、环境变量、requirements.txt、pyproject.toml、torch wheel 本地版本后缀。
- 五种状态（ok / variant / partial / blocked / unknown），未收录的包一律 `PKG-UNKNOWN-001`。
- 硬断言 `test_first_party_never_claims_dcu`：一手 ROCm 证据不得给 dcu 平台写任何状态。
- 退出码约定 0 / 1 / 2；`--prove-red` RED→GREEN 自检。
- 零依赖，Python ≥ 3.8；GitHub Actions 在 3.10–3.13 上跑测试与自检。
