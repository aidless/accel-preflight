# Changelog

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
