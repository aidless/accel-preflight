# 证据台账（docs/sources.md）

`db.py` 的 `EVIDENCE` 与本文件一一对应。任何规则的状态都必须能沿
`evidence` 指针追到这里，并沿 `source` 追到可复查的原文。

| id | kind | 来源 | as_of | 说明 |
|----|------|------|-------|------|
| S0 | meta | n/a | — | 无来源占位：数据库没收录的包/参数一律 unknown，不猜测 |
| S1 | upstream-doc | github.com/flashinfer-ai/flashinfer（README 仅「CUDA Support 12.9/13.0/13.4」） | 查证于 2026-10-31 | FlashInfer 上游只发 CUDA 轮子，无 ROCm 分发 |
| S2 | upstream-doc | github.com/Dao-AILab/flash-attention（README「CUDA toolkit or ROCm toolkit」「AMD ROCm Support」） | 查证于 2026-10-31 | FlashAttention 上游声明支持 ROCm 工具链 |
| S3 | upstream-doc | github.com/rocm/flash-attention | 查证于 2026-10-31 | AMD 官方维护的 FlashAttention ROCm 移植仓库 |
| S4 | upstream-doc | github.com/sgl-project/sglang（README 平台表：AMD Instinct MI300X/MI325X/MI350X/MI355X） | 查证于 2026-10-31 | SGLang 上游支持 AMD Instinct |
| S5 | upstream-doc | github.com/triton-lang/triton（README：LLVM target「NVPTX;AMDGPU」+ AMD backend 调试节） | 查证于 2026-10-31 | Triton 有 NVIDIA/AMD 双后端 |
| S6 | upstream-doc | github.com/vllm-project/vllm（README「full CUDA/HIP graphs」；NVIDIA/AMD GPUs） | 查证于 2026-10-31 | vLLM 上游支持 AMD GPU（HIP 路径） |
| S7 | upstream-doc | github.com/orgs/HYGON-AI/repositories（vllm-plugin-das / sglang-das / inference-cookbook-das / apex-das / Megatron-LM-das 等） | 查证于 2026-10-31 | 海光官方组织，`das` 系列 = 为 HCU 平台提供功能适配。org 现列仓库中**没有** DeepSpeed-das |
| S8 | first-party | 一手实测记录（私有日志）：AMD Radeon PRO W7900（gfx1100, RDNA3）、ROCm 7.2.1、vLLM 0.16.1.dev+rocm721 | 2026-10-26/28 | 逐项核查两套服务栈启动配置：flashinfer 系 backend、NEXTN 投机、auto-round 量化、fp8 KV cache 均不可用；`VLLM_ATTENTION_BACKEND=TRITON_ATTN` 可恢复。**仅代表该硬件/版本组合，不外推** |
| S9 | upstream-doc | pytorch.org/get-started/locally（Compute Platform 含 CUDA 11.8/12.6/12.8 与 ROCm 6.3 两条分发渠道；`--index-url .../whl/cu118`） | 复验于 2026-10-31 | PyTorch 官方按 CUDA/ROCm 渠道分发 wheel；wheel 本地版本后缀（`+cuXXX`/`+rocmXXX`）随之，装错渠道的包无法运行 |

## 为什么一手证据（S8）不给 DCU 写任何状态

S8 只在一张 AMD Radeon 消费卡上产生。DCU/HCU 是不同的硬件、不同的驱动栈、
不同厂商维护的推理插件。允许 S8 给 `dcu` 写 `blocked` 或 `ok` 都是越权，
所以 `tests/test_db_integrity.py::test_first_party_never_claims_dcu`
把「first-party 证据的规则不得含 dcu 键」写成硬断言；DCU 的答案只能来自
S7（厂商仓库）或未来的 DCU 一手实测。
