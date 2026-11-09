# 证据台账（docs/sources.md）

`db.py` 的 `EVIDENCE` 与本文件一一对应。任何规则的状态都必须能沿
`evidence` 指针追到这里，并沿 `source` 追到可复查的原文。

| id | kind | 来源 | as_of | 说明 |
|----|------|------|-------|------|
| S0 | meta | n/a | — | 无来源占位：数据库没收录的包/参数一律 unknown，不猜测 |
| S1 | upstream-doc | github.com/flashinfer-ai/flashinfer README。2026-10-31 查到 CUDA Support 12.9/13.0/13.4；2026-11-09 重读 raw main：CUDA Support 仍是这三版，GPU Support 表只有 NVIDIA SM75–SM121，安装名是 `flashinfer-python`（另有 `flashinfer-cubin`、被写成 CUDA-specific shim 的 `flashinfer-jit-cache`）。该次阅读没有 ROCm/HIP 行 | 重读于 2026-11-09 | 发行面是 CUDA/NVIDIA。本仓库没有在 ROCm 或 DCU 上重跑 FlashInfer |
| S2 | upstream-doc | github.com/Dao-AILab/flash-attention（README「CUDA toolkit or ROCm toolkit」「AMD ROCm Support」） | 查证于 2026-10-31 | FlashAttention 上游声明支持 ROCm 工具链 |
| S3 | upstream-doc | github.com/rocm/flash-attention | 查证于 2026-10-31 | AMD 官方维护的 FlashAttention ROCm 移植仓库 |
| S4 | upstream-doc | github.com/sgl-project/sglang（README 平台表：AMD Instinct MI300X/MI325X/MI350X/MI355X） | 查证于 2026-10-31 | SGLang 上游支持 AMD Instinct |
| S5 | upstream-doc | github.com/triton-lang/triton（README：LLVM target「NVPTX;AMDGPU」+ AMD backend 调试节） | 查证于 2026-10-31 | Triton 有 NVIDIA/AMD 双后端 |
| S6 | upstream-doc | github.com/vllm-project/vllm（README「full CUDA/HIP graphs」；NVIDIA/AMD GPUs） | 查证于 2026-10-31 | vLLM 上游支持 AMD GPU（HIP 路径） |
| S7 | upstream-doc | github.com/orgs/HYGON-AI/repositories（vllm-plugin-das / sglang-das / inference-cookbook-das / apex-das / Megatron-LM-das 等） | 查证于 2026-10-31 | 海光官方组织，`das` 系列 = 为 HCU 平台提供功能适配。org 现列仓库中**没有** DeepSpeed-das |
| S8 | first-party | 一手实测记录（私有日志，**未入库**）：AMD Radeon PRO W7900（gfx1100, RDNA3）、ROCm 7.2.1、vLLM 0.16.1.dev+rocm721 | 2026-10-26/28 | 该会话：flashinfer 系 backend、NEXTN、auto-round、fp8 KV cache 不可用；`TRITON_ATTN` 可恢复。`docs/notes` 同一节还写了 NVFP4，但原始台账起初没写，日志又不在仓库里，NVFP4 行更弱。**不是其他 ROCm GPU，不是 DCU，不能从本树复算** |
| S9 | upstream-doc | pytorch.org/get-started/locally。页面自述 Last updated 2026-10-28；2026-11-09 重读：Compute Platform 可见 CUDA 11.8 / 12.6 / 12.8、CPU，以及**带删除线**的 ROCm 6.3。示例命令是 `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` | 重读于 2026-11-09 | 索引 URL 是这次重读看到的。`+cu` / `+rocm` 本地版本是 2026-10-31 台账里的 PEP 440 约定，这次页面正文没有再印出这些后缀。删除线中的 ROCm 6.3 不能写成「当前仍在发行」。没有 DCU 渠道 |
| S10 | upstream-doc | github.com/vllm-project/vllm `docs/features/quantization/README.md`（Supported Hardware 表） | 查证于 2026-10-31 | 量化矩阵：AWQ / GPTQ / Marlin / llm-compressor INT8(W8A8,W4A8) / bitsandbytes / DeepSpeedFP 在 **AMD GPU 列全 ❌**；llm-compressor FP8 (W8A8) 与 GGUF **✅**。表自述 subject to change |
| S11 | upstream-doc | github.com/sgl-project/sglang（README 平台表之后的「Integrations in progress」列表） | 查证于 2026-10-31 | 上游把 **Hygon HCU/DCU** 列在「集成进行中」，未进支持表 → 裸上游 sglang 不能假设在 DCU 可用 |
| S12 | upstream-doc | github.com/HYGON-AI/inference-cookbook-das（模型列表：K100_AI / BW1000 / BW1100 × 框架，逐项 ✅ / 🚧 / -） | 查证于 2026-10-31 | 厂商按**型号×模型×框架**逐项给验证状态（vLLM / SGLang / vLLM-Omni…）：DCU 可用性有据，但口径是逐项的，不是全量 |
| S13 | upstream-doc | github.com/HYGON-AI/vllm-plugin-das（README：`export ROCM_PATH=/opt/dtk`；CI lock 用 `match: release_line` 匹配 DTK 26.04/26.10） | 查证于 2026-10-31 | DTK 装在 `/opt/dtk` 且经 `ROCM_PATH` 被构建链发现 → **DCU 软件栈属 ROCm/DTK 系**（厂商原文，非本项目推断）；插件与上游 vLLM 之间有 PEP 440 版本门 |

## 为什么一手证据（S8）不给 DCU 写任何状态

S8 只在一张 AMD Radeon 消费卡上产生。DCU/HCU 是不同的硬件、不同的驱动栈、
不同厂商维护的推理插件。允许 S8 给 `dcu` 写 `blocked` 或 `ok` 都是越权。
`tests/test_db_integrity.py::test_first_party_never_claims_dcu` 核对的是**每条平台实际引用的证据**：一手证据不能出现在 `dcu` 上；非 `unknown` 的一手结论只能挂在 `rocm`。规则可以带 `dcu` 键，但那个键必须改挂 S1/S9 这类 `upstream-doc`。

DCU 上的 `ok` / `partial` 来自厂商或上游文档，`basis=upstream-doc`，不是实测：

- **S11** 上游 sglang 仍把 Hygon 列为「集成进行中」→ 裸上游 wheel 记 `partial`；
- **S12** 厂商 cookbook 按型号×模型给出 ✅ 矩阵 → 文档里有逐项记录，不是全量，也不是我们跑的；
- **S13** 厂商 vLLM 插件把 DTK 放在 `/opt/dtk`、经 `ROCM_PATH` 取用 → 栈属 ROCm/DTK 系是厂商原文。

另外一条不来自 S8：S1/S9 把 CUDA-only 发行物和 `whl/cuNNN` 渠道在 `dcu` 上记 `blocked`。这是拒绝 CUDA 渠道字符串，不是 DCU 跑挂了的记录。厂商插件、上游包、CUDA 渠道三者都对不上时，仍然是 `unknown`。
