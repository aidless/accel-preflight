# 学习笔记：异构算力推理生态调研（2026-10-31）

建这个仓库之前做的一轮生态调研。所有条目都指向可复查的仓库 README
（查证日期 2026-10-31），结论只写「README 说了什么」，不外推运行效果。

## 1. 海光（HYGON-AI org）的适配生态

组织页：https://github.com/orgs/HYGON-AI/repositories（截至查证日共 20 个仓库）

核心发现：海光把模型侧适配统一收在 **`*-das`** 命名系列下（README 自述
"为 hcu 平台提供功能适配"——注意品牌用词是 **HCU**，不只是 "DCU"）：

| 仓库 | ★ | 内容 |
|------|---|------|
| inference-cookbook-das | 89 | LLM/VLM/Omni 部署调优最佳实践 + 模型适配矩阵（K100_AI / BW1000 / BW1100） |
| sglang-das | 47 | SGLang 的 HCU 适配 |
| vllm-plugin-das | 28 | vLLM 插件式 HCU 适配 |
| Megatron-LM-das | 7 | Megatron-LM 适配 |
| apex-das | — | NVIDIA apex 的 HCU/HIP kernel 融合优化 |
| 其余 | — | DeepEP-das、rocSHMEM-das、TileKernels-das、lightop（自研融合算子库）、one-click-quant、train-cookbook-das、mooncake-das、LMCache-das、slime-das、xla-das、tensorflow-das、pytorch-das、SkillHub、quality-gate |

值得注意的**缺口**：org 里没有 DeepSpeed-das——本仓库的规则库因此也
不给 deepspeed 写任何 DCU 状态（unknown 才是诚实答案）。

## 2. 各推理栈的异构支持面（上游 README 原文口径）

- **FlashInfer**：只有 CUDA Support（12.9 / 13.0 / 13.4）。没有 ROCm 分发 →
  一切 flashinfer 系 backend 选择在 ROCm 目标上都是死路（规则库 S1）。
- **FlashAttention**：README 明说 "CUDA toolkit or ROCm toolkit"，另有 AMD
  官方移植仓 rocm/flash-attention。但它只是 variant 不是 ok：wheel 名和
  安装路径与 CUDA 版不同（S2/S3）。
- **SGLang**：平台表含 AMD Instinct MI300X/MI325X/MI350X/MI355X（S4）。
- **Triton**：LLVM target 含 NVPTX 与 AMDGPU，官方有 AMD backend 调试节（S5）。
- **vLLM**：README 写 "full CUDA/HIP graphs"、NVIDIA/AMD GPUs + hardware
  plugins 机制（S6）——这也是 HYGON-AI 能做 vllm-plugin-das 的前提。

## 3. CUDA-only 路径清单（一手实测，2026-10-26/28）

在一台 Radeon PRO W7900（gfx1100/RDNA3, ROCm 7.2.1, vLLM 0.16.1.dev+rocm721）
上逐项核查两套服务栈的启动配置，5 类路径过不去：

1. flashinfer 系 backend（mamba / linear-attn decode+prefill）；
2. `--speculative-algorithm NEXTN`（MTP 投机）；
3. `--quantization auto-round`（W4A16 反量化走 CUDA 路径）；
4. `--kv-cache-dtype fp8_e4m3`（RDNA3 fp8 不完整）；
5. NVFP4 / modelopt_fp4：Blackwell 专属，连非 Blackwell 的 NVIDIA 卡都过不去。

恢复路径：`VLLM_ATTENTION_BACKEND=TRITON_ATTN`。

**排查教训**：flash-attn 2.8.3 缺 `flash_attn_2_cuda` 扩展时，vLLM 引擎必崩，
但表层报错是 `Failed core proc(s): {}`——什么信息都没有。真异常在
EngineCore 子进程日志的尾部。这正是「预检应发生在装环境之前」的理由：
等引擎崩了再查，浪费的是数小时的 GPU 机时。

## 4. 这个仓库学到/固化的方法

- **证据台账模式**：每条规则强制挂一条证据，`kind ∈ {upstream-doc,
  first-party, meta}`，一手证据必须带 `as_of`；
- **越权防呆**：测试强制「一手 ROCm 证据不得给 dcu 写任何状态」——
  用测试把诚实固化，而不是靠自觉；
- **unknown 是一等公民**：查不到就发 `PKG-UNKNOWN-001`，并指路（dcu → S7）；
- **RED→GREEN 自检**：`--prove-red` 注入违规断言门会红、注入干净输入断言门
  不误杀，两段都过才算门活着。
