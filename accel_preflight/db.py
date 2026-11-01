# -*- coding: utf-8 -*-
"""能力数据库：规则 + 证据台账。

每条规则的判定都挂在一条证据上（S0–S13），台账见 docs/sources.md。
一手实测证据（kind=first-party）只允许给 ROCm 平台背书——我们没摸过
DCU 硬件，所以凡是涉及 DCU 的状态一律来自上游公开文档，且查不到就写 unknown。
"""

# 平台集合。dcu 指海光 DCU/HCU 一类国产 GPGPU（其软件栈 DTK 以 ROCm/HIP 为底座）。
PLATFORMS = ("nvidia", "rocm", "dcu")

# 状态 → 严重度映射。unknown 永远不是 blocker。
STATUS_SEVERITY = {
    "ok": "info",
    "variant": "warning",
    "partial": "warning",
    "unknown": "info",
    "blocked": "blocker",
}

# ---------------------------------------------------------------------------
# 证据台账（与 docs/sources.md 一一对应）
# ---------------------------------------------------------------------------
EVIDENCE = {
    "S0": {
        "kind": "meta",
        "source": "n/a",
        "note": "无来源：数据库没收录的包/参数一律 unknown，不猜测",
    },
    "S1": {
        "kind": "upstream-doc",
        "source": "https://github.com/flashinfer-ai/flashinfer （README：仅 CUDA Support，12.9/13.0/13.4）",
        "note": "FlashInfer 上游只发 CUDA 轮子，无 ROCm 分发",
    },
    "S2": {
        "kind": "upstream-doc",
        "source": "https://github.com/Dao-AILab/flash-attention （README：『CUDA toolkit or ROCm toolkit』『### AMD ROCm Support』）",
        "note": "FlashAttention 上游声明支持 ROCm 工具链",
    },
    "S3": {
        "kind": "upstream-doc",
        "source": "https://github.com/rocm/flash-attention",
        "note": "AMD 官方维护的 FlashAttention ROCm 移植仓库",
    },
    "S4": {
        "kind": "upstream-doc",
        "source": "https://github.com/sgl-project/sglang （README 平台表：AMD Instinct MI300X/MI325X/MI350X/MI355X）",
        "note": "SGLang 上游支持 AMD Instinct 平台",
    },
    "S5": {
        "kind": "upstream-doc",
        "source": "https://github.com/triton-lang/triton （README：LLVM target 含 NVPTX;AMDGPU；AMD backend 调试一节）",
        "note": "Triton 有 NVIDIA 与 AMD 双后端",
    },
    "S6": {
        "kind": "upstream-doc",
        "source": "https://github.com/vllm-project/vllm （README：full CUDA/HIP graphs；NVIDIA/AMD GPUs 支持）",
        "note": "vLLM 上游支持 AMD GPU（HIP 路径）",
    },
    "S7": {
        "kind": "upstream-doc",
        "source": "https://github.com/orgs/HYGON-AI/repositories",
        "note": "海光官方组织：vllm-plugin-das / sglang-das / inference-cookbook-das / apex-das / Megatron-LM-das 等 das 适配仓库（README 自述『为 hcu 平台提供功能适配』）",
    },
    "S8": {
        "kind": "first-party",
        "as_of": "2026-10-26/28",
        "source": "一手实测记录（私有日志，未公开）：AMD Radeon PRO W7900 (gfx1100, RDNA3), ROCm 7.2.1, vLLM 0.16.1.dev+rocm721",
        "note": "该环境下逐项核查两套服务栈启动配置：flashinfer 系 backend、NEXTN 投机、auto-round 量化、fp8 KV cache 均不可用；VLLM_ATTENTION_BACKEND=TRITON_ATTN 可恢复。仅代表该硬件/版本组合，不外推到其他 ROCm 目标",
    },
    "S9": {
        "kind": "upstream-doc",
        "source": "https://pytorch.org/get-started/locally/ + PEP 440 本地版本号",
        "note": "PyTorch 官方按 +cuXXX / +rocmXXX 后缀区分 CUDA/ROCm wheel",
    },
    "S10": {
        "kind": "upstream-doc",
        "source": "https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/README.md （Supported Hardware 表）",
        "note": "vLLM 官方量化支持矩阵：AWQ / GPTQ / Marlin / llm-compressor INT8(W8A8,W4A8) / bitsandbytes / DeepSpeedFP 在 AMD GPU 列均为 ❌；llm-compressor FP8 (W8A8) 与 GGUF 为 ✅。文档自述该表 subject to change",
    },
    "S11": {
        "kind": "upstream-doc",
        "source": "https://github.com/sgl-project/sglang （README：Supported Hardware 表之后的 Integrations in progress 列表）",
        "note": "SGLang 上游把 Hygon HCU/DCU 列在「Integrations in progress」而非支持表——dcu 只能用下游适配仓，不能假设上游 wheel 直接可用",
    },
    "S12": {
        "kind": "upstream-doc",
        "source": "https://github.com/HYGON-AI/inference-cookbook-das （模型列表：按 K100_AI / BW1000 / BW1100 × 框架逐项标 ✅ 已验证 / 🚧 开发中 / - 暂未验证）",
        "note": "海光官方 cookbook 按型号×模型×框架逐项给出验证状态（vLLM / SGLang / vLLM-Omni 等）。含义：DCU 可用性有据可查，但是逐型号逐模型的口径，不是全量可用",
    },
    "S13": {
        "kind": "upstream-doc",
        "source": "https://github.com/HYGON-AI/vllm-plugin-das （README：export ROCM_PATH=/opt/dtk；CI lock 用 match: release_line 匹配 DTK 26.04/26.10；插件为 hcu 平台提供功能适配）",
        "note": "海光官方 vLLM 插件：DTK 装在 /opt/dtk 且经 ROCM_PATH 被构建链发现，DTK 版本按 26.04/26.10 release line 匹配 → DCU 软件栈属 ROCm/DTK 系（厂商原文，非本项目推断）；插件与上游 vLLM 之间有 PEP 440 版本门",
    },
}

# ---------------------------------------------------------------------------
# 规则
# kind=flag    : 服务栈启动参数（--xxx value / --xxx=value）
# kind=env     : 环境变量赋值（KEY=VALUE）
# kind=package : 依赖名（requirements / pyproject）
# kind=torch-wheel : torch 本地版本后缀（+cu / +rocm），逻辑规则，状态在扫描时计算
# platforms 缺某个平台 = 该平台状态 unknown（渲染时指向 S7 提示查厂商仓库）
# ---------------------------------------------------------------------------

FLAG_RULES = [
    {
        "id": "FLAG-ROCM-001",
        "kind": "flag",
        "flag": "--mamba-backend",
        "values": ["flashinfer"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "ok"},
        "evidence": "S8",
        "note": "rocm: 一手实测该 backend 不可用；nvidia: FlashInfer 是 CUDA 库（S1），默认可用",
    },
    {
        "id": "FLAG-ROCM-006",
        "kind": "flag",
        "flag": "--linear-attn-decode-backend",
        "values": ["flashinfer"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "ok"},
        "evidence": "S8",
        "note": "同 FLAG-ROCM-001（decode 侧）",
    },
    {
        "id": "FLAG-ROCM-002",
        "kind": "flag",
        "flag": "--linear-attn-prefill-backend",
        "values": ["flashinfer"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "ok"},
        "evidence": "S8",
        "note": "同 FLAG-ROCM-001（prefill 侧）",
    },
    {
        "id": "FLAG-ROCM-003",
        "kind": "flag",
        "flag": "--speculative-algorithm",
        "values": ["nextn"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "unknown"},
        "evidence": "S8",
        "note": "NEXTN/MTP 投机解码在 sglang ROCm 后端不可用（一手实测）",
    },
    {
        "id": "FLAG-ROCM-004",
        "kind": "flag",
        "flag": "--quantization",
        "values": ["auto-round"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "unknown"},
        "evidence": "S8",
        "note": "auto-round W4A16 反量化走 CUDA 路径（一手实测）",
    },
    {
        "id": "FLAG-ROCM-005",
        "kind": "flag",
        "flag": "--kv-cache-dtype",
        "values": ["fp8", "fp8_e5m2", "fp8_e4m3"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "unknown"},
        "evidence": "S8",
        "note": "fp8 KV cache 在 RDNA3 上不完整（一手实测）",
    },
    {
        "id": "FLAG-NVFP4-001",
        "kind": "flag",
        "flag": "--quantization",
        "values": ["nvfp4", "modelopt_fp4", "modelopt_fp4_per_block", "fp4"],
        "severity_hint": "blocker",
        "platforms": {"nvidia": "partial", "rocm": "blocked"},
        "evidence": "S8",
        "note": "NVFP4 是 Blackwell 级特性：非 Blackwell 的 NVIDIA 卡与 ROCm 目标均过不去（一手核查）。dcu 不写键——缺省即 unknown（查 HYGON-AI das 仓库），不给一手 ROCm 证据越权背书",
    },
    {
        "id": "FLAG-QUANT-001",
        "kind": "flag",
        "flag": "--quantization",
        "values": ["awq", "gptq", "gptq_marlin", "marlin", "bitsandbytes", "deepspeedfp"],
        "severity_hint": "blocker",
        "platforms": {"nvidia": "ok", "rocm": "blocked"},
        "evidence": "S10",
        "note": "rocm: vLLM 官方支持矩阵里这些量化格式的 AMD GPU 列全是 ❌（S10），不是配置问题而是没有对应内核；nvidia: 同一张表为 ✅。dcu 不写键——那张表的列是 AMD GPU，不是 DCU，不越权外推",
    },
    {
        "id": "FLAG-QUANT-002",
        "kind": "flag",
        "flag": "--quantization",
        "values": ["gguf"],
        "severity_hint": "info",
        "platforms": {"nvidia": "ok", "rocm": "ok"},
        "evidence": "S10",
        "note": "官方矩阵里 GGUF 在 AMD GPU 与各代 NVIDIA 均为 ✅（S10）",
    },
]

ENV_RULES = [
    {
        "id": "ENV-ROCM-001",
        "kind": "env",
        "key": "VLLM_ATTENTION_BACKEND",
        "value": "FLASH_ATTN",
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked"},
        "evidence": "S8",
        "note": "一手实测：该环境 flash-attn 2.8.3 缺 flash_attn_2_cuda 扩展，引擎初始化必崩，且表层报错是假线索（真异常在 EngineCore 子进程日志尾）",
    },
    {
        "id": "ENV-ROCM-002",
        "kind": "env",
        "key": "VLLM_ATTENTION_BACKEND",
        "value": "TRITON_ATTN",
        "severity_hint": "info",
        "platforms": {"rocm": "ok"},
        "evidence": "S8",
        "note": "一手实测：TRITON_ATTN 是该环境下可用的恢复路径",
    },
]

PACKAGE_RULES = [
    {
        "id": "PKG-VLLM-DAS-001",
        "kind": "package",
        "package": "vllm-plugin-das",
        "severity_hint": "info",
        "platforms": {"dcu": "ok", "rocm": "variant", "nvidia": "variant"},
        "evidence": "S13",
        "note": "dcu: 海光官方 vLLM 插件，是厂商给出的适配路径（S13），可用模型见 cookbook（S12）；rocm/nvidia: 面向 HCU/DTK，非目标平台 → variant",
    },
    {
        "id": "PKG-SGLANG-DAS-001",
        "kind": "package",
        "package": "sglang-das",
        "severity_hint": "info",
        "platforms": {"dcu": "ok", "rocm": "variant", "nvidia": "variant"},
        "evidence": "S12",
        "note": "dcu: 海光 sglang 下游适配仓，cookbook 按型号×模型给出 ✅（S12），且上游仍把 Hygon 列为 in progress（S11）→ 必须走下游仓；rocm/nvidia: variant",
    },
    {
        "id": "PKG-FLASHINFER-001",
        "kind": "package",
        "package": "flashinfer",
        "severity_hint": "blocker",
        "platforms": {"nvidia": "ok", "rocm": "blocked"},
        "evidence": "S1",
        "note": "上游只有 CUDA 分发；rocm: blocked（上游无 ROCm 轮子 + 一手实测相关 backend 不可用）",
    },
    {
        "id": "PKG-FLASHATTN-001",
        "kind": "package",
        "package": "flash-attn",
        "severity_hint": "warning",
        "platforms": {"nvidia": "ok", "rocm": "variant"},
        "evidence": "S2",
        "note": "上游声明支持 ROCm 工具链（S2），另有 AMD 官方移植仓库 rocm/flash-attention（S3）；但 wheel 名称/安装路径与 CUDA 版不同，属 variant 不是直接可用",
    },
    {
        "id": "PKG-VLLM-001",
        "kind": "package",
        "package": "vllm",
        "severity_hint": "info",
        "platforms": {"rocm": "ok", "dcu": "partial"},
        "evidence": "S6",
        "note": "rocm: 上游支持 AMD GPU（HIP 路径）；dcu: partial——需海光官方插件 vllm-plugin-das（S7），可用性逐型号逐模型见 S12/S13，裸上游 wheel 不能假设可用",
    },
    {
        "id": "PKG-SGLANG-001",
        "kind": "package",
        "package": "sglang",
        "severity_hint": "info",
        "platforms": {"rocm": "ok", "dcu": "partial"},
        "evidence": "S4",
        "note": "rocm: 上游平台表含 AMD Instinct；dcu: partial——上游把 Hygon 列在 Integrations in progress（S11），须用 sglang-das，验证状态见 S12",
    },
    {
        "id": "PKG-MEGATRON-001",
        "kind": "package",
        "package": "megatron",
        "severity_hint": "warning",
        "platforms": {"dcu": "partial"},
        "evidence": "S7",
        "note": "dcu: 海光维护 Megatron-LM-das 适配仓库（S7）；rocm/nvidia 状态未收录 → unknown",
    },
    {
        "id": "PKG-APEX-001",
        "kind": "package",
        "package": "apex",
        "severity_hint": "warning",
        "platforms": {"dcu": "partial"},
        "evidence": "S7",
        "note": "dcu: 海光维护 apex-das（S7，自述含 HCU/HIP Kernel 融合优化）；其余平台未收录 → unknown",
    },
    {
        "id": "PKG-TRITON-001",
        "kind": "package",
        "package": "triton",
        "severity_hint": "info",
        "platforms": {"nvidia": "ok", "rocm": "ok"},
        "evidence": "S5",
        "note": "Triton 有 NVPTX 与 AMDGPU 双后端",
    },
]

# 所有规则的总列表（完整性测试用）
ALL_RULES = FLAG_RULES + ENV_RULES + PACKAGE_RULES + [
    {
        "id": "PKG-TORCHWHEEL-001",
        "kind": "torch-wheel",
        "severity_hint": "blocker",
        "platforms": {"nvidia": "ok", "rocm": "partial", "dcu": "partial"},
        "evidence": "S9",
        "note": "torch 本地版本后缀决定 wheel 属于哪个平台：+cuXXX 装到 ROCm/DCU 目标 = 装了也跑不了",
    },
]


# ---------------------------------------------------------------------------
# 迁移建议（fix）：会判 blocker 的规则必须给出「改什么」——只说病不给药没帮上忙。
# 措辞严格守着证据：能指到实测恢复路径的就指（S8），只能「关掉这个选项」的就
# 直说关掉，绝不暗示某个未验证的替代方案一定可用。
# ---------------------------------------------------------------------------
FIXES = {
    "FLAG-ROCM-001": "去掉该参数回到栈默认 backend，或设 VLLM_ATTENTION_BACKEND=TRITON_ATTN"
                     "（后者是本项目一手实测确认可用的恢复路径，S8）",
    "FLAG-ROCM-002": "同上：去掉 --linear-attn-prefill-backend，或改用 TRITON_ATTN（S8）",
    "FLAG-ROCM-006": "同上：去掉 --linear-attn-decode-backend，或改用 TRITON_ATTN（S8）",
    "FLAG-ROCM-003": "关闭投机解码（去掉 --speculative-algorithm）；如需投机，先确认目标平台"
                     "确有对应实现再启用",
    "FLAG-ROCM-004": "不启用该量化（去掉 --quantization auto-round）；确需低比特时走厂商量化"
                     "工具链（DCU 见 S7）并自行验证精度",
    "FLAG-ROCM-005": "去掉 fp8 KV cache 回到默认精度；显存吃紧时先复核容量预算再谈量化，"
                     "别拿未验证的 fp8 换显存",
    "FLAG-NVFP4-001": "NVFP4 仅 Blackwell 可用；非 Blackwell 平台请改用其他量化并自行验证，"
                      "本项目不代为保证",
    "FLAG-QUANT-001": "换成官方矩阵里 AMD GPU 支持的格式（llm-compressor FP8 W8A8 或 GGUF，S10），"
                      "或先在目标型号上自验；本项目只做预检，不代为保证精度损失",
    "ENV-ROCM-001": "改用 VLLM_ATTENTION_BACKEND=TRITON_ATTN（S8 实测恢复路径）",
    "PKG-FLASHINFER-001": "从依赖里去掉 flashinfer，改用栈自带的 attention backend"
                          "（S1：上游无 ROCm 分发）",
    "PKG-FLASHATTN-001": "改用 AMD 官方移植仓 rocm/flash-attention 的安装方式（S3），或换回"
                         "栈自带 attention backend（S2 只是声明工具链支持，wheel 形态不同）",
    "PKG-VLLM-001": "DCU 上先装海光官方插件 vllm-plugin-das 并按其 cookbook 配置（S7）；"
                    "不要假设裸上游 wheel 可用",
    "PKG-SGLANG-001": "DCU 上使用 sglang-das 的适配分支/镜像（S7）",
    "PKG-MEGATRON-001": "DCU 上参考 Megatron-LM-das（S7）；版本与上游不一致时以厂商仓为准",
    "PKG-APEX-001": "DCU 上用 apex-das 替换上游 apex（S7）",
    "PKG-TORCHWHEEL-001": "按目标平台渠道重装：CUDA 用 download.pytorch.org/whl/cuXXX，"
                          "ROCm 用对应 rocm 渠道（S9）；两侧 wheel 不可混用",
}

for _rule in ALL_RULES:
    if _rule["id"] in FIXES:
        _rule["fix"] = FIXES[_rule["id"]]


def flag_rules():
    return [r for r in ALL_RULES if r["kind"] == "flag"]


def env_rules():
    return [r for r in ALL_RULES if r["kind"] == "env"]


def package_rules():
    return [r for r in ALL_RULES if r["kind"] == "package"]
