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
        "source": "https://github.com/flashinfer-ai/flashinfer README，2026-11-09 重读 raw main：CUDA Support 列出 12.9 / 13.0 / 13.4；GPU Support 表只有 NVIDIA SM75–SM121；安装名是 flashinfer-python，另有 flashinfer-cubin 与被写成 CUDA-specific shim 的 flashinfer-jit-cache。该次阅读没有看到 ROCm/HIP 行。",
        "note": "上游发行面是 CUDA/NVIDIA。本仓库没有在 ROCm 或 DCU 上重跑 FlashInfer。",
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
        "source": "一手实测记录（私有日志，未公开，不在本仓库）：AMD Radeon PRO W7900 (gfx1100, RDNA3), ROCm 7.2.1, vLLM 0.16.1.dev+rocm721",
        "scope": "仅此一台 gfx1100 / RDNA3 / ROCm 7.2.1 / vLLM 0.16.1.dev+rocm721。不是 MI300/MI350，不是其他 ROCm 版本，不是 DCU。原始日志不在仓库里，不能从本树复算。",
        "note": "该私有会话里：flashinfer 系 backend、NEXTN、auto-round、fp8 KV cache 不可用；VLLM_ATTENTION_BACKEND=TRITON_ATTN 可恢复。docs/notes-hygon-ecosystem.md 同一节还列出 NVFP4，但本条原始记录最初没写 NVFP4，日志又未入库，所以 NVFP4 行比重更低。不外推到其他 ROCm GPU，更不外推到 DCU。",
    },
    "S9": {
        "kind": "upstream-doc",
        "source": "https://pytorch.org/get-started/locally/ ，页面自述 Last updated 2026-10-28，本仓库于 2026-11-09 重读。Compute Platform 可见 CUDA 11.8 / 12.6 / 12.8、CPU，以及带删除线的 ROCm 6.3。页面给出的安装示例是 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118。",
        "note": "CUDA 与 CPU 是页面上分开的渠道；示例用 whl/cu118 索引，不是「装上 torch 就等于 CUDA 或 ROCm」。+cu / +rocm 本地版本标签是 2026-10-31 台账里已有的 PEP 440 约定，这次重读没有在页面正文里再看到这些后缀。ROCm 6.3 在页面上是删除线，不能把 +rocm 标签写成「当前仍在发行」。没有 DCU 渠道。",
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
        "platforms": {"rocm": "blocked", "nvidia": "ok", "dcu": "blocked"},
        "evidence": "S1",
        "platform_evidence": {"rocm": "S8", "nvidia": "S1", "dcu": "S1"},
        "note": "rocm: S8 仅一台 W7900 上该 backend 不可用（observed，不外推到其他 ROCm GPU）。nvidia: S1，上游把 FlashInfer 写成 CUDA 库，不是我们跑过 NVIDIA。dcu: S1，拒绝 CUDA-only 构件；没有 DCU 实测，不能写成在 DCU 上跑挂了。",
    },
    {
        "id": "FLAG-ROCM-006",
        "kind": "flag",
        "flag": "--linear-attn-decode-backend",
        "values": ["flashinfer"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "ok", "dcu": "blocked"},
        "evidence": "S1",
        "platform_evidence": {"rocm": "S8", "nvidia": "S1", "dcu": "S1"},
        "note": "同 FLAG-ROCM-001（decode 侧）。dcu 的 blocked 引用 S1，不引用 S8。",
    },
    {
        "id": "FLAG-ROCM-002",
        "kind": "flag",
        "flag": "--linear-attn-prefill-backend",
        "values": ["flashinfer"],
        "severity_hint": "blocker",
        "platforms": {"rocm": "blocked", "nvidia": "ok", "dcu": "blocked"},
        "evidence": "S1",
        "platform_evidence": {"rocm": "S8", "nvidia": "S1", "dcu": "S1"},
        "note": "同 FLAG-ROCM-001（prefill 侧）。dcu 的 blocked 引用 S1，不引用 S8。",
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
        "platforms": {"rocm": "blocked"},
        "evidence": "S8",
        "note": "rocm: docs/notes 把 NVFP4 列进同一段 W7900 私有会话；S8 原始 note 起初没写它，日志不在仓库内，权重大于猜测、小于可复算实测。nvidia 不写键：没有非 Blackwell NVIDIA 的单独记录，缺省 unknown。dcu 不写键。",
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
        "also": ["flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"],
        "severity_hint": "blocker",
        "platforms": {"nvidia": "ok", "rocm": "blocked", "dcu": "blocked"},
        "evidence": "S1",
        "note": "2026-11-09 README 的安装名是 flashinfer-python，GPU 表只有 NVIDIA。rocm/dcu: blocked 是拒绝这份 CUDA-only 发行物（S1），不是 DCU 或 Instinct 上的运行记录。nvidia: ok 只表示上游把自己写成 CUDA 库。",
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
        "note": "torch 的本地版本标签按渠道对照。运行时矩阵在 scan.TORCH_TAG_STATUS，以测试为准；+cpu 不是 CUDA 标签。dcu 上 +rocm 保持 unknown。",
        "status_is_computed": True,
    },
    {
        "id": "PKG-WHEELTAG-001",
        "kind": "torch-wheel",
        "severity_hint": "blocker",
        "platforms": {"nvidia": "blocked", "rocm": "blocked", "dcu": "blocked"},
        "evidence": "S9",
        "note": "非 torch 依赖上的 +cu / whl/cu / +rocm 标签。状态在扫描时按 scan.OTHER_TAG_STATUS 覆盖，静态 platforms 不是矩阵。dcu 上拒绝 CUDA 标签是渠道不匹配，不是 DCU 实测。",
        "status_is_computed": True,
    },
    {
        "id": "PKG-INDEXURL-001",
        "kind": "package",
        "severity_hint": "blocker",
        "platforms": {"nvidia": "ok", "rocm": "blocked", "dcu": "blocked"},
        "evidence": "S9",
        "note": "requirements 里的 --index-url / --extra-index-url / --find-links 指向 download.pytorch.org/whl/cuNNN。rocm/dcu 拒绝该 CUDA 渠道；nvidia 的 ok 只表示渠道名字对得上，不是装过。状态可被扫描覆盖。",
        "status_is_computed": True,
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
    "FLAG-NVFP4-001": "去掉该量化后再在目标机器上自验。本仓库没有非 Blackwell NVIDIA 的记录，"
                      "也没有 DCU 记录，不代为保证其他量化可用",
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
    "PKG-TORCHWHEEL-001": "按目标平台渠道重装：CUDA 用 download.pytorch.org/whl/cuXXX（S9 示例是 cu118）。"
                          "+rocm 只说明标签写了 rocm，不说明该版本仍在发行；不要把 CUDA 渠道装到 ROCm/DCU 目标上。",
    "PKG-WHEELTAG-001": "这条依赖的本地版本或 URL 带 CUDA 渠道标签（S9）。换目标平台的 wheel，或删掉这条；本工具没有在目标卡上装过它。",
    "PKG-INDEXURL-001": "删掉 whl/cuNNN 索引，改成目标平台的渠道（S9）。这条只认 URL 字符串，不代表已经在目标机上安装失败过。",
}

for _rule in ALL_RULES:
    if _rule["id"] in FIXES:
        _rule["fix"] = FIXES[_rule["id"]]


def evidence_id_for(rule, platform):
    """该平台实际引用的证据。缺省平台不沿用一手 ROCm 证据给 DCU 背书。"""
    per = rule.get("platform_evidence") or {}
    if platform in per:
        return per[platform]
    if platform in rule.get("platforms", {}):
        return rule["evidence"]
    if platform == "dcu":
        return "S7"
    return rule["evidence"]


def basis_for(evidence_id, status):
    """observed = 一手且状态不是 unknown；upstream-doc = 文档推断；unknown = 没有结论。"""
    if status == "unknown":
        return "unknown"
    kind = EVIDENCE[evidence_id]["kind"]
    if kind == "first-party":
        return "observed"
    return "upstream-doc"


def flag_rules():
    return [r for r in ALL_RULES if r["kind"] == "flag"]


def env_rules():
    return [r for r in ALL_RULES if r["kind"] == "env"]


def package_rules():
    return [r for r in ALL_RULES if r["kind"] == "package"]
