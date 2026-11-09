# -*- coding: utf-8 -*-
"""accel-preflight — 异构算力可移植性预检 CLI。

回答一个问题：这份 LLM 服务栈的启动配置 / 依赖，换到非 CUDA 平台
（ROCm / 国产 DCU 类硬件）上，哪些组件过不去？

设计原则（与 roofline-lint 同源）：
- 答不上来就说 unknown，绝不猜。
- 每条判定带证据指针，并用 basis 区分 observed / upstream-doc / unknown。
- observed 只来自已记录的一手会话；DCU 没有这种会话。
"""

__version__ = "0.4.0"
