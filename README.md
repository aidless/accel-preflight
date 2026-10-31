# accel-preflight

**在按下回车之前，先告诉你哪些 CUDA-only 路径在 ROCm / 国产 DCU 上过不去。**

异构算力可移植性预检 CLI。输入一段 `vllm serve` / `sglang.launch_server`
启动命令或一份 `requirements.txt`，它按「证据台账」逐项检查：
哪些参数是 CUDA 专属、哪些依赖没有对应平台的轮子、哪些 torch wheel
装到目标机器上根本跑不了。

## 为什么需要它

把一份在 NVIDIA 上调通的部署方案原样搬到 AMD ROCm 或国产 GPGPU（如海光
DCU/HCU，软件栈 DTK 以 ROCm/HIP 为底座）上，失败模式非常固定：

- `--kv-cache-dtype fp8_e4m3`、`--mamba-backend flashinfer`、
  `--speculative-algorithm NEXTN`、`--quantization auto-round` 这类参数
  在 ROCm 目标上过不去（FlashInfer 上游只有 CUDA 分发）；
- NVFP4 量化是 Blackwell 级特性，非 Blackwell 的 NVIDIA 卡和 ROCm 目标都过不去；
- torch wheel 有平台后缀：`+cu128` 的包装到 ROCm 机器上，import 都过不了；
- 更糟的是**表层报错是假线索**：`Failed core proc(s): {}` 看起来什么都没发生，
  真异常埋在 EngineCore 子进程日志的尾部。

这些结论不是猜的——每条规则都挂在 `db.py` 的证据台账上（见
[docs/sources.md](docs/sources.md)），一手实测证据只允许给 ROCm 平台背书，
DCU 相关的状态一律标 `unknown` 并指向厂商适配仓库，绝不外推。

## 快速开始

零依赖，Python ≥ 3.8：

```bash
python3 -m accel_preflight scan \
  --cmd "vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8_e4m3" \
  --platform rocm
```

输出（退出码 1，CI 应失败）：

```
accel-preflight · platform=rocm

BLOCKER (1):
  [FLAG-ROCM-005] --kv-cache-dtype fp8_e4m3  →  blocked  (证据 S8)
      fp8 KV cache 在 RDNA3 上不完整（一手实测）

summary: 1 blocker, 0 notice
```

也可以扫依赖文件与 pyproject：

```bash
python3 -m accel_preflight scan --requirements requirements.txt --platform dcu
python3 -m accel_preflight scan --pyproject pyproject.toml --platform rocm --json
```

### 退出码约定

| 退出码 | 含义 |
|-------|------|
| 0 | 无 blocker |
| 1 | 存在 blocked 项（CI 应失败） |
| 2 | 用法错误 |

### 自检（门必须能红，也要能绿）

```bash
python3 -m accel_preflight --prove-red
```

注入违规输入断言每条规则**真的会拦**，再注入干净输入断言**不误杀**。
任何一条不成立，自检退出 1。

## 方法论

1. **答案只有三态 + unknown**：`ok / variant / partial / blocked / unknown`。
   数据库没收录的包发 `PKG-UNKNOWN-001`，明说"不知道"，不猜。
2. **每条规则挂一条证据**（S0–S9）：上游 README 引用、厂商仓库、或一手实测。
3. **一手证据不越权**：只在 AMD Radeon 消费卡（gfx1100, ROCm 7.2.1）上实测过，
   所以 `kind=first-party` 的证据永远不给 `dcu` 平台写任何状态——完整性测试
   （`tests/test_db_integrity.py::test_first_party_never_claims_dcu`）强制这条。
4. **unknown 也有出路**：规则在 DCU 平台缺省为 `unknown` 时，证据指针自动转向
   海光官方组织 [HYGON-AI](https://github.com/orgs/HYGON-AI/repositories)
   的 `*-das` 适配仓库——那里才是权威答案。

## 覆盖范围（v0.1.0）

| 类别 | 规则 |
|------|------|
| 启动参数 | flashinfer 系 backend（mamba / linear-attn decode/prefill）、NEXTN 投机、auto-round 量化、fp8 KV cache、NVFP4 |
| 环境变量 | `VLLM_ATTENTION_BACKEND=FLASH_ATTN`（blocked）/ `=TRITON_ATTN`（ok） |
| 依赖 | flashinfer、flash-attn、vllm、sglang、megatron、apex、triton、torch wheel 后缀 |

规则库很小，而且**静态检查不等于跑通**：它回答"这条路大概率过不去/没证据"，
不回答"这条路一定能跑通"。删掉一个 blocker 之前，仍然要在目标机器上做一次
真实的冒烟验证。

## 相关项目

- [roofline-lint](https://github.com/aidless/roofline-lint) —— 算力够不够
  （roofline / KV cache / 精度分离），拒绝印不可运行的吞吐数；
- [paged-kv](https://github.com/aidless/paged-kv) —— PagedAttention 块管理器
  的 C++17 参考实现。

三个工具各管一段：**roofline-lint 管"装得下吗"，paged-kv 管内存怎么管，
accel-preflight 管"这份配置搬过去能跑吗"。**

## License

MIT © 2026 aidless
